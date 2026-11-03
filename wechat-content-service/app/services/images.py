import base64
import html
import io
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw

from ..config import Settings
from .video.source import SourceExtractionError, validate_public_url

logger = logging.getLogger(__name__)

MINIMAX_PROMPT_MAX_CHARS = 1500
WATERMARKED_STOCK_HOSTS = (
    "alamy.com",
    "dreamstime.com",
    "lovepik.com",
    "shutterstock.com",
    "gettyimages.com",
    "istockphoto.com",
    "depositphotos.com",
    "123rf.com",
    "bigstockphoto.com",
    "stock.adobe.com",
)


def is_stock_preview_url(url: str) -> bool:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == stock or host.endswith(f".{stock}") for stock in WATERMARKED_STOCK_HOSTS)


class ImageProviderError(RuntimeError):
    pass


class ImageProvider(ABC):
    name = "unknown"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        raise NotImplementedError


class PlaceholderImageProvider(ImageProvider):
    name = "placeholder"

    def __init__(self, settings: Settings):
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        filename = f"{uuid4().hex}.png"
        path = self.output_dir / filename
        colors = [
            ((232, 243, 235), (112, 169, 135)),
            ((255, 241, 220), (210, 158, 99)),
            ((232, 241, 250), (104, 151, 185)),
        ]
        background, accent = colors[position % len(colors)]
        image = Image.new("RGB", (1200, 630), background)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.ellipse((850, -120, 1250, 280), fill=(*accent, 85))
        draw.ellipse((-150, 420, 280, 850), fill=(255, 255, 255, 105))
        draw.rounded_rectangle(
            (180, 175, 1020, 455),
            radius=42,
            fill=(255, 255, 255, 165),
            outline=(*accent, 100),
            width=4,
        )
        draw.line((320, 315, 880, 315), fill=(*accent, 150), width=12)
        image.save(path, format="PNG", optimize=True)
        return {
            "local_path": str(path),
            "source_url": f"{self.public_base_url}/generated-assets/{filename}",
            "source_kind": "placeholder",
        }


class CompatibleImageProvider(ImageProvider):
    name = "compatible"

    def __init__(self, settings: Settings):
        if not settings.image_api_base or not settings.image_api_key or not settings.image_model:
            raise ImageProviderError(
                "IMAGE_PROVIDER=compatible requires IMAGE_API_BASE, IMAGE_API_KEY and IMAGE_MODEL"
            )
        self.api_base = settings.image_api_base.rstrip("/")
        self.api_key = settings.image_api_key
        self.model = settings.image_model
        self.size = settings.image_size
        self.timeout = settings.image_timeout_seconds
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        response = httpx.post(
            f"{self.api_base}/images/generations",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "prompt": prompt,
                "size": self.size,
                "n": 1,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            item = response.json()["data"][0]
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ImageProviderError(f"Image request failed: {exc}") from exc

        if item.get("url"):
            try:
                image_response = httpx.get(item["url"], timeout=self.timeout)
                image_response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ImageProviderError(f"Failed to persist generated image: {exc}") from exc
            filename = f"{uuid4().hex}.png"
            path = self.output_dir / filename
            path.write_bytes(image_response.content)
            return {
                "local_path": str(path),
                "source_url": f"{self.public_base_url}/generated-assets/{filename}",
                "source_kind": "ai",
            }
        if item.get("b64_json"):
            filename = f"{uuid4().hex}.png"
            path = self.output_dir / filename
            path.write_bytes(base64.b64decode(item["b64_json"]))
            return {
                "local_path": str(path),
                "source_url": f"{self.public_base_url}/generated-assets/{filename}",
                "source_kind": "ai",
            }
        raise ImageProviderError("Image API returned neither url nor b64_json")


class MiniMaxImageProvider(ImageProvider):
    name = "minimax"

    def __init__(self, settings: Settings):
        if not settings.minimax_api_key:
            raise ImageProviderError(
                "MiniMax image generation is not configured. Fill MINIMAX_API_KEY in .env"
            )
        self.api_base = settings.minimax_api_base.rstrip("/")
        self.api_key = settings.minimax_api_key
        self.model = settings.minimax_image_model
        self.timeout = settings.image_timeout_seconds
        self.max_attempts = max(1, settings.provider_max_attempts)
        self.retry_base_seconds = max(0, settings.provider_retry_base_seconds)
        self.aigc_watermark = settings.aigc_watermark
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")

    @staticmethod
    def _composition_safe_prompt(prompt: str) -> str:
        # Art direction belongs to the selected visual profile. The provider
        # layer only enforces MiniMax's request-size limit.
        return str(prompt or "").strip()[:MINIMAX_PROMPT_MAX_CHARS].rstrip()

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        payload = {
            "model": self.model,
            "prompt": self._composition_safe_prompt(prompt),
            "aspect_ratio": "16:9",
            "response_format": "base64",
            "n": 1,
            "prompt_optimizer": False,
            "aigc_watermark": self.aigc_watermark,
        }
        response: httpx.Response | None = None
        try:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = httpx.post(
                        f"{self.api_base}/image_generation",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self.timeout,
                    )
                    break
                except httpx.TransportError as exc:
                    if attempt >= self.max_attempts:
                        raise
                    delay = self.retry_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "MiniMax image transport error on attempt %s/%s; "
                        "retrying in %.1fs: %s",
                        attempt,
                        self.max_attempts,
                        delay,
                        exc,
                    )
                    if delay:
                        time.sleep(delay)
            if response is None:  # pragma: no cover - defensive guard
                raise ImageProviderError("MiniMax image request produced no response")
            response.raise_for_status()
            body = response.json()
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" — {exc.response.text[:500]}"
            raise ImageProviderError(f"MiniMax image request failed: {exc}{detail}") from exc

        if not isinstance(body, dict):
            raise ImageProviderError("MiniMax image request returned a non-object response")

        base_response = body.get("base_resp")
        status_code = base_response.get("status_code") if isinstance(base_response, dict) else None
        status_message = (
            str(base_response.get("status_msg") or "")[:300]
            if isinstance(base_response, dict)
            else ""
        )
        if status_code not in (None, 0, "0"):
            raise ImageProviderError(
                "MiniMax image request failed"
                f" (status_code={status_code}, status_msg={status_message or 'unknown'})"
            )

        data = body.get("data")
        encoded_images = data.get("image_base64") if isinstance(data, dict) else None
        if (
            not isinstance(encoded_images, list)
            or not encoded_images
            or not isinstance(encoded_images[0], str)
            or not encoded_images[0]
        ):
            metadata = body.get("metadata")
            failed_count = metadata.get("failed_count") if isinstance(metadata, dict) else None
            details = []
            if status_code is not None:
                details.append(f"status_code={status_code}")
            if status_message:
                details.append(f"status_msg={status_message}")
            if failed_count is not None:
                details.append(f"failed_count={failed_count}")
            suffix = f" ({', '.join(details)})" if details else ""
            raise ImageProviderError(f"MiniMax image request returned no image data{suffix}")

        try:
            image_bytes = base64.b64decode(encoded_images[0], validate=True)
        except (ValueError, TypeError) as exc:
            raise ImageProviderError(
                "MiniMax image request returned invalid base64 image data"
            ) from exc

        filename = f"{uuid4().hex}.jpg"
        path = self.output_dir / filename
        path.write_bytes(image_bytes)
        return {
            "local_path": str(path),
            "source_url": f"{self.public_base_url}/generated-assets/{filename}",
            "source_kind": "ai",
        }


def _plain_metadata(value: str | None) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split())


def normalize_search_query(value: object | None, fallback: str) -> str:
    fallback_query = str(fallback or "").strip()
    if isinstance(value, str):
        return value.strip() or fallback_query
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, str):
                part = item.strip()
            elif isinstance(item, (int, float)):
                part = str(item)
            else:
                continue
            if part and part not in parts:
                parts.append(part)
        return " ".join(parts) or fallback_query
    return fallback_query


class DuckDuckGoImageProvider(ImageProvider):
    """Search current web images for editorial news use.

    Unlike Openverse and Wikimedia, these results are not guaranteed to have an
    open license. The source page and an explicit review-required license label
    are persisted with every downloaded image.
    """

    name = "duckduckgo"

    def __init__(self, settings: Settings):
        self.base_url = settings.duckduckgo_image_base_url.rstrip("/")
        self.timeout = settings.image_timeout_seconds
        self.max_bytes = settings.web_image_max_bytes
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")
        self.headers = {
            "User-Agent": settings.web_image_user_agent,
            "Referer": f"{self.base_url}/",
        }

    def _search(self, query: str) -> list[dict[str, str]]:
        try:
            landing = httpx.get(
                f"{self.base_url}/",
                params={"q": query[:200]},
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout,
            )
            landing.raise_for_status()
            match = re.search(r'vqd=["\']([^"\']+)', landing.text)
            if not match:
                raise ImageProviderError("DuckDuckGo image search returned no token")
            response = httpx.get(
                f"{self.base_url}/i.js",
                params={
                    "l": "wt-wt",
                    "o": "json",
                    "q": query[:200],
                    "vqd": match.group(1),
                    "f": ",,,",
                    "p": "1",
                },
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError) as exc:
            raise ImageProviderError(f"DuckDuckGo image search failed: {exc}") from exc

        candidates: list[dict[str, str]] = []
        for item in results:
            image_url = str(item.get("image") or "")
            source_page_url = str(item.get("url") or "")
            try:
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            if not image_url or not source_page_url or width < 640 or height < 360:
                continue
            if is_stock_preview_url(image_url) or is_stock_preview_url(source_page_url):
                continue
            candidates.append(
                {
                    "download_url": image_url,
                    "source_page_url": source_page_url,
                    "source_author": str(item.get("source") or "Web publisher"),
                }
            )
            if len(candidates) >= 16:
                break
        return candidates

    def _download(
        self,
        candidate: dict[str, str],
        position: int,
    ) -> dict[str, str | None]:
        try:
            validate_public_url(candidate["download_url"])
            response = httpx.get(
                candidate["download_url"],
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            validate_public_url(str(response.url))
        except (httpx.HTTPError, SourceExtractionError) as exc:
            raise ImageProviderError(f"Web editorial image download failed: {exc}") from exc
        content = response.content
        if not content or len(content) > self.max_bytes:
            raise ImageProviderError(
                "Web editorial image is empty or exceeds WEB_IMAGE_MAX_BYTES"
            )
        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                image.verify()
        except (OSError, ValueError) as exc:
            raise ImageProviderError("Web search returned an invalid image") from exc
        suffix = ".png" if image_format == "PNG" else ".jpg"
        filename = f"{uuid4().hex}-{position}{suffix}"
        path = self.output_dir / filename
        path.write_bytes(content)
        return {
            "local_path": str(path),
            "source_url": f"{self.public_base_url}/generated-assets/{filename}",
            "source_kind": "web",
            "source_page_url": candidate["source_page_url"],
            "source_author": candidate["source_author"],
            "license_name": "版权状态待确认（仅供新闻编辑审阅）",
            "license_url": "",
        }

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        query = normalize_search_query(search_query, title)
        candidates = self._search(query)
        if not candidates and query != title:
            candidates = self._search(title)
        if not candidates:
            raise ImageProviderError(f"No editorial web image found for query: {query}")
        last_error: ImageProviderError | None = None
        for candidate in candidates:
            try:
                return self._download(candidate, position)
            except ImageProviderError as exc:
                last_error = exc
        raise last_error or ImageProviderError("No editorial web image could be downloaded")


class WikimediaImageProvider(ImageProvider):
    name = "wikimedia"
    allowed_mime_types: ClassVar[dict[str, str]] = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
    }

    def __init__(self, settings: Settings):
        self.api_url = settings.wikimedia_api_url
        self.timeout = settings.image_timeout_seconds
        self.max_bytes = settings.web_image_max_bytes
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")
        self.headers = {"User-Agent": settings.web_image_user_agent}

    @staticmethod
    def _metadata_value(metadata: dict, key: str) -> str:
        item = metadata.get(key, {})
        return str(item.get("value", "")) if isinstance(item, dict) else ""

    @staticmethod
    def _license_is_supported(license_name: str) -> bool:
        normalized = license_name.lower().replace("_", " ")
        return any(
            marker in normalized for marker in ("cc by", "cc0", "public domain", "pdm")
        ) and not any(
            marker in normalized for marker in ("noncommercial", "non-commercial", "no derivatives")
        )

    def _search(self, query: str) -> list[dict[str, str]]:
        response = httpx.get(
            self.api_url,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": 12,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": 1600,
                "format": "json",
                "formatversion": 2,
            },
            headers=self.headers,
            follow_redirects=True,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", [])
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError) as exc:
            raise ImageProviderError(f"Wikimedia image search failed: {exc}") from exc

        candidates: list[dict[str, str]] = []
        for page in pages:
            image_info = (page.get("imageinfo") or [{}])[0]
            mime = image_info.get("mime", "")
            metadata = image_info.get("extmetadata") or {}
            license_name = _plain_metadata(
                self._metadata_value(metadata, "LicenseShortName")
                or self._metadata_value(metadata, "UsageTerms")
            )
            if mime not in self.allowed_mime_types or not self._license_is_supported(license_name):
                continue
            download_url = image_info.get("thumburl") or image_info.get("url")
            if not download_url:
                continue
            candidates.append(
                {
                    "download_url": str(download_url),
                    "mime": str(mime),
                    "source_page_url": str(
                        image_info.get("descriptionurl")
                        or image_info.get("descriptionshorturl")
                        or ""
                    ),
                    "source_author": _plain_metadata(
                        self._metadata_value(metadata, "Artist")
                        or self._metadata_value(metadata, "Credit")
                    ),
                    "license_name": license_name,
                    "license_url": _plain_metadata(self._metadata_value(metadata, "LicenseUrl")),
                }
            )
        return candidates

    def _download(self, candidate: dict[str, str], position: int) -> dict[str, str | None]:
        try:
            response = httpx.get(
                candidate["download_url"],
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageProviderError(f"Wikimedia image download failed: {exc}") from exc
        content = response.content
        if not content or len(content) > self.max_bytes:
            raise ImageProviderError("Wikimedia image is empty or exceeds WEB_IMAGE_MAX_BYTES")
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ImageProviderError("Wikimedia returned an invalid image") from exc

        suffix = self.allowed_mime_types[candidate["mime"]]
        filename = f"{uuid4().hex}-{position}{suffix}"
        path = self.output_dir / filename
        path.write_bytes(content)
        return {
            "local_path": str(path),
            "source_url": f"{self.public_base_url}/generated-assets/{filename}",
            "source_kind": "web",
            "source_page_url": candidate["source_page_url"],
            "source_author": candidate["source_author"] or "Wikimedia Commons contributor",
            "license_name": candidate["license_name"],
            "license_url": candidate["license_url"],
        }

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        query = normalize_search_query(search_query, title)
        candidates = self._search(query)
        if not candidates and query != title:
            candidates = self._search(title)
        if not candidates:
            raise ImageProviderError(f"No reusable Wikimedia image found for query: {query}")
        last_error: ImageProviderError | None = None
        for candidate in candidates:
            try:
                return self._download(candidate, position)
            except ImageProviderError as exc:
                last_error = exc
        raise last_error or ImageProviderError("No Wikimedia image could be downloaded")


class OpenverseImageProvider(ImageProvider):
    name = "openverse"

    def __init__(self, settings: Settings):
        self.api_url = settings.openverse_api_url.rstrip("/")
        self.timeout = settings.image_timeout_seconds
        self.max_bytes = settings.web_image_max_bytes
        self.output_dir = Path(settings.generated_assets_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = settings.public_base_url.rstrip("/")
        self.headers = {"User-Agent": settings.web_image_user_agent}

    @staticmethod
    def _license_name(item: dict) -> str:
        license_code = str(item.get("license") or "").lower()
        license_version = str(item.get("license_version") or "").strip()
        names = {
            "cc0": "CC0",
            "by": "CC BY",
            "by-sa": "CC BY-SA",
            "pdm": "Public Domain Mark",
        }
        name = names.get(license_code, license_code.upper())
        return f"{name} {license_version}".strip()

    def _search(self, query: str) -> list[dict[str, str]]:
        response = httpx.get(
            f"{self.api_url}/images/",
            params={
                "q": query[:200],
                "license": "cc0,by,by-sa,pdm",
                "page_size": 12,
                "category": "photograph",
                "aspect_ratio": "wide",
                "mature": "false",
            },
            headers=self.headers,
            follow_redirects=True,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError) as exc:
            raise ImageProviderError(f"Openverse image search failed: {exc}") from exc

        candidates = []
        for item in results:
            download_url = item.get("thumbnail")
            source_page_url = (
                item.get("foreign_landing_url")
                or item.get("detail_url")
                or item.get("url")
            )
            license_url = item.get("license_url")
            if not download_url or not source_page_url or not license_url:
                continue
            candidates.append(
                {
                    "download_url": str(download_url),
                    "source_page_url": str(source_page_url),
                    "source_author": _plain_metadata(
                        str(item.get("creator") or "Openverse contributor")
                    ),
                    "license_name": self._license_name(item),
                    "license_url": str(license_url),
                }
            )
        return candidates

    def _download(
        self,
        candidate: dict[str, str],
        position: int,
    ) -> dict[str, str | None]:
        try:
            response = httpx.get(
                candidate["download_url"],
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageProviderError(f"Openverse image download failed: {exc}") from exc
        content = response.content
        if not content or len(content) > self.max_bytes:
            raise ImageProviderError(
                "Openverse image is empty or exceeds WEB_IMAGE_MAX_BYTES"
            )
        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                image.verify()
        except (OSError, ValueError) as exc:
            raise ImageProviderError("Openverse returned an invalid image") from exc
        suffix = ".png" if image_format == "PNG" else ".jpg"
        filename = f"{uuid4().hex}-{position}{suffix}"
        path = self.output_dir / filename
        path.write_bytes(content)
        return {
            "local_path": str(path),
            "source_url": f"{self.public_base_url}/generated-assets/{filename}",
            "source_kind": "web",
            "source_page_url": candidate["source_page_url"],
            "source_author": candidate["source_author"],
            "license_name": candidate["license_name"],
            "license_url": candidate["license_url"],
        }

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        query = normalize_search_query(search_query, title)
        candidates = self._search(query)
        if not candidates and query != title:
            candidates = self._search(title)
        if not candidates:
            raise ImageProviderError(
                f"No reusable Openverse image found for query: {query}"
            )
        last_error: ImageProviderError | None = None
        for candidate in candidates:
            try:
                return self._download(candidate, position)
            except ImageProviderError as exc:
                last_error = exc
        raise last_error or ImageProviderError("No Openverse image could be downloaded")


class FallbackWebImageProvider(ImageProvider):
    name = "web-fallback"

    def __init__(self, *providers: ImageProvider):
        self.providers = providers

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        last_error: ImageProviderError | None = None
        for provider in self.providers:
            try:
                return provider.generate(prompt, title, position, search_query)
            except ImageProviderError as exc:
                last_error = exc
        raise last_error or ImageProviderError("All web image providers failed")


class MixedImageProvider(ImageProvider):
    name = "mixed"

    def __init__(
        self,
        ai_provider: ImageProvider,
        web_provider: ImageProvider,
        fallback_to_ai: bool,
    ):
        self.ai_provider = ai_provider
        self.web_provider = web_provider
        self.fallback_to_ai = fallback_to_ai

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        # A web search fallback can return a watermarked stock preview. If the
        # cover cannot be generated, let the job retry instead of uploading it.
        if position == 0:
            return self.ai_provider.generate(prompt, title, position, search_query)
        # Use searched images for alternating inline positions.
        if position % 2 == 0:
            try:
                return self.ai_provider.generate(
                    prompt,
                    title,
                    position,
                    search_query,
                )
            except ImageProviderError as exc:
                logger.warning(
                    "AI image generation failed at position %s; using web image: %s",
                    position,
                    exc,
                )
                return self.web_provider.generate(
                    prompt,
                    title,
                    position,
                    search_query,
                )
        try:
            return self.web_provider.generate(prompt, title, position, search_query)
        except ImageProviderError as exc:
            if not self.fallback_to_ai:
                raise
            logger.warning(
                "Web image generation failed at position %s; using AI image: %s",
                position,
                exc,
            )
            return self.ai_provider.generate(prompt, title, position, search_query)


def _build_ai_provider(settings: Settings) -> ImageProvider:
    if settings.image_provider == "placeholder":
        return PlaceholderImageProvider(settings)
    if settings.image_provider == "compatible":
        return CompatibleImageProvider(settings)
    if settings.image_provider == "minimax":
        return MiniMaxImageProvider(settings)
    raise ImageProviderError(f"Unsupported IMAGE_PROVIDER: {settings.image_provider}")


def _build_web_provider(settings: Settings) -> ImageProvider:
    if settings.web_image_provider == "openverse":
        return FallbackWebImageProvider(
            OpenverseImageProvider(settings),
            WikimediaImageProvider(settings),
            DuckDuckGoImageProvider(settings),
        )
    if settings.web_image_provider == "wikimedia":
        return WikimediaImageProvider(settings)
    if settings.web_image_provider == "duckduckgo":
        return DuckDuckGoImageProvider(settings)
    raise ImageProviderError(f"Unsupported WEB_IMAGE_PROVIDER: {settings.web_image_provider}")


def build_image_provider(settings: Settings) -> ImageProvider:
    ai_provider = _build_ai_provider(settings)
    if settings.image_source_mode == "ai":
        return ai_provider
    web_provider = _build_web_provider(settings)
    if settings.image_source_mode == "web":
        return web_provider
    if settings.image_source_mode == "mixed":
        return MixedImageProvider(
            ai_provider,
            web_provider,
            settings.web_image_fallback_to_ai,
        )
    raise ImageProviderError(f"Unsupported IMAGE_SOURCE_MODE: {settings.image_source_mode}")
