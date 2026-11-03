from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urljoin, urlparse

import httpx

from ...config import Settings


class SourceExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedSource:
    url: str
    title: str
    text: str
    content_hash: str


def resolve_host(hostname: str) -> set[str]:
    try:
        return {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise SourceExtractionError(f"Cannot resolve source hostname: {hostname}") from exc


def _is_public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def validate_public_url(url: str, *, resolver=resolve_host) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SourceExtractionError("Source URL must use http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SourceExtractionError("Source URL has an invalid hostname or credentials")
    try:
        addresses = resolver(parsed.hostname)
    except ValueError as exc:
        raise SourceExtractionError("Source URL has an invalid hostname") from exc
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise SourceExtractionError("Source URL resolves to a private or unsafe address")
    return url


def _validate_connected_peer(response: httpx.Response) -> None:
    stream = response.extensions.get("network_stream")
    get_extra_info = getattr(stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return
    server_address = get_extra_info("server_addr")
    if (
        isinstance(server_address, tuple)
        and server_address
        and isinstance(server_address[0], str)
        and not _is_public_ip(server_address[0])
    ):
        raise SourceExtractionError("Source connection reached a private or unsafe address")


class _ReadableHTMLParser(HTMLParser):
    block_tags: ClassVar[set[str]] = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "main",
        "p",
        "section",
    }
    ignored_tags: ClassVar[set[str]] = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags:
            self.ignored_depth += 1
        if tag == "title":
            self.in_title = True
        if not self.ignored_depth and tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        if tag in self.ignored_tags and self.ignored_depth:
            self.ignored_depth -= 1
        if not self.ignored_depth and tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if not self.ignored_depth and not self.in_title:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)

    def result(self) -> tuple[str, str]:
        title = " ".join(" ".join(self.title_parts).split())
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        text = "\n".join(line for line in lines if len(line) >= 2)
        return title, text


def extract_readable_text(html: str, url: str) -> tuple[str, str]:
    try:
        import trafilatura
    except ImportError:
        trafilatura = None

    parser = _ReadableHTMLParser()
    parser.feed(html)
    fallback_title, fallback_text = parser.result()
    text = ""
    title = fallback_title
    if trafilatura is not None:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        ) or ""
        try:
            metadata = trafilatura.extract_metadata(html, default_url=url)
        except (AttributeError, TypeError, ValueError):
            metadata = None
        if metadata and metadata.title:
            title = str(metadata.title).strip()
    text = text.strip() or fallback_text.strip()
    if len(text) < 80:
        raise SourceExtractionError("Could not extract enough readable article text")
    return title[:500] or urlparse(url).hostname or "Untitled source", text


def fetch_article(url: str, settings: Settings) -> ExtractedSource:
    current_url = validate_public_url(url)
    headers = {"User-Agent": settings.web_image_user_agent}
    content = b""
    with httpx.Client(
        headers=headers,
        timeout=settings.video_source_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for _ in range(6):
            validate_public_url(current_url)
            try:
                with client.stream("GET", current_url) as response:
                    _validate_connected_peer(response)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SourceExtractionError("Source redirect has no location")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "html" not in content_type and "text/" not in content_type:
                        raise SourceExtractionError(
                            f"Unsupported source content type: {content_type or 'unknown'}"
                        )
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > settings.video_max_source_bytes:
                            raise SourceExtractionError("Source article exceeds maximum size")
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    break
            except httpx.HTTPError as exc:
                raise SourceExtractionError(f"Failed to download source article: {exc}") from exc
        else:
            raise SourceExtractionError("Source URL redirected too many times")
    if not content:
        raise SourceExtractionError("Source article is empty")
    html = content.decode("utf-8", errors="replace")
    title, text = extract_readable_text(html, current_url)
    return ExtractedSource(
        url=current_url,
        title=title,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
