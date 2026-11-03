from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from ...config import Settings
from ..images import FallbackWebImageProvider, ImageProvider, build_image_provider


class WebImageRatioProvider(ImageProvider):
    name = "web-ratio"

    def __init__(
        self,
        *,
        web_provider: ImageProvider,
        ai_provider: ImageProvider,
        web_image_ratio: float,
    ) -> None:
        self.web_first = FallbackWebImageProvider(web_provider, ai_provider)
        self.ai_first = FallbackWebImageProvider(ai_provider, web_provider)
        self.web_slots = min(20, max(18, int(web_image_ratio * 20 + 0.999999)))

    def generate(
        self,
        prompt: str,
        title: str,
        position: int,
        search_query: object | None = None,
    ) -> dict[str, str | None]:
        provider = self.web_first if position % 20 < self.web_slots else self.ai_first
        return provider.generate(prompt, title, position, search_query)


def build_video_image_provider(
    settings: Settings,
    *,
    use_web_images: bool = False,
    web_image_ratio: float = 0.95,
) -> ImageProvider:
    ai_provider = build_image_provider(replace(settings, image_source_mode="ai"))
    if not use_web_images:
        return ai_provider
    web_provider = build_image_provider(replace(settings, image_source_mode="web"))
    # Allocate at least the requested number of web-first slots in every 20 scenes.
    # Each slot still has a fallback so one unavailable image does not stop a video.
    return WebImageRatioProvider(
        web_provider=web_provider,
        ai_provider=ai_provider,
        web_image_ratio=web_image_ratio,
    )


def persist_video_image(
    generated: dict[str, str | None],
    output_dir: Path,
    *,
    width: int,
    height: int,
) -> str:
    source_path = generated.get("local_path")
    if not source_path:
        raise ValueError("Image provider did not persist a local image")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "image.jpg"
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        image.save(destination, format="JPEG", quality=92, optimize=True)
    return str(destination)
