from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def srt_timestamp(milliseconds: int) -> str:
    milliseconds = max(0, milliseconds)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(
    scenes: Iterable,
    *,
    gap_ms: int = 0,
    time_scale: float = 1.0,
) -> str:
    blocks = []
    cursor = 0
    for index, scene in enumerate(scenes, start=1):
        duration = int(scene.duration_ms or 0)
        if duration <= 0:
            raise ValueError(f"Scene {index - 1} has no valid audio duration")
        end = cursor + duration
        text = scene.narration.strip()
        scaled_start = round(cursor * time_scale)
        scaled_end = round(end * time_scale)
        blocks.append(
            f"{index}\n{srt_timestamp(scaled_start)} --> "
            f"{srt_timestamp(scaled_end)}\n{text}"
        )
        cursor = end + gap_ms
    return "\n\n".join(blocks) + "\n"


def write_srt(
    path: Path,
    scenes: Iterable,
    *,
    gap_ms: int = 0,
    time_scale: float = 1.0,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_srt(scenes, gap_ms=gap_ms, time_scale=time_scale),
        encoding="utf-8",
    )
    return str(path)
