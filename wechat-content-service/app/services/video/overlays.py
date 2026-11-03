from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def _ass_timestamp(milliseconds: int) -> str:
    centiseconds = max(0, round(milliseconds / 10))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_text(value: object) -> str:
    return (
        " ".join(str(value or "").split())
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def build_overlay_ass(
    scenes: Iterable,
    *,
    width: int,
    height: int,
    scene_gap_ms: int = 0,
) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: OverlayLabel,Noto Sans CJK SC,{round(height * 0.038)},&H002F2F2F,&H002F2F2F,&H00F6F2E8,&H00F6F2E8,0,0,0,0,100,100,0,0,3,9,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    cursor = 0
    events: list[str] = []
    scene_list = list(scenes)
    for index, scene in enumerate(scene_list):
        duration = int(scene.duration_ms or 0)
        try:
            labels = json.loads(scene.overlay_labels_json or "[]")
        except json.JSONDecodeError:
            labels = []
        if isinstance(labels, list):
            for label in labels[:3]:
                if not isinstance(label, dict):
                    continue
                text = _ass_text(label.get("text"))
                if not text:
                    continue
                try:
                    x = min(0.9, max(0.1, float(label.get("x"))))
                    y = min(0.82, max(0.12, float(label.get("y"))))
                except (TypeError, ValueError):
                    continue
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_timestamp(cursor + 100)},"
                    f"{_ass_timestamp(cursor + duration)},"
                    "OverlayLabel,,0,0,0,,"
                    f"{{\\pos({round(width * x)},{round(height * y)})}}{text}"
                )
        cursor += duration
        if index < len(scene_list) - 1:
            cursor += scene_gap_ms
    return header + "\n".join(events) + ("\n" if events else "")


def write_overlay_ass(
    path: Path,
    scenes: Iterable,
    *,
    width: int,
    height: int,
    scene_gap_ms: int = 0,
) -> str | None:
    content = build_overlay_ass(
        scenes,
        width=width,
        height=height,
        scene_gap_ms=scene_gap_ms,
    )
    if "Dialogue:" not in content:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)
