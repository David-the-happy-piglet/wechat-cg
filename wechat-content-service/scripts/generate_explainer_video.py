#!/usr/bin/env python3
"""Create a backend video project from a topic or public article URL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import VideoProject
from app.serializers import video_project_dict
from app.services.video.pipeline import execute_video_project
from app.services.video.visual_profiles import (
    DEFAULT_VISUAL_PROFILE,
    VisualProfileError,
    load_visual_profile,
)


def plain_text_from_markdown(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"^Title:\s*.*$|^URL Source:\s*.*$|^Markdown Content:\s*$", "", value, flags=re.MULTILINE)
    value = re.sub(r"[*_`#>]", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--topic")
    source.add_argument("--url")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--title", help="Title used with --text-file")
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--resolution", choices=("720p", "1080p"), default="720p")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--voice", default=settings.speech_voice)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--visual-profile", default=DEFAULT_VISUAL_PROFILE)
    parser.add_argument(
        "--style",
        default="",
        help="Optional per-project style notes added after the visual profile",
    )
    parser.add_argument("--scene-duration", type=float, default=3.5)
    parser.add_argument("--speech-speed", type=float, default=1.2)
    parser.add_argument("--scene-gap-ms", type=int, default=120)
    parser.add_argument(
        "--no-trim-speech-silence",
        action="store_true",
        help="Keep provider leading and trailing silence in every narration clip",
    )
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument(
        "--enqueue-only",
        action="store_true",
        help="Create the project without running it in this process.",
    )
    args = parser.parse_args()
    if not 15 <= args.duration <= 600:
        parser.error("--duration must be between 15 and 600 seconds")
    if not 24 <= args.fps <= 60:
        parser.error("--fps must be between 24 and 60")
    if not 3.0 <= args.scene_duration <= 8.0:
        parser.error("--scene-duration must be between 3.0 and 8.0 seconds")
    if not 0.5 <= args.speech_speed <= 2.0:
        parser.error("--speech-speed must be between 0.5 and 2.0")
    if not 0 <= args.scene_gap_ms <= 600:
        parser.error("--scene-gap-ms must be between 0 and 600")
    try:
        load_visual_profile(args.visual_profile)
    except VisualProfileError as exc:
        parser.error(str(exc))
    source_text = None
    source_title = None
    if args.text_file:
        if not args.text_file.is_file():
            parser.error("--text-file does not exist")
        source_text = plain_text_from_markdown(args.text_file.read_text(encoding="utf-8"))
        source_title = args.title or args.text_file.stem
        if not source_text.strip():
            parser.error("--text-file is empty")

    width, height = (1280, 720) if args.resolution == "720p" else (1920, 1080)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        project = VideoProject(
            source_type="text" if args.text_file else ("topic" if args.topic else "url"),
            source_topic=args.topic,
            source_url=args.url,
            source_title=source_title,
            source_text=source_text,
            target_duration_seconds=args.duration,
            width=width,
            height=height,
            fps=args.fps,
            voice=args.voice,
            llm_provider=args.provider,
            subtitles_enabled=not args.no_subtitles,
            settings_json=json.dumps(
                {
                    "resolution": args.resolution,
                    "visual_profile": args.visual_profile,
                    "style": args.style,
                    "scene_duration_seconds": args.scene_duration,
                    "speech_speed": args.speech_speed,
                    "scene_gap_ms": args.scene_gap_ms,
                    "trim_speech_silence": not args.no_trim_speech_silence,
                },
                ensure_ascii=False,
            ),
            next_attempt_at=datetime.now(UTC),
        )
        session.add(project)
        session.commit()
        project_id = project.id
        if not args.enqueue_only:
            execute_video_project(session, project_id, settings)
        project = session.scalar(
            select(VideoProject)
            .where(VideoProject.id == project_id)
            .options(selectinload(VideoProject.scenes))
        )
        print(json.dumps(video_project_dict(project), ensure_ascii=False, indent=2, default=str))
        return 0 if project and project.status in {"pending", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
