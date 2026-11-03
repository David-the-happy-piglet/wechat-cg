#!/usr/bin/env python3
"""Generate one standalone motion clip with MiniMax text-to-video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.video.motion import MiniMaxMotionVideoProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, choices=(6, 10), default=6)
    parser.add_argument("--resolution", choices=("768P", "1080P"), default="1080P")
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        parser.error("--prompt-file is empty")
    provider = MiniMaxMotionVideoProvider(settings)
    result = provider.generate(
        prompt,
        args.output,
        duration_seconds=args.duration,
        resolution=args.resolution,
        progress=lambda value: print(value, flush=True),
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
