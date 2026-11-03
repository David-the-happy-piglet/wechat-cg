from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-4000:])


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def subtitle_filter(path: Path) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return (
        f"subtitles=filename='{escaped}':"
        "force_style='FontName=Noto Sans CJK SC,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,"
        "Outline=1,Shadow=0,MarginV=32,Alignment=2'"
    )


def drawtext(text_file: Path, *, size: int, y: int) -> str:
    escaped = str(text_file.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return (
        f"drawtext=fontfile={FONT}:textfile='{escaped}':"
        f"fontsize={size}:fontcolor=0x333333:x=(w-text_w)/2:y={y}"
    )


def compose_body(
    project_dir: Path,
    output_dir: Path,
    minimum_scene_seconds: float,
    scene_gap_ms: int,
) -> Path:
    script = json.loads((project_dir / "script.json").read_text(encoding="utf-8"))
    scenes = script["scenes"]
    durations: list[float] = []
    inputs: list[str] = []
    filters: list[str] = []
    concat_inputs: list[str] = []
    timeline = 0.0
    subtitle_blocks: list[str] = []

    for index, scene in enumerate(scenes):
        scene_dir = project_dir / "scenes" / f"{index:03d}"
        image_path = scene_dir / "image.jpg"
        audio_path = scene_dir / "narration.mp3"
        audio_duration = max(minimum_scene_seconds, probe_duration(audio_path))
        duration = audio_duration
        if index < len(scenes) - 1:
            duration += scene_gap_ms / 1000
        durations.append(duration)
        inputs.extend(
            [
                "-loop",
                "1",
                "-framerate",
                "30",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
            ]
        )
        video_input = index * 2
        audio_input = video_input + 1
        filters.append(
            f"[{video_input}:v]scale=1280:720:force_original_aspect_ratio=increase,"
            f"crop=1280:720,fps=30,trim=duration={duration:.3f},"
            f"setpts=PTS-STARTPTS,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{audio_input}:a]apad,atrim=duration={duration:.3f},aresample=48000,"
            f"aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.extend([f"[v{index}]", f"[a{index}]"])
        subtitle_blocks.append(
            f"{index + 1}\n{srt_timestamp(timeline)} --> "
            f"{srt_timestamp(timeline + audio_duration)}\n{scene['narration']}\n"
        )
        timeline += duration

    output_dir.mkdir(parents=True, exist_ok=True)
    subtitle_path = output_dir / "subtitles-paced.srt"
    subtitle_path.write_text("\n".join(subtitle_blocks), encoding="utf-8")
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(scenes)}:v=1:a=1[vbody][abody]"
    )
    filters.append(f"[vbody]{subtitle_filter(subtitle_path)}[vout]")
    body_path = output_dir / "body-paced.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[abody]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(body_path),
        ]
    )
    return body_path


def compose_final(
    intro: Path,
    body: Path,
    outro: Path,
    output_dir: Path,
    intro_title: str,
    intro_subtitle: str,
    outro_title: str,
    outro_subtitle: str,
) -> Path:
    intro_duration = probe_duration(intro)
    outro_duration = probe_duration(outro)
    intro_title_path = output_dir / "intro-title.txt"
    intro_subtitle_path = output_dir / "intro-subtitle.txt"
    outro_title_path = output_dir / "outro-title.txt"
    outro_subtitle_path = output_dir / "outro-subtitle.txt"
    intro_title_path.write_text(intro_title, encoding="utf-8")
    intro_subtitle_path.write_text(intro_subtitle, encoding="utf-8")
    outro_title_path.write_text(outro_title, encoding="utf-8")
    outro_subtitle_path.write_text(outro_subtitle, encoding="utf-8")

    filters = [
        (
            "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setpts=PTS-STARTPTS,"
            f"{drawtext(intro_title_path, size=64, y=105)},"
            f"{drawtext(intro_subtitle_path, size=28, y=195)},format=yuv420p[vi]"
        ),
        (
            f"anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=duration={intro_duration:.3f},asetpts=PTS-STARTPTS[ai]"
        ),
        (
            "[1:v]scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720,fps=30,setpts=PTS-STARTPTS,format=yuv420p[vb]"
        ),
        (
            "[1:a]aresample=48000,aformat=channel_layouts=stereo,"
            "asetpts=PTS-STARTPTS[ab]"
        ),
        (
            "[2:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setpts=PTS-STARTPTS,"
            f"{drawtext(outro_title_path, size=46, y=105)},"
            f"{drawtext(outro_subtitle_path, size=28, y=180)},format=yuv420p[vo]"
        ),
        (
            f"anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=duration={outro_duration:.3f},asetpts=PTS-STARTPTS[ao]"
        ),
        "[vi][ai][vb][ab][vo][ao]concat=n=3:v=1:a=1[v][a]",
    ]
    final_path = output_dir / "final-with-intro-outro.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(intro),
            "-i",
            str(body),
            "-i",
            str(outro),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(final_path),
        ]
    )
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--intro", type=Path, required=True)
    parser.add_argument("--outro", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-scene-seconds", type=float, default=0.0)
    parser.add_argument("--scene-gap-ms", type=int, default=120)
    parser.add_argument("--intro-title", default="阿德勒三部曲")
    parser.add_argument("--intro-subtitle", default="三本书，看懂关系、诉求与自卑")
    parser.add_argument("--outro-title", default="分清课题 · 看懂诉求 · 在合作中找到意义")
    parser.add_argument("--outro-subtitle", default="感谢观看")
    args = parser.parse_args()

    body = compose_body(
        args.project_dir,
        args.output_dir,
        args.minimum_scene_seconds,
        max(0, min(600, args.scene_gap_ms)),
    )
    final = compose_final(
        args.intro,
        body,
        args.outro,
        args.output_dir,
        args.intro_title,
        args.intro_subtitle,
        args.outro_title,
        args.outro_subtitle,
    )
    print(
        json.dumps(
            {
                "body": str(body),
                "body_duration_seconds": round(probe_duration(body), 3),
                "final": str(final),
                "final_duration_seconds": round(probe_duration(final), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
