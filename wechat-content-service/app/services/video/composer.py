from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ...config import Settings


class VideoCompositionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComposedVideo:
    path: str
    duration_ms: int
    width: int
    height: int


def _tool_path(configured: str) -> str:
    resolved = shutil.which(configured)
    if not resolved:
        raise VideoCompositionError(
            f"Required media tool is not installed or not executable: {configured}"
        )
    return resolved


def _concat_quote(path: str) -> str:
    return path.replace("'", "'\\''")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout)[-2000:]
        raise VideoCompositionError(f"FFmpeg failed with exit code {result.returncode}: {detail}")


def probe_video(path: Path, settings: Settings) -> ComposedVideo:
    ffprobe = _tool_path(settings.video_ffprobe_path)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise VideoCompositionError(f"ffprobe failed: {result.stderr[-1000:]}")
    try:
        body = json.loads(result.stdout)
        video_stream = next(
            item for item in body["streams"] if item.get("codec_type") == "video"
        )
        audio_stream = next(
            item for item in body["streams"] if item.get("codec_type") == "audio"
        )
        duration = float(body["format"]["duration"])
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoCompositionError("Generated file is missing valid video/audio metadata") from exc
    if not audio_stream:
        raise VideoCompositionError("Generated file has no audio stream")
    return ComposedVideo(
        path=str(path),
        duration_ms=round(duration * 1000),
        width=width,
        height=height,
    )


def _probe_duration(path: Path, settings: Settings) -> float:
    ffprobe = _tool_path(settings.video_ffprobe_path)
    result = subprocess.run(
        [
            ffprobe,
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
        check=False,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise VideoCompositionError(f"Could not read media duration: {path}") from exc
    if result.returncode or duration <= 0:
        raise VideoCompositionError(f"Could not read media duration: {path}")
    return duration


def _drawtext_filter(text_path: Path, settings: Settings, *, size: int, y: int) -> str:
    font_path = Path(settings.video_font_path).resolve()
    if not font_path.is_file():
        raise VideoCompositionError(f"Video title font was not found: {font_path}")
    escaped_text = str(text_path.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    escaped_text = escaped_text.replace("'", "\\'")
    escaped_font = str(font_path).replace("\\", "\\\\").replace(":", "\\:")
    escaped_font = escaped_font.replace("'", "\\'")
    return (
        f"drawtext=fontfile='{escaped_font}':textfile='{escaped_text}':"
        f"fontsize={size}:fontcolor=0x333333:x=(w-text_w)/2:y={y}"
    )


def compose_bookended_video(
    intro_path: Path,
    body: ComposedVideo,
    outro_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    title: str,
    fps: int,
) -> ComposedVideo:
    if not intro_path.is_file() or not outro_path.is_file():
        raise VideoCompositionError("Configured intro or outro video was not found")
    output_dir.mkdir(parents=True, exist_ok=True)
    intro_duration = _probe_duration(intro_path, settings)
    outro_duration = _probe_duration(outro_path, settings)
    width, height = body.width, body.height
    title_size = round(58 * width / 1280)
    outro_size = round(46 * width / 1280)
    title_y = round(105 * height / 720)
    intro_title_path = output_dir / "intro-title.txt"
    outro_title_path = output_dir / "outro-title.txt"
    intro_title_path.write_text(title[:80], encoding="utf-8")
    outro_title_path.write_text("感谢观看", encoding="utf-8")
    filters = [
        (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},setpts=PTS-STARTPTS,"
            f"{_drawtext_filter(intro_title_path, settings, size=title_size, y=title_y)},"
            "format=yuv420p[vi]"
        ),
        (
            "anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=duration={intro_duration:.3f},asetpts=PTS-STARTPTS[ai]"
        ),
        (
            f"[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},setpts=PTS-STARTPTS,format=yuv420p[vb]"
        ),
        "[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[ab]",
        (
            f"[2:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},setpts=PTS-STARTPTS,"
            f"{_drawtext_filter(outro_title_path, settings, size=outro_size, y=title_y)},"
            "format=yuv420p[vo]"
        ),
        (
            "anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=duration={outro_duration:.3f},asetpts=PTS-STARTPTS[ao]"
        ),
        "[vi][ai][vb][ab][vo][ao]concat=n=3:v=1:a=1[v][a]",
    ]
    output_path = output_dir / "final-with-intro-outro.mp4"
    ffmpeg = _tool_path(settings.video_ffmpeg_path)
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(intro_path),
            "-i",
            body.path,
            "-i",
            str(outro_path),
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
            str(output_path),
        ]
    )
    return probe_video(output_path, settings)


def _subtitle_filter(path: Path) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return (
        f"subtitles=filename='{escaped}':"
        "force_style='FontName=Noto Sans CJK SC,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=1,"
        "Shadow=0,MarginV=32,Alignment=2'"
    )


def _ass_overlay_filter(path: Path) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return f"subtitles=filename='{escaped}'"


def _cap_duration(
    result: ComposedVideo,
    settings: Settings,
    *,
    target_duration_seconds: int | None,
    fps: int,
) -> ComposedVideo:
    if not target_duration_seconds:
        return result
    target_ms = target_duration_seconds * 1000
    if result.duration_ms <= round(target_ms * 1.03):
        return result
    playback_rate = result.duration_ms / target_ms
    if playback_rate > 2.0:
        raise VideoCompositionError(
            "Generated narration is more than twice the requested duration"
        )
    ffmpeg = _tool_path(settings.video_ffmpeg_path)
    output_path = Path(result.path)
    retimed_path = output_path.with_name(f"{output_path.stem}.retimed{output_path.suffix}")
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(output_path),
            "-filter_complex",
            (
                f"[0:v]setpts=PTS/{playback_rate:.8f},fps={fps}[v];"
                f"[0:a]atempo={playback_rate:.8f}[a]"
            ),
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
            str(retimed_path),
        ]
    )
    retimed_path.replace(output_path)
    retimed = probe_video(output_path, settings)
    if abs(retimed.duration_ms - target_ms) > 500:
        raise VideoCompositionError(
            f"Retimed video duration drifted by {retimed.duration_ms - target_ms}ms"
        )
    return retimed


def compose_video(
    scenes: Iterable,
    output_dir: Path,
    settings: Settings,
    *,
    width: int,
    height: int,
    fps: int,
    subtitle_path: Path | None,
    overlay_path: Path | None = None,
    scene_gap_ms: int = 120,
    target_duration_seconds: int | None = None,
) -> ComposedVideo:
    scene_list = list(scenes)
    scene_gap_ms = max(0, min(600, int(scene_gap_ms)))
    if not scene_list:
        raise VideoCompositionError("Cannot compose a video without scenes")
    for scene in scene_list:
        if not scene.image_path or not Path(scene.image_path).is_file():
            raise VideoCompositionError(f"Scene {scene.position} has no image file")
        if not scene.audio_path or not Path(scene.audio_path).is_file():
            raise VideoCompositionError(f"Scene {scene.position} has no audio file")
        if not scene.duration_ms or scene.duration_ms <= 0:
            raise VideoCompositionError(f"Scene {scene.position} has no audio duration")

    ffmpeg = _tool_path(settings.video_ffmpeg_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "narration.m4a"
    audio_inputs: list[str] = []
    audio_filters: list[str] = []
    audio_sequence: list[str] = []
    for index, scene in enumerate(scene_list):
        audio_inputs.extend(["-i", str(Path(scene.audio_path).resolve())])
        audio_filters.append(
            f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        audio_sequence.append(f"[a{index}]")
        if scene_gap_ms and index < len(scene_list) - 1:
            audio_filters.append(
                "anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=duration={scene_gap_ms / 1000:.3f},asetpts=PTS-STARTPTS[g{index}]"
            )
            audio_sequence.append(f"[g{index}]")
    audio_filters.append(
        "".join(audio_sequence)
        + f"concat=n={len(audio_sequence)}:v=0:a=1[aout]"
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            *audio_inputs,
            "-filter_complex",
            ";".join(audio_filters),
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(audio_path),
        ]
    )

    image_manifest = output_dir / "images.ffconcat"
    image_lines = ["ffconcat version 1.0\n"]
    for index, scene in enumerate(scene_list):
        image_lines.append(
            f"file '{_concat_quote(str(Path(scene.image_path).resolve()))}'\n"
        )
        duration_ms = int(scene.duration_ms)
        if index < len(scene_list) - 1:
            duration_ms += scene_gap_ms
        image_lines.append(f"duration {duration_ms / 1000:.3f}\n")
    image_lines.append(
        f"file '{_concat_quote(str(Path(scene_list[-1].image_path).resolve()))}'\n"
    )
    image_manifest.write_text("".join(image_lines), encoding="utf-8")

    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        f"fps={fps}",
        "format=yuv420p",
    ]
    if overlay_path:
        filters.append(_ass_overlay_filter(overlay_path))
    if subtitle_path:
        filters.append(_subtitle_filter(subtitle_path))
    output_path = output_dir / "output.mp4"
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(image_manifest),
            "-i",
            str(audio_path),
            "-vf",
            ",".join(filters),
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
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    result = probe_video(output_path, settings)
    if (result.width, result.height) != (width, height):
        raise VideoCompositionError(
            f"Generated video has unexpected resolution {result.width}x{result.height}"
        )
    expected_ms = sum(int(scene.duration_ms) for scene in scene_list)
    expected_ms += scene_gap_ms * max(0, len(scene_list) - 1)
    if abs(result.duration_ms - expected_ms) > max(1000, round(expected_ms * 0.03)):
        raise VideoCompositionError(
            f"Generated video duration drifted by {result.duration_ms - expected_ms}ms"
        )
    return _cap_duration(
        result,
        settings,
        target_duration_seconds=target_duration_seconds,
        fps=fps,
    )
