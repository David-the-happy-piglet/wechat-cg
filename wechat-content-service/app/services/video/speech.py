from __future__ import annotations

import binascii
import logging
import math
import shutil
import struct
import subprocess
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx

from ...config import Settings

logger = logging.getLogger(__name__)


MINIMAX_MANDARIN_VOICES = (
    {
        "id": "male-qn-qingse",
        "name": "青涩青年",
        "description": "轻松自然，适合知识解释和日常话题",
    },
    {
        "id": "male-qn-jingying",
        "name": "精英青年",
        "description": "清楚稳重，适合商业与方法类内容",
    },
    {
        "id": "male-qn-daxuesheng",
        "name": "青年大学生",
        "description": "年轻、有交流感，适合轻知识内容",
    },
    {
        "id": "female-chengshu",
        "name": "成熟女声",
        "description": "温和沉稳，适合心理和人文内容",
    },
    {
        "id": "female-yujie",
        "name": "清晰御姐",
        "description": "有力度、节奏明确，适合观点讲解",
    },
    {
        "id": "female-tianmei",
        "name": "甜美女声",
        "description": "明亮亲切，适合生活方式内容",
    },
)


def speech_voice_options(settings: Settings) -> list[dict[str, str]]:
    if settings.speech_provider == "minimax":
        return [dict(item) for item in MINIMAX_MANDARIN_VOICES]
    return [
        {
            "id": "test",
            "name": "本地测试音",
            "description": "仅用于不消耗外部额度的流水线测试",
        }
    ]


class SpeechProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechResult:
    path: str
    duration_ms: int
    provider: str
    model: str


class SpeechProvider(ABC):
    name = "unknown"
    model = ""

    @abstractmethod
    def synthesize(self, text: str, output_dir: Path, *, voice: str) -> SpeechResult:
        raise NotImplementedError


def trim_speech_silence(
    result: SpeechResult,
    output_dir: Path,
    settings: Settings,
) -> SpeechResult:
    """Trim only provider-added leading/trailing silence and keep natural inner pauses."""
    ffmpeg = shutil.which(settings.video_ffmpeg_path)
    ffprobe = shutil.which(settings.video_ffprobe_path)
    if not ffmpeg or not ffprobe:
        raise SpeechProviderError("FFmpeg and ffprobe are required to trim speech silence")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "narration.trimmed.mp3"
    filter_value = (
        "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-45dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.08:start_threshold=-45dB,"
        "areverse"
    )
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        result.path,
        "-af",
        filter_value,
        "-ar",
        "32000",
        "-ac",
        "1",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SpeechProviderError(f"Failed to trim speech silence: {completed.stderr[-1000:]}")
    probed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration_ms = round(float(probed.stdout.strip()) * 1000)
    except ValueError as exc:
        raise SpeechProviderError("Trimmed speech has no valid duration") from exc
    if probed.returncode or duration_ms < 200:
        raise SpeechProviderError("Trimmed speech is unexpectedly short")
    return SpeechResult(
        path=str(output_path),
        duration_ms=duration_ms,
        provider=result.provider,
        model=result.model,
    )


class MockSpeechProvider(SpeechProvider):
    name = "mock"
    model = "local-wave"

    def __init__(self, settings: Settings):
        self.speed = max(0.5, min(settings.speech_speed, 2.0))

    def synthesize(self, text: str, output_dir: Path, *, voice: str) -> SpeechResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "narration.wav"
        duration_seconds = max(1.2, len(text.strip()) / (4.2 * self.speed))
        sample_rate = 32000
        frame_count = round(duration_seconds * sample_rate)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            frames = bytearray()
            for index in range(frame_count):
                envelope = min(1.0, index / 800, (frame_count - index) / 800)
                value = int(260 * envelope * math.sin(2 * math.pi * 220 * index / sample_rate))
                frames.extend(struct.pack("<h", value))
            output.writeframes(frames)
        return SpeechResult(
            path=str(path),
            duration_ms=round(duration_seconds * 1000),
            provider=self.name,
            model=self.model,
        )


class MiniMaxSpeechProvider(SpeechProvider):
    name = "minimax"

    def __init__(self, settings: Settings):
        if not settings.minimax_api_key:
            raise SpeechProviderError(
                "MiniMax speech is not configured. Fill MINIMAX_API_KEY in .env"
            )
        self.api_base = settings.minimax_api_base.rstrip("/")
        self.api_key = settings.minimax_api_key
        self.model = settings.speech_model
        self.speed = max(0.5, min(settings.speech_speed, 2.0))
        self.timeout = settings.speech_timeout_seconds
        self.max_attempts = max(1, settings.provider_max_attempts)
        self.retry_base_seconds = max(0, settings.provider_retry_base_seconds)

    def _post(self, payload: dict) -> httpx.Response:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return httpx.post(
                    f"{self.api_base}/t2a_v2",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.TransportError as exc:
                if attempt >= self.max_attempts:
                    raise
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "MiniMax speech transport error on attempt %s/%s; retrying in %.1fs: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    exc,
                )
                if delay:
                    time.sleep(delay)
        raise SpeechProviderError("MiniMax speech request exhausted retries")

    def synthesize(self, text: str, output_dir: Path, *, voice: str) -> SpeechResult:
        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice,
                "speed": self.speed,
                "vol": 1,
                "pitch": 0,
                "emotion": "calm",
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
            "subtitle_enable": False,
            "output_format": "hex",
            "aigc_watermark": False,
        }
        try:
            response = self._post(payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" — {exc.response.text[:500]}"
            raise SpeechProviderError(f"MiniMax speech request failed: {exc}{detail}") from exc
        base_response = body.get("base_resp") or {}
        if base_response.get("status_code") not in (None, 0, "0"):
            raise SpeechProviderError(
                "MiniMax speech request failed: "
                f"{base_response.get('status_msg') or base_response.get('status_code')}"
            )
        encoded = (body.get("data") or {}).get("audio")
        if not isinstance(encoded, str) or not encoded:
            raise SpeechProviderError("MiniMax speech response contains no audio")
        try:
            audio = bytes.fromhex(encoded)
        except (ValueError, binascii.Error) as exc:
            raise SpeechProviderError("MiniMax speech returned invalid hex audio") from exc
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "narration.mp3"
        path.write_bytes(audio)
        duration_ms = (body.get("extra_info") or {}).get("audio_length")
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise SpeechProviderError("MiniMax speech response contains no valid duration")
        return SpeechResult(
            path=str(path),
            duration_ms=duration_ms,
            provider=self.name,
            model=self.model,
        )


def build_speech_provider(settings: Settings) -> SpeechProvider:
    if settings.speech_provider == "mock":
        return MockSpeechProvider(settings)
    if settings.speech_provider == "minimax":
        return MiniMaxSpeechProvider(settings)
    raise SpeechProviderError(f"Unsupported SPEECH_PROVIDER: {settings.speech_provider}")
