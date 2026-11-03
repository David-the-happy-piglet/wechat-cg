from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ...config import Settings


class MotionVideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class MotionVideoResult:
    path: str
    task_id: str
    file_id: str
    model: str
    duration_seconds: int
    resolution: str


class MiniMaxMotionVideoProvider:
    name = "minimax"

    def __init__(self, settings: Settings):
        if not settings.minimax_api_key:
            raise MotionVideoError(
                "MiniMax video is not configured. Fill MINIMAX_API_KEY in .env"
            )
        self.api_base = settings.minimax_api_base.rstrip("/")
        self.api_key = settings.minimax_api_key
        self.model = settings.minimax_video_model
        self.timeout = settings.video_generation_timeout_seconds
        self.poll_seconds = settings.video_generation_poll_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _check_response(body: dict, operation: str) -> None:
        base_response = body.get("base_resp") or {}
        if base_response.get("status_code") not in (None, 0, "0"):
            raise MotionVideoError(
                f"MiniMax {operation} failed: "
                f"{base_response.get('status_msg') or base_response.get('status_code')}"
            )

    def create_task(
        self,
        prompt: str,
        *,
        duration_seconds: int = 6,
        resolution: str = "1080P",
    ) -> str:
        try:
            response = httpx.post(
                f"{self.api_base}/video_generation",
                headers=self.headers,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "duration": duration_seconds,
                    "resolution": resolution,
                    "aigc_watermark": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MotionVideoError(f"MiniMax video task creation failed: {exc}") from exc
        self._check_response(body, "video task creation")
        task_id = body.get("task_id")
        if not task_id:
            raise MotionVideoError("MiniMax video task response contains no task_id")
        return str(task_id)

    def wait_for_file(self, task_id: str, *, progress=None) -> str:
        deadline = time.monotonic() + self.timeout
        last_status = ""
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.api_base}/query/video_generation",
                    headers=self.headers,
                    params={"task_id": task_id},
                    timeout=60,
                )
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise MotionVideoError(f"MiniMax video status query failed: {exc}") from exc
            self._check_response(body, "video status query")
            status = str(body.get("status") or "")
            if status != last_status and progress:
                progress(status)
                last_status = status
            if status == "Success":
                file_id = body.get("file_id")
                if not file_id:
                    raise MotionVideoError("MiniMax completed video contains no file_id")
                return str(file_id)
            if status == "Fail":
                raise MotionVideoError(
                    f"MiniMax video generation failed: {body.get('error_message') or 'unknown'}"
                )
            time.sleep(self.poll_seconds)
        raise MotionVideoError(
            f"MiniMax video generation timed out after {self.timeout} seconds"
        )

    def download(self, file_id: str, output_path: Path) -> None:
        try:
            response = httpx.get(
                f"{self.api_base}/files/retrieve",
                headers=self.headers,
                params={"file_id": file_id},
                timeout=60,
            )
            response.raise_for_status()
            body = response.json()
            self._check_response(body, "video file retrieval")
            download_url = (body.get("file") or {}).get("download_url")
            if not download_url:
                raise MotionVideoError("MiniMax file response contains no download URL")
            video_response = httpx.get(download_url, timeout=120)
            video_response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise MotionVideoError(f"MiniMax video download failed: {exc}") from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_response.content)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        duration_seconds: int = 6,
        resolution: str = "1080P",
        progress=None,
    ) -> MotionVideoResult:
        task_id = self.create_task(
            prompt,
            duration_seconds=duration_seconds,
            resolution=resolution,
        )
        if progress:
            progress(f"task_id={task_id}")
        file_id = self.wait_for_file(task_id, progress=progress)
        self.download(file_id, output_path)
        return MotionVideoResult(
            path=str(output_path),
            task_id=task_id,
            file_id=file_id,
            model=self.model,
            duration_seconds=duration_seconds,
            resolution=resolution,
        )
