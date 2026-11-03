#!/usr/bin/env python3
"""Probe and benchmark MiniMax Token Plan video-generation limits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE_CASES = [
    {
        "name": "probe_max_duration",
        "model": "MiniMax-Hailuo-2.3",
        "duration": 10,
        "resolution": "768P",
        "prompt": "A locked-off shot of a mechanical desk clock ticking continuously.",
    },
    {
        "name": "probe_max_resolution",
        "model": "MiniMax-Hailuo-2.3",
        "duration": 6,
        "resolution": "1080P",
        "prompt": "A locked-off shot of a mechanical desk clock ticking continuously.",
    },
]

BENCHMARK_CASES = [
    {
        "name": "human_motion",
        "model": "MiniMax-Hailuo-2.3",
        "duration": 6,
        "resolution": "768P",
        "prompt": (
            "Cinematic medium close-up of a barista in a sunlit modern cafe. "
            "She naturally picks up a white ceramic cup with her right hand, "
            "turns toward the camera, smiles once, and places the cup on a wooden "
            "counter. Preserve face, fingers, cup shape, clothing, and background "
            "continuity throughout. Natural skin texture, realistic motion blur, "
            "soft window light, shallow depth of field. [Tracking shot]"
        ),
    },
    {
        "name": "liquid_physics",
        "model": "MiniMax-Hailuo-2.3",
        "duration": 6,
        "resolution": "768P",
        "prompt": (
            "Macro product shot: a clear glass pitcher slowly pours sparkling water "
            "into a crystal tumbler containing three ice cubes and a thin lemon slice. "
            "Show a coherent stream, realistic splashes, bubbles, refraction, caustics, "
            "condensation, and stable glass geometry. Dark studio background with a "
            "single soft rim light, photorealistic, high detail. [Static shot]"
        ),
    },
    {
        "name": "camera_detail",
        "model": "MiniMax-Hailuo-2.3",
        "duration": 6,
        "resolution": "768P",
        "prompt": (
            "One continuous cinematic shot through a rainy futuristic night market. "
            "The camera begins at eye level beside a steaming noodle stall, slowly "
            "dollies forward past hanging red lanterns and reflective puddles, then "
            "gently pans right to reveal a cyclist crossing the alley. Maintain stable "
            "architecture, consistent people, believable reflections, rain, steam, "
            "parallax, and fine background detail. No cuts. [Dolly in] [Pan right]"
        ),
    },
]


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class MiniMaxClient:
    def __init__(self, api_key: str, api_base: str) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> tuple[int, dict[str, Any]]:
        url = f"{self.api_base}/{path.lstrip('/')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(response_body)
            except json.JSONDecodeError:
                parsed = {"raw_response": response_body}
            return exc.code, parsed

    def create_video(self, case: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        payload = {key: value for key, value in case.items() if key != "name"}
        return self.request("POST", "video_generation", payload=payload)

    def query_video(self, task_id: str) -> tuple[int, dict[str, Any]]:
        return self.request(
            "GET", "query/video_generation", query={"task_id": task_id}
        )

    def retrieve_file(self, file_id: str) -> tuple[int, dict[str, Any]]:
        return self.request("GET", "files/retrieve", query={"file_id": file_id})


def base_status(data: dict[str, Any]) -> tuple[int | None, str]:
    response = data.get("base_resp") or {}
    return response.get("status_code"), str(response.get("status_msg", ""))


def task_accepted(http_status: int, data: dict[str, Any]) -> bool:
    status_code, _ = base_status(data)
    return http_status < 400 and status_code == 0 and bool(data.get("task_id"))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download(url: str, destination: Path) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return destination.stat().st_size


def submit_cases(
    client: MiniMaxClient,
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
    max_accepted: int | None = None,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for case in cases:
        if max_accepted is not None and len(accepted) >= max_accepted:
            break
        print(
            f"Submitting {case['name']}: {case['duration']}s "
            f"{case['resolution']} {case['model']}",
            flush=True,
        )
        http_status, response = client.create_video(case)
        record = {
            "case": case,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "http_status": http_status,
            "response": response,
            "accepted": task_accepted(http_status, response),
        }
        manifest["attempts"].append(record)
        write_manifest(manifest_path, manifest)
        status_code, status_msg = base_status(response)
        if record["accepted"]:
            accepted.append(record)
            print(f"  accepted: task_id={response['task_id']}", flush=True)
        else:
            print(
                f"  rejected: HTTP {http_status}, "
                f"status_code={status_code}, message={status_msg}",
                flush=True,
            )
    return accepted


def poll_and_download(
    client: MiniMaxClient,
    records: list[dict[str, Any]],
    output_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    poll_interval: int,
) -> None:
    pending = {record["response"]["task_id"]: record for record in records}
    while pending:
        for task_id, record in list(pending.items()):
            http_status, response = client.query_video(task_id)
            status = response.get("status", "Unknown")
            record["last_query"] = {
                "queried_at": datetime.now(timezone.utc).isoformat(),
                "http_status": http_status,
                "response": response,
            }
            print(f"{record['case']['name']}: {status}", flush=True)
            if status == "Success":
                file_id = str(response["file_id"])
                file_http_status, file_response = client.retrieve_file(file_id)
                download_url = (file_response.get("file") or {}).get("download_url")
                record["file"] = {
                    "http_status": file_http_status,
                    "response": file_response,
                }
                if not download_url:
                    raise RuntimeError(
                        f"No download URL returned for {record['case']['name']}"
                    )
                destination = output_dir / f"{record['case']['name']}.mp4"
                record["output"] = {
                    "path": str(destination.resolve()),
                    "bytes": download(download_url, destination),
                }
                del pending[task_id]
            elif status == "Fail":
                del pending[task_id]
            write_manifest(manifest_path, manifest)
        if pending:
            time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".env",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "generated-assets"
        / "minimax-video-benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=("probe", "benchmark", "all"),
        default="all",
        help="Probe plan boundaries, run valid quality cases, or do both.",
    )
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Required because accepted tasks consume the daily video quota.",
    )
    args = parser.parse_args()
    if not args.confirm_submit:
        parser.error(
            "video generation consumes daily quota; rerun with --confirm-submit"
        )

    config = load_dotenv(args.env_file)
    api_key = os.environ.get("MINIMAX_API_KEY") or config.get("MINIMAX_API_KEY")
    api_base = (
        os.environ.get("MINIMAX_API_BASE")
        or config.get("MINIMAX_API_BASE")
        or "https://api.minimaxi.com/v1"
    )
    if not api_key:
        print("MINIMAX_API_KEY is not configured", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "api_base": api_base,
        "attempts": [],
    }
    write_manifest(manifest_path, manifest)
    client = MiniMaxClient(api_key, api_base)

    accepted: list[dict[str, Any]] = []
    if args.mode in {"probe", "all"}:
        accepted.extend(
            submit_cases(client, PROBE_CASES, manifest, manifest_path)
        )
    if args.mode in {"benchmark", "all"}:
        remaining_slots = max(0, 3 - len(accepted))
        accepted.extend(
            submit_cases(
                client,
                BENCHMARK_CASES,
                manifest,
                manifest_path,
                max_accepted=remaining_slots,
            )
        )
    poll_and_download(
        client,
        accepted,
        args.output_dir,
        manifest,
        manifest_path,
        args.poll_interval,
    )
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
