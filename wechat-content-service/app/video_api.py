from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import settings
from .database import get_db
from .models import VideoProject
from .schemas import (
    VideoProjectCreate,
    VideoProjectRead,
    VideoTextSource,
    VideoTopicSource,
    VideoURLSource,
)
from .serializers import video_project_dict
from .services.video.visual_profiles import VisualProfileError, load_visual_profile

router = APIRouter(prefix="/api/video-projects", tags=["video-projects"])


def _project_or_404(session: Session, project_id: int) -> VideoProject:
    project = session.scalar(
        select(VideoProject)
        .where(VideoProject.id == project_id)
        .options(selectinload(VideoProject.scenes))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Video project not found")
    return project


def _safe_artifact_path(project: VideoProject, value: str | None) -> Path:
    if not value:
        raise HTTPException(status_code=404, detail="Video artifact is not ready")
    root = (
        Path(settings.generated_assets_dir).resolve()
        / "video-projects"
        / str(project.id)
    )
    path = Path(value).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Video artifact was not found")
    return path


def _request_settings(data: VideoProjectCreate) -> dict:
    return {
        "resolution": data.resolution,
        "workflow": data.workflow,
        "news_review_enabled": data.news_review_enabled,
        "news_angle": data.news_angle,
        "news_model_routes": data.news_model_routes.model_dump(),
        "bookends": data.bookends,
        "use_web_images": data.use_web_images,
        "web_image_ratio": data.web_image_ratio,
        "visual_profile": data.visual_profile,
        "script_prompt": data.script_prompt,
        "style": data.style,
        "scene_duration_seconds": data.scene_duration_seconds,
        "speech_speed": data.speech_speed,
        "scene_gap_ms": data.scene_gap_ms,
        "trim_speech_silence": data.trim_speech_silence,
    }


@router.post("", response_model=VideoProjectRead, status_code=status.HTTP_202_ACCEPTED)
def create_video_project(
    data: VideoProjectCreate,
    session: Session = Depends(get_db),
):
    try:
        load_visual_profile(data.visual_profile)
    except VisualProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dimensions = {"720p": (1280, 720), "1080p": (1920, 1080)}
    width, height = dimensions[data.resolution]
    source_topic = data.source.topic if isinstance(data.source, VideoTopicSource) else None
    source_url = data.source.url if isinstance(data.source, VideoURLSource) else None
    source_text = data.source.text if isinstance(data.source, VideoTextSource) else None
    source_title = None
    source_hash = None
    if source_text:
        source_title = " ".join(source_text.split())[:80]
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    project = VideoProject(
        source_type=data.source.type,
        source_topic=source_topic,
        source_url=source_url,
        source_title=source_title,
        source_text=source_text,
        source_hash=source_hash,
        status="pending",
        current_step="queued",
        progress=0,
        target_duration_seconds=data.target_duration_seconds,
        width=width,
        height=height,
        fps=data.fps,
        voice=data.voice or settings.speech_voice,
        llm_provider=data.provider,
        subtitles_enabled=data.subtitles,
        settings_json=json.dumps(_request_settings(data), ensure_ascii=False),
        next_attempt_at=datetime.now(UTC),
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return video_project_dict(project)


@router.get("", response_model=list[VideoProjectRead])
def list_video_projects(
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_db),
):
    projects = session.scalars(
        select(VideoProject)
        .options(selectinload(VideoProject.scenes))
        .order_by(VideoProject.created_at.desc())
        .limit(limit)
    ).all()
    return [video_project_dict(project) for project in projects]


@router.get("/{project_id}", response_model=VideoProjectRead)
def get_video_project(project_id: int, session: Session = Depends(get_db)):
    return video_project_dict(_project_or_404(session, project_id))


@router.put("/{project_id}", response_model=VideoProjectRead)
def update_failed_video_project(
    project_id: int,
    data: VideoProjectCreate,
    session: Session = Depends(get_db),
):
    project = _project_or_404(session, project_id)
    if project.status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=409,
            detail="Only failed or cancelled video projects can be edited",
        )
    try:
        load_visual_profile(data.visual_profile)
    except VisualProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dimensions = {"720p": (1280, 720), "1080p": (1920, 1080)}
    width, height = dimensions[data.resolution]
    source_topic = data.source.topic if isinstance(data.source, VideoTopicSource) else None
    source_url = data.source.url if isinstance(data.source, VideoURLSource) else None
    source_text = data.source.text if isinstance(data.source, VideoTextSource) else None
    source_title = " ".join(source_text.split())[:80] if source_text else None
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest() if source_text else None
    for scene in list(project.scenes):
        session.delete(scene)
    project.source_type = data.source.type
    project.source_topic = source_topic
    project.source_url = source_url
    project.source_title = source_title
    project.source_text = source_text
    project.source_hash = source_hash
    project.title = None
    project.status = "pending"
    project.current_step = "queued"
    project.progress = 0
    project.target_duration_seconds = data.target_duration_seconds
    project.width = width
    project.height = height
    project.fps = data.fps
    project.voice = data.voice or settings.speech_voice
    project.llm_provider = data.provider
    project.subtitles_enabled = data.subtitles
    project.settings_json = json.dumps(_request_settings(data), ensure_ascii=False)
    project.research_json = "{}"
    project.script_json = "{}"
    project.review_json = "{}"
    project.safety_flags_json = "[]"
    project.output_path = None
    project.subtitle_path = None
    project.actual_duration_ms = None
    project.error_message = None
    project.retry_count = 0
    project.cancel_requested = False
    project.started_at = None
    project.heartbeat_at = None
    project.completed_at = None
    project.next_attempt_at = datetime.now(UTC)
    session.commit()
    session.refresh(project)
    return video_project_dict(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_failed_video_project(
    project_id: int,
    session: Session = Depends(get_db),
):
    project = _project_or_404(session, project_id)
    if project.status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=409,
            detail="Only failed or cancelled video projects can be deleted",
        )
    directory = (
        Path(settings.generated_assets_dir).resolve()
        / "video-projects"
        / str(project.id)
    )
    session.delete(project)
    session.commit()
    shutil.rmtree(directory, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{project_id}/retry", response_model=VideoProjectRead)
def retry_video_project(project_id: int, session: Session = Depends(get_db)):
    project = _project_or_404(session, project_id)
    if project.status == "completed":
        raise HTTPException(status_code=409, detail="Completed video projects cannot be retried")
    if project.status in {
        "running",
        "extracting_source",
        "researching_news",
        "generating_script",
        "reviewing_news",
        "generating_assets",
        "composing",
    }:
        raise HTTPException(status_code=409, detail="Video project is still running")
    if project.status == "needs_review":
        raise HTTPException(
            status_code=409,
            detail="Approve the safety review before retrying this video project",
        )
    request_settings = json.loads(project.settings_json or "{}")
    if request_settings.get("workflow") == "news_research" and project.current_step == "review_news":
        project.review_json = "{}"
    project.status = "pending"
    project.current_step = "queued"
    project.error_message = None
    project.retry_count = 0
    project.cancel_requested = False
    project.completed_at = None
    project.next_attempt_at = datetime.now(UTC)
    session.commit()
    return video_project_dict(project)


@router.post("/{project_id}/approve", response_model=VideoProjectRead)
def approve_video_project(project_id: int, session: Session = Depends(get_db)):
    project = _project_or_404(session, project_id)
    if project.status != "needs_review":
        raise HTTPException(status_code=409, detail="Video project does not need review")
    request_settings = json.loads(project.settings_json or "{}")
    review = json.loads(project.review_json or "{}")
    if request_settings.get("workflow") == "news_research" and review.get("verdict") == "blocked":
        raise HTTPException(
            status_code=409,
            detail="事实审核未通过的新闻稿不能人工跳过；请补充来源或重新研究",
        )
    request_settings["safety_approved"] = True
    project.settings_json = json.dumps(request_settings, ensure_ascii=False)
    project.status = "pending"
    project.current_step = "queued"
    project.error_message = None
    project.next_attempt_at = datetime.now(UTC)
    session.commit()
    return video_project_dict(project)


@router.post("/{project_id}/cancel", response_model=VideoProjectRead)
def cancel_video_project(project_id: int, session: Session = Depends(get_db)):
    project = _project_or_404(session, project_id)
    if project.status == "completed":
        raise HTTPException(status_code=409, detail="Completed video projects cannot be cancelled")
    project.cancel_requested = True
    if project.status in {"pending", "failed", "needs_review"}:
        project.status = "cancelled"
        project.current_step = "cancelled"
        project.next_attempt_at = None
        project.completed_at = datetime.now(UTC)
    session.commit()
    return video_project_dict(project)


@router.get("/{project_id}/output")
def download_video(project_id: int, session: Session = Depends(get_db)):
    project = _project_or_404(session, project_id)
    path = _safe_artifact_path(project, project.output_path)
    return FileResponse(path, media_type="video/mp4", filename=f"video-{project.id}.mp4")


@router.get("/{project_id}/subtitles")
def download_subtitles(project_id: int, session: Session = Depends(get_db)):
    project = _project_or_404(session, project_id)
    path = _safe_artifact_path(project, project.subtitle_path)
    return FileResponse(
        path,
        media_type="application/x-subrip; charset=utf-8",
        filename=f"video-{project.id}.srt",
    )
