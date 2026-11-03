from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from ...config import Settings
from ...models import VideoGenerationRun, VideoProject, VideoScene
from ..safety import scan_article
from .assets import build_video_image_provider, persist_video_image
from .composer import (
    VideoCompositionError,
    compose_bookended_video,
    compose_video,
    probe_video,
)
from .news_research import (
    NewsResearchError,
    audit_claim_ids,
    grounded_source_text,
    merge_review_correction,
    review_news_script,
    run_news_research,
)
from .overlays import write_overlay_ass
from .script import generate_video_script, validate_video_script
from .source import fetch_article
from .speech import SpeechProviderError, build_speech_provider, trim_speech_silence
from .subtitles import write_srt
from .visual_profiles import DEFAULT_VISUAL_PROFILE

logger = logging.getLogger(__name__)


ACTIVE_VIDEO_STATUSES = {
    "running",
    "extracting_source",
    "researching_news",
    "generating_script",
    "reviewing_news",
    "generating_assets",
    "composing",
}


def _workflow(project: VideoProject) -> str:
    return str(json.loads(project.settings_json or "{}").get("workflow") or "explainer")


def _news_routes(request_settings: dict, settings: Settings) -> dict[str, str]:
    configured = request_settings.get("news_model_routes") or {}
    return {
        "mechanical": str(configured.get("mechanical") or settings.news_mechanical_provider),
        "judgment": str(configured.get("judgment") or settings.news_judgment_provider),
        "research": str(configured.get("research") or settings.news_research_provider),
        "writer": str(configured.get("writer") or settings.news_writer_provider),
        "review": str(configured.get("review") or settings.news_review_provider),
    }


def _news_review_is_enabled(project: VideoProject, settings: Settings) -> bool:
    request_settings = json.loads(project.settings_json or "{}")
    return bool(request_settings.get("news_review_enabled", settings.news_review_enabled))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def project_directory(project: VideoProject, settings: Settings) -> Path:
    return Path(settings.generated_assets_dir).resolve() / "video-projects" / str(project.id)


def _set_progress(
    session: Session,
    project: VideoProject,
    *,
    status: str,
    step: str,
    progress: int,
) -> None:
    project.status = status
    project.current_step = step
    project.progress = min(100, max(0, progress))
    project.heartbeat_at = _utc_now()
    session.commit()


def _check_cancelled(session: Session, project: VideoProject) -> bool:
    session.refresh(project)
    if not project.cancel_requested:
        return False
    project.status = "cancelled"
    project.current_step = "cancelled"
    project.error_message = None
    project.next_attempt_at = None
    project.completed_at = _utc_now()
    session.commit()
    return True


def _add_run(
    session: Session,
    project: VideoProject,
    *,
    step: str,
    provider: str,
    model: str = "",
    scene: VideoScene | None = None,
    input_text: str = "",
    output_text: str = "",
    status: str = "completed",
    error_message: str | None = None,
    duration_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    session.add(
        VideoGenerationRun(
            video_project_id=project.id,
            video_scene_id=scene.id if scene else None,
            step=step,
            provider=provider,
            model=model,
            attempt=project.retry_count + 1,
            input_text=input_text,
            output_text=output_text,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    )


def _extract_source(session: Session, project: VideoProject, settings: Settings) -> None:
    if project.source_text:
        return
    started = time.monotonic()
    if project.source_type == "topic":
        text = project.source_topic or ""
        project.source_title = text
        project.source_text = f"请围绕以下主题组织可靠、清晰的讲解：{text}"
        project.source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        output = {"title": text, "characters": len(project.source_text)}
    elif project.source_type == "url" and project.source_url:
        extracted = fetch_article(project.source_url, settings)
        project.source_url = extracted.url
        project.source_title = extracted.title
        project.source_text = extracted.text
        project.source_hash = extracted.content_hash
        output = {"title": extracted.title, "characters": len(extracted.text)}
    else:
        raise ValueError("Video project has no valid topic or URL source")
    _add_run(
        session,
        project,
        step="extract_source",
        provider="local",
        input_text=project.source_url or project.source_topic or "",
        output_text=json.dumps(output, ensure_ascii=False),
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    session.commit()


def _record_news_calls(
    session: Session,
    project: VideoProject,
    calls,
) -> None:
    for call in calls:
        _add_run(
            session,
            project,
            step=call.step,
            provider=call.provider,
            model=call.model,
            input_text=call.prompt,
            output_text=json.dumps(call.output, ensure_ascii=False),
            duration_ms=call.duration_ms,
            prompt_tokens=call.usage.prompt_tokens,
            completion_tokens=call.usage.completion_tokens,
            total_tokens=call.usage.total_tokens,
        )


def _research_news(session: Session, project: VideoProject, settings: Settings) -> None:
    if project.research_json not in {"", "{}"}:
        return
    request_settings = json.loads(project.settings_json or "{}")
    routes = _news_routes(request_settings, settings)
    research, calls = run_news_research(
        settings,
        topic=project.source_title or project.source_topic or "新闻事件",
        source_url=project.source_url,
        source_title=project.source_title or project.source_topic or "用户提供材料",
        source_text=project.source_text or "",
        routes=routes,
        requested_angle=str(request_settings.get("news_angle") or "auto"),
    )
    project.research_json = json.dumps(research, ensure_ascii=False)
    _record_news_calls(session, project, calls)
    directory = project_directory(project, settings)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "research.json").write_text(
        json.dumps(research, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    session.commit()


def _generate_script(session: Session, project: VideoProject, settings: Settings) -> None:
    if project.scenes and project.script_json not in {"", "{}"}:
        return
    request_settings = json.loads(project.settings_json or "{}")
    style = str(request_settings.get("style") or "")
    script_prompt = str(request_settings.get("script_prompt") or "")
    use_web_images = bool(request_settings.get("use_web_images", False))
    visual_profile_name = str(
        request_settings.get("visual_profile") or DEFAULT_VISUAL_PROFILE
    )
    scene_duration_seconds = float(request_settings.get("scene_duration_seconds") or 3.5)
    speech_speed = float(request_settings.get("speech_speed") or 1.2)
    bookends = bool(request_settings.get("bookends", False))
    news_mode = _workflow(project) == "news_research"
    research = json.loads(project.research_json or "{}") if news_mode else {}
    grounded_claim_ids = (
        [
            str(item.get("claim_id"))
            for item in (research.get("fact_sheet") or {}).get("writable_claims") or []
            if item.get("claim_id")
        ]
        if news_mode
        else None
    )
    if news_mode and not grounded_claim_ids:
        raise NewsResearchError("新闻模式缺少经过证据校验的 Fact Sheet")
    if news_mode:
        source_text = grounded_source_text(research)
        provider_name = _news_routes(request_settings, settings)["writer"]
        requested_angle = str(
            research.get("selected_angle") or request_settings.get("news_angle") or "auto"
        )
        script_prompt = (
            f"{script_prompt}\n报道角度：{requested_angle}。"
            "不要越过 Fact Sheet；Hook 也必须由事实编号支撑。"
        ).strip()
    else:
        source_text = project.source_text or ""
        provider_name = project.llm_provider
    content_duration_seconds = project.target_duration_seconds
    if bookends:
        content_duration_seconds = max(
            15,
            project.target_duration_seconds - settings.video_bookend_reserve_seconds,
        )
    started = time.monotonic()
    result, provider, prompt = generate_video_script(
        settings,
        topic=project.source_title or project.source_topic or "知识讲解",
        source_text=source_text,
        target_duration_seconds=content_duration_seconds,
        style=style,
        provider_name=provider_name,
        scene_duration_seconds=scene_duration_seconds,
        speech_speed=speech_speed,
        visual_profile_name=visual_profile_name,
        script_prompt=script_prompt,
        use_web_images=use_web_images,
        grounded_claim_ids=grounded_claim_ids,
    )
    project.title = result["title"][:300]
    project.script_json = json.dumps(result, ensure_ascii=False)
    directory = project_directory(project, settings)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "script.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    project.llm_provider = provider.name
    safety_flags = scan_article("\n".join(item["narration"] for item in result["scenes"]))
    project.safety_flags_json = json.dumps(safety_flags, ensure_ascii=False)
    for scene_data in result["scenes"]:
        session.add(
            VideoScene(
                video_project_id=project.id,
                position=scene_data["position"],
                narration=scene_data["narration"],
                subtitle=scene_data["subtitle"],
                visual_intent=scene_data["visual_intent"],
                image_prompt=scene_data["image_prompt"],
                overlay_labels_json=json.dumps(
                    scene_data.get("overlay_labels") or [], ensure_ascii=False
                ),
                search_query=scene_data["search_query"],
            )
        )
    _add_run(
        session,
        project,
        step="generate_script",
        provider=provider.name,
        model=provider.model,
        input_text=prompt,
        output_text=project.script_json,
        duration_ms=round((time.monotonic() - started) * 1000),
        prompt_tokens=provider.usage.prompt_tokens,
        completion_tokens=provider.usage.completion_tokens,
        total_tokens=provider.usage.total_tokens,
    )
    session.commit()
    session.refresh(project)
    if safety_flags and not news_mode:
        project.status = "needs_review"
        project.current_step = "safety_review"
        project.error_message = "Video narration safety scan requires manual review"
        project.next_attempt_at = None
        session.commit()


def _replace_project_script(
    session: Session,
    project: VideoProject,
    result: dict,
    settings: Settings,
) -> None:
    for scene in list(project.scenes):
        session.delete(scene)
    session.flush()
    for scene_data in result["scenes"]:
        session.add(
            VideoScene(
                video_project_id=project.id,
                position=scene_data["position"],
                narration=scene_data["narration"],
                subtitle=scene_data["subtitle"],
                visual_intent=scene_data["visual_intent"],
                image_prompt=scene_data["image_prompt"],
                overlay_labels_json=json.dumps(
                    scene_data.get("overlay_labels") or [], ensure_ascii=False
                ),
                search_query=scene_data["search_query"],
            )
        )
    project.title = str(result["title"])[:300]
    project.script_json = json.dumps(result, ensure_ascii=False)
    project.safety_flags_json = json.dumps(
        scan_article("\n".join(item["narration"] for item in result["scenes"])),
        ensure_ascii=False,
    )
    directory = project_directory(project, settings)
    (directory / "script.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _review_news(session: Session, project: VideoProject, settings: Settings) -> None:
    if project.review_json not in {"", "{}"}:
        existing = json.loads(project.review_json)
        if existing.get("verdict") in {"pass", "corrected"}:
            return
    request_settings = json.loads(project.settings_json or "{}")
    research = json.loads(project.research_json or "{}")
    script = json.loads(project.script_json or "{}")
    review, call = review_news_script(
        settings,
        script=script,
        research=research,
        provider_name=_news_routes(request_settings, settings)["review"],
    )
    if review["verdict"] == "corrected":
        corrected, correction_errors = merge_review_correction(script, review)
        if correction_errors:
            review["verdict"] = "blocked"
            review["corrected_script_errors"] = correction_errors
        else:
            content_duration = project.target_duration_seconds
            if request_settings.get("bookends"):
                content_duration = max(
                    15,
                    content_duration - settings.video_bookend_reserve_seconds,
                )
            normalized = validate_video_script(
                corrected,
                content_duration,
                scene_duration_seconds=float(request_settings.get("scene_duration_seconds") or 3.5),
                style=str(request_settings.get("style") or ""),
                visual_profile_name=str(request_settings.get("visual_profile") or DEFAULT_VISUAL_PROFILE),
            )
            errors = audit_claim_ids(normalized, research)
            if errors:
                review["verdict"] = "blocked"
                review["corrected_script_errors"] = errors
            else:
                _replace_project_script(session, project, normalized, settings)
    project.review_json = json.dumps(review, ensure_ascii=False)
    _record_news_calls(session, project, [call])
    directory = project_directory(project, settings)
    (directory / "review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    session.commit()
    session.expire(project, ["scenes"])
    if review["verdict"] == "blocked":
        raise NewsResearchError("新闻事实审核未通过；请查看审核报告，不会继续生成成片")
    safety_flags = json.loads(project.safety_flags_json or "[]")
    if safety_flags:
        project.status = "needs_review"
        project.current_step = "safety_review"
        project.error_message = "事实审核已通过，但安全扫描仍需人工确认"
        project.next_attempt_at = None
        session.commit()


def _skip_news_review(session: Session, project: VideoProject, settings: Settings) -> None:
    research = json.loads(project.research_json or "{}")
    script = json.loads(project.script_json or "{}")
    claim_id_errors = audit_claim_ids(script, research, require_claim_ids=False)
    review = {
        "verdict": "blocked" if claim_id_errors else "skipped",
        "reason": "independent_llm_review_disabled",
        "local_claim_id_errors": claim_id_errors,
        "notes": "独立 LLM 新闻事实审核已关闭；仅执行本地事实编号校验。",
    }
    project.review_json = json.dumps(review, ensure_ascii=False)
    _add_run(
        session,
        project,
        step="news_review_skipped",
        provider="local",
        output_text=json.dumps(review, ensure_ascii=False),
    )
    directory = project_directory(project, settings)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    session.commit()
    if claim_id_errors:
        raise NewsResearchError("新闻稿引用了无效事实编号；请查看本地校验报告")


def _generate_scene_assets(
    session: Session,
    project: VideoProject,
    settings: Settings,
) -> None:
    request_settings = json.loads(project.settings_json or "{}")
    speech_speed = float(request_settings.get("speech_speed") or settings.speech_speed)
    trim_silence = bool(request_settings.get("trim_speech_silence", True))
    use_web_images = bool(request_settings.get("use_web_images", False))
    web_image_ratio = float(request_settings.get("web_image_ratio", 0.95))
    image_provider = build_video_image_provider(
        settings,
        use_web_images=use_web_images,
        web_image_ratio=web_image_ratio,
    )
    speech_provider = build_speech_provider(replace(settings, speech_speed=speech_speed))
    scenes = sorted(project.scenes, key=lambda item: item.position)
    for index, scene in enumerate(scenes):
        if _check_cancelled(session, project):
            return
        scene_dir = project_directory(project, settings) / "scenes" / f"{scene.position:03d}"
        scene.status = "generating"
        scene.error_message = None
        session.commit()
        if not scene.image_path or not Path(scene.image_path).is_file():
            started = time.monotonic()
            try:
                generated = image_provider.generate(
                    scene.image_prompt,
                    project.title or "视频配图",
                    scene.position,
                    scene.search_query,
                )
                scene.image_path = persist_video_image(
                    generated,
                    scene_dir,
                    width=project.width,
                    height=project.height,
                )
                scene.image_source_url = generated.get("source_url")
                scene.image_source_kind = generated.get("source_kind")
                scene.image_source_page_url = generated.get("source_page_url")
                scene.image_source_author = generated.get("source_author")
                scene.image_license_name = generated.get("license_name")
                scene.image_license_url = generated.get("license_url")
                _add_run(
                    session,
                    project,
                    step="generate_image",
                    provider=image_provider.name,
                    scene=scene,
                    input_text=scene.image_prompt,
                    output_text=json.dumps(generated, ensure_ascii=False),
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                session.commit()
            except Exception as exc:
                scene.status = "failed"
                scene.error_message = str(exc)
                _add_run(
                    session,
                    project,
                    step="generate_image",
                    provider=image_provider.name,
                    scene=scene,
                    input_text=scene.image_prompt,
                    status="failed",
                    error_message=str(exc),
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                session.commit()
                raise
        if not scene.audio_path or not Path(scene.audio_path).is_file() or not scene.duration_ms:
            started = time.monotonic()
            try:
                speech = speech_provider.synthesize(
                    scene.narration,
                    scene_dir,
                    voice=project.voice,
                )
                if trim_silence and speech_provider.name != "mock":
                    try:
                        speech = trim_speech_silence(speech, scene_dir, settings)
                    except SpeechProviderError as exc:
                        logger.warning(
                            "Could not trim speech silence for scene %s: %s",
                            scene.position,
                            exc,
                        )
                scene.audio_path = speech.path
                scene.duration_ms = speech.duration_ms
                _add_run(
                    session,
                    project,
                    step="generate_speech",
                    provider=speech.provider,
                    model=speech.model,
                    scene=scene,
                    input_text=scene.narration,
                    output_text=json.dumps(
                        {"path": speech.path, "duration_ms": speech.duration_ms},
                        ensure_ascii=False,
                    ),
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                session.commit()
            except Exception as exc:
                scene.status = "failed"
                scene.error_message = str(exc)
                _add_run(
                    session,
                    project,
                    step="generate_speech",
                    provider=speech_provider.name,
                    model=speech_provider.model,
                    scene=scene,
                    input_text=scene.narration,
                    status="failed",
                    error_message=str(exc),
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                session.commit()
                raise
        scene.status = "completed"
        scene.error_message = None
        project.progress = 30 + round(50 * (index + 1) / len(scenes))
        project.heartbeat_at = _utc_now()
        session.commit()


def _compose(session: Session, project: VideoProject, settings: Settings) -> None:
    if project.output_path and Path(project.output_path).is_file():
        try:
            existing = probe_video(Path(project.output_path), settings)
        except VideoCompositionError:
            pass
        else:
            project.actual_duration_ms = existing.duration_ms
            session.commit()
            return
    directory = project_directory(project, settings)
    scenes = sorted(project.scenes, key=lambda item: item.position)
    request_settings = json.loads(project.settings_json or "{}")
    bookends = bool(request_settings.get("bookends", False))
    content_duration_seconds = project.target_duration_seconds
    if bookends:
        content_duration_seconds = max(
            15,
            project.target_duration_seconds - settings.video_bookend_reserve_seconds,
        )
    scene_gap_ms = max(0, min(600, int(request_settings.get("scene_gap_ms", 120))))
    subtitle_path = Path(
        write_srt(directory / "subtitles.srt", scenes, gap_ms=scene_gap_ms)
    )
    overlay_value = write_overlay_ass(
        directory / "overlay-labels.ass",
        scenes,
        width=project.width,
        height=project.height,
        scene_gap_ms=scene_gap_ms,
    )
    overlay_path = Path(overlay_value) if overlay_value else None
    project.subtitle_path = str(subtitle_path)
    session.commit()
    started = time.monotonic()
    result = compose_video(
        scenes,
        directory,
        settings,
        width=project.width,
        height=project.height,
        fps=project.fps,
        subtitle_path=subtitle_path if project.subtitles_enabled else None,
        overlay_path=overlay_path,
        scene_gap_ms=scene_gap_ms,
        target_duration_seconds=content_duration_seconds,
    )
    source_timeline_ms = sum(int(scene.duration_ms or 0) for scene in scenes)
    source_timeline_ms += scene_gap_ms * max(0, len(scenes) - 1)
    if source_timeline_ms > 0 and abs(result.duration_ms - source_timeline_ms) > 100:
        write_srt(
            subtitle_path,
            scenes,
            gap_ms=scene_gap_ms,
            time_scale=result.duration_ms / source_timeline_ms,
        )
    if bookends:
        result = compose_bookended_video(
            Path(settings.video_intro_path),
            result,
            Path(settings.video_outro_path),
            directory,
            settings,
            title=project.title or "知识解说",
            fps=project.fps,
        )
    project.output_path = result.path
    project.actual_duration_ms = result.duration_ms
    _add_run(
        session,
        project,
        step="compose_video",
        provider="ffmpeg",
        input_text=json.dumps(
            {"scene_count": len(scenes), "width": project.width, "height": project.height}
        ),
        output_text=json.dumps(
            {"path": result.path, "duration_ms": result.duration_ms}, ensure_ascii=False
        ),
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    manifest = {
        "project_id": project.id,
        "title": project.title,
        "source_type": project.source_type,
        "source_url": project.source_url,
        "source_hash": project.source_hash,
        "research": json.loads(project.research_json or "{}"),
        "review": json.loads(project.review_json or "{}"),
        "script": json.loads(project.script_json),
        "assets": [
            {
                "position": scene.position,
                "source_kind": scene.image_source_kind,
                "source_page_url": scene.image_source_page_url,
                "source_author": scene.image_source_author,
                "license_name": scene.image_license_name,
                "license_url": scene.image_license_url,
            }
            for scene in scenes
        ],
        "output": {
            "path": result.path,
            "duration_ms": result.duration_ms,
            "width": result.width,
            "height": result.height,
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    session.commit()


def execute_video_project(
    session: Session,
    project_id: int,
    settings: Settings,
) -> VideoProject:
    project = session.scalar(
        select(VideoProject)
        .where(VideoProject.id == project_id)
        .options(selectinload(VideoProject.scenes))
    )
    if not project:
        raise ValueError(f"Video project {project_id} not found")
    if project.status in {"completed", "cancelled", "needs_review"}:
        return project
    if project.cancel_requested:
        _check_cancelled(session, project)
        return project
    if not project.started_at:
        project.started_at = _utc_now()
    project.error_message = None
    project.heartbeat_at = _utc_now()
    session.commit()
    try:
        _set_progress(
            session, project, status="extracting_source", step="extract_source", progress=5
        )
        _extract_source(session, project, settings)
        if _check_cancelled(session, project):
            return project
        if _workflow(project) == "news_research":
            _set_progress(
                session, project, status="researching_news", step="research_news", progress=10
            )
            _research_news(session, project, settings)
            if _check_cancelled(session, project):
                return project
        _set_progress(
            session,
            project,
            status="generating_script",
            step="generate_script",
            progress=35 if _workflow(project) == "news_research" else 15,
        )
        _generate_script(session, project, settings)
        if project.status == "needs_review" or _check_cancelled(session, project):
            return project
        if _workflow(project) == "news_research":
            if _news_review_is_enabled(project, settings):
                _set_progress(
                    session,
                    project,
                    status="reviewing_news",
                    step="review_news",
                    progress=48,
                )
                _review_news(session, project, settings)
            else:
                _skip_news_review(session, project, settings)
            if project.status == "needs_review" or _check_cancelled(session, project):
                return project
        _set_progress(
            session,
            project,
            status="generating_assets",
            step="generate_assets",
            progress=58 if _workflow(project) == "news_research" else 30,
        )
        _generate_scene_assets(session, project, settings)
        if project.status == "cancelled" or _check_cancelled(session, project):
            return project
        _set_progress(session, project, status="composing", step="compose_video", progress=85)
        _compose(session, project, settings)
        project.status = "completed"
        project.current_step = "completed"
        project.progress = 100
        project.error_message = None
        project.next_attempt_at = None
        project.heartbeat_at = _utc_now()
        project.completed_at = _utc_now()
        session.commit()
        return project
    except Exception as exc:
        logger.exception("video project %s failed at %s", project.id, project.current_step)
        failed_step = project.current_step
        session.rollback()
        project = session.get(VideoProject, project_id)
        if not project:
            raise
        project.status = "failed"
        project.error_message = str(exc)[:4000]
        _add_run(
            session,
            project,
            step=failed_step,
            provider="pipeline",
            status="failed",
            error_message=str(exc)[:4000],
        )
        project.retry_count += 1
        project.heartbeat_at = _utc_now()
        if project.retry_count < settings.video_max_retries:
            project.next_attempt_at = _utc_now() + timedelta(
                seconds=min(300, 2 ** project.retry_count * 5)
            )
        else:
            project.next_attempt_at = None
        session.commit()
        return project


def claim_next_video_project(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> int | None:
    now = now or _utc_now()
    statement = (
        select(VideoProject)
        .where(
            VideoProject.cancel_requested.is_(False),
            VideoProject.retry_count < settings.video_max_retries,
            VideoProject.status.in_(["pending", "failed"]),
            or_(VideoProject.next_attempt_at.is_(None), VideoProject.next_attempt_at <= now),
        )
        .order_by(VideoProject.created_at, VideoProject.id)
        .limit(1)
    )
    if session.bind and session.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)
    project = session.scalar(statement)
    if not project:
        return None
    project.status = "running"
    project.current_step = "claimed"
    project.heartbeat_at = now
    session.commit()
    return project.id


def process_next_video_project(session: Session, settings: Settings) -> VideoProject | None:
    project_id = claim_next_video_project(session, settings)
    if project_id is None:
        return None
    return execute_video_project(session, project_id, settings)


def recover_stale_video_projects(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> int:
    now = now or _utc_now()
    cutoff = now - timedelta(minutes=settings.video_stale_minutes)
    projects = session.scalars(
        select(VideoProject).where(
            VideoProject.status.in_(ACTIVE_VIDEO_STATUSES),
            VideoProject.heartbeat_at < cutoff,
        )
    ).all()
    for project in projects:
        project.status = "failed"
        project.error_message = "Video worker stopped before completing the project"
        project.retry_count += 1
        project.next_attempt_at = now if project.retry_count < settings.video_max_retries else None
    session.commit()
    return len(projects)
