import json

from .models import Article, AudienceProfile, AutomationPlan, AutomationRun, VideoProject
from .services.video.visual_profiles import DEFAULT_VISUAL_PROFILE


def audience_dict(audience: AudienceProfile) -> dict:
    return {
        "id": audience.id,
        "slug": audience.slug,
        "name": audience.name,
        "age_range": audience.age_range,
        "description": audience.description,
        "interests": json.loads(audience.interests_json),
        "writing_style": audience.writing_style,
        "title_preferences": json.loads(audience.title_preferences_json),
        "article_length": audience.article_length,
        "image_style": audience.image_style,
        "prohibited_topics": json.loads(audience.prohibited_topics_json),
        "enabled": audience.enabled,
        "created_at": audience.created_at,
        "updated_at": audience.updated_at,
    }


def article_dict(article: Article) -> dict:
    return {
        "id": article.id,
        "content_type": article.content_type or "health_article",
        "story_mode": article.story_mode or "auto",
        "audience_profile_id": article.audience_profile_id,
        "topic": article.topic,
        "category": article.category,
        "target_length": article.target_length,
        "title_candidates": json.loads(article.title_candidates_json),
        "selected_title": article.selected_title,
        "summary": article.summary,
        "content": json.loads(article.content_json),
        "content_html": article.content_html,
        "status": article.status,
        "llm_provider": article.llm_provider,
        "model_name": article.model_name,
        "prompt_version": article.prompt_version,
        "safety_flags": json.loads(article.safety_flags_json),
        "wechat_media_id": article.wechat_media_id,
        "wechat_publish_id": article.wechat_publish_id,
        "created_at": article.created_at,
        "published_at": article.published_at,
        "assets": [
            {
                "id": asset.id,
                "asset_type": asset.asset_type,
                "position": asset.position,
                "prompt": asset.prompt,
                "local_path": asset.local_path,
                "source_url": asset.source_url,
                "source_kind": asset.source_kind,
                "source_page_url": asset.source_page_url,
                "source_author": asset.source_author,
                "license_name": asset.license_name,
                "license_url": asset.license_url,
                "wechat_url": asset.wechat_url,
            }
            for asset in sorted(article.assets, key=lambda item: item.position)
        ],
    }


def automation_plan_dict(plan: AutomationPlan) -> dict:
    return {
        "id": plan.id,
        "name": plan.name,
        "content_type": plan.content_type or "health_article",
        "story_mode": plan.story_mode or "auto",
        "audience_profile_id": plan.audience_profile_id,
        "audience_slug": plan.audience.slug,
        "directions": plan.directions,
        "category": plan.category,
        "provider": plan.provider,
        "run_time": plan.run_time,
        "timezone": plan.timezone,
        "title_count": plan.title_count,
        "image_count": plan.image_count,
        "target_length": plan.target_length,
        "headline_intensity": plan.headline_intensity,
        "emotional_title_reference": plan.emotional_title_reference,
        "humanize": plan.humanize,
        "auto_upload_draft": plan.auto_upload_draft,
        "enabled": plan.enabled,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


def automation_run_dict(run: AutomationRun) -> dict:
    return {
        "id": run.id,
        "automation_plan_id": run.automation_plan_id,
        "article_id": run.article_id,
        "scheduled_for": run.scheduled_for,
        "status": run.status,
        "topic": run.topic,
        "topic_rationale": run.topic_rationale,
        "error_message": run.error_message,
        "retry_count": run.retry_count,
        "next_attempt_at": run.next_attempt_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "created_at": run.created_at,
    }


def video_project_dict(project: VideoProject) -> dict:
    request_settings = json.loads(project.settings_json or "{}")
    sourced_scenes = [scene for scene in project.scenes if scene.image_path]
    actual_web_image_ratio = (
        round(
            sum(scene.image_source_kind == "web" for scene in sourced_scenes)
            / len(sourced_scenes),
            4,
        )
        if sourced_scenes
        else None
    )
    output_url = (
        f"/api/video-projects/{project.id}/output" if project.output_path else None
    )
    subtitle_url = (
        f"/api/video-projects/{project.id}/subtitles" if project.subtitle_path else None
    )
    return {
        "id": project.id,
        "source_type": project.source_type,
        "source_url": project.source_url,
        "source_topic": project.source_topic,
        "source_title": project.source_title,
        "source_text": project.source_text,
        "title": project.title,
        "status": project.status,
        "current_step": project.current_step,
        "progress": project.progress,
        "target_duration_seconds": project.target_duration_seconds,
        "width": project.width,
        "height": project.height,
        "fps": project.fps,
        "voice": project.voice,
        "llm_provider": project.llm_provider,
        "workflow": request_settings.get("workflow", "explainer"),
        "news_review_enabled": bool(request_settings.get("news_review_enabled", False)),
        "news_angle": request_settings.get("news_angle", "auto"),
        "news_model_routes": request_settings.get("news_model_routes", {}),
        "research": json.loads(project.research_json or "{}"),
        "review": json.loads(project.review_json or "{}"),
        "visual_profile": request_settings.get("visual_profile", DEFAULT_VISUAL_PROFILE),
        "script_prompt": request_settings.get("script_prompt", ""),
        "style": request_settings.get("style", ""),
        "subtitles_enabled": project.subtitles_enabled,
        "bookends": bool(request_settings.get("bookends", False)),
        "use_web_images": bool(request_settings.get("use_web_images", False)),
        "web_image_ratio": float(request_settings.get("web_image_ratio", 0.95)),
        "actual_web_image_ratio": actual_web_image_ratio,
        "resolution": request_settings.get(
            "resolution", "1080p" if project.width >= 1920 else "720p"
        ),
        "scene_duration_seconds": float(
            request_settings.get("scene_duration_seconds", 3.5)
        ),
        "speech_speed": float(request_settings.get("speech_speed", 1.2)),
        "scene_gap_ms": int(request_settings.get("scene_gap_ms", 120)),
        "trim_speech_silence": bool(
            request_settings.get("trim_speech_silence", True)
        ),
        "safety_flags": json.loads(project.safety_flags_json or "[]"),
        "output_url": output_url,
        "subtitle_url": subtitle_url,
        "actual_duration_ms": project.actual_duration_ms,
        "error_message": project.error_message,
        "retry_count": project.retry_count,
        "cancel_requested": project.cancel_requested,
        "created_at": project.created_at,
        "started_at": project.started_at,
        "completed_at": project.completed_at,
        "scenes": [
            {
                "id": scene.id,
                "position": scene.position,
                "narration": scene.narration,
                "subtitle": scene.subtitle,
                "visual_intent": scene.visual_intent,
                "image_prompt": scene.image_prompt,
                "overlay_labels": json.loads(scene.overlay_labels_json or "[]"),
                "search_query": scene.search_query,
                "image_path": scene.image_path,
                "image_source_url": scene.image_source_url,
                "image_source_kind": scene.image_source_kind,
                "image_source_page_url": scene.image_source_page_url,
                "image_source_author": scene.image_source_author,
                "image_license_name": scene.image_license_name,
                "image_license_url": scene.image_license_url,
                "audio_path": scene.audio_path,
                "duration_ms": scene.duration_ms,
                "status": scene.status,
                "error_message": scene.error_message,
            }
            for scene in sorted(project.scenes, key=lambda item: item.position)
        ],
    }
