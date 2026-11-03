"""Persist reviewed topics and queue only explicitly approved writing requests."""

import json
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from ..application.audiences import require_enabled_audience
from ..models import Article, AutomationRun, EmotionalTopicReview, GenerationRun
from ..schemas import ArticleGenerateRequest
from .generator import audience_to_context, generate_article, generate_article_images
from .llm import build_llm_provider
from .prompts import build_emotional_topic_prompt
from .publishing import upload_article_to_draft


def suggest_topic(session, request, audience, settings, directions):
    recent = list(
        session.scalars(
            select(Article.topic)
            .where(
                Article.content_type == "emotional_article",
                Article.audience_profile_id == audience.id,
            )
            .order_by(Article.created_at.desc())
            .limit(30)
        )
    )
    prompt = build_emotional_topic_prompt(
        directions=directions,
        category=request.category,
        audience=audience_to_context(audience),
        recent_topics=recent,
        local_date=datetime.now(UTC).date().isoformat(),
    )
    llm = build_llm_provider(settings, request.provider)
    started = time.monotonic()
    result = llm.generate_json(
        prompt, system_prompt="只生成一个具体、口语化的恋爱选题，不写文章。只输出合法 JSON。"
    )
    for key in ("topic", "category", "rationale"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"选题缺少 {key}")
    if len(result["topic"]) > 300 or result["topic"] in recent:
        raise ValueError("选题过长或与已有文章重复，请换一个")
    record = GenerationRun(
        step="topic",
        provider=llm.name,
        model=llm.model,
        input_text=prompt,
        output_text=json.dumps(result, ensure_ascii=False),
        status="completed",
        duration_ms=int((time.monotonic() - started) * 1000),
        prompt_tokens=llm.usage.prompt_tokens,
        completion_tokens=llm.usage.completion_tokens,
        total_tokens=llm.usage.total_tokens,
    )
    session.add(record)
    session.flush()
    return result, record


def create_review(
    session,
    request,
    *,
    topic,
    rationale,
    directions,
    auto_upload,
    generation_run=None,
    automation_run_id=None,
):
    review = EmotionalTopicReview(
        topic=topic,
        category=request.category,
        rationale=rationale,
        automation_run_id=automation_run_id,
        generation_run_id=generation_run.id if generation_run else None,
        settings_json=json.dumps(
            {
                "request": request.model_dump(),
                "directions": directions,
                "auto_upload_draft": auto_upload,
            },
            ensure_ascii=False,
        ),
    )
    session.add(review)
    session.flush()
    return review


def reviewed_request(review):
    data = json.loads(review.settings_json)
    request = ArticleGenerateRequest.model_validate(data["request"])
    request.topic = review.topic
    request.category = review.category
    request.source_prompt = (
        request.source_prompt
        + "\n审阅通过的选题："
        + review.topic
        + "\n审阅通过的切入点与佐证方向："
        + review.rationale
        + "\n严格围绕已审选题写作，不重新选择主题；佐证仍须匹配具体处境并有可靠出处。"
    )
    return request


def approve_review(session, review_id, topic, rationale):
    topic, rationale = topic.strip(), rationale.strip()
    if not 3 <= len(topic) <= 300 or len(rationale) > 3000:
        raise ValueError("主题需3至300字，切入点最多3000字")
    review = session.get(EmotionalTopicReview, review_id)
    if review is None:
        raise LookupError("选题不存在")
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.status == "awaiting_review",
        )
        .values(
            status="approved",
            topic=topic,
            rationale=rationale,
            approved_at=datetime.now(UTC),
            error_message=None,
        )
    )
    if not changed.rowcount:
        session.rollback()
        return False
    if review.automation_run_id:
        session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == review.automation_run_id,
                AutomationRun.status == "awaiting_review",
            )
            .values(
                status="pending",
                topic=topic,
                topic_rationale=rationale,
                next_attempt_at=datetime.now(UTC),
                started_at=None,
                error_message=None,
                retry_count=0,
            )
        )
    session.commit()
    return True


def save_review(session, review_id, topic, rationale):
    topic, rationale = topic.strip(), rationale.strip()
    if not 3 <= len(topic) <= 300 or len(rationale) > 3000:
        raise ValueError("主题需3至300字，切入点最多3000字")
    review = session.get(EmotionalTopicReview, review_id)
    if review is None:
        raise LookupError("选题不存在")
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.status == "awaiting_review",
        )
        .values(topic=topic, rationale=rationale)
    )
    if changed.rowcount and review.automation_run_id:
        session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == review.automation_run_id,
                AutomationRun.status == "awaiting_review",
            )
            .values(topic=topic, topic_rationale=rationale)
        )
    session.commit()


def reject_review(session, review_id):
    review = session.get(EmotionalTopicReview, review_id)
    if review is None:
        raise LookupError("选题不存在")
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.status == "awaiting_review",
        )
        .values(status="rejected")
    )
    if changed.rowcount and review.automation_run_id:
        session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == review.automation_run_id,
                AutomationRun.status == "awaiting_review",
            )
            .values(status="cancelled", next_attempt_at=None)
        )
    session.commit()


def regenerate_review(session, review_id, settings):
    review = session.get(EmotionalTopicReview, review_id)
    if review is None:
        raise LookupError("选题不存在")
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.status == "awaiting_review",
        )
        .values(status="regenerating", started_at=datetime.now(UTC))
    )
    if not changed.rowcount:
        session.rollback()
        return False
    session.commit()
    try:
        data = json.loads(review.settings_json)
        request = ArticleGenerateRequest.model_validate(data["request"])
        audience = require_enabled_audience(session, request.audience_slug)
        result, record = suggest_topic(
            session,
            request,
            audience,
            settings,
            data["directions"] + "\n不要复用这个旧选题或只换词：" + review.topic,
        )
        review.topic = result["topic"]
        review.rationale = result["rationale"]
        review.generation_run_id = record.id
        review.status = "awaiting_review"
        review.error_message = None
        if review.automation_run_id:
            run = session.get(AutomationRun, review.automation_run_id)
            run.topic, run.topic_rationale = review.topic, review.rationale
        session.commit()
        return True
    except Exception as exc:
        session.rollback()
        review = session.get(EmotionalTopicReview, review_id)
        review.status = "awaiting_review"
        review.error_message = str(exc)[:2000]
        session.commit()
        raise


def process_next_manual_review(session, settings):
    review_id = session.scalar(
        select(EmotionalTopicReview.id)
        .where(
            EmotionalTopicReview.automation_run_id.is_(None),
            EmotionalTopicReview.status == "approved",
        )
        .order_by(EmotionalTopicReview.id)
        .limit(1)
    )
    if review_id is None:
        return None
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.status == "approved",
        )
        .values(status="running", started_at=datetime.now(UTC))
    )
    session.commit()
    if not changed.rowcount:
        return None
    review = session.get(EmotionalTopicReview, review_id)
    try:
        request = reviewed_request(review)
        audience = require_enabled_audience(session, request.audience_slug)
        if review.article_id is None:
            article = generate_article(
                session, request, audience, settings, commit=False, generate_images=False
            )
            review.article_id = article.id
            if review.generation_run_id:
                session.get(GenerationRun, review.generation_run_id).article_id = article.id
            session.commit()
        article = session.get(Article, review.article_id)
        generate_article_images(
            session, article, settings, image_count=request.image_count, checkpoint_each=True
        )
        if json.loads(review.settings_json)["auto_upload_draft"]:
            upload_article_to_draft(session, article, settings)
        review.status, review.error_message = "completed", None
        session.commit()
    except Exception as exc:  # noqa: BLE001 - persist worker failures for explicit retry
        session.rollback()
        review = session.get(EmotionalTopicReview, review_id)
        review.status, review.error_message = "failed", str(exc)[:2000]
        session.commit()
    return review


def retry_manual_review(session, review_id):
    changed = session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.id == review_id,
            EmotionalTopicReview.automation_run_id.is_(None),
            EmotionalTopicReview.status == "failed",
            EmotionalTopicReview.approved_at.is_not(None),
        )
        .values(status="approved", error_message=None)
    )
    session.commit()
    return bool(changed.rowcount)


def recover_stale_reviews(session, settings):
    cutoff = datetime.now(UTC) - timedelta(minutes=settings.automation_stale_minutes)
    session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.status == "regenerating",
            EmotionalTopicReview.started_at < cutoff,
        )
        .values(status="awaiting_review", error_message="换题中断，已保留上次选题")
    )
    session.execute(
        update(EmotionalTopicReview)
        .where(
            EmotionalTopicReview.automation_run_id.is_(None),
            EmotionalTopicReview.status == "running",
            EmotionalTopicReview.started_at < cutoff,
        )
        .values(status="failed", error_message="生成中断，可重试；已保存的正文和图片会复用")
    )
    session.commit()
