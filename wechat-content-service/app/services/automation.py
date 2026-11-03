import json
import logging
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import case, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from ..config import Settings
from ..content_workflows import get_content_workflow
from ..emotional_defaults import (
    CLEAR_SENTENCE_CATEGORY,
    DEFAULT_EMOTIONAL_CATEGORY,
    HEARTBREAK_CATEGORY,
)
from ..models import (
    Article,
    AutomationPlan,
    AutomationRun,
    EmotionalTopicReview,
    GenerationRun,
)
from ..schemas import ArticleGenerateRequest
from .generator import audience_to_context, generate_article, generate_article_images
from .llm import build_llm_provider
from .publishing import upload_article_to_draft
from .topic_reviews import create_review, reviewed_request

logger = logging.getLogger(__name__)


def _resolved_article_category(
    content_type: str,
    configured_category: str,
    generated_category: str | None,
) -> str:
    if (
        content_type == "emotional_article"
        and configured_category.strip() in {HEARTBREAK_CATEGORY, DEFAULT_EMOTIONAL_CATEGORY, CLEAR_SENTENCE_CATEGORY}
    ):
        return configured_category.strip()
    return generated_category or configured_category


def _timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"Unknown timezone: {name}") from exc


def _run_clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("run_time must use HH:MM format") from exc


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def scheduled_datetime(
    plan: AutomationPlan,
    local_date,
) -> datetime:
    local = datetime.combine(
        local_date,
        _run_clock(plan.run_time),
        tzinfo=_timezone(plan.timezone),
    )
    return local.astimezone(UTC)


def enqueue_due_plans(
    session: Session,
    *,
    now: datetime | None = None,
) -> list[int]:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    plans = session.scalars(select(AutomationPlan).where(AutomationPlan.enabled.is_(True))).all()
    created: list[int] = []
    for plan in plans:
        try:
            timezone = _timezone(plan.timezone)
            local_date = now.astimezone(timezone).date()
            scheduled_for = scheduled_datetime(plan, local_date)
        except (OverflowError, ValueError) as exc:
            logger.error(
                "Skipping invalid automation plan id=%s name=%r timezone=%r run_time=%r: %s",
                plan.id,
                plan.name,
                plan.timezone,
                plan.run_time,
                exc,
            )
            continue
        if scheduled_for > now:
            continue
        run = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=scheduled_for,
            status="pending",
            next_attempt_at=now,
        )
        session.add(run)
        try:
            session.commit()
            session.refresh(run)
            created.append(run.id)
        except IntegrityError:
            session.rollback()
    return created


def enqueue_run_now(
    session: Session,
    plan: AutomationPlan,
    *,
    now: datetime | None = None,
) -> AutomationRun:
    now = now or datetime.now(UTC)
    run = AutomationRun(
        automation_plan_id=plan.id,
        scheduled_for=now,
        status="pending",
        next_attempt_at=now,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def retry_article_images(
    session: Session,
    article_id: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Requeue the original failed image job, including exhausted jobs.

    Use the same guarded update as worker claiming so concurrent clicks or a
    worker winning the race cannot reset an already queued/running job.
    """
    now = now or datetime.now(UTC)
    result = session.execute(
        update(AutomationRun)
        .where(
            AutomationRun.article_id == article_id,
            AutomationRun.status == "failed",
            AutomationRun.article.has(
                (Article.status == "image_pending")
                & Article.wechat_media_id.is_(None)
            ),
        )
        .values(
            status="pending",
            retry_count=0,
            next_attempt_at=now,
            started_at=None,
            completed_at=None,
            error_message=None,
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return bool(result.rowcount)


def recover_stale_runs(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(minutes=settings.automation_stale_minutes)
    next_retry_count = AutomationRun.retry_count + 1
    result = session.execute(
        update(AutomationRun)
        .where(
            AutomationRun.status == "running",
            AutomationRun.started_at < cutoff,
        )
        .values(
            status="failed",
            error_message="Worker stopped before completing the run",
            retry_count=next_retry_count,
            next_attempt_at=case(
                (next_retry_count < settings.automation_max_retries, now),
                else_=None,
            ),
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount or 0


def _runnable_filters(settings: Settings, now: datetime) -> tuple:
    return (
        AutomationRun.retry_count < settings.automation_max_retries,
        AutomationRun.status.in_(["pending", "failed"]),
        or_(
            AutomationRun.next_attempt_at.is_(None),
            AutomationRun.next_attempt_at <= now,
        ),
    )


def next_runnable_run(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> AutomationRun | None:
    now = now or datetime.now(UTC)
    return session.scalar(
        select(AutomationRun)
        .where(*_runnable_filters(settings, now))
        .order_by(AutomationRun.scheduled_for, AutomationRun.id)
        .limit(1)
    )


def claim_next_automation_run(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> int | None:
    """Atomically claim the oldest runnable job and return its stable ID."""
    now = now or datetime.now(UTC)

    while True:
        statement = (
            select(AutomationRun.id)
            .where(*_runnable_filters(settings, now))
            .order_by(AutomationRun.scheduled_for, AutomationRun.id)
            .limit(1)
        )
        if session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)

        run_id = session.scalar(statement)
        if run_id is None:
            return None

        claimed = session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == run_id,
                *_runnable_filters(settings, now),
            )
            .values(
                status="running",
                started_at=now,
                error_message=None,
            )
            .execution_options(synchronize_session=False)
        )
        if claimed.rowcount == 1:
            session.commit()
            return run_id

        # SQLite has no row-level SKIP LOCKED support. The guarded update still
        # prevents a stale candidate from being claimed twice; retry the lookup
        # in a fresh transaction when another worker won the race.
        session.rollback()


def _choose_topic(
    session: Session,
    plan: AutomationPlan,
    settings: Settings,
    local_date: str,
) -> tuple[dict, GenerationRun]:
    workflow = get_content_workflow(plan.content_type)
    recent_topics = session.scalars(
        select(Article.topic)
        .where(
            Article.audience_profile_id == plan.audience_profile_id,
            Article.content_type == plan.content_type,
        )
        .order_by(Article.created_at.desc())
        .limit(30)
    ).all()
    prompt = workflow.build_topic_prompt(
        directions=plan.directions,
        category=plan.category,
        audience=audience_to_context(plan.audience),
        recent_topics=list(recent_topics),
        local_date=local_date,
        story_mode=plan.story_mode,
    )
    llm = build_llm_provider(settings, plan.provider)
    started = datetime.now(UTC)
    result = llm.generate_json(
        prompt,
        system_prompt=workflow.topic_system_prompt,
    )
    for key in ("topic", "category", "rationale"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Topic response missing valid {key}")
    if len(result["topic"]) > 300:
        raise ValueError("Generated topic is longer than 300 characters")
    if result["topic"] in recent_topics:
        raise ValueError("Generated topic duplicates a recent article")
    generation_run = GenerationRun(
        step="topic",
        provider=llm.name,
        model=llm.model,
        input_text=prompt,
        output_text=json.dumps(result, ensure_ascii=False),
        status="completed",
        duration_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
        prompt_tokens=llm.usage.prompt_tokens,
        completion_tokens=llm.usage.completion_tokens,
        total_tokens=llm.usage.total_tokens,
    )
    return result, generation_run


def execute_automation_run(
    session: Session,
    run_id: int,
    settings: Settings,
) -> AutomationRun:
    run = session.scalar(
        select(AutomationRun)
        .where(AutomationRun.id == run_id)
        .options(
            selectinload(AutomationRun.plan).selectinload(AutomationPlan.audience),
            selectinload(AutomationRun.article).selectinload(Article.assets),
        )
    )
    if not run:
        raise ValueError(f"Automation run {run_id} not found")
    plan = run.plan

    review = session.scalar(select(EmotionalTopicReview).where(
        EmotionalTopicReview.automation_run_id == run.id,
    )) if plan.content_type == "emotional_article" else None
    if not run.article_id:
        if review is not None:
            if review.status != "approved":
                return run
            article_request = reviewed_request(review)
            topic_result = {"topic": review.topic, "rationale": review.rationale}
            topic_generation = session.get(GenerationRun, review.generation_run_id)
        else:
            local_date = (
                _as_utc(run.scheduled_for).astimezone(_timezone(plan.timezone)).date().isoformat()
            )
            topic_result, topic_generation = _choose_topic(session, plan, settings, local_date)
            article_request = ArticleGenerateRequest(
                content_type=plan.content_type,
                story_mode=plan.story_mode,
                topic=topic_result["topic"],
                audience_slug=plan.audience.slug,
                category=_resolved_article_category(
                    plan.content_type, plan.category, topic_result.get("category"),
                ),
                title_count=plan.title_count, image_count=plan.image_count,
                target_length=plan.target_length, headline_intensity=plan.headline_intensity,
                emotional_title_reference=plan.emotional_title_reference,
                humanize=plan.humanize, provider=plan.provider,
            )
            if plan.content_type == "emotional_article":
                session.add(topic_generation)
                session.flush()
                create_review(
                    session, article_request, topic=topic_result["topic"],
                    rationale=topic_result["rationale"], directions=plan.directions,
                    auto_upload=plan.auto_upload_draft, generation_run=topic_generation,
                    automation_run_id=run.id,
                )
                run.topic, run.topic_rationale = topic_result["topic"], topic_result["rationale"]
                run.status, run.next_attempt_at = "awaiting_review", None
                session.commit()
                return run
        article = generate_article(
            session, article_request, plan.audience, settings,
            commit=False, generate_images=False,
        )
        if review is not None:
            review.article_id = article.id
        topic_generation.article_id = article.id
        session.add(topic_generation)
        run.article_id = article.id
        run.topic = topic_result["topic"]
        run.topic_rationale = topic_result["rationale"]
        session.commit()
        run.article = article

    article = run.article or session.get(Article, run.article_id)
    if not article:
        raise ValueError("Automation run article could not be loaded")

    generate_article_images(
        session,
        article,
        settings,
        image_count=(reviewed_request(review).image_count if review else plan.image_count),
        checkpoint_each=True,
    )

    if (json.loads(review.settings_json)["auto_upload_draft"] if review else plan.auto_upload_draft):
        upload_article_to_draft(session, article, settings)

    run.status = "completed"
    if review is not None:
        review.status = "completed"
    run.error_message = None
    run.completed_at = datetime.now(UTC)
    run.next_attempt_at = None
    session.commit()
    session.refresh(run)
    return run


def process_next_run(
    session: Session,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> AutomationRun | None:
    now = now or datetime.now(UTC)
    run_id = claim_next_automation_run(session, settings, now=now)
    if run_id is None:
        return None
    try:
        return execute_automation_run(session, run_id, settings)
    except Exception as exc:
        session.rollback()
        failed = session.get(AutomationRun, run_id)
        if not failed:
            raise
        failed.retry_count += 1
        failed.status = "failed"
        failed.error_message = str(exc)[:2000]
        if failed.retry_count < settings.automation_max_retries:
            delay_seconds = min(60 * (2 ** (failed.retry_count - 1)), 3600)
            failed.next_attempt_at = now + timedelta(seconds=delay_seconds)
        else:
            failed.next_attempt_at = None
        session.commit()
        return failed
