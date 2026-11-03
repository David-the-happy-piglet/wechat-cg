import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .application.audiences import (
    AudienceNotAvailableError,
    require_enabled_audience,
)
from .application.automation_plans import (
    AutomationPlanNotFoundError,
    InvalidTimezoneError,
    create_automation_plan,
    update_automation_plan,
)
from .config import settings, wechat_settings_for_content
from .content_workflows import content_workflow_metadata
from .database import get_db
from .models import (
    Article,
    ArticleMetricDaily,
    AudienceProfile,
    AutomationPlan,
    AutomationRun,
    PublishJob,
)
from .repositories import create_audience, update_audience
from .schemas import (
    ArticleComparisonRequest,
    ArticleGenerateRequest,
    ArticleRead,
    AudienceCreate,
    AudienceRead,
    AudienceUpdate,
    AutomationPlanCreate,
    AutomationPlanRead,
    AutomationPlanUpdate,
    AutomationRunRead,
    MetricCreate,
    PublishResult,
)
from .serializers import (
    article_dict,
    audience_dict,
    automation_plan_dict,
    automation_run_dict,
)
from .services.automation import enqueue_run_now
from .services.generator import generate_article
from .services.llm import provider_catalog
from .services.publishing import upload_article_to_draft
from .services.usage import token_usage_summary
from .services.video.speech import speech_voice_options
from .services.wechat import WechatClient

router = APIRouter(prefix="/api")
wechat_client = WechatClient(settings)
emotional_wechat_client = WechatClient(wechat_settings_for_content(settings, "emotional_article"))


def _wechat_client_for(content_type: str) -> WechatClient:
    return emotional_wechat_client if content_type == "emotional_article" else wechat_client


@router.get("/wechat/status")
def wechat_connection_status(content_type: str = "health_article"):
    if content_type not in {"health_article", "emotional_article", "micro_horror"}:
        raise HTTPException(status_code=422, detail="Unsupported content_type")
    try:
        return _wechat_client_for(content_type).check_connection()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _enabled_audience_or_404(session: Session, slug: str) -> AudienceProfile:
    try:
        return require_enabled_audience(session, slug)
    except AudienceNotAvailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def get_article_or_404(session: Session, article_id: int) -> Article:
    article = session.scalar(
        select(Article).where(Article.id == article_id).options(selectinload(Article.assets))
    )
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.get("/audiences", response_model=list[AudienceRead])
def list_audiences(session: Session = Depends(get_db)):
    audiences = session.scalars(select(AudienceProfile).order_by(AudienceProfile.id)).all()
    return [audience_dict(item) for item in audiences]


@router.post(
    "/audiences",
    response_model=AudienceRead,
    status_code=status.HTTP_201_CREATED,
)
def add_audience(data: AudienceCreate, session: Session = Depends(get_db)):
    try:
        audience = create_audience(session, data)
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="Audience slug already exists") from exc
    return audience_dict(audience)


@router.put("/audiences/{audience_id}", response_model=AudienceRead)
def edit_audience(
    audience_id: int,
    data: AudienceUpdate,
    session: Session = Depends(get_db),
):
    audience = session.get(AudienceProfile, audience_id)
    if not audience:
        raise HTTPException(status_code=404, detail="Audience not found")
    return audience_dict(update_audience(session, audience, data))


@router.get("/articles", response_model=list[ArticleRead])
def list_articles(session: Session = Depends(get_db)):
    articles = session.scalars(
        select(Article).options(selectinload(Article.assets)).order_by(Article.created_at.desc())
    ).all()
    return [article_dict(item) for item in articles]


@router.get("/providers")
def list_providers():
    return {
        "text": provider_catalog(settings),
        "image": {
            "provider": settings.image_provider,
            "source_mode": settings.image_source_mode,
            "web_provider": (
                settings.web_image_provider
                if settings.image_source_mode in {"mixed", "web"}
                else None
            ),
            "model": (
                settings.minimax_image_model
                if settings.image_provider == "minimax"
                else settings.image_model or "local-placeholder"
            ),
            "configured": (
                (
                    bool(settings.minimax_api_key)
                    if settings.image_provider == "minimax"
                    else settings.image_provider == "placeholder" or bool(settings.image_api_key)
                )
                and (
                    settings.web_image_provider in {"openverse", "wikimedia"}
                    if settings.image_source_mode in {"mixed", "web"}
                    else True
                )
            ),
        },
        "speech": {
            "provider": settings.speech_provider,
            "model": (
                settings.speech_model
                if settings.speech_provider == "minimax"
                else "local-wave"
            ),
            "voice": settings.speech_voice,
            "voices": speech_voice_options(settings),
            "configured": (
                settings.speech_provider == "mock"
                or (
                    settings.speech_provider == "minimax"
                    and bool(settings.minimax_api_key)
                )
            ),
        },
    }


@router.get("/content-workflows")
def list_content_workflow_metadata():
    return content_workflow_metadata()


@router.get("/token-usage")
def get_token_usage(session: Session = Depends(get_db)):
    return token_usage_summary(session)


@router.get("/automation/plans", response_model=list[AutomationPlanRead])
def list_automation_plans(session: Session = Depends(get_db)):
    plans = session.scalars(
        select(AutomationPlan)
        .options(selectinload(AutomationPlan.audience))
        .order_by(AutomationPlan.id)
    ).all()
    return [automation_plan_dict(plan) for plan in plans]


@router.post(
    "/automation/plans",
    response_model=AutomationPlanRead,
    status_code=status.HTTP_201_CREATED,
)
def create_automation_plan_endpoint(
    data: AutomationPlanCreate,
    session: Session = Depends(get_db),
):
    try:
        plan = create_automation_plan(session, data)
    except AudienceNotAvailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidTimezoneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return automation_plan_dict(plan)


@router.put(
    "/automation/plans/{plan_id}",
    response_model=AutomationPlanRead,
)
def update_automation_plan_endpoint(
    plan_id: int,
    data: AutomationPlanUpdate,
    session: Session = Depends(get_db),
):
    try:
        plan = update_automation_plan(session, plan_id, data)
    except (AutomationPlanNotFoundError, AudienceNotAvailableError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidTimezoneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return automation_plan_dict(plan)


@router.post(
    "/automation/plans/{plan_id}/run-now",
    response_model=AutomationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_automation_plan_now(
    plan_id: int,
    session: Session = Depends(get_db),
):
    plan = session.get(AutomationPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Automation plan not found")
    return automation_run_dict(enqueue_run_now(session, plan))


@router.get("/automation/runs", response_model=list[AutomationRunRead])
def list_automation_runs(
    limit: int = 30,
    session: Session = Depends(get_db),
):
    runs = session.scalars(
        select(AutomationRun)
        .order_by(AutomationRun.created_at.desc())
        .limit(min(max(limit, 1), 200))
    ).all()
    return [automation_run_dict(run) for run in runs]


@router.get("/articles/{article_id}", response_model=ArticleRead)
def get_article(article_id: int, session: Session = Depends(get_db)):
    return article_dict(get_article_or_404(session, article_id))


@router.post(
    "/articles/generate",
    response_model=ArticleRead,
    status_code=status.HTTP_201_CREATED,
)
def create_article(
    data: ArticleGenerateRequest,
    session: Session = Depends(get_db),
):
    audience = _enabled_audience_or_404(session, data.audience_slug)
    try:
        article = generate_article(session, data, audience, settings)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
    return article_dict(article)


@router.post("/articles/compare")
def compare_articles(
    data: ArticleComparisonRequest,
    session: Session = Depends(get_db),
):
    audience = _enabled_audience_or_404(session, data.audience_slug)

    base_request = data.model_dump(exclude={"providers"})
    articles = []
    errors = []
    for provider in data.providers:
        try:
            article = generate_article(
                session,
                ArticleGenerateRequest(**base_request, provider=provider),
                audience,
                settings,
            )
            articles.append(article_dict(article))
        except Exception as exc:  # noqa: BLE001 - report providers independently
            session.rollback()
            errors.append({"provider": provider, "error": str(exc)})
    return {
        "topic": data.topic,
        "content_type": data.content_type,
        "story_mode": data.story_mode,
        "audience_slug": data.audience_slug,
        "articles": articles,
        "errors": errors,
    }


@router.post("/articles/{article_id}/wechat-draft", response_model=PublishResult)
def upload_wechat_draft(
    article_id: int,
    session: Session = Depends(get_db),
):
    article = get_article_or_404(session, article_id)
    try:
        media_id, message = upload_article_to_draft(
            session,
            article,
            settings,
            client=_wechat_client_for(article.content_type),
        )
        return PublishResult(
            article_id=article.id,
            status=article.status,
            media_id=media_id,
            message=message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"WeChat draft failed: {exc}") from exc


@router.post("/articles/{article_id}/publish", response_model=PublishResult)
def publish_article(
    article_id: int,
    session: Session = Depends(get_db),
):
    article = get_article_or_404(session, article_id)
    if not article.wechat_media_id:
        raise HTTPException(status_code=409, detail="Create a WeChat draft first")
    account_client = _wechat_client_for(article.content_type)
    if account_client.enabled and article.wechat_media_id.startswith("mock-draft-"):
        raise HTTPException(status_code=409, detail="请先上传真实微信草稿")
    try:
        if account_client.enabled:
            publish_id = account_client.publish(article.wechat_media_id)
            message = "Article submitted for WeChat publishing"
            article.status = "publishing"
        else:
            publish_id = f"mock-publish-{article.id}"
            message = "WeChat is disabled; mock publish completed"
            article.status = "published"
            article.published_at = datetime.now(UTC)
        article.wechat_publish_id = publish_id
        session.add(
            PublishJob(
                article_id=article.id,
                job_type="publish",
                status="completed",
                wechat_media_id=article.wechat_media_id,
                wechat_publish_id=publish_id,
                completed_at=datetime.now(UTC),
            )
        )
        session.commit()
        return PublishResult(
            article_id=article.id,
            status=article.status,
            media_id=article.wechat_media_id,
            publish_id=publish_id,
            message=message,
        )
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=502, detail=f"WeChat publish failed: {exc}") from exc


@router.post("/articles/{article_id}/metrics", status_code=status.HTTP_201_CREATED)
def add_metric(
    article_id: int,
    data: MetricCreate,
    session: Session = Depends(get_db),
):
    get_article_or_404(session, article_id)
    existing = session.scalar(
        select(ArticleMetricDaily).where(
            ArticleMetricDaily.article_id == article_id,
            ArticleMetricDaily.metric_date == data.metric_date,
        )
    )
    values = data.model_dump()
    raw = values.pop("raw")
    if existing:
        metric = existing
        for key, value in values.items():
            setattr(metric, key, value)
        metric.raw_json = json.dumps(raw, ensure_ascii=False)
    else:
        metric = ArticleMetricDaily(
            article_id=article_id,
            **values,
            raw_json=json.dumps(raw, ensure_ascii=False),
        )
        session.add(metric)
    session.commit()
    return {"article_id": article_id, **data.model_dump()}
