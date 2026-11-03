from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, wechat_settings_for_content
from ..models import Article, PublishJob
from .wechat import WechatClient


def upload_article_to_draft(
    session: Session,
    article: Article,
    settings: Settings,
    *,
    client: WechatClient | None = None,
) -> tuple[str, str]:
    wechat = client or WechatClient(wechat_settings_for_content(settings, article.content_type))
    # Serialize uploads for this article on PostgreSQL to avoid double clicks
    # creating two drafts. Refresh IDs after waiting for another upload's lock.
    session.refresh(article, with_for_update=True)
    if article.llm_provider == "manual-claude" and article.status not in {
        "approved", "wechat_draft", "wechat_draft_unverified"
    }:
        raise ValueError("请先在微恐手动写作工作台审核成稿并确认配图")
    is_mock = bool(article.wechat_media_id and article.wechat_media_id.startswith("mock-draft-"))
    if article.wechat_media_id and not (wechat.enabled and is_mock):
        if wechat.enabled:
            try:
                wechat.verify_draft(article.wechat_media_id, article.selected_title)
            except Exception:
                article.status = "wechat_draft_unverified"
                session.commit()
                raise
            article.status = "wechat_draft"
            for previous in session.scalars(select(PublishJob).where(
                PublishJob.article_id == article.id,
                PublishJob.wechat_media_id == article.wechat_media_id,
                PublishJob.status == "completed_unverified",
            )):
                previous.status = "completed"
                previous.error_message = None
            session.commit()
            return article.wechat_media_id, "已核验现有微信草稿，无需重复上传"
        session.commit()
        return article.wechat_media_id, "当前为模拟模式，未上传新草稿"

    job = PublishJob(article_id=article.id, job_type="draft", status="running")
    session.add(job)
    session.flush()
    try:
        if wechat.enabled:
            media_id = wechat.create_draft(article)
            message = "已上传到公众号草稿箱"
        else:
            media_id = f"mock-draft-{article.id}"
            message = "当前为模拟模式，仅创建本地模拟草稿，未上传微信"
        article.wechat_media_id = media_id
        article.status = "wechat_draft"
        job.wechat_media_id = media_id
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        session.commit()

        if wechat.enabled:
            try:
                wechat.verify_draft(media_id, article.selected_title)
                message = "已上传到公众号草稿箱，并通过微信接口核验成功"
            except Exception as exc:  # noqa: BLE001 - preserve a successfully created draft
                article.status = "wechat_draft_unverified"
                job.status = "completed_unverified"
                job.error_message = str(exc)[:2000]
                session.commit()
                message = f"微信已返回草稿 ID，但核验暂未通过；请重新核验，无需重复上传：{exc}"
        return media_id, message
    except Exception as exc:
        session.rollback()
        failed_job = PublishJob(
            article_id=article.id,
            job_type="draft",
            status="failed",
            error_message=str(exc)[:2000],
        )
        session.add(failed_job)
        session.commit()
        raise
