from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import Article, Asset, AudienceProfile, AutomationPlan, AutomationRun
from app.schemas import ArticleGenerateRequest
from app.services import automation, generator


@pytest.fixture
def failed_image_job():
    with TestClient(app) as client:
        with SessionLocal() as session:
            audience = session.scalar(select(AudienceProfile).where(
                AudienceProfile.slug == "middle-age-health"
            ))
            article = generator.generate_article(
                session,
                ArticleGenerateRequest(
                    topic="睡眠健康", audience_slug=audience.slug,
                    provider="mock", image_count=3,
                ),
                audience, settings, generate_images=False,
            )
            plan = AutomationPlan(
                name=f"retry-images-{uuid4()}", audience_profile_id=audience.id,
                directions="睡眠健康", provider="mock", image_count=3,
                auto_upload_draft=True,
            )
            session.add(plan)
            session.flush()
            run = AutomationRun(
                automation_plan_id=plan.id, article_id=article.id,
                scheduled_for=datetime.now(UTC), status="failed",
                retry_count=settings.automation_max_retries,
                error_message="MiniMax image quota exhausted",
            )
            session.add_all([run, Asset(
                article_id=article.id, position=0, prompt="saved cover",
                source_url="https://images.test/saved-cover.jpg",
            )])
            session.commit()
            article_id, run_id, plan_id = article.id, run.id, plan.id
        try:
            yield client, article_id, run_id
        finally:
            with SessionLocal() as session:
                session.delete(session.get(AutomationPlan, plan_id))
                session.delete(session.get(Article, article_id))
                session.commit()


def test_retry_images_resumes_exhausted_job_and_preserves_partial_progress(
    failed_image_job, monkeypatch,
):
    client, article_id, run_id = failed_image_job
    endpoint = f"/admin/articles/{article_id}/retry-images"
    preview = client.get(f"/admin/articles/{article_id}")
    assert "重试配图" in preview.text
    assert "自动上传草稿箱" in preview.text
    assert "MiniMax image quota exhausted" in preview.text
    assert endpoint in client.get("/admin").text

    positions, uploads = [], []
    fail_last = True

    class Images:
        name = "test"

        def generate(self, prompt, title, position, search_query=None):
            positions.append(position)
            if position == 2 and fail_last:
                raise RuntimeError("quota still exhausted")
            return {"source_url": f"https://images.test/{position}.jpg"}

    def reject_text(*args, **kwargs):
        pytest.fail("retry must not regenerate text")

    monkeypatch.setattr(generator, "build_image_provider", lambda _: Images())
    monkeypatch.setattr(automation, "generate_article", reject_text)
    monkeypatch.setattr(automation, "upload_article_to_draft",
                        lambda session, article, settings: uploads.append(article.id))
    with SessionLocal() as session:
        saved_text = session.get(Article, article_id).content_json
        saved_cover_id = session.scalar(select(Asset.id).where(
            Asset.article_id == article_id, Asset.position == 0,
        ))
        run_count = session.scalar(select(func.count(AutomationRun.id)))

    response = client.post(endpoint, follow_redirects=False)
    assert response.status_code == 303
    assert "已加入重试队列" in client.get(response.headers["location"]).text
    assert "未重复提交" in client.post(endpoint).text
    with SessionLocal() as session:
        run = session.get(AutomationRun, run_id)
        assert run.status == "pending"
        assert run.retry_count == 0
        assert run.error_message is None
        assert run.next_attempt_at is not None
        assert session.scalar(select(func.count(AutomationRun.id))) == run_count
        assert automation.claim_next_automation_run(session, settings) == run_id

    assert "未重复提交" in client.post(endpoint).text
    with SessionLocal() as session:
        assert session.get(AutomationRun, run_id).status == "running"
        with pytest.raises(RuntimeError, match="quota still exhausted"):
            automation.execute_automation_run(session, run_id, settings)
        session.rollback()
        run = session.get(AutomationRun, run_id)
        run.status = "failed"
        run.retry_count = settings.automation_max_retries
        run.error_message = "quota still exhausted"
        session.commit()
        assert session.scalars(select(Asset.position).where(
            Asset.article_id == article_id,
        ).order_by(Asset.position)).all() == [0, 1]
    assert uploads == []

    fail_last = False
    assert "已加入重试队列" in client.post(endpoint).text
    with SessionLocal() as session:
        assert automation.claim_next_automation_run(session, settings) == run_id
        completed = automation.execute_automation_run(session, run_id, settings)
        assert completed.status == "completed"
        article = session.get(Article, article_id)
        assert article.content_json == saved_text
        assert article.status == "generated"
        assert saved_cover_id in [asset.id for asset in article.assets]
        assert len(article.assets) == 3
        assert "https://images.test/2.jpg" in article.content_html
    assert positions == [1, 2, 2]
    assert uploads == [article_id]
    assert endpoint not in client.get(f"/admin/articles/{article_id}").text
    assert "未重复提交" in client.post(endpoint).text
    assert uploads == [article_id]


@pytest.mark.parametrize("run_status,article_status,media_id", [
    ("pending", "image_pending", None),
    ("running", "image_pending", None),
    ("completed", "generated", None),
    ("failed", "generated", None),  # e.g. a draft upload failure
    ("failed", "image_pending", "existing-draft"),
])
def test_retry_images_rejects_ineligible_jobs(
    failed_image_job, run_status, article_status, media_id,
):
    client, article_id, run_id = failed_image_job
    with SessionLocal() as session:
        run = session.get(AutomationRun, run_id)
        run.status = run_status
        article = session.get(Article, article_id)
        article.status, article.wechat_media_id = article_status, media_id
        session.commit()
    endpoint = f"/admin/articles/{article_id}/retry-images"
    assert endpoint not in client.get(f"/admin/articles/{article_id}").text
    assert "未重复提交" in client.post(endpoint).text
    with SessionLocal() as session:
        run = session.get(AutomationRun, run_id)
        assert run.status == run_status
        assert run.retry_count == settings.automation_max_retries


def test_retry_images_missing_article(failed_image_job):
    client, _, _ = failed_image_job
    assert client.post("/admin/articles/99999999/retry-images").status_code == 404


def test_emotional_workspace_has_retry_button(failed_image_job):
    client, article_id, run_id = failed_image_job
    with SessionLocal() as session:
        session.get(AutomationRun, run_id).plan.content_type = "emotional_article"
        session.get(Article, article_id).content_type = "emotional_article"
        session.commit()
    assert f"/admin/articles/{article_id}/retry-images" in client.get("/admin/emotions").text
