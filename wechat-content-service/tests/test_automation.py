import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, SessionLocal
from app.main import app
from app.models import (
    Article,
    Asset,
    AudienceProfile,
    AutomationPlan,
    AutomationRun,
    GenerationRun,
)
from app.services import automation as automation_service
from app.services import generator as generator_service
from app.services.automation import (
    _resolved_article_category,
    claim_next_automation_run,
    enqueue_due_plans,
    execute_automation_run,
    recover_stale_runs,
)


def test_heartbreak_automation_keeps_configured_category():
    assert _resolved_article_category(
        "emotional_article", "失恋主题", "情感安慰"
    ) == "失恋主题"
    assert _resolved_article_category(
        "emotional_article", "恋爱心理", "情感安慰"
    ) == "恋爱心理"
    assert _resolved_article_category(
        "emotional_article", "恋爱相处", "情感安慰"
    ) == "情感安慰"


def _create_plan(session, *, name: str) -> AutomationPlan:
    audience = session.scalar(
        select(AudienceProfile).where(AudienceProfile.slug == "middle-age-health")
    )
    assert audience is not None
    plan = AutomationPlan(
        name=name,
        audience_profile_id=audience.id,
        directions="可靠性测试专用内容方向",
        provider="mock",
        image_count=1,
        auto_upload_draft=False,
    )
    session.add(plan)
    session.flush()
    return plan


def test_claim_next_automation_run_claims_each_run_once():
    with TestClient(app):
        pass

    now = datetime.now(UTC)
    with SessionLocal() as session:
        plan = _create_plan(session, name=f"claim-{uuid4()}")
        first = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now - timedelta(minutes=2),
            status="pending",
            next_attempt_at=now,
        )
        second = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now - timedelta(minutes=1),
            status="pending",
            next_attempt_at=now,
        )
        session.add_all([first, second])
        session.commit()
        plan_id = plan.id
        first_id = first.id
        second_id = second.id

    with SessionLocal() as first_worker:
        claimed_first = claim_next_automation_run(first_worker, settings, now=now)
    with SessionLocal() as second_worker:
        claimed_second = claim_next_automation_run(second_worker, settings, now=now)

    assert claimed_first == first_id
    assert claimed_second == second_id

    with SessionLocal() as session:
        stored_first = session.get(AutomationRun, first_id)
        stored_second = session.get(AutomationRun, second_id)
        assert stored_first is not None
        assert stored_second is not None
        assert stored_first.status == "running"
        assert stored_second.status == "running"
        assert stored_first.started_at is not None
        assert stored_second.started_at is not None
        plan = session.get(AutomationPlan, plan_id)
        assert plan is not None
        session.delete(plan)
        session.commit()


def test_recover_stale_runs_counts_attempt_and_stops_at_retry_limit():
    with TestClient(app):
        pass

    now = datetime.now(UTC)
    recovery_settings = replace(
        settings,
        automation_max_retries=2,
        automation_stale_minutes=30,
    )
    with SessionLocal() as session:
        plan = _create_plan(session, name=f"recovery-{uuid4()}")
        retryable = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now - timedelta(hours=3),
            status="running",
            retry_count=0,
            started_at=now - timedelta(hours=2),
        )
        exhausted = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now - timedelta(hours=2),
            status="running",
            retry_count=1,
            started_at=now - timedelta(hours=2),
        )
        fresh = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now - timedelta(hours=1),
            status="running",
            retry_count=0,
            started_at=now - timedelta(minutes=5),
        )
        session.add_all([retryable, exhausted, fresh])
        session.commit()
        plan_id = plan.id
        retryable_id = retryable.id
        exhausted_id = exhausted.id
        fresh_id = fresh.id

        assert recover_stale_runs(session, recovery_settings, now=now) == 2
        assert recover_stale_runs(session, recovery_settings, now=now) == 0

        session.expire_all()
        retryable = session.get(AutomationRun, retryable_id)
        exhausted = session.get(AutomationRun, exhausted_id)
        fresh = session.get(AutomationRun, fresh_id)
        assert retryable is not None
        assert exhausted is not None
        assert fresh is not None
        assert retryable.status == "failed"
        assert retryable.retry_count == 1
        assert retryable.next_attempt_at is not None
        assert retryable.next_attempt_at.replace(tzinfo=UTC) == now
        assert exhausted.status == "failed"
        assert exhausted.retry_count == 2
        assert exhausted.next_attempt_at is None
        assert fresh.status == "running"
        assert fresh.retry_count == 0

        plan = session.get(AutomationPlan, plan_id)
        assert plan is not None
        session.delete(plan)
        session.commit()


def test_automation_article_and_run_link_commit_together(monkeypatch):
    with TestClient(app):
        pass

    now = datetime.now(UTC)
    with SessionLocal() as session:
        plan = _create_plan(session, name=f"article-checkpoint-{uuid4()}")
        run = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now,
            status="running",
            started_at=now,
        )
        session.add(run)
        session.commit()
        plan_id = plan.id
        run_id = run.id
        article_count_before = session.scalar(select(func.count(Article.id)))

    real_generate_article = automation_service.generate_article

    def fail_after_article_generation(*args, **kwargs):
        assert kwargs.get("commit") is False
        real_generate_article(*args, **kwargs)
        raise RuntimeError("simulated worker crash before run checkpoint")

    monkeypatch.setattr(
        automation_service,
        "generate_article",
        fail_after_article_generation,
    )
    with SessionLocal() as session:
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            execute_automation_run(session, run_id, settings)
        session.rollback()
        stored = session.get(AutomationRun, run_id)
        assert stored is not None
        assert stored.article_id is None
        assert session.scalar(select(func.count(Article.id))) == article_count_before

    monkeypatch.setattr(
        automation_service,
        "generate_article",
        real_generate_article,
    )
    with SessionLocal() as session:
        completed = execute_automation_run(session, run_id, settings)
        assert completed.status == "completed"
        assert completed.article_id is not None
        article_id = completed.article_id
        assert session.scalar(select(func.count(Article.id))) == article_count_before + 1

        def reject_duplicate_generation(*args, **kwargs):
            pytest.fail("a checkpointed automation run must reuse its article")

        monkeypatch.setattr(
            automation_service,
            "generate_article",
            reject_duplicate_generation,
        )
        repeated = execute_automation_run(session, run_id, settings)
        assert repeated.article_id == article_id
        assert session.scalar(select(func.count(Article.id))) == article_count_before + 1

        article = session.get(Article, article_id)
        plan = session.get(AutomationPlan, plan_id)
        assert article is not None
        assert plan is not None
        session.delete(plan)
        session.delete(article)
        session.commit()


def test_automation_reuses_checkpointed_text_and_only_retries_missing_images(
    monkeypatch,
):
    with TestClient(app):
        pass

    now = datetime.now(UTC)
    with SessionLocal() as session:
        plan = _create_plan(session, name=f"image-checkpoint-{uuid4()}")
        plan.image_count = 2
        run = AutomationRun(
            automation_plan_id=plan.id,
            scheduled_for=now,
            status="running",
            started_at=now,
        )
        session.add(run)
        session.commit()
        plan_id = plan.id
        run_id = run.id

    generated_positions = []
    fail_second_image = True

    class QuotaLimitedImages:
        name = "quota-limited"

        def generate(self, prompt, title, position, search_query=None):
            nonlocal fail_second_image
            generated_positions.append(position)
            if position == 1 and fail_second_image:
                fail_second_image = False
                raise RuntimeError("image quota exhausted")
            return {
                "source_kind": "ai",
                "source_url": f"https://images.test/{position}.jpg",
            }

    monkeypatch.setattr(
        generator_service,
        "build_image_provider",
        lambda settings: QuotaLimitedImages(),
    )

    with SessionLocal() as session:
        with pytest.raises(RuntimeError, match="image quota exhausted"):
            execute_automation_run(session, run_id, settings)
        session.rollback()
        stored_run = session.get(AutomationRun, run_id)
        assert stored_run is not None
        assert stored_run.article_id is not None
        article_id = stored_run.article_id
        article = session.get(Article, article_id)
        assert article is not None
        saved_text = article.content_json
        assert article.status == "image_pending"
        assert session.scalars(
            select(Asset.position)
            .where(Asset.article_id == article_id)
            .order_by(Asset.position)
        ).all() == [0]
        assert session.scalar(
            select(func.count(GenerationRun.id)).where(
                GenerationRun.article_id == article_id,
                GenerationRun.step == "article",
            )
        ) == 1

    with SessionLocal() as session:
        completed = execute_automation_run(session, run_id, settings)
        assert completed.status == "completed"
        article = session.get(Article, article_id)
        assert article is not None
        assert article.content_json == saved_text
        assert article.status == "generated"
        assert session.scalars(
            select(Asset.position)
            .where(Asset.article_id == article_id)
            .order_by(Asset.position)
        ).all() == [0, 1]
        assert session.scalar(
            select(func.count(GenerationRun.id)).where(
                GenerationRun.article_id == article_id,
                GenerationRun.step == "article",
            )
        ) == 1
        assert generated_positions == [0, 1, 1]

        plan = session.get(AutomationPlan, plan_id)
        assert plan is not None
        session.delete(plan)
        session.delete(article)
        session.commit()


def test_api_and_admin_share_automation_plan_boundaries():
    suffix = uuid4().hex
    disabled_slug = f"disabled-{suffix}"
    plan_name = f"boundary-{suffix}"
    plan_id = None

    with TestClient(app) as client:
        with SessionLocal() as session:
            disabled = AudienceProfile(
                slug=disabled_slug,
                name="已停用测试人群",
                age_range="18+",
                enabled=False,
            )
            session.add(disabled)
            session.commit()
            disabled_id = disabled.id

        api_disabled = client.post(
            "/api/automation/plans",
            json={
                "name": f"api-disabled-{suffix}",
                "audience_slug": disabled_slug,
                "directions": "不可使用停用人群",
                "provider": "mock",
            },
        )
        admin_disabled = client.post(
            "/admin/automation-plans",
            data={
                "name": f"admin-disabled-{suffix}",
                "audience_slug": disabled_slug,
                "directions": "不可使用停用人群",
                "provider": "mock",
            },
            follow_redirects=False,
        )
        assert api_disabled.status_code == 404
        assert admin_disabled.status_code == 404
        assert api_disabled.json()["detail"] == "Audience not found or disabled"
        assert admin_disabled.json()["detail"] == "Audience not found or disabled"

        api_bad_timezone = client.post(
            "/api/automation/plans",
            json={
                "name": f"api-timezone-{suffix}",
                "audience_slug": "middle-age-health",
                "directions": "拒绝无效时区",
                "provider": "mock",
                "timezone": "Mars/Olympus",
            },
        )
        admin_bad_timezone = client.post(
            "/admin/automation-plans",
            data={
                "name": f"admin-timezone-{suffix}",
                "audience_slug": "middle-age-health",
                "directions": "拒绝无效时区",
                "provider": "mock",
                "timezone": "Mars/Olympus",
            },
            follow_redirects=False,
        )
        assert api_bad_timezone.status_code == 422
        assert admin_bad_timezone.status_code == 422

        created = client.post(
            "/api/automation/plans",
            json={
                "name": plan_name,
                "audience_slug": "middle-age-health",
                "directions": "更新边界测试",
                "provider": "mock",
                "timezone": "UTC",
            },
        )
        assert created.status_code == 201
        plan_id = created.json()["id"]

        api_timezone_update = client.put(
            f"/api/automation/plans/{plan_id}",
            json={"timezone": "Mars/Olympus"},
        )
        admin_timezone_update = client.post(
            f"/admin/automation-plans/{plan_id}",
            data={
                "name": plan_name,
                "audience_slug": "middle-age-health",
                "directions": "更新边界测试",
                "provider": "mock",
                "run_time": "06:30",
                "timezone": "Mars/Olympus",
            },
            follow_redirects=False,
        )
        assert api_timezone_update.status_code == 422
        assert admin_timezone_update.status_code == 422

        api_update = client.put(
            f"/api/automation/plans/{plan_id}",
            json={"audience_slug": disabled_slug},
        )
        admin_update = client.post(
            f"/admin/automation-plans/{plan_id}",
            data={
                "name": plan_name,
                "audience_slug": disabled_slug,
                "directions": "更新边界测试",
                "provider": "mock",
                "run_time": "06:30",
                "timezone": "UTC",
            },
            follow_redirects=False,
        )
        assert api_update.status_code == 404
        assert admin_update.status_code == 404

        api_generate = client.post(
            "/api/articles/generate",
            json={
                "topic": "停用人群不能生成文章",
                "audience_slug": disabled_slug,
                "provider": "mock",
            },
        )
        admin_generate = client.post(
            "/admin/generate",
            data={
                "topic": "停用人群不能生成文章",
                "source_type": "topic",
                "audience_slug": disabled_slug,
                "provider": "mock",
            },
            follow_redirects=False,
        )
        assert api_generate.status_code == 404
        assert admin_generate.status_code == 404

        stored = next(
            item
            for item in client.get("/api/automation/plans").json()
            if item["id"] == plan_id
        )
        assert stored["audience_slug"] == "middle-age-health"
        assert stored["timezone"] == "UTC"

        admin_valid_update = client.post(
            f"/admin/automation-plans/{plan_id}",
            data={
                "name": f"{plan_name}-admin",
                "audience_slug": "middle-age-health",
                "directions": "后台成功更新",
                "provider": "mock",
                "run_time": "07:45",
                "timezone": "Asia/Shanghai",
                "enabled": "true",
            },
            follow_redirects=False,
        )
        assert admin_valid_update.status_code == 303
        api_valid_update = client.put(
            f"/api/automation/plans/{plan_id}",
            json={
                "name": f"{plan_name}-api",
                "timezone": "America/Los_Angeles",
            },
        )
        assert api_valid_update.status_code == 200
        assert api_valid_update.json()["name"] == f"{plan_name}-api"
        assert api_valid_update.json()["timezone"] == "America/Los_Angeles"

    with SessionLocal() as session:
        if plan_id is not None:
            plan = session.get(AutomationPlan, plan_id)
            if plan is not None:
                session.delete(plan)
        disabled = session.get(AudienceProfile, disabled_id)
        if disabled is not None:
            session.delete(disabled)
        session.commit()


def test_enqueue_due_plans_skips_invalid_legacy_plan(caplog):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    now = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)

    with Session(engine) as session:
        audience = AudienceProfile(
            slug="scheduler-test",
            name="调度测试人群",
            age_range="18+",
        )
        session.add(audience)
        session.flush()
        invalid = AutomationPlan(
            name="遗留坏计划",
            audience_profile_id=audience.id,
            directions="无效时区不能阻塞队列",
            provider="mock",
            run_time="00:00",
            timezone="Mars/Olympus",
            enabled=True,
        )
        valid = AutomationPlan(
            name="有效计划",
            audience_profile_id=audience.id,
            directions="应该正常进入队列",
            provider="mock",
            run_time="00:00",
            timezone="UTC",
            enabled=True,
        )
        session.add_all([invalid, valid])
        session.commit()
        invalid_id = invalid.id
        valid_id = valid.id

        with caplog.at_level(logging.ERROR, logger="app.services.automation"):
            created = enqueue_due_plans(session, now=now)

        assert len(created) == 1
        run = session.get(AutomationRun, created[0])
        assert run is not None
        assert run.automation_plan_id == valid_id
        assert f"id={invalid_id}" in caplog.text
        assert "Mars/Olympus" in caplog.text
