from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import Article, EmotionalTopicReview, GenerationRun
from app.services import automation, topic_reviews
from app.services.topic_reviews import (
    approve_review,
    process_next_manual_review,
    regenerate_review,
    reject_review,
    retry_manual_review,
    save_review,
)


def manual_review(client, idea="每次约会都是她先问几点见"):
    response = client.post(
        "/admin/emotions/generate",
        data={
            "idea": idea,
            "provider": "mock",
            "category": "恋爱心理",
            "auto_upload_draft": "true",
            "humanize": "true",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/emotions/topics/")
    return int(response.headers["location"].rsplit("/", 1)[1])


def test_manual_supplied_topic_costs_nothing_until_approved(monkeypatch):
    with TestClient(app) as client:
        calls = []

        def unexpected(*args, **kwargs):
            calls.append(True)
            pytest.fail("no model, images or upload before approval")

        with monkeypatch.context() as blocked:
            blocked.setattr(topic_reviews, "suggest_topic", unexpected)
            blocked.setattr(topic_reviews, "generate_article", unexpected)
            blocked.setattr(topic_reviews, "generate_article_images", unexpected)
            blocked.setattr(topic_reviews, "upload_article_to_draft", unexpected)
            review_id = manual_review(client)
            with SessionLocal() as session:
                review = session.get(EmotionalTopicReview, review_id)
                assert review.status == "awaiting_review"
                assert review.article_id is None and review.generation_run_id is None
                assert process_next_manual_review(session, settings) is None
            page = client.get(f"/admin/emotions/topics/{review_id}")
            assert page.status_code == 200
            assert "只保存修改" in page.text
            edited = "他又忘了几点见，她这次没有再提醒"
            response = client.post(
                f"/admin/emotions/topics/{review_id}/save",
                data={
                    "topic": edited,
                    "rationale": "写等待时的不安，不断言对方故意。",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303
            with SessionLocal() as session:
                review = session.get(EmotionalTopicReview, review_id)
                assert review.topic == edited
                assert review.status == "awaiting_review"
            assert not calls
        client.post(
            f"/admin/emotions/topics/{review_id}/approve",
            data={
                "topic": edited,
                "rationale": "写等待时的不安，结合拒绝敏感性研究。",
            },
        )
        # A second click cannot overwrite the approved topic or queue another article.
        client.post(
            f"/admin/emotions/topics/{review_id}/approve",
            data={
                "topic": "另一个完全不同的选题",
                "rationale": "",
            },
        )
        with SessionLocal() as session:
            review = session.get(EmotionalTopicReview, review_id)
            assert review.topic == edited and review.approved_at
            assert review.status == "approved"
            generated = process_next_manual_review(session, settings)
            assert generated.status == "completed", generated.error_message
            article = session.get(Article, generated.article_id)
            assert article.topic == edited
            prompt = session.scalar(
                select(GenerationRun.input_text).where(
                    GenerationRun.article_id == article.id,
                    GenerationRun.step == "emotional_article",
                )
            )
            assert "审阅通过的选题：" + edited in prompt
            assert "拒绝敏感性研究" in prompt
            assert process_next_manual_review(session, settings) is None
            page = client.get(f"/admin/emotions/topics/{review_id}")
            assert "查看文章" in page.text


def test_manual_blank_topic_and_regeneration_only_make_topic_calls(monkeypatch):
    with TestClient(app) as client:
        with SessionLocal() as session:
            before = session.scalar(select(func.count(Article.id)))
        review_id = manual_review(client, idea="")
        with SessionLocal() as session:
            review = session.get(EmotionalTopicReview, review_id)
            first_id, first_topic = review.generation_run_id, review.topic
            assert first_id and review.status == "awaiting_review"
            regenerate_review(session, review_id, settings)
            session.refresh(review)
            assert review.generation_run_id != first_id
            assert review.topic != first_topic
            assert session.scalar(select(func.count(Article.id))) == before
            assert all(
                session.get(GenerationRun, item).step == "topic"
                for item in (first_id, review.generation_run_id)
            )
            reject_review(session, review_id)
            assert process_next_manual_review(session, settings) is None
            assert not approve_review(session, review_id, first_topic, "")


def test_automation_waits_and_reuses_reviewed_topic_without_selecting_again(monkeypatch):
    with TestClient(app) as client:
        plan = client.post(
            "/api/automation/plans",
            json={
                "name": "选题审阅测试",
                "content_type": "emotional_article",
                "audience_slug": "relationship-readers",
                "category": "恋爱心理",
                "directions": "等不到回复时的担心",
                "provider": "mock",
                "enabled": False,
                "auto_upload_draft": False,
                "image_count": 1,
            },
        ).json()
        run = client.post(f"/api/automation/plans/{plan['id']}/run-now").json()
        with SessionLocal() as session:
            before = session.scalar(select(func.count(Article.id)))
            waiting = automation.process_next_run(session, settings)
            assert waiting.id == run["id"] and waiting.status == "awaiting_review"
            assert waiting.article_id is None
            assert session.scalar(select(func.count(Article.id))) == before
            review = session.scalar(
                select(EmotionalTopicReview).where(
                    EmotionalTopicReview.automation_run_id == waiting.id
                )
            )
            review_id, generation_id = review.id, review.generation_run_id
        # A new session simulates a restart; waiting topics never enter the worker queue.
        with SessionLocal() as session:
            assert automation.claim_next_automation_run(session, settings) is None
            review = session.get(EmotionalTopicReview, review_id)
            edited = "他说想见她，却从来没定过下一次的时间"
            save_review(session, review_id, edited, "写期待落空后的反复确认。")
            approve_review(session, review_id, edited, "写期待落空后的反复确认。")
            monkeypatch.setattr(
                automation, "_choose_topic", lambda *a: pytest.fail("must reuse reviewed topic")
            )
            completed = automation.process_next_run(session, settings)
            assert completed.status == "completed", completed.error_message
            assert session.get(Article, completed.article_id).topic == edited
            assert session.get(GenerationRun, generation_id).article_id == completed.article_id
            assert session.get(EmotionalTopicReview, review_id).status == "completed"


def test_manual_image_retry_reuses_checkpointed_article(monkeypatch):
    with TestClient(app) as client:
        review_id = manual_review(client, "他把周末又留给了加班，她把两张票收进抽屉")
        with SessionLocal() as session:
            review = session.get(EmotionalTopicReview, review_id)
            approve_review(session, review_id, review.topic, "")
        original = topic_reviews.generate_article_images

        def fail_images(*args, **kwargs):
            raise RuntimeError("image quota exhausted")

        monkeypatch.setattr(topic_reviews, "generate_article_images", fail_images)
        with SessionLocal() as session:
            failed = process_next_manual_review(session, settings)
            assert failed.status == "failed" and failed.article_id
            article_id = failed.article_id
            assert session.get(Article, article_id).status == "image_pending"
        monkeypatch.setattr(topic_reviews, "generate_article_images", original)
        monkeypatch.setattr(
            topic_reviews, "generate_article", lambda *a, **k: pytest.fail("must keep text")
        )
        with SessionLocal() as session:
            assert retry_manual_review(session, review_id)
            completed = process_next_manual_review(session, settings)
            assert completed.status == "completed"
            assert completed.article_id == article_id


def test_waiting_topics_are_not_stale_worker_failures():
    with TestClient(app) as client:
        review_id = manual_review(client, "说好一起吃饭，她在餐厅等到了打烊")
        with SessionLocal() as session:
            review = session.get(EmotionalTopicReview, review_id)
            review.started_at = datetime.now(UTC) - timedelta(days=3)
            session.commit()
            topic_reviews.recover_stale_reviews(session, settings)
            session.refresh(review)
            assert review.status == "awaiting_review"
            reject_review(session, review_id)
