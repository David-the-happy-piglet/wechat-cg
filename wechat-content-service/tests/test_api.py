from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.admin import ARTICLE_PAGE_SIZE
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import (
    Article,
    AudienceProfile,
    EmotionalTopicReview,
    GenerationRun,
    StoryOutlineDraft,
)
from app.services import generator as generator_service
from app.services.automation import process_next_run
from app.services.llm import MockLLMProvider, TokenUsage
from app.services.topic_reviews import approve_review


def test_generation_to_mock_publish_flow():
    with TestClient(app) as client:
        audiences = client.get("/api/audiences")
        assert audiences.status_code == 200
        assert {item["slug"] for item in audiences.json()} == {
            "senior-wellness",
            "middle-age-health",
            "micro-horror-readers",
            "relationship-readers",
        }

        response = client.post(
            "/api/articles/generate",
            json={
                "topic": "人到中年如何改善睡眠",
                "audience_slug": "middle-age-health",
                "category": "睡眠健康",
                "title_count": 5,
                "image_count": 3,
                "target_length": 800,
            },
        )
        assert response.status_code == 201
        article = response.json()
        assert article["status"] == "generated"
        assert article["content_type"] == "health_article"
        assert len(article["title_candidates"]) == 5
        assert len(article["assets"]) == 3
        assert article["target_length"] == 800
        assert "过了35岁" in article["selected_title"]

        preview = client.get(f"/admin/articles/{article['id']}")
        assert preview.status_code == 200
        assert "一键上传到公众号草稿箱" in preview.text

        draft = client.post(f"/api/articles/{article['id']}/wechat-draft")
        assert draft.status_code == 200
        assert draft.json()["media_id"].startswith("mock-draft-")
        preview = client.get(f"/admin/articles/{article['id']}")
        assert "重新核验微信草稿" in preview.text

        publish = client.post(f"/api/articles/{article['id']}/publish")
        assert publish.status_code == 200
        assert publish.json()["publish_id"].startswith("mock-publish-")

        metric = client.post(
            f"/api/articles/{article['id']}/metrics",
            json={
                "metric_date": "2026-07-25",
                "read_count": 1200,
                "read_users": 980,
                "share_count": 75,
                "share_users": 61,
            },
        )
        assert metric.status_code == 201
        assert metric.json()["read_count"] == 1200


def test_micro_horror_generation_runs_outline_draft_and_review():
    with TestClient(app) as client:
        response = client.post(
            "/api/articles/generate",
            json={
                "content_type": "micro_horror",
                "story_mode": "near_future",
                "topic": "智能门锁每天替独居者拒绝一个不存在的访客",
                "audience_slug": "micro-horror-readers",
                "provider": "mock",
                "title_count": 5,
                "image_count": 2,
                "humanize": True,
            },
        )
        assert response.status_code == 201
        article = response.json()
        assert article["content_type"] == "micro_horror"
        assert article["story_mode"] == "near_future"
        assert article["category"] == "微恐短篇"
        assert article["prompt_version"] == "micro-horror-v1-reviewed"
        assert article["content"]["content_type"] == "micro_horror"
        assert len(article["title_candidates"]) == 5
        assert len(article["assets"]) == 2
        assert "background:#fbfaf8" in article["content_html"]
        assert "background:#f4f8f5" not in article["content_html"]

        preview = client.get(f"/admin/articles/{article['id']}")
        assert preview.status_code == 200
        assert "故事方案（含剧透）" in preview.text
        assert "审稿报告" in preview.text

        with SessionLocal() as session:
            steps = session.scalars(
                select(GenerationRun.step)
                .where(GenerationRun.article_id == article["id"])
                .order_by(GenerationRun.id)
            ).all()
        assert list(steps) == ["story_outline", "story_draft", "story_review"]


def test_content_type_validation_and_admin_control():
    with TestClient(app) as client:
        admin = client.get("/admin")
        assert admin.status_code == 200
        assert 'name="content_type"' in admin.text
        assert 'value="health_article"' in admin.text
        assert "养生文章工作台" in admin.text
        assert ">养生文章</a>" in admin.text

        story_entry = client.get("/admin/stories", follow_redirects=False)
        assert story_entry.status_code == 200
        assert "微恐故事工作台" in story_entry.text
        assert "读者边界" in story_entry.text
        assert "大纲预览" in story_entry.text
        assert "留空则自由生成" in story_entry.text
        assert "第一人称回忆式叙事" in story_entry.text
        assert 'name="audience_slug"' not in story_entry.text
        assert "添加目标受众" not in story_entry.text

        old_story_entry = client.get(
            "/admin?content_type=micro_horror",
            follow_redirects=False,
        )
        assert old_story_entry.status_code == 302
        assert old_story_entry.headers["location"] == "/admin/stories"

        emotional_entry = client.get("/admin/emotions")
        assert emotional_entry.status_code == 200
        assert "情感文章工作台" in emotional_entry.text
        assert "分手创伤自我修复" in emotional_entry.text
        assert "失恋主题" in emotional_entry.text
        assert "选择“清醒短句”：不描写具体场景" in emotional_entry.text
        assert '<option value="前任">前任</option>' in emotional_entry.text
        assert '<option value="友情">' not in emotional_entry.text
        assert '<option value="职场里的人和事">' not in emotional_entry.text
        assert "追求技巧与套路拆解" in emotional_entry.text
        assert "星座用于娱乐观察" in emotional_entry.text
        assert "每日自动生产" in emotional_entry.text
        assert "自动任务记录" in emotional_entry.text
        assert "生成记录" in emotional_entry.text
        assert 'value="emotional_article"' in emotional_entry.text
        assert 'action="/admin/automation-plans"' in emotional_entry.text
        assert '<select name="audience_slug"' not in emotional_entry.text

        emotional_generated = client.post(
            "/admin/emotions/generate",
            data={
                "idea": "分手后为什么总想反复查看对方的动态",
                "review_topic": "false",
                "category": "分手创伤自我修复",
                "reader_context": "刚刚分手、正在恢复的人",
                "angle": "解释信息戒断，并给出七天可执行的小步骤",
                "provider": "mock",
                "target_length": "1200",
                "headline_intensity": "4",
                "humanize": "true",
            },
            follow_redirects=False,
        )
        assert emotional_generated.status_code == 303
        assert emotional_generated.headers["location"].startswith("/admin/articles/")
        emotional_preview = client.get(emotional_generated.headers["location"])
        assert emotional_preview.status_code == 200
        assert "情感文章" in emotional_preview.text
        assert "恋爱与情感成长读者" in emotional_preview.text
        emotional_article_id = int(
            emotional_generated.headers["location"].rsplit("/", 1)[-1]
        )
        with SessionLocal() as session:
            emotional_steps = session.scalars(
                select(GenerationRun.step)
                .where(GenerationRun.article_id == emotional_article_id)
                .order_by(GenerationRun.id)
            ).all()
        assert list(emotional_steps) == ["emotional_article"]

        invalid = client.post(
            "/api/articles/generate",
            json={"content_type": "ghost_story", "topic": "一个足够长的故事主题"},
        )
        assert invalid.status_code == 422

        outlined = client.post(
            "/admin/stories/outlines",
            data={
                "story_mode": "everyday_uncanny",
                "idea": "",
                "reader_boundary": "成年读者；克制微恐，不写露骨血腥，标题不得剧透。",
                "style_prompt": "第一人称，用日常细节递进，结尾停在具体动作。",
                "category": "微恐短篇",
                "provider": "mock",
                "target_length": "1800",
                "headline_intensity": "4",
                "polish": "true",
            },
            follow_redirects=False,
        )
        assert outlined.status_code == 303
        assert outlined.headers["location"].startswith("/admin/stories/outlines/")
        outline_page = client.get(outlined.headers["location"])
        assert outline_page.status_code == 200
        assert "因果节拍" in outline_page.text
        assert "大纲审核通过，写作并润色" in outline_page.text

        draft_id = int(outlined.headers["location"].rsplit("/", 1)[-1])
        revised = client.post(
            f"/admin/stories/outlines/{draft_id}/revise",
            data={"review_notes": "让第二个伏笔更自然，结尾不要解释规则。"},
            follow_redirects=False,
        )
        assert revised.status_code == 303
        revised_page = client.get(revised.headers["location"])
        assert "让第二个伏笔更自然" in revised_page.text

        generated = client.post(
            f"/admin/stories/outlines/{draft_id}/generate",
            data={"polish": "true"},
            follow_redirects=False,
        )
        assert generated.status_code == 303
        assert generated.headers["location"].startswith("/admin/articles/")
        preview = client.get(generated.headers["location"])
        assert "故事方案（含剧透）" in preview.text
        assert "微恐短篇读者" not in preview.text
        assert "成人微恐 · 克制表达" in preview.text

        article_id = int(generated.headers["location"].rsplit("/", 1)[-1])
        with SessionLocal() as session:
            draft = session.get(StoryOutlineDraft, draft_id)
            assert draft is not None
            assert draft.status == "completed"
            assert draft.article_id == article_id
            steps = session.scalars(
                select(GenerationRun.step)
                .where(GenerationRun.article_id == article_id)
                .order_by(GenerationRun.id)
            ).all()
        assert list(steps) == [
            "story_outline",
            "story_outline",
            "story_draft",
            "story_review",
        ]

        story_records = client.get("/admin/stories")
        assert story_records.status_code == 200
        assert "生成记录" in story_records.text
        assert "大纲生成 / 修订" in story_records.text
        assert "正文初稿" in story_records.text
        assert "编辑审稿与润色" in story_records.text
        assert f"大纲 #{draft_id}" in story_records.text
        assert f'/admin/articles/{article_id}' in story_records.text


def test_emotional_daily_automation_admin_flow():
    plan_name = "测试每日情感文章"
    with TestClient(app) as client:
        created = client.post(
            "/admin/automation-plans",
            data={
                "name": plan_name,
                "content_type": "emotional_article",
                "story_mode": "auto",
                "audience_slug": "relationship-readers",
                "directions": "分手修复、关系边界与沟通冲突",
                "category": "分手创伤自我修复",
                "provider": "mock",
                "run_time": "08:30",
                "timezone": "Asia/Shanghai",
                "target_length": "1200",
                "headline_intensity": "4",
                "humanize": "true",
            },
            follow_redirects=False,
        )
        assert created.status_code == 303
        assert created.headers["location"] == "/admin/emotions"

        plans = client.get("/api/automation/plans").json()
        plan = next(item for item in plans if item["name"] == plan_name)
        assert plan["content_type"] == "emotional_article"
        assert plan["audience_slug"] == "relationship-readers"

        emotional_page = client.get("/admin/emotions")
        assert plan_name in emotional_page.text
        assert f'id="emotional-plan-{plan["id"]}"' in emotional_page.text

        queued = client.post(
            f'/admin/automation-plans/{plan["id"]}/run-now',
            follow_redirects=False,
        )
        assert queued.status_code == 303
        assert queued.headers["location"] == "/admin/emotions"

        with SessionLocal() as session:
            completed = process_next_run(session, settings)
            assert completed is not None
            assert completed.status == "awaiting_review"
            review = session.scalar(select(EmotionalTopicReview).where(
                EmotionalTopicReview.automation_run_id == completed.id))
            assert approve_review(session, review.id, review.topic, review.rationale)
            completed = process_next_run(session, settings)
            assert completed.status == "completed"
            article_id = completed.article_id

        article = client.get(f"/api/articles/{article_id}").json()
        assert article["content_type"] == "emotional_article"
        emotional_page = client.get("/admin/emotions")
        assert plan_name in emotional_page.text
        assert "completed" in emotional_page.text
        assert f'/admin/articles/{article_id}' in emotional_page.text


def test_micro_horror_review_failure_keeps_valid_draft(monkeypatch):
    class ReviewFailProvider(MockLLMProvider):
        def generate_json(
            self,
            prompt,
            *,
            system_prompt="只输出合法 JSON，不要输出 Markdown。",
            operation=None,
            context=None,
        ):
            if operation == "micro_horror_review":
                raise RuntimeError("review unavailable")
            return super().generate_json(
                prompt,
                system_prompt=system_prompt,
                operation=operation,
                context=context,
            )

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: ReviewFailProvider(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/articles/generate",
            json={
                "content_type": "micro_horror",
                "topic": "楼上的住户每天借走一分钟安静",
                "audience_slug": "micro-horror-readers",
                "provider": "mock",
                "image_count": 1,
                "humanize": True,
            },
        )
        assert response.status_code == 201
        article = response.json()
        assert article["prompt_version"] == "micro-horror-v1-draft"
        with SessionLocal() as session:
            failed = session.scalar(
                select(GenerationRun).where(
                    GenerationRun.article_id == article["id"],
                    GenerationRun.step == "story_review",
                )
            )
            assert failed is not None
            assert failed.status == "failed"
            assert "review unavailable" in (failed.error_message or "")


def test_provider_status_and_mock_comparison():
    with TestClient(app) as client:
        providers = client.get("/api/providers")
        assert providers.status_code == 200
        assert {item["provider"] for item in providers.json()["text"]} == {
            "openai",
            "openai-terra",
            "openai-luna",
            "deepseek",
            "minimax",
            "kimi",
        }
        assert all("api_key" not in item for item in providers.json()["text"])

        response = client.post(
            "/api/articles/compare",
            json={
                "topic": "晚饭后如何安排散步和休息",
                "providers": ["mock"],
                "image_count": 1,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["errors"] == []
        assert len(result["articles"]) == 1
        assert result["articles"][0]["llm_provider"] == "mock"

        usage = client.get("/api/token-usage")
        assert usage.status_code == 200
        assert "models" in usage.json()
        assert "totals" in usage.json()


def test_generation_normalizes_list_image_search_queries(monkeypatch):
    class ListQueryMockProvider(MockLLMProvider):
        def generate_article(self, prompt, context):
            result = super().generate_article(prompt, context)
            result["image_prompts"][0]["search_query"] = [
                "healthy family",
                "dinner",
            ]
            return result

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: ListQueryMockProvider(),
    )
    with TestClient(app) as client:
        generated = client.post(
            "/api/articles/generate",
            json={
                "topic": "家庭晚饭如何搭配得更轻松",
                "provider": "mock",
                "image_count": 1,
            },
        )
        assert generated.status_code == 201
        search_query = generated.json()["content"]["image_prompts"][0][
            "search_query"
        ]
        assert search_query == "healthy family dinner"


def test_article_generation_does_not_run_risk_scan(monkeypatch):
    class RiskyWordingMockProvider(MockLLMProvider):
        def generate_article(self, prompt, context):
            result = super().generate_article(prompt, context)
            result["opening"] = "不用吃药，也能彻底治愈。"
            return result

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: RiskyWordingMockProvider(),
    )
    with TestClient(app) as client:
        generated = client.post(
            "/api/articles/generate",
            json={
                "topic": "测试图文生成不再触发风险扫描",
                "provider": "mock",
                "image_count": 1,
            },
        )

        assert generated.status_code == 201
        assert generated.json()["status"] == "generated"
        assert generated.json()["safety_flags"] == []
        preview = client.get(f"/admin/articles/{generated.json()['id']}")
        assert "安全提醒" not in preview.text


def test_headline_intensity_validation_and_admin_control():
    with TestClient(app) as client:
        admin = client.get("/admin")
        assert admin.status_code == 200
        assert 'name="headline_intensity"' in admin.text
        assert "4 — 强吸引（推荐）" in admin.text
        assert 'name="humanize"' in admin.text
        assert "Humanizer-zh 二次润色" in admin.text
        assert 'name="target_length"' in admin.text
        assert 'name="auto_upload_draft"' in admin.text
        assert "生成完成后立即推送到公众号草稿箱" in admin.text
        assert "短篇 — 约 800 字" in admin.text
        assert "深度 — 约 2500 字" in admin.text
        assert "GPT-5.6 Sol（旗舰质量）" in admin.text
        assert "GPT-5.6 Terra（均衡）" in admin.text
        assert "GPT-5.6 Luna（快速经济）" in admin.text
        assert 'name="source_type"' in admin.text
        assert "给定文本 / 原稿" in admin.text
        assert 'href="/admin/videos"' in admin.text

        invalid = client.post(
            "/api/articles/generate",
            json={
                "topic": "晚饭后如何合理活动",
                "provider": "mock",
                "headline_intensity": 6,
            },
        )
        assert invalid.status_code == 422

        invalid_length = client.post(
            "/api/articles/generate",
            json={
                "topic": "晚饭后如何合理活动",
                "provider": "mock",
                "target_length": 200,
            },
        )
        assert invalid_length.status_code == 422


def test_article_generation_accepts_reference_text_and_custom_rewrite_prompt(monkeypatch):
    captured = {}

    class CapturingMockProvider(MockLLMProvider):
        def generate_article(self, prompt, context):
            captured["prompt"] = prompt
            captured["context"] = context
            return super().generate_article(prompt, context)

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: CapturingMockProvider(),
    )
    source_text = "稳定改变往往不是靠意志硬撑，而是把行动拆小，并根据每天的反馈持续调整。"
    with TestClient(app) as client:
        response = client.post(
            "/api/articles/generate",
            json={
                "source_type": "text",
                "source_text": source_text,
                "source_prompt": "保留原观点，换成普通家庭能理解的生活例子。",
                "provider": "mock",
                "image_count": 1,
            },
        )
    assert response.status_code == 201
    article = response.json()
    assert article["topic"].startswith("稳定改变")
    assert source_text in captured["prompt"]
    assert "换成普通家庭能理解的生活例子" in captured["prompt"]
    assert captured["context"]["topic"] == article["topic"]


def test_admin_generation_can_upload_draft_immediately():
    with TestClient(app) as client:
        generated = client.post(
            "/admin/generate",
            data={
                "topic": "晚饭后如何安排散步和休息",
                "category": "生活方式",
                "audience_slug": "middle-age-health",
                "provider": "mock",
                "target_length": "800",
                "headline_intensity": "3",
                "auto_upload_draft": "true",
            },
            follow_redirects=False,
        )
        assert generated.status_code == 303
        location = generated.headers["location"]
        assert location.startswith("/admin/articles/")
        assert "draft_message=" in location

        preview = client.get(location)
        assert preview.status_code == 200
        assert "mock-draft-" in preview.text
        assert "重新核验微信草稿" in preview.text

        article_id = int(location.split("?", 1)[0].rstrip("/").split("/")[-1])
        admin = client.get("/admin")
        assert (
            f'action="/admin/articles/{article_id}/wechat-draft"'
            in admin.text
        )
        assert "重新核验" in admin.text


def test_admin_generation_does_not_upload_when_option_is_unchecked():
    with TestClient(app) as client:
        generated = client.post(
            "/admin/generate",
            data={
                "topic": "周末如何安排轻松的家庭活动",
                "category": "生活方式",
                "audience_slug": "middle-age-health",
                "provider": "mock",
                "target_length": "800",
                "headline_intensity": "3",
            },
            follow_redirects=False,
        )
        assert generated.status_code == 303
        location = generated.headers["location"]
        assert "draft_message=" not in location

        preview = client.get(location)
        assert preview.status_code == 200
        assert "一键上传到公众号草稿箱" in preview.text
        assert "mock-draft-" not in preview.text

        article_id = int(location.rstrip("/").split("/")[-1])
        admin = client.get("/admin")
        assert (
            f'action="/admin/articles/{article_id}/wechat-draft"'
            in admin.text
        )
        assert "推送草稿箱" in admin.text


def test_admin_generation_shows_draft_upload_error(monkeypatch):
    def fail_upload(*args, **kwargs):
        raise RuntimeError("微信草稿接口测试失败")

    monkeypatch.setattr("app.admin.upload_article_to_draft", fail_upload)
    with TestClient(app) as client:
        generated = client.post(
            "/admin/generate",
            data={
                "topic": "每天如何安排简单的伸展活动",
                "category": "生活方式",
                "audience_slug": "middle-age-health",
                "provider": "mock",
                "target_length": "800",
                "headline_intensity": "3",
                "auto_upload_draft": "true",
            },
            follow_redirects=False,
        )
        assert generated.status_code == 303
        location = generated.headers["location"]
        assert "draft_error=" in location

        preview = client.get(location)
        assert preview.status_code == 200
        assert "草稿上传失败" in preview.text
        assert "微信草稿接口测试失败" in preview.text


def test_admin_can_create_and_use_a_target_audience():
    with TestClient(app) as client:
        form = client.get("/admin/audiences/new")
        assert form.status_code == 200
        assert "添加目标受众" in form.text
        assert 'name="slug"' in form.text

        created = client.post(
            "/admin/audiences",
            data={
                "slug": "young-family-health",
                "name": "年轻家庭健康",
                "age_range": "28-42",
                "description": "关注家庭饮食、睡眠和亲子活动的年轻父母",
                "interests": "家庭饮食\n睡眠\n亲子运动",
                "writing_style": "自然、简洁、重视家庭生活场景",
                "title_preferences": "有孩子的家庭……\n下班后坚持这几个习惯……",
                "article_length": "900",
                "image_style": "真实自然的中国年轻家庭生活场景",
                "prohibited_topics": "宣称治愈疾病\n制造健康焦虑",
                "enabled": "true",
            },
            follow_redirects=False,
        )
        assert created.status_code == 303
        assert created.headers["location"].startswith("/admin?audience_created=")

        admin = client.get("/admin")
        assert "年轻家庭健康" in admin.text
        assert "添加目标受众" in admin.text

        generated = client.post(
            "/api/articles/generate",
            json={
                "topic": "年轻家庭如何安排晚饭后的活动",
                "audience_slug": "young-family-health",
                "provider": "mock",
                "image_count": 1,
            },
        )
        assert generated.status_code == 201
        article = generated.json()
        assert article["target_length"] == 900
        assert "过了28岁" in article["selected_title"]

        duplicate = client.post(
            "/admin/audiences",
            data={
                "slug": "young-family-health",
                "name": "重复画像",
                "age_range": "28-42",
                "article_length": "900",
                "enabled": "true",
            },
        )
        assert duplicate.status_code == 200
        assert "该唯一标识已存在" in duplicate.text


def test_token_usage_is_persisted_and_aggregated(monkeypatch):
    class MeteredMockProvider(MockLLMProvider):
        def generate_article(self, prompt, context):
            result = super().generate_article(prompt, context)
            self.usage = TokenUsage(
                prompt_tokens=120,
                completion_tokens=80,
                total_tokens=200,
            )
            return result

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: MeteredMockProvider(),
    )
    with TestClient(app) as client:
        generated = client.post(
            "/api/articles/generate",
            json={
                "topic": "晚饭后如何合理活动",
                "provider": "mock",
                "image_count": 1,
            },
        )
        assert generated.status_code == 201

        usage = client.get("/api/token-usage").json()
        mock = next(item for item in usage["models"] if item["provider"] == "mock")
        assert mock["metered_requests"] >= 1
        assert mock["prompt_tokens"] >= 120
        assert mock["completion_tokens"] >= 80
        assert mock["total_tokens"] >= 200


def test_humanizer_runs_as_second_metered_model_call(monkeypatch):
    class HumanizingProvider(MockLLMProvider):
        name = "test-humanizer"
        model = "test-model"

        def generate_article(self, prompt, context):
            result = super().generate_article(prompt, context)
            self.usage = TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
            if "Humanizer-zh" in prompt:
                result["opening"] = "吃过晚饭，先别急着往沙发上一躺。"
            return result

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: HumanizingProvider(),
    )
    with TestClient(app) as client:
        generated = client.post(
            "/api/articles/generate",
            json={
                "topic": "晚饭后如何合理活动",
                "provider": "mock",
                "image_count": 1,
                "humanize": True,
            },
        )
        assert generated.status_code == 201
        article = generated.json()
        assert article["prompt_version"] == "v2-humanizer-zh"
        assert article["content"]["opening"] == "吃过晚饭，先别急着往沙发上一躺。"

        usage = client.get("/api/token-usage").json()
        humanizer = next(item for item in usage["models"] if item["provider"] == "test-humanizer")
        assert humanizer["metered_requests"] == 2
        assert humanizer["total_tokens"] == 300


def test_emotional_style_reaches_generation_and_polishing(monkeypatch):
    captured = []

    class EmotionalProvider(MockLLMProvider):
        name = "test-emotional-style"

        def generate_article(self, prompt, context):
            captured.append(prompt)
            result = super().generate_article(prompt, context)
            for section in result["sections"]:
                section["heading"] = ""
            return result

    monkeypatch.setattr(
        generator_service,
        "build_llm_provider",
        lambda settings, provider_name=None: EmotionalProvider(),
    )
    with TestClient(app) as client:
        for category in ("情感安慰", "恋爱心理", "星座恋爱性格", "人生哲理", "失恋主题", "前任"):
            captured.clear()
            response = client.post(
                "/admin/emotions/generate",
                data={
                    "category": category,
                    "idea": "一个人回家时想起过去的相处",
                    "review_topic": "false",
                    "angle": "围绕本篇具体主题写，不用模板化结尾",
                    "provider": "mock",
                    "humanize": "true",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert response.headers["location"].startswith("/admin/articles/")
            assert len(captured) == 2
            for prompt in captured:
                assert "本篇主要写法：情感共鸣型" in prompt
                assert "消息改了六遍" in prompt
                assert "不新增“你应该、建议你、学会……”" in captured[1]
                if category == "恋爱心理":
                    assert "每篇必须至少有一处心理学家观点或研究佐证" in prompt
                    assert "https://pubmed.ncbi.nlm.nih.gov/8667172/" in prompt
                assert "围绕本篇具体主题写" in prompt
                assert "润色不得改写引号内原句或丢失出处" in prompt
            article_id = response.headers["location"].rsplit("/", 1)[-1]
            article = client.get(f"/api/articles/{article_id}").json()
            assert article["prompt_version"] == "emotional-v4-resonance-humanized"
            assert "<h2" not in article["content_html"]
            assert article["content"]["sections"][0]["paragraphs"][0] in article["content_html"]


def test_daily_automation_generates_and_uploads_mock_draft():
    with TestClient(app) as client:
        created = client.post(
            "/api/automation/plans",
            json={
                "name": "测试每日计划",
                "audience_slug": "middle-age-health",
                "directions": "睡眠、适度活动和普通家庭可执行的健康习惯",
                "category": "生活方式",
                "provider": "mock",
                "run_time": "06:30",
                "timezone": "Asia/Shanghai",
                "image_count": 1,
                "target_length": 1800,
                "auto_upload_draft": True,
                "enabled": True,
            },
        )
        assert created.status_code == 201
        plan = created.json()
        assert plan["target_length"] == 1800

        queued = client.post(f"/api/automation/plans/{plan['id']}/run-now")
        assert queued.status_code == 202
        assert queued.json()["status"] == "pending"

        with SessionLocal() as session:
            completed = process_next_run(session, settings)
            assert completed is not None
            assert completed.status == "completed"
            assert completed.article_id is not None

        runs = client.get("/api/automation/runs").json()
        run = next(item for item in runs if item["id"] == queued.json()["id"])
        assert run["status"] == "completed"
        article = client.get(f"/api/articles/{run['article_id']}").json()
        assert article["status"] == "wechat_draft"
        assert article["target_length"] == 1800
        assert article["wechat_media_id"].startswith("mock-draft-")
        assert "人工智能辅助生成" not in article["content_html"]


def test_micro_horror_automation_uses_story_topic_and_pipeline():
    with TestClient(app) as client:
        created = client.post(
            "/api/automation/plans",
            json={
                "name": "每日微恐计划",
                "content_type": "micro_horror",
                "story_mode": "everyday_uncanny",
                "audience_slug": "micro-horror-readers",
                "directions": "普通住宅、亲密关系与一个被忽视的生活规则",
                "category": "微恐短篇",
                "provider": "mock",
                "image_count": 1,
                "humanize": True,
                "auto_upload_draft": False,
            },
        )
        assert created.status_code == 201
        plan = created.json()
        assert plan["content_type"] == "micro_horror"
        assert plan["story_mode"] == "everyday_uncanny"

        queued = client.post(f"/api/automation/plans/{plan['id']}/run-now")
        assert queued.status_code == 202
        with SessionLocal() as session:
            completed = process_next_run(session, settings)
            assert completed is not None
            assert completed.status == "completed"
            article_id = completed.article_id

        article = client.get(f"/api/articles/{article_id}").json()
        assert article["content_type"] == "micro_horror"
        assert article["story_mode"] == "everyday_uncanny"
        assert article["category"] == "微恐短篇"
        assert article["prompt_version"] == "micro-horror-v1-reviewed"

        admin_before_second = client.get("/admin")
        assert 'id="new-automation-plan"' in admin_before_second.text
        assert 'action="/admin/automation-plans"' in admin_before_second.text
        assert "新增定时任务" in admin_before_second.text

        second = client.post(
            "/admin/automation-plans",
            data={
                "name": "家庭饮食晚间计划",
                "audience_slug": "senior-wellness",
                "directions": "普通家庭容易执行的晚饭搭配和厨房习惯",
                "category": "家庭饮食",
                "provider": "mock",
                "run_time": "20:15",
                "timezone": "Asia/Shanghai",
                "target_length": "800",
                "headline_intensity": "3",
            },
            follow_redirects=False,
        )
        assert second.status_code == 303
        plans = client.get("/api/automation/plans").json()
        second_plan = next(
            item for item in plans if item["name"] == "家庭饮食晚间计划"
        )
        assert len(plans) >= 2

        admin = client.get("/admin")
        assert admin.status_code == 200
        assert "定时任务列表" in admin.text
        assert "执行时间" in admin.text
        assert "模型 / 篇幅" in admin.text
        assert "测试每日计划" in admin.text
        assert "家庭饮食晚间计划" in admin.text
        assert "每天 20:15" in admin.text
        assert "仅生成" in admin.text
        assert "已停用" in admin.text
        assert f'id="plan-{plan["id"]}"' not in admin.text
        assert f'id="plan-{second_plan["id"]}"' in admin.text
        assert "立即运行一次" in admin.text


def test_admin_article_list_uses_server_side_pagination():
    topic_prefix = "admin-pagination-test-"
    oldest_title = "分页测试文章 00"
    newest_title = f"分页测试文章 {ARTICLE_PAGE_SIZE:02d}"

    with TestClient(app) as client:
        with SessionLocal() as session:
            audience = session.scalar(select(AudienceProfile).order_by(AudienceProfile.id))
            assert audience is not None
            created_at = datetime(2099, 1, 1, tzinfo=UTC)
            for index in range(ARTICLE_PAGE_SIZE + 1):
                session.add(
                    Article(
                        audience_profile_id=audience.id,
                        topic=f"{topic_prefix}{index}",
                        selected_title=f"分页测试文章 {index:02d}",
                        content_json="{}",
                        content_html="<p>分页测试</p>",
                        llm_provider="mock",
                        model_name="pagination-test",
                        created_at=created_at + timedelta(seconds=index),
                    )
                )
            session.commit()

        try:
            first_page = client.get("/admin?article_page=1")
            assert first_page.status_code == 200
            assert newest_title in first_page.text
            assert oldest_title not in first_page.text
            assert "下一页" in first_page.text
            assert "第 1 /" in first_page.text

            second_page = client.get("/admin?article_page=2")
            assert second_page.status_code == 200
            assert oldest_title in second_page.text
            assert "上一页" in second_page.text
        finally:
            with SessionLocal() as session:
                session.execute(
                    delete(Article).where(Article.topic.startswith(topic_prefix))
                )
                session.commit()


def test_folklore_workbench_keeps_material_through_revision_and_review():
    with TestClient(app) as client:
        page = client.get('/admin/stories')
        assert 'value="folk_horror"' in page.text
        assert 'name="folklore_element"' in page.text
        created = client.post(
            '/admin/stories/outlines',
            data={
                'story_mode': 'folk_horror',
                'folklore_element': 'ancestral',
                'provider': 'mock',
                'idea': '回乡修祖屋，族谱每多一个名字就少一扇门',
            },
            follow_redirects=False,
        )
        assert created.status_code == 303
        assert created.headers['location'].startswith('/admin/stories/outlines/')
        outline_url = created.headers['location']
        draft_id = int(outline_url.rsplit('/', 1)[-1])
        revised = client.post(
            f'{outline_url}/revise',
            data={'review_notes': '让族谱和木尺两处伏笔更自然。'},
            follow_redirects=False,
        )
        assert revised.headers['location'] == outline_url
        generated = client.post(
            f'{outline_url}/generate', data={'polish': 'true'}, follow_redirects=False,
        )
        assert generated.headers['location'].startswith('/admin/articles/')
        article_id = int(generated.headers['location'].rsplit('/', 1)[-1])
        article = client.get(f'/api/articles/{article_id}').json()
        assert article['story_mode'] == 'folk_horror'
        assert '族谱' in article['content_html']
        assert '民俗志异' in client.get(generated.headers['location']).text
        with SessionLocal() as session:
            draft = session.get(StoryOutlineDraft, draft_id)
            assert draft.story_mode == 'folk_horror'
            assert '本篇民俗素材：祠堂、族谱与祖先祭祀传说' in draft.style_prompt
            runs = session.scalars(
                select(GenerationRun)
                .where(GenerationRun.article_id == article_id)
                .order_by(GenerationRun.id)
            ).all()
            assert [run.step for run in runs] == [
                'story_outline', 'story_outline', 'story_draft', 'story_review',
            ]
            for run in runs:
                assert '本篇民俗素材：祠堂、族谱与祖先祭祀传说' in run.input_text
                assert '民俗志异写法' in run.input_text
                assert '不要杜撰经文' in run.input_text

        plan = client.post('/api/automation/plans', json={
            'name': '民俗志异测试计划', 'content_type': 'micro_horror',
            'story_mode': 'folk_horror', 'audience_slug': 'micro-horror-readers',
            'directions': '祠堂族谱与旧屋', 'provider': 'mock',
            'auto_upload_draft': False, 'enabled': False,
        })
        assert plan.status_code == 201
        assert plan.json()['story_mode'] == 'folk_horror'


def test_emotional_title_reference_is_opt_in_and_survives_polishing(monkeypatch):
    captured = []

    class TitleReferenceProvider(MockLLMProvider):
        name = "test-title-reference"

        def generate_article(self, prompt, context):
            captured.append(prompt)
            return super().generate_article(prompt, context)

    monkeypatch.setattr(
        generator_service, "build_llm_provider",
        lambda settings, provider_name=None: TitleReferenceProvider(),
    )
    with TestClient(app) as client:
        page = client.get("/admin/emotions")
        assert 'name="emotional_title_reference"' in page.text
        for enabled in (False, True):
            captured.clear()
            data = {"review_topic": "false", "idea": "分手后反复点开聊天框", "humanize": "true", "provider": "mock"}
            if enabled:
                data["emotional_title_reference"] = "true"
            response = client.post("/admin/emotions/generate", data=data, follow_redirects=False)
            assert response.headers["location"].startswith("/admin/articles/")
            assert len(captured) == 2
            assert all(("【可选情感标题参考】" in prompt) == enabled for prompt in captured)
            if enabled:
                assert "不能为标题补造经历" in captured[0]
                assert "必须保留正文中兑现候选标题" in captured[1]

        captured.clear()
        response = client.post("/api/articles/generate", json={
            "topic": "饭后散步的注意事项", "provider": "mock",
            "emotional_title_reference": True, "humanize": True,
        })
        assert response.status_code == 201
        assert all("【可选情感标题参考】" not in prompt for prompt in captured)


def test_emotional_title_reference_automation_persists_runs_and_can_be_disabled():
    import re

    with TestClient(app) as client:
        form = {
            "name": "情感标题参考测试", "content_type": "emotional_article",
            "audience_slug": "relationship-readers", "directions": "分手后反复点开聊天框",
            "provider": "mock", "emotional_title_reference": "true",
        }
        response = client.post("/admin/automation-plans", data=form, follow_redirects=False)
        assert response.status_code == 303
        plan = next(p for p in client.get("/api/automation/plans").json() if p["name"] == form["name"])
        assert plan["emotional_title_reference"] is True
        page = client.get("/admin/emotions").text
        plan_form = page.split(f'id="emotional-plan-{plan["id"]}"', 1)[1].split("</form>", 1)[0]
        checkbox = re.search(r'<input[^>]*name="emotional_title_reference"[^>]*>', plan_form)
        assert checkbox is not None
        assert "checked" in checkbox.group()
        client.post(f'/admin/automation-plans/{plan["id"]}/run-now', follow_redirects=False)
        with SessionLocal() as session:
            run = process_next_run(session, settings)
            assert run.status == "awaiting_review"
            review = session.scalar(select(EmotionalTopicReview).where(
                EmotionalTopicReview.automation_run_id == run.id))
            approve_review(session, review.id, review.topic, review.rationale)
            run = process_next_run(session, settings)
            assert run.status == "completed"
            prompt = session.scalar(select(GenerationRun.input_text).where(
                GenerationRun.article_id == run.article_id,
                GenerationRun.step == "emotional_article",
            ))
            assert "【可选情感标题参考】" in prompt
        form.pop("emotional_title_reference")
        response = client.post(
            f'/admin/automation-plans/{plan["id"]}', data=form, follow_redirects=False,
        )
        assert response.status_code == 303
        plan_id = plan["id"]
        plan = next(p for p in client.get("/api/automation/plans").json() if p["id"] == plan_id)
        assert plan["emotional_title_reference"] is False
