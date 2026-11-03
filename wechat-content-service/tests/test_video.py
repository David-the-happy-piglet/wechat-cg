import json
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import Settings, settings
from app.database import SessionLocal
from app.main import app
from app.models import VideoProject
from app.services.images import ImageProviderError
from app.services.video.assets import build_video_image_provider
from app.services.video.composer import ComposedVideo
from app.services.video.motion import MiniMaxMotionVideoProvider
from app.services.video.news_research import (
    audit_claim_ids,
    merge_review_correction,
    run_news_research,
)
from app.services.video.overlays import build_overlay_ass
from app.services.video.pipeline import execute_video_project, recover_stale_video_projects
from app.services.video.script import (
    build_video_script_prompt,
    target_scene_count,
    validate_video_script,
)
from app.services.video.source import (
    ExtractedSource,
    SourceExtractionError,
    extract_readable_text,
    validate_public_url,
)
from app.services.video.speech import (
    MiniMaxSpeechProvider,
    MockSpeechProvider,
    speech_voice_options,
)
from app.services.video.subtitles import build_srt, srt_timestamp
from app.services.video.visual_profiles import load_visual_profile


def test_source_url_safety_and_readable_html_extraction():
    with pytest.raises(SourceExtractionError):
        validate_public_url("http://127.0.0.1/private")
    assert (
        validate_public_url(
            "https://example.com/article",
            resolver=lambda hostname: {"93.184.216.34"},
        )
        == "https://example.com/article"
    )
    title, text = extract_readable_text(
        """
        <html><head><title>测试文章</title><style>hidden</style></head>
        <body><nav>导航内容</nav><article>
        <h1>为什么好习惯需要时间</h1>
        <p>真正稳定的改变，往往不是一天完成的。我们需要先选择一个足够具体的小目标。</p>
        <p>记录每天的实际感受，再根据反馈逐步调整，通常比频繁更换方法更可靠。</p>
        </article><script>alert(1)</script></body></html>
        """,
        "https://example.com/article",
    )
    assert title in {"测试文章", "为什么好习惯需要时间"}
    assert "真正稳定的改变" in text
    assert "导航内容" not in text
    assert "alert" not in text


def test_video_script_contract_and_prompt():
    assert target_scene_count(80) == 23
    profile = load_visual_profile()
    prompt = build_video_script_prompt(
        topic="改善睡眠",
        source_text="从固定作息开始。",
        target_duration_seconds=60,
        style="真实自然",
        scene_duration_seconds=3.5,
        speech_speed=1.2,
        script_prompt="多使用生活化例子，避免课程腔。",
        use_web_images=True,
    )
    assert "建议场景数" in prompt
    assert "逐句照搬" in prompt
    assert "聪明但没读过原文的朋友" in prompt
    assert "无意义抽象线条" in prompt
    assert "多使用生活化例子，避免课程腔" in prompt
    assert "优先使用互联网真实图片" in prompt
    assert "具体人物、地点、事件" in prompt
    assert "、".join(profile.allowed_types) in prompt
    result = validate_video_script(
        {
            "title": "改善睡眠",
            "summary": "从日常习惯开始",
            "style_bible": "真实自然",
            "scenes": [
                {
                    "narration": "睡眠并不是一个孤立的问题，我们可以先观察每天的作息和精神状态。",
                    "subtitle": "错误的精简字幕",
                    "image_prompt": "夜晚安静整洁的卧室",
                    "visual_type": "flowchart",
                    "overlay_labels": [
                        {"text": "现在的作息", "x": 0.25, "y": 0.4},
                        {"text": "精神状态", "x": 0.75, "y": 0.4},
                    ],
                },
                {
                    "narration": "先选择固定起床时间，坚持记录一周，再根据真实反馈逐步调整。",
                    "image_prompt": "清晨卧室中的闹钟和自然光",
                },
            ],
        },
        15,
    )
    assert [scene["position"] for scene in result["scenes"]] == [0, 1]
    assert result["scenes"][0]["subtitle"] == result["scenes"][0]["narration"]
    assert result["scenes"][0]["visual_type"] == "flowchart"
    assert result["scenes"][0]["overlay_labels"][0]["text"] == "现在的作息"
    assert "不要在图片中渲染标签文字" in result["scenes"][0]["image_prompt"]


def test_visual_profile_types_are_loaded_from_one_config_file(tmp_path):
    (tmp_path / "custom.json").write_text(
        json.dumps(
            {
                "id": "custom",
                "name": "自定义范式",
                "base_prompt": "统一基础规范",
                "default_type": "diagram",
                "types": {
                    "diagram": {
                        "use_when": "表达关系时",
                        "prompt": "使用粗线关系图",
                    },
                    "cutout": {
                        "use_when": "表达人物时",
                        "prompt": "使用纸片剪影",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    profile = load_visual_profile("custom", root=tmp_path)
    assert profile.allowed_types == ("diagram", "cutout")
    assert profile.normalize_type("unknown") == "diagram"
    prompt, visual_type = profile.render_prompt("cutout", "两个人面对面")
    assert visual_type == "cutout"
    assert "使用纸片剪影" in prompt
    assert "两个人面对面" in prompt


def test_news_documentary_profile_allows_safe_photorealistic_prompts():
    profile = load_visual_profile("news_documentary")
    assert profile.name == "新闻纪实照片"
    assert profile.allow_photorealism is True
    assert profile.default_type == "documentary_scene"
    assert set(profile.allowed_types) == {
        "documentary_scene",
        "location_wide",
        "object_closeup",
        "people_context",
    }
    prompt = build_video_script_prompt(
        topic="城市公共交通事件说明",
        source_text="材料描述了事件发生的地点、时间和公开处置过程。",
        target_duration_seconds=60,
        style="自然克制",
        visual_profile_name="news_documentary",
        use_web_images=True,
    )
    assert "允许写实摄影和新闻纪实构图" in prompt
    assert "不得要求复刻其精确面孔" in prompt
    assert "避免任何摄影、写实" not in prompt
    rendered, visual_type = profile.render_prompt(
        "location_wide",
        "雨后的城市道路和远处车站",
    )
    assert visual_type == "location_wide"
    assert "新闻纪实摄影风格" in rendered
    assert "不能伪装成真实现场证据" in rendered


def test_mock_speech_and_srt(tmp_path):
    provider = MockSpeechProvider(Settings(speech_speed=1.0))
    first = provider.synthesize("这是第一段用于测试的旁白。", tmp_path / "one", voice="test")
    second = provider.synthesize("这是第二段用于测试的旁白。", tmp_path / "two", voice="test")
    with wave.open(first.path, "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 32000
    scenes = [
        type(
            "Scene",
            (),
            {
                "duration_ms": first.duration_ms,
                "subtitle": "错误的精简字幕",
                "narration": "这是第一段用于测试的旁白。",
            },
        ),
        type(
            "Scene",
            (),
            {
                "duration_ms": second.duration_ms,
                "subtitle": "第二段",
                "narration": "这是第二段用于测试的旁白。",
            },
        ),
    ]
    subtitles = build_srt(scenes, gap_ms=120)
    assert srt_timestamp(3_661_007) == "01:01:01,007"
    assert "这是第一段用于测试的旁白。" in subtitles
    assert "错误的精简字幕" not in subtitles
    assert srt_timestamp(first.duration_ms) in subtitles
    assert srt_timestamp(first.duration_ms + 120) in subtitles
    retimed = build_srt(scenes, gap_ms=120, time_scale=0.5)
    assert srt_timestamp(round(first.duration_ms * 0.5)) in retimed


def test_overlay_labels_are_rendered_locally_with_scene_timing():
    scenes = [
        type(
            "Scene",
            (),
            {
                "duration_ms": 2000,
                "overlay_labels_json": json.dumps(
                    [{"text": "自己能控制", "x": 0.25, "y": 0.4}],
                    ensure_ascii=False,
                ),
            },
        ),
        type(
            "Scene",
            (),
            {
                "duration_ms": 1800,
                "overlay_labels_json": json.dumps(
                    [{"text": "别人来决定", "x": 0.75, "y": 0.4}],
                    ensure_ascii=False,
                ),
            },
        ),
    ]
    overlay = build_overlay_ass(scenes, width=1280, height=720, scene_gap_ms=120)
    assert "自己能控制" in overlay
    assert "别人来决定" in overlay
    assert r"\pos(320,288)" in overlay
    assert "0:00:02.22" in overlay


def test_minimax_speech_provider_persists_hex_audio(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {"audio": "49443304"},
                "extra_info": {"audio_length": 1234},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "payload": json})
        return FakeResponse()

    monkeypatch.setattr("app.services.video.speech.httpx.post", fake_post)
    provider = MiniMaxSpeechProvider(
        Settings(minimax_api_key="test-key", provider_max_attempts=1)
    )
    result = provider.synthesize("测试语音", tmp_path, voice="male-qn-qingse")
    assert Path(result.path).read_bytes() == bytes.fromhex("49443304")
    assert result.duration_ms == 1234
    assert captured["url"].endswith("/v1/t2a_v2")
    assert captured["payload"]["voice_setting"]["voice_id"] == "male-qn-qingse"


def test_minimax_speech_exposes_recommended_mandarin_voices():
    voices = speech_voice_options(Settings(speech_provider="minimax"))
    assert {item["id"] for item in voices} >= {
        "male-qn-qingse",
        "male-qn-jingying",
        "female-chengshu",
        "female-tianmei",
    }


def test_video_web_image_mode_prefers_web_and_falls_back_to_ai(monkeypatch):
    attempts = []

    class FakeProvider:
        def __init__(self, mode):
            self.mode = mode

        def generate(self, prompt, title, position, search_query=None):
            attempts.append(self.mode)
            if self.mode == "web":
                raise ImageProviderError("no suitable licensed result")
            return {"local_path": "/tmp/fallback.jpg", "source_kind": "ai"}

    monkeypatch.setattr(
        "app.services.video.assets.build_image_provider",
        lambda configured: FakeProvider(configured.image_source_mode),
    )
    provider = build_video_image_provider(
        Settings(), use_web_images=True, web_image_ratio=0.95
    )
    generated = provider.generate("prompt", "title", 0, "current event")
    assert attempts == ["web", "ai"]
    assert generated["source_kind"] == "ai"


def test_video_web_image_ratio_reserves_one_ai_first_slot_per_twenty(monkeypatch):
    attempts = []

    class FakeProvider:
        def __init__(self, mode):
            self.mode = mode

        def generate(self, prompt, title, position, search_query=None):
            attempts.append((position, self.mode))
            return {"local_path": f"/tmp/{self.mode}.jpg", "source_kind": self.mode}

    monkeypatch.setattr(
        "app.services.video.assets.build_image_provider",
        lambda configured: FakeProvider(configured.image_source_mode),
    )
    provider = build_video_image_provider(
        Settings(), use_web_images=True, web_image_ratio=0.95
    )
    assert provider.generate("prompt", "title", 18)["source_kind"] == "web"
    assert provider.generate("prompt", "title", 19)["source_kind"] == "ai"
    assert attempts == [(18, "web"), (19, "ai")]


def test_minimax_motion_video_provider_creates_polls_and_downloads(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        def __init__(self, body=None, content=b""):
            self._body = body
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_post(url, *, headers, json, timeout):
        captured.update({"create_url": url, "payload": json})
        return FakeResponse({"task_id": "task-1", "base_resp": {"status_code": 0}})

    def fake_get(url, *, timeout, headers=None, params=None):
        if url.endswith("/query/video_generation"):
            return FakeResponse(
                {
                    "status": "Success",
                    "file_id": "file-1",
                    "base_resp": {"status_code": 0},
                }
            )
        if url.endswith("/files/retrieve"):
            return FakeResponse(
                {
                    "file": {"download_url": "https://cdn.example/video.mp4"},
                    "base_resp": {"status_code": 0},
                }
            )
        return FakeResponse(content=b"mock-video")

    monkeypatch.setattr("app.services.video.motion.httpx.post", fake_post)
    monkeypatch.setattr("app.services.video.motion.httpx.get", fake_get)
    provider = MiniMaxMotionVideoProvider(
        Settings(
            minimax_api_key="test-key",
            minimax_video_model="MiniMax-Hailuo-2.3",
            video_generation_poll_seconds=0,
        )
    )
    output = tmp_path / "intro.mp4"
    result = provider.generate("minimal motion", output)
    assert output.read_bytes() == b"mock-video"
    assert result.task_id == "task-1"
    assert result.file_id == "file-1"
    assert captured["create_url"].endswith("/v1/video_generation")
    assert captured["payload"]["model"] == "MiniMax-Hailuo-2.3"
    assert captured["payload"]["aigc_watermark"] is False


def test_video_api_create_get_and_cancel():
    with TestClient(app) as client:
        providers = client.get("/api/providers").json()
        assert providers["speech"]["provider"] == "mock"
        assert providers["speech"]["configured"] is True
        assert providers["speech"]["voices"][0]["id"] == "test"
        response = client.post(
            "/api/video-projects",
            json={
                "source": {"type": "topic", "topic": "为什么习惯需要循序渐进"},
                "target_duration_seconds": 30,
                "subtitles": True,
            },
        )
        assert response.status_code == 202
        project = response.json()
        assert project["status"] == "pending"
        assert project["width"] == 1280
        assert project["scenes"] == []

        fetched = client.get(f"/api/video-projects/{project['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["source_topic"] == "为什么习惯需要循序渐进"

        cancelled = client.post(f"/api/video-projects/{project['id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        invalid_profile = client.post(
            "/api/video-projects",
            json={
                "source": {"type": "topic", "topic": "测试不存在的视觉范式"},
                "visual_profile": "does_not_exist",
            },
        )
        assert invalid_profile.status_code == 422


def test_video_frontend_and_text_source_with_editable_prompts():
    with TestClient(app) as client:
        page = client.get("/admin/videos")
        assert page.status_code == 200
        assert "养生文章" in page.text
        assert "情感文章" in page.text
        assert "视频生成" in page.text
        assert 'id="voice"' in page.text
        assert 'id="script_prompt"' in page.text
        assert 'id="style"' in page.text
        assert 'id="bookends"' in page.text
        assert 'id="use_web_images"' in page.text
        assert 'id="web_image_ratio"' in page.text
        assert 'id="workflow"' in page.text
        assert 'id="news_mechanical"' in page.text
        assert 'id="news_review"' in page.text
        assert 'id="news_review_enabled"' in page.text
        assert "查看新闻 Fact Sheet、来源和审核报告" in page.text
        assert "优先使用互联网上的真实图片" in page.text
        assert "新闻纪实照片（news_documentary）" in page.text
        assert "文本 / 原稿" in page.text

        source_text = "别人对你的评价属于他的课题，你能做的是决定自己接下来怎样行动。"
        response = client.post(
            "/api/video-projects",
            json={
                "source": {"type": "text", "text": source_text},
                "target_duration_seconds": 30,
                "voice": "test",
                "bookends": True,
                "use_web_images": True,
                "web_image_ratio": 0.95,
                "script_prompt": "语气像朋友聊天，避免课程腔。",
                "style": "多用人物动作和常见生活图标。",
            },
        )
        assert response.status_code == 202
        project = response.json()
        assert project["source_type"] == "text"
        assert project["bookends"] is True
        assert project["use_web_images"] is True
        assert project["web_image_ratio"] == 0.95
        assert project["actual_web_image_ratio"] is None
        assert project["script_prompt"] == "语气像朋友聊天，避免课程腔。"
        assert project["style"] == "多用人物动作和常见生活图标。"

    with SessionLocal() as session:
        stored = session.get(VideoProject, project["id"])
        assert stored.source_text == source_text
        assert stored.source_hash


def test_news_project_api_persists_role_routes_and_workflow():
    with TestClient(app) as client:
        response = client.post(
            "/api/video-projects",
            json={
                "source": {"type": "topic", "topic": "某公司发布最新公开说明"},
                "workflow": "news_research",
                "news_angle": "timeline",
                "news_model_routes": {
                    "mechanical": "openai-luna",
                    "judgment": "openai-terra",
                    "research": "openai",
                    "writer": "minimax",
                    "review": "openai-terra",
                },
            },
        )
        assert response.status_code == 202
        project = response.json()
        assert project["workflow"] == "news_research"
        assert project["news_angle"] == "timeline"
        assert project["news_model_routes"]["writer"] == "minimax"
        assert project["news_review_enabled"] is False
        assert project["research"] == {}
        assert project["review"] == {}


def test_news_research_requires_sources_and_builds_evidence_checked_fact_sheet(monkeypatch):
    def fake_search(query, configured):
        return [{"url": "https://news.example.org/report", "title": "独立报道"}]

    def fake_fetch(url, configured):
        return ExtractedSource(
            url=url,
            title="独立报道",
            text="独立媒体确认这项公开安排已经公布。报道同时说明后续时间仍可能调整。",
            content_hash="hash-2",
        )

    monkeypatch.setattr("app.services.video.news_research.search_news", fake_search)
    monkeypatch.setattr("app.services.video.news_research.fetch_article", fake_fetch)
    research, calls = run_news_research(
        Settings(
            news_min_independent_sources=2,
            news_max_documents=4,
            news_search_max_results=4,
        ),
        topic="公开安排说明",
        source_url="https://official.example.com/notice",
        source_title="公开说明",
        source_text="官方说明这项公开安排已经公布。具体执行时间以之后通知为准。",
        routes={
            "mechanical": "mock",
            "judgment": "mock",
            "research": "mock",
            "writer": "mock",
            "review": "mock",
        },
        requested_angle="straight_news",
    )
    assert len(research["sources"]) == 2
    assert {call.step for call in calls} == {
        "news_query_expansion",
        "news_mechanical",
        "news_judgment",
        "news_fact_sheet",
    }
    writable = research["fact_sheet"]["writable_claims"]
    assert writable
    assert all(item["evidence"] for item in writable)
    script = {"scenes": [{"claim_ids": [writable[0]["claim_id"]]}]}
    assert audit_claim_ids(script, research) == []
    assert audit_claim_ids({"scenes": [{"claim_ids": ["C999"]}]}, research)
    assert audit_claim_ids(
        {"scenes": [{"claim_ids": []}]}, research, require_claim_ids=False
    ) == []


def test_review_correction_merges_narration_patch_without_losing_visual_fields():
    original = {
        "title": "原标题",
        "summary": "原摘要",
        "scenes": [
            {
                "position": 0,
                "narration": "不够准确的原旁白",
                "claim_ids": ["C1"],
                "image_prompt": "保留的完整图片提示词",
                "visual_type": "location_wide",
                "search_query": "Henan news",
            },
            {
                "position": 1,
                "narration": "无需修改的旁白",
                "claim_ids": ["C2"],
                "image_prompt": "第二幕图片提示词",
                "visual_type": "object_closeup",
                "search_query": "official notice",
            },
        ],
    }
    review = {
        "unsupported_claims": [{"scene_position": 0, "issue": "表述不准确"}],
        "corrected_script": {
            "scenes": [
                {"position": 0, "narration": "据工作专班通报，嫌疑人已被抓获。", "claim_ids": ["C3"]}
            ]
        },
    }
    merged, errors = merge_review_correction(original, review)
    assert errors == []
    assert merged["scenes"][0]["narration"].startswith("据工作专班通报")
    assert merged["scenes"][0]["image_prompt"] == "保留的完整图片提示词"
    assert merged["scenes"][0]["search_query"] == "Henan news"
    assert merged["scenes"][1] == original["scenes"][1]


def test_failed_video_project_can_be_edited_and_deleted():
    with TestClient(app) as client:
        created = client.post(
            "/api/video-projects",
            json={"source": {"type": "topic", "topic": "需要修改的失败任务"}},
        ).json()
        with SessionLocal() as session:
            stored = session.get(VideoProject, created["id"])
            stored.status = "failed"
            stored.current_step = "review_news"
            stored.error_message = "old failure"
            stored.research_json = json.dumps({"old": True})
            stored.review_json = json.dumps({"verdict": "blocked"})
            session.commit()
        updated = client.put(
            f"/api/video-projects/{created['id']}",
            json={
                "source": {"type": "topic", "topic": "修改后的新闻任务"},
                "workflow": "news_research",
                "news_model_routes": {
                    "mechanical": "minimax",
                    "judgment": "openai-terra",
                    "research": "openai",
                    "writer": "minimax",
                    "review": "openai-terra",
                },
            },
        )
        assert updated.status_code == 200
        body = updated.json()
        assert body["id"] == created["id"]
        assert body["status"] == "pending"
        assert body["source_topic"] == "修改后的新闻任务"
        assert body["research"] == {}
        assert body["review"] == {}
        with SessionLocal() as session:
            stored = session.get(VideoProject, created["id"])
            stored.status = "failed"
            session.commit()
        deleted = client.delete(f"/api/video-projects/{created['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/video-projects/{created['id']}").status_code == 404


def test_stale_active_video_project_is_recovered_during_worker_polling():
    with TestClient(app) as client:
        created = client.post(
            "/api/video-projects",
            json={"source": {"type": "topic", "topic": "中断恢复测试"}},
        ).json()
    now = datetime.now(UTC)
    with SessionLocal() as session:
        stored = session.get(VideoProject, created["id"])
        stored.status = "generating_assets"
        stored.current_step = "generate_assets"
        stored.heartbeat_at = now - timedelta(minutes=31)
        session.commit()

        recovered = recover_stale_video_projects(session, settings, now=now)
        session.refresh(stored)
        assert recovered == 1
        assert stored.status == "failed"
        assert stored.retry_count == 1
        assert stored.next_attempt_at.replace(tzinfo=UTC) == now
        assert "worker stopped" in str(stored.error_message).lower()


def test_mock_video_pipeline_runs_all_steps_without_external_providers(monkeypatch):
    with TestClient(app) as client:
        created = client.post(
            "/api/video-projects",
            json={
                "source": {"type": "topic", "topic": "如何把一个小习惯坚持下来"},
                "target_duration_seconds": 30,
                "subtitles": False,
            },
        ).json()

    def fake_compose(scenes, output_dir, settings, **kwargs):
        scene_list = list(scenes)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output.mp4"
        path.write_bytes(b"mock-mp4")
        return ComposedVideo(
            path=str(path),
            duration_ms=sum(scene.duration_ms for scene in scene_list),
            width=kwargs["width"],
            height=kwargs["height"],
        )

    monkeypatch.setattr("app.services.video.pipeline.compose_video", fake_compose)
    with SessionLocal() as session:
        result = execute_video_project(session, created["id"], settings)
        assert result.status == "completed"
        project = session.scalar(
            select(VideoProject)
            .where(VideoProject.id == created["id"])
            .options(selectinload(VideoProject.scenes))
        )
        assert project.progress == 100
        assert project.actual_duration_ms is not None
        assert len(project.scenes) >= 3
        assert all(scene.status == "completed" for scene in project.scenes)
        assert Path(project.output_path).read_bytes() == b"mock-mp4"
        assert Path(project.subtitle_path).is_file()
        manifest = json.loads(
            (Path(project.output_path).parent / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["project_id"] == project.id
