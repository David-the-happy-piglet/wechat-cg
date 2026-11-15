import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.models import Article, Asset, AudienceProfile
from app.services.generator import audience_to_context
from app.services.humanizer import build_humanizer_prompt
from app.services.images import (
    DuckDuckGoImageProvider,
    ImageProviderError,
    MiniMaxImageProvider,
    MixedImageProvider,
    OpenverseImageProvider,
    WikimediaImageProvider,
    normalize_search_query,
)
from app.services.llm import (
    LLMError,
    MockLLMProvider,
    build_llm_provider,
    parse_json_response,
    parse_token_usage,
)
from app.services.prompts import (
    build_article_prompt,
    build_emotional_article_prompt,
    build_emotional_topic_prompt,
    build_story_draft_prompt,
    build_story_outline_prompt,
    build_story_review_prompt,
)
from app.services.renderer import render_emotional_html, render_wechat_html
from app.services.safety import scan_article
from app.services.wechat import WechatClient, WechatError


def sample_audience() -> AudienceProfile:
    return AudienceProfile(
        id=1,
        slug="middle-age-health",
        name="中年健康管理",
        age_range="35-55",
        description="工作和家庭压力较大的人群",
        interests_json=json.dumps(["睡眠", "职场压力"], ensure_ascii=False),
        writing_style="节奏明快，强调可执行性",
        title_preferences_json=json.dumps(
            ["人到中年……", "每天坐8小时的人……"],
            ensure_ascii=False,
        ),
        article_length=1000,
        image_style="现代都市生活中的中国中年人物",
        prohibited_topics_json=json.dumps(["建议停药"], ensure_ascii=False),
        enabled=True,
    )


def test_audience_context_is_dynamic():
    context = audience_to_context(sample_audience())
    assert context["name"] == "中年健康管理"
    assert context["age_range"] == "35-55"
    assert context["interests"] == ["睡眠", "职场压力"]


def test_prompt_contains_selected_audience():
    context = audience_to_context(sample_audience())
    prompt = build_article_prompt(
        topic="改善久坐带来的疲劳",
        category="职场健康",
        audience=context,
        title_count=5,
        image_count=3,
    )
    assert "中年健康管理" in prompt
    assert "35-55" in prompt
    assert "人到中年" in prompt
    assert "目标长度：约1000字" in prompt
    assert "不得返回数组、对象或其他类型" in prompt


def test_prompt_applies_headline_intensity():
    context = audience_to_context(sample_audience())
    prompt = build_article_prompt(
        topic="改善久坐带来的疲劳",
        category="职场健康",
        audience=context,
        title_count=5,
        image_count=3,
        headline_intensity=5,
    )
    assert "标题吸睛程度：5/5" in prompt
    assert "信息缺口" in prompt
    assert "不得编造数字" in prompt


def test_emotional_prompt_covers_relationship_topics_with_clear_boundaries():
    context = audience_to_context(sample_audience())
    context["description"] = "恋爱中的男女，以女性读者为主"
    context["article_length"] = 1500
    prompt = build_emotional_article_prompt(
        topic="分手后总想反复查看对方动态",
        category="分手创伤自我修复",
        audience=context,
        title_count=5,
        image_count=3,
        headline_intensity=4,
        source_instruction="写她反复点开头像又退出来的动作。",
    )
    assert "[EMOTIONAL_ARTICLE]" in prompt
    assert "恋爱中的男女，以女性读者为主" in prompt
    assert "禁止 PUA、操控" in prompt
    assert "星座内容属于轻松的情感观察" in prompt
    assert "写她反复点开头像又退出来的动作" in prompt
    assert "本篇主要写法：情感共鸣型" in prompt
    assert "每个感受都有动作或场景支撑" in prompt
    assert "不教读者怎么做，不写建议清单" in prompt


@pytest.mark.parametrize("title_reference", [False, True])
@pytest.mark.parametrize("category", ["失恋主题", "恋爱心理", "星座恋爱性格", "前任", "情感综合"])
def test_all_emotional_categories_use_resonance_prompt(category, title_reference):
    prompt = build_emotional_article_prompt(
        topic="消息改了六遍才发出去，发完又把手机扣在桌上",
        category=category,
        audience=audience_to_context(sample_audience()),
        title_count=5,
        image_count=3,
        emotional_title_reference=title_reference,
    )
    assert "本篇主要写法：情感共鸣型" in prompt
    assert "把一条消息改了六遍才发出去" in prompt
    assert "每个候选标题和 selected_title 都是一句有具体依据的判断，25字以内" in prompt
    assert "不用问号" in prompt
    assert "不额外输出“主题、处境、标题、正文”等工作标签" in prompt
    assert "只选下面一种文风" not in prompt
    assert "2至4个有信息的小标题" not in prompt
    assert "heading 全部为空字符串" in prompt
    assert "你查他的星座，是想知道他会不会留下" in prompt
    assert "只写恋爱和两个人之间的事，不写亲情、友情、职场" in prompt
    assert "他把前任删了，却留着她送的那块表" in prompt
    assert "我们没绝交，只是再也约不上了" not in prompt
    if title_reference:
        assert "共鸣型 Prompt 的标题规则始终优先" in prompt


def test_heartbreak_topic_prompt_preserves_selected_category():
    context = audience_to_context(sample_audience())
    prompt = build_emotional_topic_prompt(
        directions="写失恋后的具体生活片段",
        category="失恋主题",
        audience=context,
        recent_topics=[],
        local_date="2026-09-26",
    )
    assert "category 必须原样返回“失恋主题”" in prompt
    assert "不得改成情感安慰或其他分类" in prompt


def test_emotional_topic_prompt_limits_legacy_directions_to_romance():
    prompt = build_emotional_topic_prompt(
        directions="亲情、友情、职场、恋爱里的钱",
        category="友情",
        audience=audience_to_context(sample_audience()),
        recent_topics=[],
        local_date="2026-09-26",
    )
    assert "只写恋爱和两个人之间的事，不写亲情、友情、职场" in prompt
    assert "只选其中与恋爱相关的部分" in prompt
    assert "恋爱里的钱" in prompt
    assert "领导一句" not in prompt


def test_micro_horror_prompts_separate_planning_writing_and_review():
    outline_prompt = build_story_outline_prompt(
        topic="门锁开始拒绝屋主回家",
        category="微恐短篇",
        story_mode="near_future",
        target_length=1800,
        source_material="一篇参考故事",
        recent_signatures=["旧故事｜旧标题｜旧结尾"],
    )
    assert "[MICRO_HORROR_OUTLINE]" in outline_prompt
    assert "不要写正文或标题" in outline_prompt
    assert "不得复用参考材料" in outline_prompt
    assert "2至4个" in outline_prompt
    assert "目标读者" not in outline_prompt
    assert "读者边界" in outline_prompt

    outline = MockLLMProvider().generate_json(
        outline_prompt,
        operation="micro_horror_outline",
        context={"topic": "门锁开始拒绝屋主回家", "story_mode": "near_future"},
    )
    draft_prompt = build_story_draft_prompt(
        topic="门锁开始拒绝屋主回家",
        category="微恐短篇",
        outline=outline,
        title_count=5,
        image_count=2,
        target_length=1800,
        headline_intensity=4,
    )
    assert "[MICRO_HORROR_DRAFT]" in draft_prompt
    assert "不得故意隐瞒" in draft_prompt
    assert "标题和摘要不得泄露真相" in draft_prompt
    assert "目标读者" not in draft_prompt

    draft = MockLLMProvider().generate_json(
        draft_prompt,
        operation="micro_horror_draft",
        context={"topic": "门锁开始拒绝屋主回家", "title_count": 5, "image_count": 2},
    )
    review_prompt = build_story_review_prompt(
        draft=draft,
        outline=outline,
        target_length=1800,
        title_count=5,
        image_count=2,
    )
    assert "[MICRO_HORROR_REVIEW]" in review_prompt
    assert "视角公平" in review_prompt
    assert "机械连接词" in review_prompt


def test_humanizer_prompt_preserves_structure_and_removes_ai_patterns():
    prompt = build_humanizer_prompt(
        {
            "title_candidates": ["标题一"],
            "selected_title": "标题一",
            "summary": "此外，这是一个至关重要的习惯。",
            "opening": "值得注意的是，睡眠十分重要。",
            "sections": [{"heading": "建议", "paragraphs": ["第一，早点睡。"]}],
            "conclusion": "总而言之，未来可期。",
            "image_prompts": [{"position": 0, "prompt": "图片"}],
        }
    )
    assert "Humanizer-zh" in prompt
    assert "title_candidates、selected_title 和 image_prompts 原样保留" in prompt
    assert "机械连接词" in prompt
    assert "不虚构个人经历" in prompt
    assert "目标至少 45/50" in prompt


def test_emotional_humanizer_preserves_comfort_before_advice():
    prompt = build_humanizer_prompt(
        {
            "title_candidates": ["标题一"],
            "selected_title": "标题一",
            "summary": "你还在难过。",
            "opening": "夜里又点开了聊天记录。",
            "sections": [{"heading": "舍不得", "paragraphs": ["你没有睡着。"]}],
            "conclusion": "今晚先把手机放远一点。",
            "image_prompts": [{"position": 0, "prompt": "图片"}],
        },
        content_type="emotional_article",
    )
    assert "中文情感编辑" in prompt
    assert "保留有出处、贴合处境的心理解释及其适用范围" in prompt
    assert "不新增“你应该、建议你、学会……”" in prompt
    assert "结尾停在一句话或画面上" in prompt


def test_heartbreak_humanizer_uses_new_resonance_rules():
    prompt = build_humanizer_prompt(
        {
            "title_candidates": ["分手第三天，我把他的外套收进纸箱"],
            "selected_title": "分手第三天，我把他的外套收进纸箱",
            "summary": "一件外套留在门口。",
            "opening": "凌晨一点，门口的外套还挂着。",
            "sections": [{"heading": "分手第三天", "paragraphs": ["我把它折好。"]}],
            "conclusion": "纸箱合上了。",
            "image_prompts": [{"position": 0, "prompt": "门口的外套"}],
        },
        content_type="emotional_article",
        category="失恋主题",
        topic="收起前任留下的外套",
    )
    assert "本篇主要写法：情感共鸣型" in prompt
    assert "保留一个具体处境和其中准确的小动作" in prompt
    assert "只写恋爱和两个人之间的事，不写亲情、友情、职场" in prompt
    assert "heading 全部为空字符串" in prompt
    assert "不补总结、升华、口号" in prompt
    assert "删除机械连接词、反复的三段式、否定式排比和刻意金句" not in prompt


def test_mock_generation_and_rendering():
    context = audience_to_context(sample_audience())
    result = MockLLMProvider().generate_article(
        "prompt",
        {
            "topic": "改善久坐带来的疲劳",
            "category": "职场健康",
            "audience": context,
            "title_count": 5,
            "image_count": 3,
        },
    )
    assert len(result["title_candidates"]) == 5
    assert len(result["image_prompts"]) == 3
    html = render_wechat_html(result, ["cover.png", "one.png", "two.png"])
    assert result["sections"][0]["heading"] in html
    assert "one.png" in html


def test_renderer_adds_web_image_attribution():
    html = render_wechat_html(
        {
            "opening": "开头",
            "sections": [{"heading": "小标题", "paragraphs": ["正文"]}],
            "conclusion": "结尾",
        },
        ["cover.png"],
        [
            {
                "source_kind": "web",
                "source_page_url": "https://commons.wikimedia.org/wiki/File:Tea.jpg",
                "source_author": "Example Author",
                "license_name": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            }
        ],
    )
    assert "图片来源与许可" in html
    assert "Example Author" in html
    assert "CC BY-SA 4.0" in html
    assert "人工智能辅助生成" not in html


def test_emotional_renderer_inserts_images_in_single_section():
    paragraphs = [f"第{i}段" for i in range(1, 10)]
    html = render_emotional_html(
        {
            "opening": "开头",
            "sections": [{"heading": "", "paragraphs": paragraphs}],
            "conclusion": "结尾",
        },
        ["cover.jpg", "first.jpg", "second.jpg"],
    )
    assert html.count("<img") == 2
    assert "first.jpg" in html and "second.jpg" in html
    assert html.index("第3段") < html.index("first.jpg") < html.index("第4段")
    assert html.index("第6段") < html.index("second.jpg") < html.index("第7段")


def test_renderer_uses_story_layout_for_micro_horror():
    html = render_wechat_html(
        {
            "content_type": "micro_horror",
            "opening": "门锁亮起红灯。",
            "sections": [{"heading": "访客", "paragraphs": ["屋里有人走动。"]}],
            "conclusion": "门内传来他的声音。",
        },
        ["cover.png"],
    )
    assert "background:#fbfaf8" in html
    assert "font-family:Georgia" in html
    assert "background:#f4f8f5" not in html


def test_mock_headline_changes_with_intensity():
    context = audience_to_context(sample_audience())
    base_context = {
        "topic": "改善久坐带来的疲劳",
        "category": "职场健康",
        "audience": context,
        "title_count": 5,
        "image_count": 1,
    }
    calm = MockLLMProvider().generate_article(
        "prompt",
        {**base_context, "headline_intensity": 1},
    )
    strong = MockLLMProvider().generate_article(
        "prompt",
        {**base_context, "headline_intensity": 5},
    )
    assert calm["selected_title"] != strong["selected_title"]
    assert "提醒家里人" in strong["selected_title"]


def test_safety_scan_blocks_high_risk_claims():
    flags = scan_article("每天吃一片，不用吃药，也能彻底治愈。")
    assert "包含绝对治疗承诺" in flags
    assert "可能诱导停止或替代正规治疗" in flags


def test_safety_scan_does_not_treat_food_portions_as_medication_dosage():
    flags = scan_article(
        "每天水果可以参考200到350克，根据全天饮食调整。"
        "水果不能代替一日三餐，也不能替代治疗。"
    )
    assert "可能包含具体药物或补充剂剂量" not in flags


def test_safety_scan_still_flags_specific_medication_dosage():
    flags = scan_article("维生素D补充剂每日服用1片，具体剂量无需咨询医生。")
    assert "可能包含具体药物或补充剂剂量" in flags


def test_parse_json_response_handles_reasoning_and_markdown():
    content = """<think>先规划文章结构</think>
```json
{"selected_title": "晚饭后的三个小习惯"}
```"""
    assert parse_json_response(content) == {"selected_title": "晚饭后的三个小习惯"}


def test_parse_json_response_ignores_trailing_text():
    content = '{"topic": "夏季睡眠"}\n以上是今天的选题。'
    assert parse_json_response(content) == {"topic": "夏季睡眠"}


def test_parse_json_response_uses_first_complete_object():
    content = (
        '{"topic": "夏季睡眠", "category": "健康"}'
        '{"topic": "补充选题", "category": "生活"}'
    )
    assert parse_json_response(content) == {
        "topic": "夏季睡眠",
        "category": "健康",
    }


def test_parse_json_response_finds_object_after_leading_explanation():
    content = '按要求返回如下结果：\n{"topic": "夏季睡眠"}'
    assert parse_json_response(content) == {"topic": "夏季睡眠"}


def test_kimi_uses_required_temperature():
    provider = build_llm_provider(
        Settings(moonshot_api_key="test-key"),
        "kimi",
    )
    assert provider.temperature == 1


def test_deepseek_uses_larger_output_budget():
    provider = build_llm_provider(
        Settings(deepseek_api_key="test-key"),
        "deepseek",
    )
    assert provider.max_tokens_field == "max_tokens"
    assert provider.max_output_tokens == 8192


def test_minimax_m3_uses_standard_endpoint_and_larger_output_budget(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"topic": "测试选题"}'},
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        captured.update({"url": url, "payload": json})
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    provider = build_llm_provider(
        Settings(
            minimax_api_key="test-key",
            minimax_model="MiniMax-M3",
            provider_max_attempts=1,
        ),
        "minimax",
    )

    assert provider.generate_json("prompt") == {"topic": "测试选题"}
    assert captured["url"] == "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert captured["payload"]["max_completion_tokens"] == 16384


@pytest.mark.parametrize(("model", "effort", "expected"), [
    ("MiniMax-M3", "max", {"thinking": {"type": "disabled"}}),
    ("MiniMax-M3.1-Flash", "", {}),
    ("MiniMax-M3.1-Flash", "xhigh", {"reasoning_effort": "xhigh"}),
    ("MiniMax-M2.7", "high", {}),
])
def test_minimax_reasoning_controls_follow_the_model(monkeypatch, model, effort, expected):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"topic": "测试"}'}}]}

    def fake_post(url, *, json, headers, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    provider = build_llm_provider(
        Settings(minimax_api_key="test-key", minimax_model=model, minimax_reasoning_effort=effort,
                 provider_max_attempts=1),
        "minimax",
    )
    assert provider.generate_json("prompt") == {"topic": "测试"}
    controls = {key: captured[key] for key in ("thinking", "reasoning_effort") if key in captured}
    assert controls == expected


def test_minimax_text_reports_business_error_when_choices_are_null(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.headers = {"content-type": "application/json"}
            self.content = b"{}"
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": None,
                "usage": {"total_tokens": 0},
                "base_resp": {
                    "status_code": 2056,
                    "status_msg": "Token Plan usage limit reached",
                },
            }

    monkeypatch.setattr(
        "app.services.llm.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    provider = build_llm_provider(
        Settings(minimax_api_key="test-key", provider_max_attempts=1),
        "minimax",
    )
    try:
        provider.generate_json("prompt")
    except LLMError as exc:
        message = str(exc)
        assert "status_code=2056" in message
        assert "Token Plan usage limit reached" in message
    else:
        raise AssertionError("Expected MiniMax business error")


def test_llm_retries_transient_transport_errors(monkeypatch):
    attempts = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"topic": "重试成功"}'},
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("SSL EOF")
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    provider = build_llm_provider(
        Settings(
            moonshot_api_key="test-key",
            provider_max_attempts=3,
            provider_retry_base_seconds=0,
        ),
        "kimi",
    )

    assert provider.generate_json("prompt") == {"topic": "重试成功"}
    assert attempts == 3


def test_openai_models_use_expected_request_shape(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"topic": "测试选题"}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

    def fake_post(url, *, json, headers, timeout):
        captured.clear()
        captured.update({"url": url, "payload": json})
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    settings = Settings(openai_api_key="test-key")
    cases = (
        ("openai", "gpt-5.6-sol", "medium"),
        ("openai-terra", "gpt-5.6-terra", "low"),
        ("openai-luna", "gpt-5.6-luna", "none"),
    )

    for provider_name, expected_model, expected_effort in cases:
        provider = build_llm_provider(settings, provider_name)
        assert provider.generate_json("prompt") == {"topic": "测试选题"}
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["payload"]["model"] == expected_model
        assert captured["payload"]["reasoning_effort"] == expected_effort
        assert captured["payload"]["max_completion_tokens"] == 8192
        assert captured["payload"]["response_format"] == {"type": "json_object"}
        assert "temperature" not in captured["payload"]


def test_token_usage_supports_openai_and_alias_fields():
    assert (
        parse_token_usage(
            {
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 80,
                    "total_tokens": 200,
                }
            }
        ).total_tokens
        == 200
    )
    aliased = parse_token_usage({"usage": {"input_tokens": 25, "output_tokens": 15}})
    assert aliased.prompt_tokens == 25
    assert aliased.completion_tokens == 15
    assert aliased.total_tokens == 40


def test_minimax_images_respect_disabled_aigc_watermark(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"image_base64": ["anBlZw=="]}}

    def fake_post(url, *, headers, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("app.services.images.httpx.post", fake_post)
    provider = MiniMaxImageProvider(
        Settings(
            minimax_api_key="test-key",
            aigc_watermark=False,
            generated_assets_dir=str(tmp_path),
        )
    )
    generated = provider.generate("prompt", "title", 0)
    assert captured["aigc_watermark"] is False
    assert captured["prompt"] == "prompt"
    assert captured["prompt_optimizer"] is False
    assert Path(str(generated["local_path"])).read_bytes() == b"jpeg"


def test_minimax_images_retry_transient_transport_errors(monkeypatch, tmp_path: Path):
    attempts = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"image_base64": ["anBlZw=="]}}

    def fake_post(url, *, headers, json, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.RemoteProtocolError("incomplete chunked read")
        return FakeResponse()

    monkeypatch.setattr("app.services.images.httpx.post", fake_post)
    provider = MiniMaxImageProvider(
        Settings(
            minimax_api_key="test-key",
            generated_assets_dir=str(tmp_path),
            provider_max_attempts=3,
            provider_retry_base_seconds=0,
        )
    )

    generated = provider.generate("prompt", "title", 0)
    assert attempts == 3
    assert Path(str(generated["local_path"])).read_bytes() == b"jpeg"


def test_minimax_composition_constraint_respects_prompt_length_limit():
    constrained = MiniMaxImageProvider._composition_safe_prompt("a" * 2000)
    assert constrained == "a" * 1500


def test_minimax_images_report_business_error(monkeypatch, tmp_path: Path):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": None,
                "metadata": {"failed_count": 1},
                "base_resp": {
                    "status_code": 1008,
                    "status_msg": "insufficient balance",
                },
            }

    monkeypatch.setattr(
        "app.services.images.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    provider = MiniMaxImageProvider(
        Settings(
            minimax_api_key="test-key",
            generated_assets_dir=str(tmp_path),
        )
    )

    try:
        provider.generate("prompt", "title", 0)
    except ImageProviderError as exc:
        message = str(exc)
        assert "status_code=1008" in message
        assert "insufficient balance" in message
    else:
        raise AssertionError("Expected MiniMax business error")


def test_minimax_images_report_empty_data(monkeypatch, tmp_path: Path):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": None,
                "metadata": {"failed_count": 1},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }

    monkeypatch.setattr(
        "app.services.images.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    provider = MiniMaxImageProvider(
        Settings(
            minimax_api_key="test-key",
            generated_assets_dir=str(tmp_path),
        )
    )

    try:
        provider.generate("prompt", "title", 0)
    except ImageProviderError as exc:
        message = str(exc)
        assert "returned no image data" in message
        assert "failed_count=1" in message
    else:
        raise AssertionError("Expected missing MiniMax image data error")


def test_wikimedia_provider_downloads_reusable_image(monkeypatch, tmp_path: Path):
    image_buffer = BytesIO()
    Image.new("RGB", (800, 600), "white").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()

    class FakeResponse:
        def __init__(self, *, payload=None, content=b""):
            self._payload = payload
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    search_payload = {
        "query": {
            "pages": [
                {
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "thumburl": "https://upload.wikimedia.test/tea.jpg",
                            "descriptionurl": ("https://commons.wikimedia.org/wiki/File:Tea.jpg"),
                            "extmetadata": {
                                "Artist": {"value": "<b>Example Author</b>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {
                                    "value": ("https://creativecommons.org/licenses/by-sa/4.0/")
                                },
                            },
                        }
                    ]
                }
            ]
        }
    }

    def fake_get(url, **kwargs):
        if "api.php" in url:
            return FakeResponse(payload=search_payload)
        return FakeResponse(content=image_bytes)

    monkeypatch.setattr("app.services.images.httpx.get", fake_get)
    provider = WikimediaImageProvider(
        Settings(
            generated_assets_dir=str(tmp_path),
            public_base_url="http://content.test",
        )
    )
    result = provider.generate("prompt", "tea", 1, "green tea cup")

    assert result["source_kind"] == "web"
    assert result["source_author"] == "Example Author"
    assert result["license_name"] == "CC BY-SA 4.0"
    assert Path(str(result["local_path"])).read_bytes() == image_bytes


def test_openverse_provider_preserves_source_and_license(monkeypatch, tmp_path: Path):
    image_buffer = BytesIO()
    Image.new("RGB", (800, 600), "white").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()

    class FakeResponse:
        def __init__(self, *, payload=None, content=b""):
            self._payload = payload
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    search_payload = {
        "results": [
            {
                "thumbnail": "https://api.openverse.test/image/thumb/",
                "foreign_landing_url": "https://example.test/source-image",
                "creator": "Example Photographer",
                "license": "by",
                "license_version": "4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ]
    }

    def fake_get(url, **kwargs):
        if url.endswith("/images/"):
            return FakeResponse(payload=search_payload)
        return FakeResponse(content=image_bytes)

    monkeypatch.setattr("app.services.images.httpx.get", fake_get)
    provider = OpenverseImageProvider(
        Settings(
            generated_assets_dir=str(tmp_path),
            public_base_url="http://content.test",
        )
    )
    result = provider.generate(
        "prompt",
        "tea",
        1,
        ["green tea", "cup"],
    )

    assert result["source_kind"] == "web"
    assert result["source_author"] == "Example Photographer"
    assert result["license_name"] == "CC BY 4.0"
    assert result["source_page_url"] == "https://example.test/source-image"
    assert Path(str(result["local_path"])).read_bytes() == image_bytes


def test_duckduckgo_provider_preserves_editorial_source_and_review_label(
    monkeypatch, tmp_path: Path
):
    image_buffer = BytesIO()
    Image.new("RGB", (1280, 720), "white").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()

    class FakeResponse:
        def __init__(self, *, payload=None, content=b"", text="", url="https://image.test"):
            self._payload = payload
            self.content = content
            self.text = text
            self.url = url

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if url.endswith("/i.js"):
            return FakeResponse(
                payload={
                    "results": [
                        {
                            "image": "https://images.example/alamy-preview.jpg",
                            "url": "https://www.alamy.com/stock-photo/example.html",
                            "source": "Stock preview",
                            "width": 1280,
                            "height": 720,
                        },
                        {
                            "image": "https://www.dreamstime.com/preview.jpg",
                            "url": "https://publisher.test/story",
                            "source": "Stock preview",
                            "width": 1280,
                            "height": 720,
                        },
                        {
                            "image": "https://image.test/news.jpg",
                            "url": "https://publisher.test/story",
                            "source": "Example News",
                            "width": 1280,
                            "height": 720,
                        }
                    ]
                }
            )
        if "duckduckgo" in url:
            return FakeResponse(text='vqd="test-token"')
        return FakeResponse(content=image_bytes, url=url)

    monkeypatch.setattr("app.services.images.httpx.get", fake_get)
    monkeypatch.setattr("app.services.images.validate_public_url", lambda url: url)
    provider = DuckDuckGoImageProvider(
        Settings(
            generated_assets_dir=str(tmp_path),
            public_base_url="http://content.test",
        )
    )
    result = provider.generate("prompt", "news", 1, "current news event")

    assert result["source_kind"] == "web"
    assert result["source_author"] == "Example News"
    assert result["source_page_url"] == "https://publisher.test/story"
    assert "版权状态待确认" in str(result["license_name"])
    assert Path(str(result["local_path"])).read_bytes() == image_bytes


def test_search_query_normalization_handles_model_type_drift():
    assert normalize_search_query(" healthy family ", "fallback") == "healthy family"
    assert (
        normalize_search_query(
            ["healthy", " family ", "healthy", 2026, {"ignored": True}],
            "fallback",
        )
        == "healthy family 2026"
    )
    assert normalize_search_query({"unexpected": "object"}, "fallback") == "fallback"
    assert normalize_search_query([], "fallback") == "fallback"


def test_mixed_provider_uses_web_for_alternating_inline_images():
    class StubProvider:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, title, position, search_query=None):
            return {"source_kind": self.name}

    provider = MixedImageProvider(
        StubProvider("ai"),
        StubProvider("web"),
        fallback_to_ai=True,
    )
    assert provider.generate("p", "t", 0)["source_kind"] == "ai"
    assert provider.generate("p", "t", 1)["source_kind"] == "web"
    assert provider.generate("p", "t", 2)["source_kind"] == "ai"


def test_mixed_provider_falls_back_to_ai():
    class AiProvider:
        def generate(self, prompt, title, position, search_query=None):
            return {"source_kind": "ai"}

    class FailingWebProvider:
        def generate(self, prompt, title, position, search_query=None):
            raise ImageProviderError("no result")

    provider = MixedImageProvider(
        AiProvider(),
        FailingWebProvider(),
        fallback_to_ai=True,
    )
    assert provider.generate("p", "t", 1)["source_kind"] == "ai"


def test_mixed_provider_never_falls_back_to_web_for_cover():
    class FailingAiProvider:
        def generate(self, prompt, title, position, search_query=None):
            raise ImageProviderError("no AI image")

    class WebProvider:
        def generate(self, prompt, title, position, search_query=None):
            return {"source_kind": "web"}

    provider = MixedImageProvider(
        FailingAiProvider(),
        WebProvider(),
        fallback_to_ai=True,
    )
    with pytest.raises(ImageProviderError, match="no AI image"):
        provider.generate("p", "t", 0)
    assert provider.generate("p", "t", 2)["source_kind"] == "web"


def test_wechat_rejects_existing_stock_preview_cover():
    article = Article(selected_title="测试标题")
    article.assets = [
        Asset(
            position=0,
            source_kind="web",
            source_page_url="https://www.alamy.com/stock-photo/example.html",
        )
    ]
    with pytest.raises(WechatError, match="带水印的图库预览图"):
        WechatClient(Settings()).create_draft(article)


def test_wechat_draft_media_id_is_verified(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"news_item": [{"title": "已上传标题"}]}

    monkeypatch.setattr(
        "app.services.wechat.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    client = WechatClient(
        Settings(
            wechat_enabled=True,
            wechat_app_id="app-id",
            wechat_app_secret="secret",
        )
    )
    client._token = "cached-token"
    client._token_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    client.verify_draft("media-id", "已上传标题")

    try:
        client.verify_draft("media-id", "另一个标题")
    except WechatError as exc:
        assert "title mismatch" in str(exc)
    else:
        raise AssertionError("Expected title mismatch to fail verification")


def test_psychology_generation_and_polishing_receive_citable_sources():
    prompt = build_emotional_article_prompt(
        topic="等不到回复时，她把聊天记录从头翻了一遍",
        category="恋爱心理",
        audience=audience_to_context(sample_audience()),
        title_count=5,
        image_count=3,
        emotional_title_reference=True,
    )
    polished = build_humanizer_prompt(
        {"opening": "她把消息删了又写。"},
        content_type="emotional_article",
        category="恋爱心理",
    )
    for text in (prompt, polished):
        assert "每篇必须至少有一处心理学家观点或研究佐证" in text
        assert "至少一处佐证须来自下方已核对来源库" in text
        assert "Geraldine Downey" in text
        assert "https://pubmed.ncbi.nlm.nih.gov/8667172/" in text
        assert "不把概述放进引号" in text
        assert "不输出论文链接、网址、DOI或参考文献清单" in text
        assert "25字以内" in text
    assert "移除论文链接、网址、DOI和独立参考文献清单" in polished
    topic_prompt = build_emotional_topic_prompt(
        directions="等待回复时的不安", category="恋爱心理",
        audience=audience_to_context(sample_audience()),
        recent_topics=[], local_date="2026-09-26",
    )
    assert "category 必须返回“恋爱心理”" in topic_prompt
    assert "rationale 简短写出作者和年份" in topic_prompt
    assert "Geraldine Downey" in topic_prompt
    assert "https://pubmed.ncbi.nlm.nih.gov/8667172/" not in topic_prompt
