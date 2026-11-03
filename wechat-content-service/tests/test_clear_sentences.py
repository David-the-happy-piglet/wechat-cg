import json
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from app.content_workflows import get_content_workflow
from app.database import SessionLocal
from app.emotional_defaults import CLEAR_SENTENCE_CATEGORY
from app.main import app
from app.models import EmotionalTopicReview
from app.schemas import ArticleGenerateRequest
from app.services.automation import _resolved_article_category
from app.services.generator import _generate_emotional_article
from app.services.humanizer import build_humanizer_prompt
from app.services.llm import MockLLMProvider
from app.services.prompts import build_emotional_article_prompt, build_emotional_topic_prompt
from app.services.renderer import render_emotional_html


def test_short_style_has_no_conflicting_scene_instructions():
    audience = {"description": "恋爱中的读者", "article_length": 800, "age_range": "20-40", "image_style": "无水印"}
    prompts = [
        build_emotional_topic_prompt(directions="恋爱关系", category=CLEAR_SENTENCE_CATEGORY,
                                     audience=audience, recent_topics=[], local_date="2026-09-29"),
        build_emotional_article_prompt(topic="舍不得已付出的时间", category=CLEAR_SENTENCE_CATEGORY,
                                       audience=audience, title_count=5, image_count=3,
                                       emotional_title_reference=True),
        build_humanizer_prompt({}, content_type="emotional_article", category=CLEAR_SENTENCE_CATEGORY,
                               emotional_title_reference=True),
    ]
    for prompt in prompts:
        assert "不描写具体场景" in prompt or "不设计具体场景" in prompt
        assert "不超过50字" in prompt
        assert "每个感受都有场景支撑" not in prompt
        assert "每个感受都有动作或场景支撑" not in prompt
        assert "只写一个小场景" not in prompt
        assert "结尾停在一句话或画面上" not in prompt
    assert _resolved_article_category("emotional_article", CLEAR_SENTENCE_CATEGORY, "恋爱心理") == CLEAR_SENTENCE_CATEGORY


@pytest.mark.parametrize("polish", ["off", "success", "failure"])
def test_generated_short_article_keeps_text_order_and_50_character_limit(polish):
    audience = {"description": "恋爱中的读者", "article_length": 800, "age_range": "20-40", "image_style": "无水印"}
    context = {"topic": "舍不得已付出的时间", "category": CLEAR_SENTENCE_CATEGORY,
               "audience": audience, "title_count": 5, "image_count": 3,
               "content_type": "emotional_article"}
    provider = MockLLMProvider()
    draft = provider.generate_article("[EMOTIONAL_ARTICLE]", context)
    draft["opening"] = "已经付出的时间不能决定还要付出多少。" * 5
    draft["sections"] = [{"heading": "", "paragraphs": ["放弃解释不等于承认自己不值得。" * 7, "界" * 101]}]
    draft["conclusion"] = "舍不得是感受，继续付出是选择。" * 6
    provider.name = "test"
    calls = []

    def generate(prompt, ctx):
        calls.append(prompt)
        if len(calls) == 2 and polish == "failure":
            raise RuntimeError("polish unavailable")
        return deepcopy(draft)

    provider.generate_article = generate
    request = ArticleGenerateRequest(**{k: context[k] for k in ["topic", "category", "content_type"]},
                                     humanize=polish != "off")
    article, records, _ = _generate_emotional_article(
        workflow=get_content_workflow("emotional_article"), session=None, llm=provider,
        request=request, effective_topic=context["topic"], audience_context=audience, source_material="",
    )
    paragraphs = [article["opening"], *[p for s in article["sections"] for p in s["paragraphs"]], article["conclusion"]]
    original = draft["opening"] + "".join(draft["sections"][0]["paragraphs"]) + draft["conclusion"]
    assert "".join(paragraphs) == original
    assert all(0 < len(p) <= 50 for p in paragraphs)
    assert len(calls) == (1 if polish == "off" else 2)
    if polish == "failure":
        assert records[-1]["status"] == "failed"

    class ParagraphParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paragraphs = []
            self.active = False
        def handle_starttag(self, tag, attrs):
            if tag == "p":
                self.active = True
                self.paragraphs.append("")
        def handle_endtag(self, tag):
            if tag == "p":
                self.active = False
        def handle_data(self, text):
            if self.active:
                self.paragraphs[-1] += text

    parser = ParagraphParser()
    parser.feed(render_emotional_html(article, [], []))
    assert parser.paragraphs == paragraphs


def test_manual_short_style_is_selectable_and_preserved_for_review():
    with TestClient(app) as client:
        page = client.get("/admin/emotions")
        assert '<option value="清醒短句"' in page.text
        response = client.post("/admin/emotions/generate", data={
            "category": CLEAR_SENTENCE_CATEGORY, "idea": "舍不得已付出的时间", "provider": "mock",
        }, follow_redirects=False)
        location = response.headers["location"]
        assert location.startswith("/admin/emotions/topics/")
        review_page = client.get(location)
        assert "不必设计场景" in review_page.text
        with SessionLocal() as session:
            review = session.get(EmotionalTopicReview, int(location.rsplit("/", 1)[1]))
            assert review.status == "awaiting_review" and review.article_id is None
            assert json.loads(review.settings_json)["request"]["category"] == CLEAR_SENTENCE_CATEGORY
