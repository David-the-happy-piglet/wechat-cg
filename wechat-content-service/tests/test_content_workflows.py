import pytest
from fastapi.testclient import TestClient

from app.content_workflows import (
    UnknownContentWorkflowError,
    content_workflow_metadata,
    get_content_workflow,
    list_content_workflows,
)
from app.main import app
from app.services.renderer import render_wechat_html


def test_content_workflow_registry_centralizes_metadata_and_strategies():
    workflows = list_content_workflows()
    assert [workflow.content_type for workflow in workflows] == [
        "health_article",
        "emotional_article",
        "micro_horror",
    ]

    health = get_content_workflow("health_article")
    story = get_content_workflow("micro_horror")
    emotional = get_content_workflow("emotional_article")
    assert health.display_name == "健康生活文章"
    assert health.default_audience_slug == "senior-wellness"
    assert health.default_category == "日常养生"
    assert health.prompt_versions == ("v2", "v2-humanizer-zh")
    assert health.renderer_name == "health"
    assert emotional.display_name == "情感文章"
    assert emotional.default_audience_slug == "relationship-readers"
    assert emotional.default_category == "恋爱心理"
    assert emotional.renderer_name == "emotional"
    assert story.display_name == "微恐短篇小说"
    assert story.default_audience_slug == "micro-horror-readers"
    assert story.default_category == "微恐短篇"
    assert [mode.value for mode in story.story_modes] == [
        "auto",
        "everyday_uncanny",
        "near_future",
        "folk_horror",
    ]
    assert story.generation_steps == (
        "story_outline",
        "story_draft",
        "story_review",
    )

    audience = {
        "name": "测试读者",
        "age_range": "18+",
        "interests": ["日常异样"],
        "prohibited_topics": ["血腥"],
    }
    health_prompt = health.build_topic_prompt(
        directions="生活习惯",
        category=health.default_category,
        audience=audience,
        recent_topics=[],
        local_date="2026-09-15",
        story_mode=health.default_story_mode,
    )
    story_prompt = story.build_topic_prompt(
        directions="住宅中的异常规则",
        category=story.default_category,
        audience=audience,
        recent_topics=[],
        local_date="2026-09-15",
        story_mode="everyday_uncanny",
    )
    emotional_prompt = emotional.build_topic_prompt(
        directions="分手修复与关系边界",
        category=emotional.default_category,
        audience={
            **audience,
            "description": "恋爱中的男女，以女性为主",
        },
        recent_topics=[],
        local_date="2026-09-15",
        story_mode=emotional.default_story_mode,
    )
    assert "低风险、实用、具体" in health_prompt
    assert "[EMOTIONAL_ARTICLE_TOPIC]" in emotional_prompt
    assert "[MICRO_HORROR_TOPIC]" in story_prompt


def test_unknown_content_workflow_fails_explicitly():
    with pytest.raises(UnknownContentWorkflowError, match="unknown-theme"):
        get_content_workflow("unknown-theme")
    with pytest.raises(UnknownContentWorkflowError, match="unknown-theme"):
        render_wechat_html({"content_type": "unknown-theme"})


def test_content_workflow_metadata_api_matches_registry():
    with TestClient(app) as client:
        response = client.get("/api/content-workflows")

    assert response.status_code == 200
    assert response.json() == content_workflow_metadata()
    story = next(
        item for item in response.json() if item["content_type"] == "micro_horror"
    )
    assert story["supports_story_modes"] is True
    assert story["renderer"] == "story"
    assert story["prompt_versions"] == [
        "micro-horror-v1-draft",
        "micro-horror-v1-reviewed",
    ]


@pytest.mark.parametrize(
    ("payload", "expected_version", "html_marker"),
    [
        (
            {
                "content_type": "emotional_article",
                "topic": "为什么关系里越懂事的人越容易内耗",
                "audience_slug": "relationship-readers",
                "provider": "mock",
                "image_count": 1,
                "humanize": True,
            },
            "emotional-v4-resonance",
            "background:#fffaf9",
        ),
        (
            {
                "content_type": "health_article",
                "topic": "如何安排晚饭后的轻度活动",
                "audience_slug": "middle-age-health",
                "provider": "mock",
                "image_count": 1,
                "humanize": False,
            },
            "v2",
            "background:#f4f8f5",
        ),
        (
            {
                "content_type": "micro_horror",
                "story_mode": "everyday_uncanny",
                "topic": "电梯每天替住户跳过一个不存在的楼层",
                "audience_slug": "micro-horror-readers",
                "provider": "mock",
                "image_count": 1,
                "humanize": True,
            },
            "micro-horror-v1-reviewed",
            "background:#fbfaf8",
        ),
    ],
)
def test_registered_workflows_keep_generation_behavior(
    payload,
    expected_version,
    html_marker,
):
    with TestClient(app) as client:
        response = client.post("/api/articles/generate", json=payload)

    assert response.status_code == 201
    article = response.json()
    assert article["content_type"] == payload["content_type"]
    assert article["prompt_version"] == expected_version
    assert html_marker in article["content_html"]
