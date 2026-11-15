from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .services.prompts import (
    build_emotional_topic_prompt,
    build_health_topic_prompt,
    build_story_topic_prompt,
)
from .services.renderer import (
    render_emotional_html,
    render_health_html,
    render_story_html,
)

DEFAULT_CONTENT_TYPE = "health_article"

TopicPromptBuilder = Callable[..., str]
Renderer = Callable[
    [dict[str, Any], list[str], list[dict[str, str | None]]],
    str,
]
GenerationHandler = Callable[..., tuple[dict[str, Any], list[dict[str, Any]], str]]


class UnknownContentWorkflowError(ValueError):
    """Raised when no workflow is registered for a content type."""


@dataclass(frozen=True)
class StoryModeDefinition:
    value: str
    label: str

    def to_metadata(self) -> dict[str, str]:
        return {"value": self.value, "label": self.label}


@dataclass(frozen=True)
class ContentWorkflow:
    content_type: str
    display_name: str
    short_name: str
    description: str
    default_audience_slug: str
    default_category: str
    default_story_mode: str
    fallback_story_mode: str
    story_modes: tuple[StoryModeDefinition, ...]
    supports_story_modes: bool
    topic_prompt_builder: TopicPromptBuilder
    topic_system_prompt: str
    prompt_versions: tuple[str, ...]
    base_prompt_version: str
    enhanced_prompt_version: str
    renderer_name: str
    renderer: Renderer
    generation_steps: tuple[str, ...]
    preview_steps: tuple[str, ...]
    ui: Mapping[str, Any]

    def build_topic_prompt(self, **kwargs: Any) -> str:
        return self.topic_prompt_builder(**kwargs)

    def generate(self, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        handler = _GENERATION_HANDLERS.get(self.content_type)
        if handler is None:
            raise RuntimeError(
                f"Content workflow {self.content_type!r} has no generation handler"
            )
        return handler(workflow=self, **kwargs)

    def render_article(
        self,
        article: dict[str, Any],
        asset_urls: list[str],
        asset_credits: list[dict[str, str | None]],
    ) -> str:
        return self.renderer(article, asset_urls, asset_credits)

    def story_mode_label(self, value: str) -> str:
        labels = {mode.value: mode.label for mode in self.story_modes}
        return labels.get(value, labels[self.default_story_mode])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "content_type": self.content_type,
            "display_name": self.display_name,
            "short_name": self.short_name,
            "description": self.description,
            "default_audience_slug": self.default_audience_slug,
            "default_category": self.default_category,
            "default_story_mode": self.default_story_mode,
            "fallback_story_mode": self.fallback_story_mode,
            "supports_story_modes": self.supports_story_modes,
            "story_modes": [mode.to_metadata() for mode in self.story_modes],
            "prompt_versions": list(self.prompt_versions),
            "renderer": self.renderer_name,
            "generation_steps": list(self.generation_steps),
            "ui": dict(self.ui),
        }


def _health_topic_prompt(**kwargs: Any) -> str:
    kwargs.pop("story_mode", None)
    return build_health_topic_prompt(**kwargs)


def _story_topic_prompt(**kwargs: Any) -> str:
    return build_story_topic_prompt(**kwargs)


def _emotional_topic_prompt(**kwargs: Any) -> str:
    kwargs.pop("story_mode", None)
    return build_emotional_topic_prompt(**kwargs)


_AUTO_MODE = StoryModeDefinition("auto", "自动选择")

_WORKFLOWS = {
    "health_article": ContentWorkflow(
        content_type="health_article",
        display_name="健康生活文章",
        short_name="健康文章",
        description="面向特定受众生成低风险、可执行的健康生活内容。",
        default_audience_slug="senior-wellness",
        default_category="日常养生",
        default_story_mode="auto",
        fallback_story_mode="auto",
        story_modes=(_AUTO_MODE,),
        supports_story_modes=False,
        topic_prompt_builder=_health_topic_prompt,
        topic_system_prompt="你是谨慎的微信公众号选题编辑。只输出合法 JSON，不要输出 Markdown。",
        prompt_versions=("v2", "v2-humanizer-zh"),
        base_prompt_version="v2",
        enhanced_prompt_version="v2-humanizer-zh",
        renderer_name="health",
        renderer=render_health_html,
        generation_steps=("article", "humanize"),
        preview_steps=(),
        ui={
            "source_modes": {
                "topic": ["文章主题", "例如：人到中年如何改善睡眠"],
                "text": ["粘贴原稿或参考文本", "粘贴希望改写的完整文本…"],
                "url": ["文章链接", "https://example.com/article"],
            },
            "source_help": "原材料只作为观点和事实依据，系统会重新组织结构与表达。",
            "headline_help": (
                "等级越高，越强调数字、反差、悬念和人群代入；始终禁止虚假健康承诺。"
            ),
            "humanize_label": "使用 Humanizer-zh 二次润色",
            "humanize_help": (
                "减少机械排比、AI 高频词和模板化结尾；会增加一次文字模型调用。"
            ),
        },
    ),
    "emotional_article": ContentWorkflow(
        content_type="emotional_article",
        display_name="情感文章",
        short_name="情感文章",
        description="以女性读者为主，侧重恋爱心理，从具体处境写共鸣，用有出处的观点帮助理解感受。",
        default_audience_slug="relationship-readers",
        default_category="恋爱心理",
        default_story_mode="auto",
        fallback_story_mode="auto",
        story_modes=(_AUTO_MODE,),
        supports_story_modes=False,
        topic_prompt_builder=_emotional_topic_prompt,
        topic_system_prompt="你是克制、尊重边界的情感内容选题编辑。只输出合法 JSON，不要输出 Markdown。",
        prompt_versions=(
            "emotional-v3", "emotional-v3-humanized",
            "emotional-v4-resonance", "emotional-v4-resonance-humanized",
            "dialogue-essay-v1", "dialogue-essay-v1-polished",
        ),
        base_prompt_version="emotional-v4-resonance",
        enhanced_prompt_version="emotional-v4-resonance-humanized",
        renderer_name="emotional",
        renderer=render_emotional_html,
        generation_steps=("emotional_article", "humanize"),
        preview_steps=(),
        ui={
            "source_modes": {
                "topic": ["情感主题", "例如：为什么关系里越懂事的人越容易内耗"],
                "text": ["参考文字", "粘贴希望分析或改写的情感内容…"],
                "url": ["参考文章链接", "https://example.com/article"],
            },
            "source_help": "参考材料只用于提炼观点和写法，不冒充原作者经历，也不逐句改写。",
            "headline_help": "标题可以有情绪张力，但不得制造羞耻、恐慌、性别对立或承诺操控结果。",
            "humanize_label": "使用 Humanizer-zh 二次润色",
            "humanize_help": "减少机械排比、空泛金句和说教感；会增加一次文字模型调用。",
        },
    ),
    "micro_horror": ContentWorkflow(
        content_type="micro_horror",
        display_name="微恐短篇小说",
        short_name="微恐短篇",
        description="先设计规则、因果和伏笔，再完成正文与公平性审稿。",
        default_audience_slug="micro-horror-readers",
        default_category="微恐短篇",
        default_story_mode="auto",
        fallback_story_mode="everyday_uncanny",
        story_modes=(
            _AUTO_MODE,
            StoryModeDefinition("everyday_uncanny", "日常异样"),
            StoryModeDefinition("near_future", "近未来寓言"),
            StoryModeDefinition("folk_horror", "民俗志异"),
        ),
        supports_story_modes=True,
        topic_prompt_builder=_story_topic_prompt,
        topic_system_prompt="你是微恐短篇选题编辑。只输出合法 JSON，不要输出 Markdown。",
        prompt_versions=("micro-horror-v1-draft", "micro-horror-v1-reviewed"),
        base_prompt_version="micro-horror-v1-draft",
        enhanced_prompt_version="micro-horror-v1-reviewed",
        renderer_name="story",
        renderer=render_story_html,
        generation_steps=("story_outline", "story_draft", "story_review"),
        preview_steps=("story_outline", "story_review"),
        ui={
            "source_modes": {
                "topic": [
                    "故事种子",
                    "例如：智能门锁每天替独居者拒绝一个不存在的访客",
                ],
                "text": [
                    "粘贴参考文本",
                    "粘贴参考故事；系统只分析节奏、伏笔与反转手法…",
                ],
                "url": ["参考文章链接", "https://example.com/story"],
            },
            "source_help": (
                "只借鉴叙事节奏、信息控制和不安机制，不复用人物、意象、情节组合或"
                "结局。正文在图片里的链接请改为粘贴文本。"
            ),
            "headline_help": (
                "等级越高，标题的信息缺口越强；标题、摘要和配图始终不能泄露反转。"
            ),
            "humanize_label": "执行伏笔与反转审稿修订",
            "humanize_help": (
                "检查因果、人物动机、规则一致性和视角公平，并去除机械表达；"
                "会增加一次文字模型调用。"
            ),
        },
    ),
}

_GENERATION_HANDLERS: dict[str, GenerationHandler] = {}


def get_content_workflow(content_type: str) -> ContentWorkflow:
    try:
        return _WORKFLOWS[content_type]
    except KeyError as exc:
        raise UnknownContentWorkflowError(
            f"Unknown content workflow: {content_type}"
        ) from exc


def list_content_workflows() -> tuple[ContentWorkflow, ...]:
    return tuple(_WORKFLOWS.values())


def content_workflow_metadata() -> list[dict[str, Any]]:
    return [workflow.to_metadata() for workflow in list_content_workflows()]


def register_generation_handler(
    content_type: str,
) -> Callable[[GenerationHandler], GenerationHandler]:
    get_content_workflow(content_type)

    def decorator(handler: GenerationHandler) -> GenerationHandler:
        _GENERATION_HANDLERS[content_type] = handler
        return handler

    return decorator


def normalize_default_category(content_type: str, category: str) -> str:
    workflow = get_content_workflow(content_type)
    default_category = get_content_workflow(DEFAULT_CONTENT_TYPE).default_category
    if category == default_category:
        return workflow.default_category
    return category
