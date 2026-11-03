from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .content_workflows import normalize_default_category
from .services.video.visual_profiles import DEFAULT_VISUAL_PROFILE
from .story_defaults import DEFAULT_READER_BOUNDARY

ProviderName = Literal[
    "mock",
    "compatible",
    "openai",
    "openai-terra",
    "openai-luna",
    "deepseek",
    "minimax",
    "kimi",
]
ContentType = Literal["health_article", "emotional_article", "micro_horror"]
StoryMode = Literal["auto", "everyday_uncanny", "near_future", "folk_horror"]


class AudienceBase(BaseModel):
    slug: str = Field(min_length=2, max_length=80, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=2, max_length=120)
    age_range: str = Field(min_length=2, max_length=40)
    description: str = ""
    interests: list[str] = Field(default_factory=list)
    writing_style: str = ""
    title_preferences: list[str] = Field(default_factory=list)
    article_length: int = Field(default=1200, ge=300, le=5000)
    image_style: str = ""
    prohibited_topics: list[str] = Field(default_factory=list)
    enabled: bool = True


class AudienceCreate(AudienceBase):
    pass


class AudienceUpdate(BaseModel):
    name: str | None = None
    age_range: str | None = None
    description: str | None = None
    interests: list[str] | None = None
    writing_style: str | None = None
    title_preferences: list[str] | None = None
    article_length: int | None = Field(default=None, ge=300, le=5000)
    image_style: str | None = None
    prohibited_topics: list[str] | None = None
    enabled: bool | None = None


class AudienceRead(AudienceBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class ArticleGenerateRequest(BaseModel):
    content_type: ContentType = "health_article"
    story_mode: StoryMode = "auto"
    topic: str = Field(default="", max_length=300)
    source_type: Literal["topic", "text", "url"] = "topic"
    source_text: str | None = Field(default=None, min_length=3, max_length=30000)
    source_url: str | None = Field(default=None, min_length=10, max_length=2048)
    source_prompt: str = Field(default="", max_length=4000)
    audience_slug: str = "senior-wellness"
    category: str = Field(default="日常养生", min_length=2, max_length=100)
    title_count: int = Field(default=5, ge=3, le=10)
    image_count: int = Field(default=3, ge=1, le=6)
    target_length: int | None = Field(default=None, ge=300, le=5000)
    headline_intensity: int = Field(default=4, ge=1, le=5)
    emotional_title_reference: bool = False
    humanize: bool = True
    provider: ProviderName | None = None
    reader_boundary: str = Field(default=DEFAULT_READER_BOUNDARY, max_length=2000)

    @model_validator(mode="after")
    def validate_source(self):
        self.category = normalize_default_category(self.content_type, self.category)
        if self.source_type == "topic" and len(self.topic.strip()) < 3:
            raise ValueError("Article topic must contain at least 3 characters")
        if self.source_type == "text" and not (self.source_text or "").strip():
            raise ValueError("Article text source cannot be empty")
        if self.source_type == "url":
            url = (self.source_url or "").strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("Article source URL must use http or https")
        return self


class StoryOutlineCreate(BaseModel):
    idea: str = Field(default="", max_length=1000)
    story_mode: StoryMode = "auto"
    reader_boundary: str = Field(min_length=10, max_length=2000)
    reference_text: str = Field(default="", max_length=12000)
    style_prompt: str = Field(default="", max_length=4000)
    category: str = Field(default="微恐短篇", min_length=2, max_length=100)
    provider: ProviderName | None = None
    target_length: int = Field(default=1800, ge=300, le=5000)
    headline_intensity: int = Field(default=4, ge=1, le=5)
    title_count: int = Field(default=5, ge=3, le=10)
    image_count: int = Field(default=3, ge=1, le=6)
    polish: bool = True

    @model_validator(mode="after")
    def normalize_text(self):
        self.idea = self.idea.strip()
        self.reader_boundary = self.reader_boundary.strip()
        self.reference_text = self.reference_text.strip()
        self.style_prompt = self.style_prompt.strip()
        return self


class ArticleComparisonRequest(BaseModel):
    content_type: ContentType = "health_article"
    story_mode: StoryMode = "auto"
    topic: str = Field(min_length=3, max_length=300)
    audience_slug: str = "senior-wellness"
    category: str = Field(default="日常养生", min_length=2, max_length=100)
    title_count: int = Field(default=5, ge=3, le=10)
    image_count: int = Field(default=3, ge=1, le=6)
    target_length: int | None = Field(default=None, ge=300, le=5000)
    headline_intensity: int = Field(default=4, ge=1, le=5)
    emotional_title_reference: bool = False
    humanize: bool = True
    providers: list[ProviderName] = Field(
        default_factory=lambda: ["deepseek", "minimax", "kimi"], min_length=1
    )

    @model_validator(mode="after")
    def normalize_category(self):
        self.category = normalize_default_category(self.content_type, self.category)
        return self


class AssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_type: str
    position: int
    prompt: str
    local_path: str | None
    source_url: str | None
    source_kind: str
    source_page_url: str | None
    source_author: str | None
    license_name: str | None
    license_url: str | None
    wechat_url: str | None


class ArticleRead(BaseModel):
    id: int
    content_type: ContentType
    story_mode: StoryMode
    audience_profile_id: int
    topic: str
    category: str
    target_length: int
    title_candidates: list[str]
    selected_title: str
    summary: str
    content: dict[str, Any]
    content_html: str
    status: str
    llm_provider: str
    model_name: str
    prompt_version: str
    safety_flags: list[str]
    wechat_media_id: str | None
    wechat_publish_id: str | None
    created_at: datetime
    published_at: datetime | None
    assets: list[AssetRead] = Field(default_factory=list)


class MetricCreate(BaseModel):
    metric_date: date
    read_count: int = Field(default=0, ge=0)
    read_users: int = Field(default=0, ge=0)
    share_count: int = Field(default=0, ge=0)
    share_users: int = Field(default=0, ge=0)
    raw: dict[str, Any] = Field(default_factory=dict)


class PublishResult(BaseModel):
    article_id: int
    status: str
    media_id: str | None = None
    publish_id: str | None = None
    message: str


class AutomationPlanBase(BaseModel):
    name: str = Field(default="每日内容计划", min_length=2, max_length=120)
    content_type: ContentType = "health_article"
    story_mode: StoryMode = "auto"
    audience_slug: str = "senior-wellness"
    directions: str = Field(min_length=2, max_length=5000)
    category: str = Field(default="日常养生", min_length=2, max_length=100)
    provider: ProviderName = "deepseek"
    run_time: str = Field(default="06:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(default="Asia/Shanghai", min_length=3, max_length=80)
    title_count: int = Field(default=5, ge=3, le=10)
    image_count: int = Field(default=3, ge=1, le=6)
    target_length: int | None = Field(default=None, ge=300, le=5000)
    headline_intensity: int = Field(default=4, ge=1, le=5)
    emotional_title_reference: bool = False
    humanize: bool = True
    auto_upload_draft: bool = True
    enabled: bool = True

    @model_validator(mode="after")
    def normalize_content_defaults(self):
        self.category = normalize_default_category(self.content_type, self.category)
        return self


class AutomationPlanCreate(AutomationPlanBase):
    pass


class AutomationPlanUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    content_type: ContentType | None = None
    story_mode: StoryMode | None = None
    audience_slug: str | None = None
    directions: str | None = Field(default=None, min_length=2, max_length=5000)
    category: str | None = Field(default=None, min_length=2, max_length=100)
    provider: ProviderName | None = None
    run_time: str | None = Field(
        default=None,
        pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
    )
    timezone: str | None = Field(default=None, min_length=3, max_length=80)
    title_count: int | None = Field(default=None, ge=3, le=10)
    image_count: int | None = Field(default=None, ge=1, le=6)
    target_length: int | None = Field(default=None, ge=300, le=5000)
    headline_intensity: int | None = Field(default=None, ge=1, le=5)
    emotional_title_reference: bool | None = None
    humanize: bool | None = None
    auto_upload_draft: bool | None = None
    enabled: bool | None = None


class AutomationPlanRead(AutomationPlanBase):
    id: int
    audience_profile_id: int
    created_at: datetime
    updated_at: datetime


class AutomationRunRead(BaseModel):
    id: int
    automation_plan_id: int
    article_id: int | None
    scheduled_for: datetime
    status: str
    topic: str | None
    topic_rationale: str | None
    error_message: str | None
    retry_count: int
    next_attempt_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class VideoTopicSource(BaseModel):
    type: Literal["topic"]
    topic: str = Field(min_length=3, max_length=300)


class VideoURLSource(BaseModel):
    type: Literal["url"]
    url: str = Field(min_length=10, max_length=2048)


class VideoTextSource(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=3, max_length=30000)


VideoSource = Annotated[
    VideoTopicSource | VideoURLSource | VideoTextSource,
    Field(discriminator="type"),
]


class NewsModelRoutes(BaseModel):
    mechanical: ProviderName = "minimax"
    judgment: ProviderName = "openai-terra"
    research: ProviderName = "openai"
    writer: ProviderName = "minimax"
    review: ProviderName = "openai-terra"


class VideoProjectCreate(BaseModel):
    source: VideoSource
    workflow: Literal["explainer", "news_research"] = "explainer"
    news_review_enabled: bool = False
    news_angle: Literal[
        "auto",
        "straight_news",
        "curiosity",
        "drama",
        "internet_reaction",
        "timeline",
        "explainer",
        "follow_up",
        "nostalgia",
        "rumor_investigation",
    ] = "auto"
    news_model_routes: NewsModelRoutes = Field(default_factory=NewsModelRoutes)
    target_duration_seconds: int = Field(default=90, ge=15, le=600)
    resolution: Literal["720p", "1080p"] = "720p"
    fps: int = Field(default=30, ge=24, le=60)
    voice: str | None = Field(default=None, min_length=2, max_length=120)
    provider: ProviderName | None = None
    subtitles: bool = True
    bookends: bool = False
    use_web_images: bool = False
    web_image_ratio: float = Field(default=0.95, ge=0.9, le=1.0)
    visual_profile: str = Field(
        default=DEFAULT_VISUAL_PROFILE,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    script_prompt: str = Field(default="", max_length=4000)
    style: str = Field(default="", max_length=4000)
    scene_duration_seconds: float = Field(default=3.5, ge=3.0, le=8.0)
    speech_speed: float = Field(default=1.2, ge=0.5, le=2.0)
    scene_gap_ms: int = Field(default=120, ge=0, le=600)
    trim_speech_silence: bool = True

    @model_validator(mode="after")
    def validate_source_url(self):
        if isinstance(self.source, VideoURLSource) and not self.source.url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("Video source URL must use http or https")
        return self


class VideoSceneRead(BaseModel):
    id: int
    position: int
    narration: str
    subtitle: str
    visual_intent: str
    image_prompt: str
    overlay_labels: list[dict[str, str | float]] = Field(default_factory=list)
    search_query: str
    image_path: str | None
    image_source_url: str | None
    image_source_kind: str | None
    image_source_page_url: str | None
    image_source_author: str | None
    image_license_name: str | None
    image_license_url: str | None
    audio_path: str | None
    duration_ms: int | None
    status: str
    error_message: str | None


class VideoProjectRead(BaseModel):
    id: int
    source_type: str
    source_url: str | None
    source_topic: str | None
    source_title: str | None
    source_text: str | None
    title: str | None
    status: str
    current_step: str
    progress: int
    target_duration_seconds: int
    width: int
    height: int
    fps: int
    voice: str
    llm_provider: str | None
    workflow: str
    news_angle: str
    news_review_enabled: bool
    news_model_routes: dict[str, str]
    research: dict[str, Any]
    review: dict[str, Any]
    visual_profile: str
    script_prompt: str
    style: str
    subtitles_enabled: bool
    bookends: bool
    use_web_images: bool
    web_image_ratio: float
    actual_web_image_ratio: float | None
    resolution: str
    scene_duration_seconds: float
    speech_speed: float
    scene_gap_ms: int
    trim_speech_silence: bool
    safety_flags: list[str]
    output_url: str | None
    subtitle_url: str | None
    actual_duration_ms: int | None
    error_message: str | None
    retry_count: int
    cancel_requested: bool
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    scenes: list[VideoSceneRead] = Field(default_factory=list)
