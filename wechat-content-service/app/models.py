from datetime import UTC, date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class AudienceProfile(Base):
    __tablename__ = "audience_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    age_range: Mapped[str] = mapped_column(String(40))
    description: Mapped[str] = mapped_column(Text, default="")
    interests_json: Mapped[str] = mapped_column(Text, default="[]")
    writing_style: Mapped[str] = mapped_column(Text, default="")
    title_preferences_json: Mapped[str] = mapped_column(Text, default="[]")
    article_length: Mapped[int] = mapped_column(Integer, default=1200)
    image_style: Mapped[str] = mapped_column(Text, default="")
    prohibited_topics_json: Mapped[str] = mapped_column(Text, default="[]")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    articles: Mapped[list["Article"]] = relationship(back_populates="audience")


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    content_type: Mapped[str] = mapped_column(
        String(40), default="health_article", index=True
    )
    story_mode: Mapped[str] = mapped_column(String(40), default="auto")
    audience_profile_id: Mapped[int] = mapped_column(
        ForeignKey("audience_profiles.id"),
        index=True,
    )
    topic: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(100), default="日常养生")
    target_length: Mapped[int] = mapped_column(Integer, default=1200)
    title_candidates_json: Mapped[str] = mapped_column(Text, default="[]")
    selected_title: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str] = mapped_column(Text, default="")
    content_json: Mapped[str] = mapped_column(Text)
    content_html: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default="generated", index=True)
    llm_provider: Mapped[str] = mapped_column(String(80), default="mock")
    model_name: Mapped[str] = mapped_column(String(120), default="")
    prompt_version: Mapped[str] = mapped_column(String(40), default="v1")
    safety_flags_json: Mapped[str] = mapped_column(Text, default="[]")
    wechat_media_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    wechat_publish_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    audience: Mapped[AudienceProfile] = relationship(back_populates="articles")
    assets: Mapped[list["Asset"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )
    generation_runs: Mapped[list["GenerationRun"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )
    metrics: Mapped[list["ArticleMetricDaily"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )


class StoryOutlineDraft(Base):
    __tablename__ = "story_outline_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    audience_profile_id: Mapped[int] = mapped_column(
        ForeignKey("audience_profiles.id"),
        index=True,
    )
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id"),
        nullable=True,
        unique=True,
    )
    status: Mapped[str] = mapped_column(
        String(40),
        default="outline_ready",
        index=True,
    )
    idea: Mapped[str] = mapped_column(Text, default="")
    story_mode: Mapped[str] = mapped_column(String(40), default="auto")
    reader_boundary: Mapped[str] = mapped_column(Text, default="")
    reference_text: Mapped[str] = mapped_column(Text, default="")
    style_prompt: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(100), default="微恐短篇")
    provider: Mapped[str] = mapped_column(String(40), default="openai-luna")
    target_length: Mapped[int] = mapped_column(Integer, default=1800)
    headline_intensity: Mapped[int] = mapped_column(Integer, default=4)
    title_count: Mapped[int] = mapped_column(Integer, default=5)
    image_count: Mapped[int] = mapped_column(Integer, default=3)
    polish: Mapped[bool] = mapped_column(Boolean, default=True)
    outline_json: Mapped[str] = mapped_column(Text)
    review_notes: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    audience: Mapped[AudienceProfile] = relationship()
    article: Mapped[Article | None] = relationship()
    generation_runs: Mapped[list["GenerationRun"]] = relationship(
        back_populates="story_outline_draft",
        cascade="all, delete-orphan",
        order_by="GenerationRun.id",
    )


class ManualStoryDraft(Base):
    __tablename__ = "manual_story_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id"), nullable=True, unique=True
    )
    idea: Mapped[str] = mapped_column(Text)
    story_mode: Mapped[str] = mapped_column(String(40), default="auto")
    folklore_element: Mapped[str] = mapped_column(String(80), default="auto")
    reader_boundary: Mapped[str] = mapped_column(Text)
    style_prompt: Mapped[str] = mapped_column(Text)
    target_length: Mapped[int] = mapped_column(Integer, default=1800)
    prompt_text: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(300), default="")
    push_title: Mapped[str] = mapped_column(String(300), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    body_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(40), default="prompt_ready", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    article: Mapped[Article | None] = relationship()


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    asset_type: Mapped[str] = mapped_column(String(40), default="inline")
    position: Mapped[int] = mapped_column(Integer, default=0)
    prompt: Mapped[str] = mapped_column(Text)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_kind: Mapped[str] = mapped_column(String(40), default="generated")
    source_page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_author: Mapped[str | None] = mapped_column(Text, nullable=True)
    license_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    license_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    wechat_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    wechat_media_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    article: Mapped[Article] = relationship(back_populates="assets")


class GenerationRun(Base):
    __tablename__ = "generation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id"),
        nullable=True,
        index=True,
    )
    story_outline_draft_id: Mapped[int | None] = mapped_column(
        ForeignKey("story_outline_drafts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    step: Mapped[str] = mapped_column(String(40))
    provider: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(120), default="")
    input_text: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    article: Mapped[Article | None] = relationship(back_populates="generation_runs")
    story_outline_draft: Mapped[StoryOutlineDraft | None] = relationship(
        back_populates="generation_runs"
    )


class PublishJob(Base):
    __tablename__ = "publish_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    job_type: Mapped[str] = mapped_column(String(30), default="draft")
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    wechat_media_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    wechat_publish_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )


class ArticleMetricDaily(Base):
    __tablename__ = "article_metrics_daily"
    __table_args__ = (UniqueConstraint("article_id", "metric_date", name="uq_article_metric_day"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    metric_date: Mapped[date] = mapped_column(Date)
    read_count: Mapped[int] = mapped_column(Integer, default=0)
    read_users: Mapped[int] = mapped_column(Integer, default=0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)
    share_users: Mapped[int] = mapped_column(Integer, default=0)
    raw_json: Mapped[str] = mapped_column(Text, default="{}")

    article: Mapped[Article] = relationship(back_populates="metrics")


class ContentStrategy(Base):
    __tablename__ = "content_strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    audience_profile_id: Mapped[int] = mapped_column(
        ForeignKey("audience_profiles.id"),
        index=True,
    )
    category: Mapped[str] = mapped_column(String(100))
    title_pattern: Mapped[str] = mapped_column(Text, default="")
    article_structure: Mapped[str] = mapped_column(Text, default="")
    image_style: Mapped[str] = mapped_column(Text, default="")
    target_length: Mapped[int] = mapped_column(Integer, default=1200)
    performance_score: Mapped[float] = mapped_column(Float, default=0.0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class AutomationPlan(Base):
    __tablename__ = "automation_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="每日内容计划")
    content_type: Mapped[str] = mapped_column(
        String(40), default="health_article", index=True
    )
    story_mode: Mapped[str] = mapped_column(String(40), default="auto")
    audience_profile_id: Mapped[int] = mapped_column(
        ForeignKey("audience_profiles.id"),
        index=True,
    )
    directions: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), default="日常养生")
    provider: Mapped[str] = mapped_column(String(40), default="deepseek")
    run_time: Mapped[str] = mapped_column(String(5), default="06:30")
    timezone: Mapped[str] = mapped_column(String(80), default="Asia/Shanghai")
    title_count: Mapped[int] = mapped_column(Integer, default=5)
    image_count: Mapped[int] = mapped_column(Integer, default=3)
    target_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    headline_intensity: Mapped[int] = mapped_column(Integer, default=4)
    emotional_title_reference: Mapped[bool] = mapped_column(Boolean, default=False)
    humanize: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_upload_draft: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    audience: Mapped[AudienceProfile] = relationship()
    runs: Mapped[list["AutomationRun"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
    )


class AutomationRun(Base):
    __tablename__ = "automation_runs"
    __table_args__ = (
        UniqueConstraint(
            "automation_plan_id",
            "scheduled_for",
            name="uq_automation_plan_schedule",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    automation_plan_id: Mapped[int] = mapped_column(
        ForeignKey("automation_plans.id"),
        index=True,
    )
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id"),
        nullable=True,
        index=True,
    )
    scheduled_for: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(40),
        default="pending",
        index=True,
    )
    topic: Mapped[str | None] = mapped_column(String(300), nullable=True)
    topic_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    plan: Mapped[AutomationPlan] = relationship(back_populates="runs")
    article: Mapped[Article | None] = relationship()


class VideoProject(Base):
    __tablename__ = "video_projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str] = mapped_column(String(20), index=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_topic: Mapped[str | None] = mapped_column(String(300), nullable=True)
    source_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    current_step: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    target_duration_seconds: Mapped[int] = mapped_column(Integer, default=90)
    width: Mapped[int] = mapped_column(Integer, default=1280)
    height: Mapped[int] = mapped_column(Integer, default=720)
    fps: Mapped[int] = mapped_column(Integer, default=30)
    voice: Mapped[str] = mapped_column(String(120), default="male-qn-qingse")
    llm_provider: Mapped[str | None] = mapped_column(String(80), nullable=True)
    subtitles_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    settings_json: Mapped[str] = mapped_column(Text, default="{}")
    research_json: Mapped[str] = mapped_column(Text, default="{}")
    script_json: Mapped[str] = mapped_column(Text, default="{}")
    review_json: Mapped[str] = mapped_column(Text, default="{}")
    safety_flags_json: Mapped[str] = mapped_column(Text, default="[]")
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    subtitle_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    scenes: Mapped[list["VideoScene"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="VideoScene.position",
    )
    generation_runs: Mapped[list["VideoGenerationRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class VideoScene(Base):
    __tablename__ = "video_scenes"
    __table_args__ = (
        UniqueConstraint("video_project_id", "position", name="uq_video_scene_position"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    video_project_id: Mapped[int] = mapped_column(
        ForeignKey("video_projects.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    narration: Mapped[str] = mapped_column(Text)
    subtitle: Mapped[str] = mapped_column(Text, default="")
    visual_intent: Mapped[str] = mapped_column(Text, default="")
    image_prompt: Mapped[str] = mapped_column(Text)
    overlay_labels_json: Mapped[str] = mapped_column(Text, default="[]")
    search_query: Mapped[str] = mapped_column(String(300), default="")
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_source_kind: Mapped[str | None] = mapped_column(String(40), nullable=True)
    image_source_page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_source_author: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_license_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    image_license_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    project: Mapped[VideoProject] = relationship(back_populates="scenes")
    generation_runs: Mapped[list["VideoGenerationRun"]] = relationship(
        back_populates="scene"
    )


class VideoGenerationRun(Base):
    __tablename__ = "video_generation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_project_id: Mapped[int] = mapped_column(
        ForeignKey("video_projects.id", ondelete="CASCADE"), index=True
    )
    video_scene_id: Mapped[int | None] = mapped_column(
        ForeignKey("video_scenes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    step: Mapped[str] = mapped_column(String(40), index=True)
    provider: Mapped[str] = mapped_column(String(80), default="local")
    model: Mapped[str] = mapped_column(String(120), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    input_text: Mapped[str] = mapped_column(Text, default="")
    output_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(40), default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    project: Mapped[VideoProject] = relationship(back_populates="generation_runs")
    scene: Mapped[VideoScene | None] = relationship(back_populates="generation_runs")


class EmotionalTopicReview(Base):
    __tablename__ = "emotional_topic_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    automation_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("automation_runs.id", ondelete="CASCADE"), unique=True, nullable=True,
    )
    generation_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("generation_runs.id", ondelete="SET NULL"), nullable=True,
    )
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), nullable=True,
    )
    topic: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(100))
    rationale: Mapped[str] = mapped_column(Text, default="")
    settings_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default="awaiting_review", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
