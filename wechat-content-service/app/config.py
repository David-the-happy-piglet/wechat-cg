import os
from dataclasses import dataclass, replace


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./wechat-content.db",
    )
    auto_create_schema: bool = _as_bool(os.getenv("AUTO_CREATE_SCHEMA"), True)
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    llm_api_base: str = os.getenv("LLM_API_BASE", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    provider_max_attempts: int = int(os.getenv("PROVIDER_MAX_ATTEMPTS", "3"))
    provider_retry_base_seconds: float = float(
        os.getenv("PROVIDER_RETRY_BASE_SECONDS", "1")
    )

    openai_api_base: str = os.getenv(
        "OPENAI_API_BASE",
        "https://api.openai.com/v1",
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
    openai_terra_model: str = os.getenv("OPENAI_TERRA_MODEL", "gpt-5.6-terra")
    openai_luna_model: str = os.getenv("OPENAI_LUNA_MODEL", "gpt-5.6-luna")

    deepseek_api_base: str = os.getenv(
        "DEEPSEEK_API_BASE",
        "https://api.deepseek.com",
    )
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    minimax_api_base: str = os.getenv(
        "MINIMAX_API_BASE",
        "https://api.minimaxi.com/v1",
    )
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    minimax_model: str = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
    minimax_max_completion_tokens: int = int(
        os.getenv("MINIMAX_MAX_COMPLETION_TOKENS", "16384")
    )
    minimax_image_model: str = os.getenv("MINIMAX_IMAGE_MODEL", "image-01")
    minimax_video_model: str = os.getenv(
        "MINIMAX_VIDEO_MODEL", "MiniMax-Hailuo-2.3"
    )

    kimi_api_base: str = os.getenv(
        "KIMI_API_BASE",
        "https://api.moonshot.cn/v1",
    )
    moonshot_api_key: str = os.getenv("MOONSHOT_API_KEY", "")
    kimi_model: str = os.getenv("KIMI_MODEL", "kimi-k3")

    image_provider: str = os.getenv("IMAGE_PROVIDER", "placeholder")
    image_api_base: str = os.getenv("IMAGE_API_BASE", "")
    image_api_key: str = os.getenv("IMAGE_API_KEY", "")
    image_model: str = os.getenv("IMAGE_MODEL", "")
    image_size: str = os.getenv("IMAGE_SIZE", "1024x1024")
    image_timeout_seconds: float = float(os.getenv("IMAGE_TIMEOUT_SECONDS", "120"))
    aigc_watermark: bool = _as_bool(os.getenv("AIGC_WATERMARK"), True)
    image_source_mode: str = os.getenv("IMAGE_SOURCE_MODE", "ai")
    web_image_provider: str = os.getenv("WEB_IMAGE_PROVIDER", "openverse")
    web_image_fallback_to_ai: bool = _as_bool(
        os.getenv("WEB_IMAGE_FALLBACK_TO_AI"),
        True,
    )
    wikimedia_api_url: str = os.getenv(
        "WIKIMEDIA_API_URL",
        "https://commons.wikimedia.org/w/api.php",
    )
    openverse_api_url: str = os.getenv(
        "OPENVERSE_API_URL",
        "https://api.openverse.org/v1",
    )
    duckduckgo_image_base_url: str = os.getenv(
        "DUCKDUCKGO_IMAGE_BASE_URL",
        "https://duckduckgo.com",
    )
    web_image_user_agent: str = os.getenv(
        "WEB_IMAGE_USER_AGENT",
        os.getenv("WIKIMEDIA_USER_AGENT", "ContentGenerator/0.1"),
    )
    web_image_max_bytes: int = int(os.getenv("WEB_IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))
    generated_assets_dir: str = os.getenv(
        "GENERATED_ASSETS_DIR",
        "/app/generated-assets" if os.path.isdir("/app") else "./generated-assets",
    )

    video_worker_poll_seconds: int = int(os.getenv("VIDEO_WORKER_POLL_SECONDS", "5"))
    video_max_retries: int = int(os.getenv("VIDEO_MAX_RETRIES", "3"))
    video_stale_minutes: int = int(os.getenv("VIDEO_STALE_MINUTES", "30"))
    video_default_duration_seconds: int = int(
        os.getenv("VIDEO_DEFAULT_DURATION_SECONDS", "90")
    )
    video_default_width: int = int(os.getenv("VIDEO_DEFAULT_WIDTH", "1280"))
    video_default_height: int = int(os.getenv("VIDEO_DEFAULT_HEIGHT", "720"))
    video_default_fps: int = int(os.getenv("VIDEO_DEFAULT_FPS", "30"))
    video_max_source_bytes: int = int(
        os.getenv("VIDEO_MAX_SOURCE_BYTES", str(2 * 1024 * 1024))
    )
    video_source_timeout_seconds: float = float(
        os.getenv("VIDEO_SOURCE_TIMEOUT_SECONDS", "15")
    )
    video_ffmpeg_path: str = os.getenv("VIDEO_FFMPEG_PATH", "ffmpeg")
    video_ffprobe_path: str = os.getenv("VIDEO_FFPROBE_PATH", "ffprobe")
    video_intro_path: str = os.getenv("VIDEO_INTRO_PATH", "media/video/intro.mp4")
    video_outro_path: str = os.getenv("VIDEO_OUTRO_PATH", "media/video/outro.mp4")
    video_font_path: str = os.getenv(
        "VIDEO_FONT_PATH",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    video_bookend_reserve_seconds: int = int(
        os.getenv("VIDEO_BOOKEND_RESERVE_SECONDS", "12")
    )
    video_generation_poll_seconds: int = int(
        os.getenv("VIDEO_GENERATION_POLL_SECONDS", "10")
    )
    video_generation_timeout_seconds: int = int(
        os.getenv("VIDEO_GENERATION_TIMEOUT_SECONDS", "1200")
    )

    news_search_provider: str = os.getenv("NEWS_SEARCH_PROVIDER", "duckduckgo")
    news_search_timeout_seconds: float = float(
        os.getenv("NEWS_SEARCH_TIMEOUT_SECONDS", "20")
    )
    news_search_max_results: int = int(os.getenv("NEWS_SEARCH_MAX_RESULTS", "12"))
    news_max_documents: int = int(os.getenv("NEWS_MAX_DOCUMENTS", "8"))
    news_min_independent_sources: int = int(
        os.getenv("NEWS_MIN_INDEPENDENT_SOURCES", "2")
    )
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    news_mechanical_provider: str = os.getenv(
        "NEWS_MECHANICAL_PROVIDER", "minimax"
    )
    news_judgment_provider: str = os.getenv(
        "NEWS_JUDGMENT_PROVIDER", "openai-terra"
    )
    news_research_provider: str = os.getenv("NEWS_RESEARCH_PROVIDER", "openai")
    news_writer_provider: str = os.getenv("NEWS_WRITER_PROVIDER", "minimax")
    news_review_provider: str = os.getenv(
        "NEWS_REVIEW_PROVIDER", "openai-terra"
    )
    news_review_enabled: bool = _as_bool(os.getenv("NEWS_REVIEW_ENABLED"), False)

    speech_provider: str = os.getenv("SPEECH_PROVIDER", "mock")
    speech_model: str = os.getenv("SPEECH_MODEL", "speech-2.8-hd")
    speech_voice: str = os.getenv("SPEECH_VOICE", "male-qn-qingse")
    speech_speed: float = float(os.getenv("SPEECH_SPEED", "1.0"))
    speech_timeout_seconds: float = float(os.getenv("SPEECH_TIMEOUT_SECONDS", "120"))

    wechat_enabled: bool = _as_bool(os.getenv("WECHAT_ENABLED"), False)
    wechat_app_id: str = os.getenv("WECHAT_APP_ID", "")
    wechat_app_secret: str = os.getenv("WECHAT_APP_SECRET", "")
    wechat_author: str = os.getenv("WECHAT_AUTHOR", "")
    wechat_emotional_enabled: bool = _as_bool(os.getenv("WECHAT_EMOTIONAL_ENABLED"), False)
    wechat_emotional_app_id: str = os.getenv("WECHAT_EMOTIONAL_APP_ID", "")
    wechat_emotional_app_secret: str = os.getenv("WECHAT_EMOTIONAL_APP_SECRET", "")
    wechat_emotional_author: str = os.getenv("WECHAT_EMOTIONAL_AUTHOR", "")
    wechat_api_base: str = os.getenv(
        "WECHAT_API_BASE",
        "https://api.weixin.qq.com",
    )
    automation_poll_seconds: int = int(os.getenv("AUTOMATION_POLL_SECONDS", "30"))
    automation_max_retries: int = int(os.getenv("AUTOMATION_MAX_RETRIES", "3"))
    automation_stale_minutes: int = int(os.getenv("AUTOMATION_STALE_MINUTES", "120"))


def wechat_settings_for_content(settings: Settings, content_type: str) -> Settings:
    """The legacy WECHAT_* account remains unchanged for existing workflows."""
    if content_type == "emotional_article":
        return replace(
            settings,
            wechat_enabled=settings.wechat_emotional_enabled,
            wechat_app_id=settings.wechat_emotional_app_id,
            wechat_app_secret=settings.wechat_emotional_app_secret,
            wechat_author=settings.wechat_emotional_author,
        )
    return settings


settings = Settings()
