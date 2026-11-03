from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_bridge_handles_partially_runtime_patched_database(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "partial-legacy.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(database_url)

    command.upgrade(config, "20260915_01")

    engine = sa.create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                """
                INSERT INTO audience_profiles (
                    id, slug, name, age_range, description, interests_json,
                    writing_style, title_preferences_json, article_length,
                    image_style, prohibited_topics_json, enabled, created_at,
                    updated_at
                ) VALUES (
                    1, 'legacy', 'Legacy', '18-60', '', '[]', '', '[]', 1200,
                    '', '[]', 1, '2026-09-15 00:00:00',
                    '2026-09-15 00:00:00'
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO articles (
                    id, audience_profile_id, topic, category, target_length,
                    title_candidates_json, selected_title, summary, content_json,
                    content_html, status, llm_provider, model_name, prompt_version,
                    safety_flags_json, created_at
                ) VALUES (
                    1, 1, 'legacy topic', 'legacy', NULL, '[]', 'Legacy title',
                    '', '{}', '<p>legacy</p>', 'generated', 'mock', '', 'v1',
                    '[]', '2026-09-15 00:00:00'
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO assets (
                    id, article_id, asset_type, position, prompt, source_kind,
                    created_at
                ) VALUES (
                    1, 1, 'inline', 0, 'legacy asset', NULL,
                    '2026-09-15 00:00:00'
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO video_projects (
                    id, source_type, status, current_step, progress,
                    target_duration_seconds, width, height, fps, voice,
                    subtitles_enabled, settings_json, script_json,
                    safety_flags_json, retry_count, cancel_requested, created_at,
                    updated_at
                ) VALUES (
                    1, 'topic', 'pending', 'queued', 0, 90, 1280, 720, 30,
                    'test', 1, '{}', '{}', '[]', 0, 0,
                    '2026-09-15 00:00:00', '2026-09-15 00:00:00'
                )
                """
            )
        )

        # Reproduce an interrupted/partial ensure_runtime_schema rollout.
        connection.execute(
            sa.text(
                "ALTER TABLE articles ADD COLUMN content_type VARCHAR(40) DEFAULT 'health_article'"
            )
        )
        connection.execute(
            sa.text("CREATE INDEX ix_articles_content_type ON articles (content_type)")
        )
        connection.execute(
            sa.text("ALTER TABLE automation_plans ADD COLUMN story_mode VARCHAR(40) DEFAULT 'auto'")
        )
        connection.execute(
            sa.text("ALTER TABLE video_projects ADD COLUMN research_json TEXT DEFAULT '{}'")
        )
        connection.execute(sa.text("UPDATE video_projects SET research_json = NULL WHERE id = 1"))
        connection.execute(
            sa.text("ALTER TABLE video_scenes ADD COLUMN image_source_page_url TEXT")
        )
    engine.dispose()

    command.upgrade(config, "head")
    command.check(config)

    engine = sa.create_engine(database_url)
    inspector = sa.inspect(engine)
    article_columns = {column["name"]: column for column in inspector.get_columns("articles")}
    asset_columns = {column["name"]: column for column in inspector.get_columns("assets")}
    video_columns = {column["name"]: column for column in inspector.get_columns("video_projects")}
    generation_run_columns = {
        column["name"]: column for column in inspector.get_columns("generation_runs")
    }

    assert article_columns["content_type"]["nullable"] is False
    assert article_columns["content_type"]["default"] is None
    assert article_columns["story_mode"]["nullable"] is False
    assert article_columns["target_length"]["nullable"] is False
    assert article_columns["target_length"]["default"] is None
    assert asset_columns["source_kind"]["nullable"] is False
    assert asset_columns["source_kind"]["default"] is None
    assert video_columns["research_json"]["nullable"] is False
    assert video_columns["research_json"]["default"] is None
    assert video_columns["review_json"]["nullable"] is False
    assert "story_outline_drafts" in inspector.get_table_names()
    assert generation_run_columns["story_outline_draft_id"]["nullable"] is True
    assert "ix_articles_content_type" in {
        index["name"] for index in inspector.get_indexes("articles")
    }
    assert "ix_automation_plans_content_type" in {
        index["name"] for index in inspector.get_indexes("automation_plans")
    }

    with engine.connect() as connection:
        article = connection.execute(
            sa.text("SELECT content_type, story_mode, target_length FROM articles WHERE id = 1")
        ).one()
        source_kind = connection.execute(
            sa.text("SELECT source_kind FROM assets WHERE id = 1")
        ).scalar_one()
        research = connection.execute(
            sa.text("SELECT research_json, review_json FROM video_projects WHERE id = 1")
        ).one()
        revision = connection.execute(
            sa.text("SELECT version_num FROM alembic_version")
        ).scalar_one()

    assert article == ("health_article", "auto", 1200)
    assert source_kind == "generated"
    assert research == ("{}", "{}")
    assert "manual_story_drafts" in inspector.get_table_names()
    assert "push_title" in {
        column["name"] for column in inspector.get_columns("manual_story_drafts")
    }
    assert revision == "20260927_07"
    assert "emotional_topic_reviews" in inspector.get_table_names()


def test_manual_story_title_migration_preserves_existing_title(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'manual-title.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(database_url)
    command.upgrade(config, "20260916_04")
    engine = sa.create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(sa.text("""
            INSERT INTO manual_story_drafts (
                idea, story_mode, folklore_element, reader_boundary, style_prompt,
                target_length, prompt_text, title, summary, body_text, status,
                created_at, updated_at
            ) VALUES (
                '旧灵感', 'auto', 'auto', '成年读者', '克制表达',
                1800, '旧 Prompt', '原有标题', '', '', 'prompt_ready',
                '2026-09-16 00:00:00', '2026-09-16 00:00:00'
            )
        """))
    engine.dispose()
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url)
    with engine.connect() as connection:
        assert connection.scalar(
            sa.text("SELECT push_title FROM manual_story_drafts LIMIT 1")
        ) == "原有标题"
    engine.dispose()
    engine.dispose()


def test_title_reference_migration_defaults_existing_plans_to_disabled(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'title-reference.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(database_url)
    command.upgrade(config, "20260916_05")
    engine = sa.create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(sa.text("""
            INSERT INTO automation_plans (
                name, content_type, story_mode, audience_profile_id, directions,
                category, provider, run_time, timezone, title_count, image_count,
                headline_intensity, humanize, auto_upload_draft, enabled,
                created_at, updated_at
            ) VALUES (
                '旧情感计划', 'emotional_article', 'auto', 1, '关系沟通',
                '情感综合', 'mock', '08:30', 'Asia/Shanghai', 5, 3,
                4, 1, 0, 0, '2026-09-16 00:00:00', '2026-09-16 00:00:00'
            )
        """))
    command.upgrade(config, "head")
    command.check(config)
    with engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT name, emotional_title_reference FROM automation_plans"
        )).one() == ("旧情感计划", False)
    command.downgrade(config, "20260916_05")
    assert "emotional_title_reference" not in {
        column["name"] for column in sa.inspect(engine).get_columns("automation_plans")
    }
    engine.dispose()
