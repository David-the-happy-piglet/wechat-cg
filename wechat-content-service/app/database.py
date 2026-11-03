from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def ensure_runtime_schema() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    required_by_table = {
        "generation_runs": {
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "total_tokens": "INTEGER",
        },
        "assets": {
            "source_kind": "VARCHAR(40) DEFAULT 'generated'",
            "source_page_url": "TEXT",
            "source_author": "TEXT",
            "license_name": "VARCHAR(120)",
            "license_url": "TEXT",
        },
        "articles": {
            "target_length": "INTEGER DEFAULT 1200",
            "content_type": "VARCHAR(40) DEFAULT 'health_article'",
            "story_mode": "VARCHAR(40) DEFAULT 'auto'",
        },
        "automation_plans": {
            "emotional_title_reference": "BOOLEAN NOT NULL DEFAULT FALSE",
            "target_length": "INTEGER",
            "content_type": "VARCHAR(40) DEFAULT 'health_article'",
            "story_mode": "VARCHAR(40) DEFAULT 'auto'",
        },
        "video_projects": {
            "actual_duration_ms": "INTEGER",
            "research_json": "TEXT DEFAULT '{}'",
            "review_json": "TEXT DEFAULT '{}'",
        },
        "video_scenes": {
            "overlay_labels_json": "TEXT DEFAULT '[]'",
            "image_source_page_url": "TEXT",
            "image_source_author": "TEXT",
            "image_license_name": "VARCHAR(120)",
            "image_license_url": "TEXT",
        },
    }
    with engine.begin() as connection:
        for table_name, required in required_by_table.items():
            if table_name not in tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for name, sql_type in required.items():
                if name in existing:
                    continue
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {sql_type}"))


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
