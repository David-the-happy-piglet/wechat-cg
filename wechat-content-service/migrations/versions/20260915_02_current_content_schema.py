"""bridge legacy create_all databases to the current content schema

Revision ID: 20260915_02
Revises: 20260915_01
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "20260915_02"
down_revision: str | Sequence[str] | None = "20260915_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(connection, table_name: str) -> dict[str, dict]:
    if context.is_offline_mode():
        return {}
    return {column["name"]: column for column in sa.inspect(connection).get_columns(table_name)}


def _index_names(connection, table_name: str) -> set[str]:
    if context.is_offline_mode():
        return set()
    return {
        index["name"]
        for index in sa.inspect(connection).get_indexes(table_name)
        if index.get("name")
    }


def _add_missing_columns(
    table_name: str,
    existing: dict[str, dict],
    columns: list[sa.Column],
) -> None:
    missing = [column for column in columns if column.name not in existing]
    if not missing:
        return
    with op.batch_alter_table(table_name) as batch_op:
        for column in missing:
            batch_op.add_column(column)


def _needs_normalization(columns: dict[str, dict], column_name: str) -> bool:
    if context.is_offline_mode():
        return True
    column = columns[column_name]
    return bool(column["nullable"]) or column.get("default") is not None


def upgrade() -> None:
    connection = op.get_bind()

    _add_missing_columns(
        "articles",
        _columns(connection, "articles"),
        [
            sa.Column("content_type", sa.String(length=40), nullable=True),
            sa.Column("story_mode", sa.String(length=40), nullable=True),
        ],
    )
    _add_missing_columns(
        "automation_plans",
        _columns(connection, "automation_plans"),
        [
            sa.Column("content_type", sa.String(length=40), nullable=True),
            sa.Column("story_mode", sa.String(length=40), nullable=True),
        ],
    )
    _add_missing_columns(
        "video_projects",
        _columns(connection, "video_projects"),
        [
            sa.Column("research_json", sa.Text(), nullable=True),
            sa.Column("review_json", sa.Text(), nullable=True),
        ],
    )
    _add_missing_columns(
        "video_scenes",
        _columns(connection, "video_scenes"),
        [
            sa.Column("image_source_page_url", sa.Text(), nullable=True),
            sa.Column("image_source_author", sa.Text(), nullable=True),
            sa.Column("image_license_name", sa.String(length=120), nullable=True),
            sa.Column("image_license_url", sa.Text(), nullable=True),
        ],
    )

    articles = sa.table(
        "articles",
        sa.column("content_type", sa.String(length=40)),
        sa.column("story_mode", sa.String(length=40)),
        sa.column("target_length", sa.Integer()),
    )
    automation_plans = sa.table(
        "automation_plans",
        sa.column("content_type", sa.String(length=40)),
        sa.column("story_mode", sa.String(length=40)),
    )
    assets = sa.table(
        "assets",
        sa.column("source_kind", sa.String(length=40)),
    )
    video_projects = sa.table(
        "video_projects",
        sa.column("research_json", sa.Text()),
        sa.column("review_json", sa.Text()),
    )
    connection.execute(
        sa.update(articles)
        .where(articles.c.content_type.is_(None))
        .values(content_type="health_article")
    )
    connection.execute(
        sa.update(articles).where(articles.c.story_mode.is_(None)).values(story_mode="auto")
    )
    connection.execute(
        sa.update(articles).where(articles.c.target_length.is_(None)).values(target_length=1200)
    )
    connection.execute(
        sa.update(automation_plans)
        .where(automation_plans.c.content_type.is_(None))
        .values(content_type="health_article")
    )
    connection.execute(
        sa.update(automation_plans)
        .where(automation_plans.c.story_mode.is_(None))
        .values(story_mode="auto")
    )
    connection.execute(
        sa.update(assets).where(assets.c.source_kind.is_(None)).values(source_kind="generated")
    )
    connection.execute(
        sa.update(video_projects)
        .where(video_projects.c.research_json.is_(None))
        .values(research_json="{}")
    )
    connection.execute(
        sa.update(video_projects)
        .where(video_projects.c.review_json.is_(None))
        .values(review_json="{}")
    )

    article_columns = _columns(connection, "articles")
    article_indexes = _index_names(connection, "articles")
    normalize_article_content_type = _needs_normalization(article_columns, "content_type")
    normalize_article_story_mode = _needs_normalization(article_columns, "story_mode")
    normalize_target_length = _needs_normalization(article_columns, "target_length")
    add_article_index = "ix_articles_content_type" not in article_indexes
    if any(
        (
            normalize_article_content_type,
            normalize_article_story_mode,
            normalize_target_length,
            add_article_index,
        )
    ):
        with op.batch_alter_table("articles") as batch_op:
            if normalize_article_content_type:
                batch_op.alter_column(
                    "content_type",
                    existing_type=sa.String(length=40),
                    server_default=None,
                    nullable=False,
                )
            if normalize_article_story_mode:
                batch_op.alter_column(
                    "story_mode",
                    existing_type=sa.String(length=40),
                    server_default=None,
                    nullable=False,
                )
            if normalize_target_length:
                batch_op.alter_column(
                    "target_length",
                    existing_type=sa.Integer(),
                    server_default=None,
                    nullable=False,
                )
            if add_article_index:
                batch_op.create_index(
                    "ix_articles_content_type",
                    ["content_type"],
                    unique=False,
                )

    asset_columns = _columns(connection, "assets")
    if _needs_normalization(asset_columns, "source_kind"):
        with op.batch_alter_table("assets") as batch_op:
            batch_op.alter_column(
                "source_kind",
                existing_type=sa.String(length=40),
                server_default=None,
                nullable=False,
            )

    automation_columns = _columns(connection, "automation_plans")
    automation_indexes = _index_names(connection, "automation_plans")
    normalize_plan_content_type = _needs_normalization(automation_columns, "content_type")
    normalize_plan_story_mode = _needs_normalization(automation_columns, "story_mode")
    add_plan_index = "ix_automation_plans_content_type" not in automation_indexes
    if any((normalize_plan_content_type, normalize_plan_story_mode, add_plan_index)):
        with op.batch_alter_table("automation_plans") as batch_op:
            if normalize_plan_content_type:
                batch_op.alter_column(
                    "content_type",
                    existing_type=sa.String(length=40),
                    server_default=None,
                    nullable=False,
                )
            if normalize_plan_story_mode:
                batch_op.alter_column(
                    "story_mode",
                    existing_type=sa.String(length=40),
                    server_default=None,
                    nullable=False,
                )
            if add_plan_index:
                batch_op.create_index(
                    "ix_automation_plans_content_type",
                    ["content_type"],
                    unique=False,
                )

    video_project_columns = _columns(connection, "video_projects")
    normalize_research_json = _needs_normalization(video_project_columns, "research_json")
    normalize_review_json = _needs_normalization(video_project_columns, "review_json")
    if normalize_research_json or normalize_review_json:
        with op.batch_alter_table("video_projects") as batch_op:
            if normalize_research_json:
                batch_op.alter_column(
                    "research_json",
                    existing_type=sa.Text(),
                    server_default=None,
                    nullable=False,
                )
            if normalize_review_json:
                batch_op.alter_column(
                    "review_json",
                    existing_type=sa.Text(),
                    server_default=None,
                    nullable=False,
                )


def downgrade() -> None:
    with op.batch_alter_table("video_scenes") as batch_op:
        batch_op.drop_column("image_license_url")
        batch_op.drop_column("image_license_name")
        batch_op.drop_column("image_source_author")
        batch_op.drop_column("image_source_page_url")

    with op.batch_alter_table("video_projects") as batch_op:
        batch_op.drop_column("review_json")
        batch_op.drop_column("research_json")

    with op.batch_alter_table("automation_plans") as batch_op:
        batch_op.drop_index("ix_automation_plans_content_type")
        batch_op.drop_column("story_mode")
        batch_op.drop_column("content_type")

    with op.batch_alter_table("assets") as batch_op:
        batch_op.alter_column(
            "source_kind",
            existing_type=sa.String(length=40),
            server_default=sa.text("'generated'"),
            nullable=True,
        )

    with op.batch_alter_table("articles") as batch_op:
        batch_op.drop_index("ix_articles_content_type")
        batch_op.drop_column("story_mode")
        batch_op.drop_column("content_type")
        batch_op.alter_column(
            "target_length",
            existing_type=sa.Integer(),
            server_default=sa.text("1200"),
            nullable=True,
        )
