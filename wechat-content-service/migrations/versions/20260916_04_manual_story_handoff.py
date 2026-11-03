"""Persist reviewed prompts and manually imported stories.

Revision ID: 20260916_04
Revises: 20260915_03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260916_04"
down_revision: str | Sequence[str] | None = "20260915_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "manual_story_drafts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("article_id", sa.Integer(), sa.ForeignKey("articles.id"), unique=True),
        sa.Column("idea", sa.Text(), nullable=False),
        sa.Column("story_mode", sa.String(40), nullable=False),
        sa.Column("folklore_element", sa.String(80), nullable=False),
        sa.Column("reader_boundary", sa.Text(), nullable=False),
        sa.Column("style_prompt", sa.Text(), nullable=False),
        sa.Column("target_length", sa.Integer(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_manual_story_drafts_status", "manual_story_drafts", ["status"])


def downgrade() -> None:
    op.drop_index("ix_manual_story_drafts_status", table_name="manual_story_drafts")
    op.drop_table("manual_story_drafts")
