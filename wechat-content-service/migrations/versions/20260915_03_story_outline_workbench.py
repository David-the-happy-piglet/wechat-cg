"""add reviewable story outline drafts

Revision ID: 20260915_03
Revises: 20260915_02
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260915_03"
down_revision: str | Sequence[str] | None = "20260915_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "story_outline_drafts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("audience_profile_id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("idea", sa.Text(), nullable=False),
        sa.Column("story_mode", sa.String(length=40), nullable=False),
        sa.Column("reader_boundary", sa.Text(), nullable=False),
        sa.Column("reference_text", sa.Text(), nullable=False),
        sa.Column("style_prompt", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("target_length", sa.Integer(), nullable=False),
        sa.Column("headline_intensity", sa.Integer(), nullable=False),
        sa.Column("title_count", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("polish", sa.Boolean(), nullable=False),
        sa.Column("outline_json", sa.Text(), nullable=False),
        sa.Column("review_notes", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"]),
        sa.ForeignKeyConstraint(["audience_profile_id"], ["audience_profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("article_id"),
    )
    op.create_index(
        op.f("ix_story_outline_drafts_audience_profile_id"),
        "story_outline_drafts",
        ["audience_profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_story_outline_drafts_status"),
        "story_outline_drafts",
        ["status"],
        unique=False,
    )
    with op.batch_alter_table("generation_runs") as batch_op:
        batch_op.add_column(
            sa.Column("story_outline_draft_id", sa.Integer(), nullable=True)
        )
        batch_op.create_index(
            "ix_generation_runs_story_outline_draft_id",
            ["story_outline_draft_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_generation_runs_story_outline_draft_id",
            "story_outline_drafts",
            ["story_outline_draft_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("generation_runs") as batch_op:
        batch_op.drop_constraint(
            "fk_generation_runs_story_outline_draft_id",
            type_="foreignkey",
        )
        batch_op.drop_index("ix_generation_runs_story_outline_draft_id")
        batch_op.drop_column("story_outline_draft_id")
    op.drop_index(
        op.f("ix_story_outline_drafts_status"),
        table_name="story_outline_drafts",
    )
    op.drop_index(
        op.f("ix_story_outline_drafts_audience_profile_id"),
        table_name="story_outline_drafts",
    )
    op.drop_table("story_outline_drafts")
