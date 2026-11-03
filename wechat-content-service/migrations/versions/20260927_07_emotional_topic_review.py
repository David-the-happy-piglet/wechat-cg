"""Persist emotional topics for human review before article generation."""

import sqlalchemy as sa
from alembic import op

revision = "20260927_07"
down_revision = "20260917_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "emotional_topic_reviews" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "emotional_topic_reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "automation_run_id",
            sa.Integer(),
            sa.ForeignKey("automation_runs.id", ondelete="CASCADE"),
            unique=True,
            nullable=True,
        ),
        sa.Column(
            "generation_run_id",
            sa.Integer(),
            sa.ForeignKey("generation_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "article_id",
            sa.Integer(),
            sa.ForeignKey("articles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("topic", sa.String(300), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("settings_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_emotional_topic_reviews_status", "emotional_topic_reviews", ["status"])


def downgrade() -> None:
    op.drop_table("emotional_topic_reviews")
