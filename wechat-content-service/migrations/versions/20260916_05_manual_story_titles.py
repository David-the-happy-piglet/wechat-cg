"""Store separate body and push titles for manually authored stories.

Revision ID: 20260916_05
Revises: 20260916_04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260916_05"
down_revision: str | Sequence[str] | None = "20260916_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "manual_story_drafts",
        sa.Column("push_title", sa.String(300), nullable=False, server_default=""),
    )
    op.execute("UPDATE manual_story_drafts SET push_title = title WHERE push_title = ''")
    with op.batch_alter_table("manual_story_drafts") as batch_op:
        batch_op.alter_column("push_title", server_default=None)


def downgrade() -> None:
    op.drop_column("manual_story_drafts", "push_title")
