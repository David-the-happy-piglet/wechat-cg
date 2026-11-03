"""Add optional emotional title reference to automation plans.

Revision ID: 20260917_06
Revises: 20260916_05
"""

import sqlalchemy as sa
from alembic import op

revision = "20260917_06"
down_revision = "20260916_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = sa.inspect(op.get_bind()).get_columns("automation_plans")
    if "emotional_title_reference" not in {column["name"] for column in columns}:
        op.add_column(
            "automation_plans",
            sa.Column(
                "emotional_title_reference", sa.Boolean(),
                nullable=False, server_default=sa.false(),
            ),
        )
    with op.batch_alter_table("automation_plans") as batch_op:
        batch_op.alter_column("emotional_title_reference", server_default=None)


def downgrade() -> None:
    op.drop_column("automation_plans", "emotional_title_reference")
