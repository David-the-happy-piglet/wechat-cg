from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import AutomationPlan
from ..schemas import AutomationPlanCreate, AutomationPlanUpdate
from .audiences import require_enabled_audience


class AutomationPlanNotFoundError(LookupError):
    """Raised when the requested automation plan does not exist."""


class InvalidTimezoneError(ValueError):
    """Raised when a plan references an unknown IANA timezone."""


def _validate_timezone(name: str) -> None:
    try:
        ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise InvalidTimezoneError(f"Unknown timezone: {name}") from exc


def _commit_plan(session: Session, plan: AutomationPlan) -> AutomationPlan:
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(plan)
    return plan


def create_automation_plan(
    session: Session,
    data: AutomationPlanCreate,
) -> AutomationPlan:
    _validate_timezone(data.timezone)
    audience = require_enabled_audience(session, data.audience_slug)
    values = data.model_dump(exclude={"audience_slug"})
    plan = AutomationPlan(audience_profile_id=audience.id, **values)
    plan.audience = audience
    session.add(plan)
    return _commit_plan(session, plan)


def update_automation_plan(
    session: Session,
    plan_id: int,
    data: AutomationPlanUpdate,
) -> AutomationPlan:
    plan = session.scalar(
        select(AutomationPlan)
        .where(AutomationPlan.id == plan_id)
        .options(selectinload(AutomationPlan.audience))
    )
    if plan is None:
        raise AutomationPlanNotFoundError("Automation plan not found")

    values = data.model_dump(exclude_unset=True)
    audience_slug = values.pop("audience_slug", None)
    if "timezone" in values:
        _validate_timezone(values["timezone"])
    audience = (
        require_enabled_audience(session, audience_slug)
        if audience_slug is not None
        else None
    )

    if audience is not None:
        plan.audience_profile_id = audience.id
        plan.audience = audience
    for key, value in values.items():
        setattr(plan, key, value)
    return _commit_plan(session, plan)
