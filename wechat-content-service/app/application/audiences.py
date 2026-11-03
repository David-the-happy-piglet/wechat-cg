from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AudienceProfile


class AudienceNotAvailableError(LookupError):
    """Raised when an audience does not exist or is disabled."""


def require_enabled_audience(session: Session, slug: str) -> AudienceProfile:
    audience = session.scalar(
        select(AudienceProfile).where(
            AudienceProfile.slug == slug,
            AudienceProfile.enabled.is_(True),
        )
    )
    if audience is None:
        raise AudienceNotAvailableError("Audience not found or disabled")
    return audience
