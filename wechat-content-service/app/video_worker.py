import logging
import time

from .config import settings
from .database import Base, SessionLocal, engine
from .services.video.pipeline import (
    process_next_video_project,
    recover_stale_video_projects,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("video-worker")


def worker_tick() -> bool:
    with SessionLocal() as session:
        recovered = recover_stale_video_projects(session, settings)
        if recovered:
            logger.warning("recovered %s stale video projects", recovered)
        project = process_next_video_project(session, settings)
        if not project:
            return False
        logger.info(
            "video project %s finished tick with status=%s step=%s",
            project.id,
            project.status,
            project.current_step,
        )
        return True


def main() -> None:
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        recovered = recover_stale_video_projects(session, settings)
        if recovered:
            logger.warning("recovered %s stale video projects", recovered)
    logger.info(
        "video worker started; poll interval=%ss, max retries=%s",
        settings.video_worker_poll_seconds,
        settings.video_max_retries,
    )
    while True:
        try:
            processed = worker_tick()
        except Exception:
            logger.exception("video worker tick failed")
            processed = False
        if not processed:
            time.sleep(max(1, settings.video_worker_poll_seconds))


if __name__ == "__main__":
    main()
