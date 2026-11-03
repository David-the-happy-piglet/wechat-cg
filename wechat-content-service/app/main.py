from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text

from .admin import router as admin_router
from .api import router as api_router
from .config import settings
from .database import Base, SessionLocal, engine, ensure_runtime_schema
from .models import Article
from .repositories import seed_audiences
from .video_api import router as video_api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=engine)
        ensure_runtime_schema()
    with SessionLocal() as session:
        seed_audiences(session)
    yield


app = FastAPI(
    title="WeChat Content Generator",
    version="0.1.0",
    lifespan=lifespan,
)

assets_dir = Path(settings.generated_assets_dir)
assets_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/generated-assets",
    StaticFiles(directory=str(assets_dir)),
    name="generated-assets",
)

app.include_router(api_router)
app.include_router(video_api_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/admin")


@app.get("/health")
def health():
    readiness()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "image_provider": settings.image_provider,
        "image_source_mode": settings.image_source_mode,
        "web_image_provider": settings.web_image_provider,
        "wechat_enabled": settings.wechat_enabled,
        "wechat_emotional_enabled": settings.wechat_emotional_enabled,
    }


@app.get("/livez", include_in_schema=False)
def liveness():
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readiness():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            connection.execute(
                select(Article.id, Article.content_type, Article.story_mode).limit(1)
            )
        if not assets_dir.is_dir():
            raise RuntimeError("generated asset directory is unavailable")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="service is not ready") from exc
    return {"status": "ready"}
