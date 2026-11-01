from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
from app.db.session import Base, engine
from app.models import tables  # noqa: F401 - register SQLAlchemy models

Base.metadata.create_all(bind=engine)
app = FastAPI(title=settings.app_name, version="1.0.0", description="Standalone student behavior adaptation and multi-agent orchestration service.")
app.include_router(router, prefix=settings.api_v1_prefix)

@app.get("/")
def root():
    return {"name": settings.app_name, "docs": "/docs", "health": f"{settings.api_v1_prefix}/health"}
