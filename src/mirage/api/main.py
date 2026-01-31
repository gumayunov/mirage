import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mirage.api.dependencies import get_settings
from mirage.api.routers import documents, projects, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure logging
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    yield
    # Shutdown: nothing to do


app = FastAPI(title="miRAGe", version="0.1.0", lifespan=lifespan)

app.include_router(projects.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
