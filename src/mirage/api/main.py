from fastapi import FastAPI

from mirage.api.routers import projects

app = FastAPI(title="miRAGe", version="0.1.0")

app.include_router(projects.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
