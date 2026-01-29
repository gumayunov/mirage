from fastapi import FastAPI

from mirage.api.routers import documents, projects, search

app = FastAPI(title="miRAGe", version="0.1.0")

app.include_router(projects.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
