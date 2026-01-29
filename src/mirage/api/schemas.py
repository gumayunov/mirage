from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    created_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    project_id: str
    filename: str
    file_type: str
    status: str
    error_message: str | None = None
    metadata: dict | None = Field(default=None, validation_alias="metadata_json")
    created_at: datetime
    indexed_at: datetime | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    structure: dict | None = None
    document: dict


class SearchResponse(BaseModel):
    results: list[ChunkResult]
