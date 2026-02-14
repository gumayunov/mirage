# Multi-Model Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multiple embedding models per project with separate vector tables for each model.

**Architecture:** Each model gets its own embeddings table (`embeddings_{model}_{dim}`). Projects specify which models to use via `project_models` table. Embedding status tracked per chunk per model in `embedding_status` table. Search unions results across all enabled models.

**Tech Stack:** PostgreSQL + pgvector, SQLAlchemy, Alembic migrations, FastAPI

---

## Task 1: Define Supported Models Registry

**Files:**
- Create: `src/mirage/shared/models_registry.py`
- Test: `tests/shared/test_models_registry.py`

**Step 1: Write the failing test**

```python
# tests/shared/test_models_registry.py
import pytest
from mirage.shared.models_registry import (
    SupportedModel,
    get_model,
    get_all_models,
    get_model_table_name,
)


def test_get_all_models_returns_three():
    models = get_all_models()
    assert len(models) == 3
    names = [m.name for m in models]
    assert "nomic-embed-text" in names
    assert "bge-m3" in names
    assert "mxbai-embed-large" in names


def test_get_model_by_name():
    model = get_model("bge-m3")
    assert model is not None
    assert model.name == "bge-m3"
    assert model.dimensions == 1024
    assert model.context_length == 8192


def test_get_model_unknown_returns_none():
    model = get_model("unknown-model")
    assert model is None


def test_get_model_table_name():
    assert get_model_table_name("nomic-embed-text") == "embeddings_nomic_768"
    assert get_model_table_name("bge-m3") == "embeddings_bge_m3_1024"
    assert get_model_table_name("mxbai-embed-large") == "embeddings_mxbai_1024"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_models_registry.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/mirage/shared/models_registry.py
from dataclasses import dataclass

SUPPORTED_MODELS: dict[str, "SupportedModel"] = {}


@dataclass(frozen=True)
class SupportedModel:
    name: str
    dimensions: int
    context_length: int
    ollama_name: str


# Define supported models
_nomic = SupportedModel(
    name="nomic-embed-text",
    dimensions=768,
    context_length=8192,
    ollama_name="nomic-embed-text",
)
_bge_m3 = SupportedModel(
    name="bge-m3",
    dimensions=1024,
    context_length=8192,
    ollama_name="bge-m3",
)
_mxbai = SupportedModel(
    name="mxbai-embed-large",
    dimensions=1024,
    context_length=512,
    ollama_name="mxbai-embed-large",
)

SUPPORTED_MODELS = {
    _nomic.name: _nomic,
    _bge_m3.name: _bge_m3,
    _mxbai.name: _mxbai,
}


def get_all_models() -> list[SupportedModel]:
    return list(SUPPORTED_MODELS.values())


def get_model(name: str) -> SupportedModel | None:
    return SUPPORTED_MODELS.get(name)


def get_model_table_name(model_name: str) -> str:
    model = get_model(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    # e.g., "bge-m3" -> "embeddings_bge_m3_1024"
    safe_name = model_name.replace("-", "_")
    return f"embeddings_{safe_name}_{model.dimensions}"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_models_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/models_registry.py tests/shared/test_models_registry.py
git commit -m "feat: add supported models registry"
```

---

## Task 2: Add Database Models for Multi-Model Support

**Files:**
- Modify: `src/mirage/shared/db.py`
- Test: `tests/shared/test_db.py`

**Step 1: Write the failing test**

```python
# Add to tests/shared/test_db.py

import pytest
from mirage.shared.db import (
    ProjectTable,
    ProjectModelTable,
    EmbeddingStatusTable,
)
from mirage.shared.models_registry import SUPPORTED_MODELS


@pytest.mark.asyncio
async def test_project_model_table():
    """Test ProjectModelTable can be instantiated."""
    pm = ProjectModelTable(
        project_id="test-project-id",
        model_name="bge-m3",
        enabled=True,
    )
    assert pm.model_name == "bge-m3"
    assert pm.enabled is True


@pytest.mark.asyncio
async def test_embedding_status_table():
    """Test EmbeddingStatusTable can be instantiated."""
    es = EmbeddingStatusTable(
        chunk_id="test-chunk-id",
        model_name="nomic-embed-text",
        status="pending",
    )
    assert es.model_name == "nomic-embed-text"
    assert es.status == "pending"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_db.py::test_project_model_table -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to src/mirage/shared/db.py (after existing tables)

class ProjectModelTable(Base):
    __tablename__ = "project_models"

    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=True)


class EmbeddingStatusTable(Base):
    __tablename__ = "embedding_status"

    chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
```

Also update `ProjectTable` to add `ollama_url`:

```python
class ProjectTable(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True)
    ollama_url: Mapped[str] = mapped_column(String(512), default="http://ollama:11434")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    documents: Mapped[list["DocumentTable"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    models: Mapped[list["ProjectModelTable"]] = relationship(back_populates="project", cascade="all, delete-orphan")
```

Add relationship to `ProjectModelTable`:

```python
class ProjectModelTable(Base):
    __tablename__ = "project_models"

    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=True)

    project: Mapped["ProjectTable"] = relationship(back_populates="models")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/db.py tests/shared/test_db.py
git commit -m "feat: add project_models and embedding_status tables"
```

---

## Task 3: Create Embedding Tables Dynamically

**Files:**
- Modify: `src/mirage/shared/db.py`
- Test: `tests/shared/test_db.py`

**Step 1: Write the failing test**

```python
# Add to tests/shared/test_db.py

from mirage.shared.db import get_embeddings_table_class
from mirage.shared.models_registry import get_model


def test_get_embeddings_table_class_nomic():
    """Test dynamic embeddings table class for nomic."""
    model = get_model("nomic-embed-text")
    TableClass = get_embeddings_table_class(model)
    assert TableClass.__tablename__ == "embeddings_nomic_768"


def test_get_embeddings_table_class_bge_m3():
    """Test dynamic embeddings table class for bge-m3."""
    model = get_model("bge-m3")
    TableClass = get_embeddings_table_class(model)
    assert TableClass.__tablename__ == "embeddings_bge_m3_1024"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_db.py::test_get_embeddings_table_class -v`
Expected: FAIL with "NameError"

**Step 3: Write minimal implementation**

```python
# Add to src/mirage/shared/db.py

from typing import Type
from mirage.shared.models_registry import SupportedModel

# Cache for dynamically created table classes
_embeddings_table_classes: dict[str, Type] = {}


def get_embeddings_table_class(model: SupportedModel) -> Type:
    """Get or create an embeddings table class for a model."""
    table_name = f"embeddings_{model.name.replace('-', '_')}_{model.dimensions}"

    if table_name in _embeddings_table_classes:
        return _embeddings_table_classes[table_name]

    class EmbeddingsTable(Base):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        chunk_id: Mapped[str] = mapped_column(
            String(36), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
        )
        embedding = mapped_column(Vector(model.dimensions), nullable=False)

    EmbeddingsTable.__name__ = f"EmbeddingsTable_{model.name.replace('-', '_').title()}"
    _embeddings_table_classes[table_name] = EmbeddingsTable

    return EmbeddingsTable
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_db.py::test_get_embeddings_table_class -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/db.py tests/shared/test_db.py
git commit -m "feat: add dynamic embeddings table generation per model"
```

---

## Task 4: Create Alembic Migration

**Files:**
- Create: `src/mirage/migrations/versions/002_multi_model_embeddings.py`

**Step 1: Create migration file**

```python
# src/mirage/migrations/versions/002_multi_model_embeddings.py
"""Add multi-model embeddings support.

Revision ID: 002
Revises: 001
Create Date: 2026-02-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add ollama_url to projects
    op.add_column(
        "projects",
        sa.Column("ollama_url", sa.String(512), server_default="http://ollama:11434"),
    )

    # Create project_models table
    op.create_table(
        "project_models",
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_name", sa.String(100), primary_key=True),
        sa.Column("enabled", sa.Boolean, server_default="true"),
    )

    # Create embedding_status table
    op.create_table(
        "embedding_status",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_name", sa.String(100), primary_key=True),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # Create embeddings tables for each supported model
    # nomic-embed-text: 768 dimensions
    op.create_table(
        "embeddings_nomic_768",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(768), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_nomic_768 ON embeddings_nomic_768 "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # bge-m3: 1024 dimensions
    op.create_table(
        "embeddings_bge_m3_1024",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(1024), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_bge_m3_1024 ON embeddings_bge_m3_1024 "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # mxbai-embed-large: 1024 dimensions
    op.create_table(
        "embeddings_mxbai_1024",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(1024), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_mxbai_1024 ON embeddings_mxbai_1024 "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("embeddings_mxbai_1024")
    op.drop_table("embeddings_bge_m3_1024")
    op.drop_table("embeddings_nomic_768")
    op.drop_table("embedding_status")
    op.drop_table("project_models")
    op.drop_column("projects", "ollama_url")
```

**Step 2: Commit**

```bash
git add src/mirage/migrations/versions/002_multi_model_embeddings.py
git commit -m "feat: add migration for multi-model embeddings tables"
```

---

## Task 5: Update API Schemas

**Files:**
- Modify: `src/mirage/api/schemas.py`
- Test: `tests/api/test_projects.py`

**Step 1: Write the failing test**

```python
# Add to tests/api/test_projects.py

@pytest.mark.asyncio
async def test_create_project_with_models(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "multi-model-project",
                "models": ["nomic-embed-text", "bge-m3"],
                "ollama_url": "http://custom-ollama:11434",
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "multi-model-project"
    assert data["ollama_url"] == "http://custom-ollama:11434"
    assert set(data["models"]) == {"nomic-embed-text", "bge-m3"}


@pytest.mark.asyncio
async def test_create_project_default_models(test_db, override_settings):
    """If models not specified, all supported models should be enabled."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "default-models-project"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["models"]) == 3  # All 3 supported models
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_projects.py::test_create_project_with_models -v`
Expected: FAIL (schema doesn't support models yet)

**Step 3: Write minimal implementation**

```python
# Update src/mirage/api/schemas.py

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    models: list[str] | None = None  # If None, all supported models
    ollama_url: str | None = None


class ProjectModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    enabled: bool = True


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    ollama_url: str = "http://ollama:11434"
    created_at: datetime
    models: list[ProjectModelResponse] = []


# ... rest unchanged, add models to SearchRequest
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    models: list[str] | None = None  # Optional filter for specific models
```

**Step 4: Run test to verify it fails (now on API logic)**

Run: `uv run pytest tests/api/test_projects.py::test_create_project_with_models -v`
Expected: FAIL (API not updated yet)

**Step 5: Commit**

```bash
git add src/mirage/api/schemas.py tests/api/test_projects.py
git commit -m "feat: add models field to project API schemas"
```

---

## Task 6: Update Projects API Endpoint

**Files:**
- Modify: `src/mirage/api/routers/projects.py`

**Step 1: Update create_project endpoint**

```python
# Update src/mirage/api/routers/projects.py

from mirage.shared.db import ProjectTable, ProjectModelTable
from mirage.shared.models_registry import get_all_models, get_model


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    existing = await db.execute(
        select(ProjectTable).where(ProjectTable.name == project.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project with this name already exists",
        )

    # Create project
    db_project = ProjectTable(
        name=project.name,
        ollama_url=project.ollama_url or "http://ollama:11434",
    )
    db.add(db_project)
    await db.flush()  # Get project ID

    # Determine which models to enable
    if project.models:
        model_names = project.models
    else:
        model_names = [m.name for m in get_all_models()]

    # Create project_models entries
    for model_name in model_names:
        if get_model(model_name) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model: {model_name}",
            )
        db_model = ProjectModelTable(
            project_id=db_project.id,
            model_name=model_name,
            enabled=True,
        )
        db.add(db_model)

    await db.commit()
    await db.refresh(db_project)

    # Load models relationship
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == db_project.id)
    )
    db_project = result.scalar_one()

    return db_project
```

**Step 2: Update list_projects to include models**

```python
@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable)
    )
    projects = result.scalars().all()
    # Models are loaded via relationship
    return projects
```

**Step 3: Run tests**

Run: `uv run pytest tests/api/test_projects.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/mirage/api/routers/projects.py
git commit -m "feat: update projects API to support model selection"
```

---

## Task 7: Update Embedding Worker for Multi-Model

**Files:**
- Modify: `src/mirage/indexer/embedding_worker.py`
- Modify: `src/mirage/indexer/worker.py` (to create embedding_status rows)
- Test: `tests/indexer/test_embedding_worker.py`

**Step 1: Write the failing test**

```python
# Add to tests/indexer/test_embedding_worker.py

import pytest
from unittest.mock import AsyncMock, patch
from mirage.indexer.embedding_worker import MultiModelEmbeddingWorker
from mirage.shared.db import EmbeddingStatusTable, ChunkTable
from mirage.shared.models_registry import get_model


@pytest.mark.asyncio
async def test_worker_processes_all_enabled_models(test_db_session):
    """Worker should embed chunk with all enabled models for project."""
    # This test will verify the worker queries enabled models
    # and creates embeddings for each
    pass  # Implementation depends on test fixtures
```

**Step 2: Create MultiModelEmbeddingWorker**

```python
# Update src/mirage/indexer/embedding_worker.py

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import (
    ChunkTable,
    DocumentTable,
    EmbeddingStatusTable,
    ProjectModelTable,
    ProjectTable,
    get_embeddings_table_class,
    get_engine,
)
from mirage.shared.embedding import OllamaEmbedding
from mirage.shared.models_registry import get_model

logger = logging.getLogger(__name__)


@dataclass
class PendingEmbedding:
    chunk_id: str
    model_name: str
    content: str
    ollama_url: str


class MultiModelEmbeddingWorker:
    """Claims pending embeddings and processes them via Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _claim_pending(self, session: AsyncSession) -> PendingEmbedding | None:
        """Find a pending embedding_status row with project's enabled model."""
        # Query for pending embeddings where:
        # - embedding_status.status = 'pending'
        # - project_models.enabled = true
        result = await session.execute(
            select(EmbeddingStatusTable, ChunkTable, ProjectModelTable, ProjectTable)
            .join(ChunkTable, EmbeddingStatusTable.chunk_id == ChunkTable.id)
            .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
            .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
            .join(
                ProjectModelTable,
                (ProjectModelTable.project_id == ProjectTable.id)
                & (ProjectModelTable.model_name == EmbeddingStatusTable.model_name),
            )
            .where(
                EmbeddingStatusTable.status == "pending",
                ProjectModelTable.enabled == True,
            )
            .limit(1)
        )
        row = result.first()
        if not row:
            return None

        embedding_status, chunk, _, project = row

        # Claim it
        embedding_status.status = "processing"
        await session.flush()

        return PendingEmbedding(
            chunk_id=chunk.id,
            model_name=embedding_status.model_name,
            content=chunk.content,
            ollama_url=project.ollama_url,
        )

    async def process_one(self, session: AsyncSession) -> bool:
        """Process a single embedding. Returns True if processed."""
        pending = await self._claim_pending(session)
        if not pending:
            return False

        model = get_model(pending.model_name)
        if not model:
            logger.error(f"Unknown model: {pending.model_name}")
            return False

        client = OllamaEmbedding(pending.ollama_url, model.ollama_name)
        result = await client.get_embedding(pending.content, prefix="search_document: ")

        # Get the status record
        status_record = await session.execute(
            select(EmbeddingStatusTable).where(
                EmbeddingStatusTable.chunk_id == pending.chunk_id,
                EmbeddingStatusTable.model_name == pending.model_name,
            )
        )
        status = status_record.scalar_one()

        if result is None:
            status.status = "failed"
            status.error_message = "Embedding request failed"
        else:
            # Insert into appropriate embeddings table
            TableClass = get_embeddings_table_class(model)
            embedding_row = TableClass(
                chunk_id=pending.chunk_id,
                embedding=result.embedding,
            )
            session.add(embedding_row)
            status.status = "ready"

        await session.commit()
        logger.info(f"Embedded chunk {pending.chunk_id} with {pending.model_name}")
        return True

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("MultiModelEmbeddingWorker started")

        while True:
            async with async_session() as session:
                processed = await self.process_one(session)

            if not processed:
                await asyncio.sleep(2)
```

**Step 3: Run tests**

Run: `uv run pytest tests/indexer/test_embedding_worker.py -v`
Expected: PASS (after fixture setup)

**Step 4: Commit**

```bash
git add src/mirage/indexer/embedding_worker.py tests/indexer/test_embedding_worker.py
git commit -m "feat: update embedding worker for multi-model support"
```

---

## Task 8: Update Worker to Create Embedding Status Rows

**Files:**
- Modify: `src/mirage/indexer/worker.py`
- Test: `tests/indexer/test_worker.py`

**Step 1: Update ChunkWorker to create embedding_status rows**

When a chunk is created, create `embedding_status` rows for all enabled models.

```python
# In src/mirage/indexer/worker.py, after creating a chunk:

from mirage.shared.db import EmbeddingStatusTable
from mirage.shared.models_registry import get_all_models

async def _create_embedding_status_for_chunk(
    session: AsyncSession,
    chunk_id: str,
    project_id: str,
) -> None:
    """Create embedding_status rows for all enabled models in project."""
    # Get enabled models for project
    result = await session.execute(
        select(ProjectModelTable.model_name).where(
            ProjectModelTable.project_id == project_id,
            ProjectModelTable.enabled == True,
        )
    )
    enabled_models = [row[0] for row in result.fetchall()]

    # Create status rows
    for model_name in enabled_models:
        status = EmbeddingStatusTable(
            chunk_id=chunk_id,
            model_name=model_name,
            status="pending",
        )
        session.add(status)
```

**Step 2: Commit**

```bash
git add src/mirage/indexer/worker.py tests/indexer/test_worker.py
git commit -m "feat: create embedding_status rows when chunks are created"
```

---

## Task 9: Update Search for Multi-Model

**Files:**
- Modify: `src/mirage/api/routers/search.py`
- Test: `tests/api/test_search.py`

**Step 1: Update search endpoint**

```python
# Update src/mirage/api/routers/search.py

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, verify_api_key
from mirage.api.schemas import ChunkResult, SearchRequest, SearchResponse
from mirage.shared.db import ChunkTable, DocumentTable, ProjectModelTable, ProjectTable
from mirage.shared.embedding import OllamaEmbedding
from mirage.shared.models_registry import get_model

router = APIRouter(prefix="/projects/{project_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    project_id: str,
    request: SearchRequest,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    # Verify project exists
    project_result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    project = project_result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get enabled models (optionally filtered by request.models)
    if request.models:
        models_result = await db.execute(
            select(ProjectModelTable.model_name).where(
                ProjectModelTable.project_id == project_id,
                ProjectModelTable.enabled == True,
                ProjectModelTable.model_name.in_(request.models),
            )
        )
    else:
        models_result = await db.execute(
            select(ProjectModelTable.model_name).where(
                ProjectModelTable.project_id == project_id,
                ProjectModelTable.enabled == True,
            )
        )

    model_names = [row[0] for row in models_result.fetchall()]
    if not model_names:
        raise HTTPException(status_code=400, detail="No enabled models for search")

    # Embed query with each model (parallel)
    async def embed_query(model_name: str) -> tuple[str, list[float] | None]:
        model = get_model(model_name)
        if not model:
            return model_name, None
        client = OllamaEmbedding(project.ollama_url, model.ollama_name)
        result = await client.get_embedding(request.query, prefix="search_query: ")
        return model_name, result.embedding if result else None

    embedding_tasks = [embed_query(m) for m in model_names]
    embeddings = await asyncio.gather(*embedding_tasks)

    # Search each model's embeddings table
    all_results: list[tuple[str, str, str, float, str]] = []  # (chunk_id, content, parent_content, distance, doc_id)

    for model_name, query_embedding in embeddings:
        if query_embedding is None:
            continue

        model = get_model(model_name)
        if not model:
            continue

        table_name = f"embeddings_{model_name.replace('-', '_')}_{model.dimensions}"

        try:
            sql = text(f"""
                SELECT DISTINCT ON (child.parent_id)
                       child.id, child.content, child.structure,
                       e.embedding <=> :embedding AS distance,
                       parent.content AS parent_content,
                       d.id as doc_id, d.filename
                FROM {table_name} e
                JOIN chunks child ON e.chunk_id = child.id
                JOIN chunks parent ON child.parent_id = parent.id
                JOIN documents d ON child.document_id = d.id
                WHERE d.project_id = :project_id
                  AND d.status IN ('ready', 'partial')
                ORDER BY child.parent_id, e.embedding <=> :embedding
                LIMIT :limit
            """)
            result = await db.execute(
                sql,
                {
                    "embedding": str(query_embedding),
                    "project_id": project_id,
                    "limit": request.limit,
                },
            )
            rows = result.fetchall()

            for row in rows:
                all_results.append((
                    row.id,
                    row.content,
                    row.parent_content,
                    row.distance,
                    row.doc_id,
                    row.filename,
                    row.structure,
                ))
        except Exception as e:
            logger.warning(f"Search failed for model {model_name}: {e}")
            continue

    # Deduplicate by chunk_id, keep minimum distance
    seen_chunks: dict[str, tuple] = {}
    for chunk_id, content, parent_content, distance, doc_id, filename, structure in all_results:
        if chunk_id not in seen_chunks or seen_chunks[chunk_id][3] > distance:
            seen_chunks[chunk_id] = (content, parent_content, distance, doc_id, filename, structure)

    # Sort by distance and limit
    sorted_results = sorted(seen_chunks.values(), key=lambda x: x[2])[:request.limit]

    # Build response
    results = []
    for content, parent_content, distance, doc_id, filename, structure in sorted_results:
        score = 1 - distance
        if score >= request.threshold:
            results.append(
                ChunkResult(
                    chunk_id=chunk_id,  # Need to track this
                    content=content,
                    parent_content=parent_content,
                    score=score,
                    structure=structure,
                    document={"id": doc_id, "filename": filename},
                )
            )

    return SearchResponse(results=results)
```

**Step 2: Commit**

```bash
git add src/mirage/api/routers/search.py tests/api/test_search.py
git commit -m "feat: update search to query multiple embedding tables"
```

---

## Task 10: Update CLI Commands

**Files:**
- Modify: `src/mirage/cli/commands/projects.py`
- Test: `tests/cli/test_cli.py`

**Step 1: Update project create command**

```python
# In src/mirage/cli/commands/projects.py

@app.command("create")
def create_project(
    name: str,
    models: list[str] | None = typer.Option(None, "--model", "-m", help="Embedding models to use"),
    ollama_url: str | None = typer.Option(None, "--ollama-url", help="Ollama server URL"),
):
    """Create a new project."""
    payload = {"name": name}
    if models:
        payload["models"] = models
    if ollama_url:
        payload["ollama_url"] = ollama_url

    response = httpx.post(f"{API_URL}/projects", json=payload, headers=HEADERS)
    # ... handle response
```

**Step 2: Commit**

```bash
git add src/mirage/cli/commands/projects.py tests/cli/test_cli.py
git commit -m "feat: add models option to project create CLI command"
```

---

## Task 11: Integration Tests

**Files:**
- Create: `tests/integration/test_multi_model.py`

**Step 1: Write integration test**

```python
# tests/integration/test_multi_model.py

import pytest
from httpx import AsyncClient, ASGITransport

from mirage.api.main import app
from mirage.api.dependencies import get_db_session
from mirage.shared.config import Settings
from mirage.shared.db import Base, get_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker


@pytest.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db
    yield engine
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_project_with_custom_models(test_db):
    """Test creating project with specific models."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "custom-models-project",
                "models": ["nomic-embed-text"],
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["model_name"] == "nomic-embed-text"
```

**Step 2: Run tests**

Run: `uv run pytest tests/integration/test_multi_model.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_multi_model.py
git commit -m "test: add integration tests for multi-model support"
```

---

## Task 12: Update Documentation

**Files:**
- Update: `AGENTS.md`
- Update: `CLAUDE.md`

**Step 1: Update AGENTS.md**

Add note about multi-model configuration in Configuration section.

**Step 2: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: update agent docs for multi-model embeddings"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Models registry | +2 files |
| 2 | Database models | 2 files |
| 3 | Dynamic tables | 2 files |
| 4 | Migration | +1 file |
| 5 | API schemas | 2 files |
| 6 | Projects API | 1 file |
| 7 | Embedding worker | 2 files |
| 8 | Chunk worker | 2 files |
| 9 | Search API | 2 files |
| 10 | CLI commands | 2 files |
| 11 | Integration tests | +1 file |
| 12 | Documentation | 2 files |

**Estimated time:** 3-4 hours for full implementation.
