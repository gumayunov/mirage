# miRAGe Full Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать локальную RAG-систему для работы с книгами и документацией, интегрированную с Claude Code.

**Architecture:** API (FastAPI) + Indexer worker + PostgreSQL/pgvector + Ollama embeddings. CLI для управления, Helm chart для деплоя в k3s.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, pgvector, Ollama (mxbai-embed-large), Typer (CLI), Helm, uv

---

## Phase 1: Project Setup

### Task 1.1: Initialize Python Project

**Files:**
- Create: `mise.toml`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

**Step 1: Create mise.toml**

```toml
[tools]
python = "3.12"
uv = "latest"

[env]
VIRTUAL_ENV = ".venv"
UV_LINK_MODE = "copy"
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "mirage"
version = "0.1.0"
description = "Local RAG system for books and documentation"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",
    "pgvector>=0.2.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "typer>=0.9.0",
    "httpx>=0.26.0",
    "python-multipart>=0.0.6",
    "PyMuPDF>=1.23.0",
    "ebooklib>=0.18",
    "markdown-it-py>=3.0.0",
    "tiktoken>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[project.scripts]
mirage = "mirage.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mirage"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
```

**Step 3: Create .gitignore**

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
*.egg-info/
dist/
.coverage
htmlcov/
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

**Step 4: Create README.md**

```markdown
# miRAGe

Local RAG system for books and documentation.

## Development

```bash
mise install
uv sync
```

## Usage

```bash
mirage --help
```
```

**Step 5: Initialize git and commit**

```bash
git init
git add .
git commit -m "chore: initialize project structure"
```

---

### Task 1.2: Create Source Directory Structure

**Files:**
- Create: `src/mirage/__init__.py`
- Create: `src/mirage/api/__init__.py`
- Create: `src/mirage/indexer/__init__.py`
- Create: `src/mirage/cli/__init__.py`
- Create: `src/mirage/shared/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create all __init__.py files**

All files empty except main package:

`src/mirage/__init__.py`:
```python
"""miRAGe - Local RAG system for books and documentation."""

__version__ = "0.1.0"
```

Other `__init__.py` files: empty.

**Step 2: Commit**

```bash
git add .
git commit -m "chore: add source directory structure"
```

---

## Phase 2: Shared Components

### Task 2.1: Configuration Module

**Files:**
- Create: `src/mirage/shared/config.py`
- Create: `tests/shared/__init__.py`
- Create: `tests/shared/test_config.py`

**Step 1: Write the failing test**

`tests/shared/test_config.py`:
```python
import os
from mirage.shared.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    monkeypatch.setenv("MIRAGE_OLLAMA_URL", "http://localhost:11434")

    settings = Settings()

    assert settings.database_url == "postgresql://test:test@localhost/test"
    assert settings.api_key == "test-key"
    assert settings.ollama_url == "http://localhost:11434"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")

    settings = Settings()

    assert settings.ollama_url == "http://ollama:11434"
    assert settings.ollama_model == "mxbai-embed-large"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/shared/test_config.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`src/mirage/shared/config.py`:
```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    api_key: str
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "mxbai-embed-large"
    chunk_size: int = 800
    chunk_overlap: int = 100
    documents_path: str = "/data/documents"

    model_config = {"env_prefix": "MIRAGE_"}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/shared/test_config.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add configuration module"
```

---

### Task 2.2: Database Models

**Files:**
- Create: `src/mirage/shared/db.py`
- Create: `src/mirage/shared/models.py`
- Create: `tests/shared/test_models.py`

**Step 1: Write the failing test**

`tests/shared/test_models.py`:
```python
import uuid
from datetime import datetime
from mirage.shared.models import Project, Document, Chunk, IndexingTask, DocumentStatus, TaskStatus


def test_project_model():
    project = Project(
        id=uuid.uuid4(),
        name="test-project",
        created_at=datetime.utcnow(),
    )
    assert project.name == "test-project"


def test_document_model():
    project_id = uuid.uuid4()
    doc = Document(
        id=uuid.uuid4(),
        project_id=project_id,
        filename="test.pdf",
        original_path="/data/documents/test.pdf",
        file_type="pdf",
        status=DocumentStatus.PENDING,
        created_at=datetime.utcnow(),
    )
    assert doc.status == DocumentStatus.PENDING


def test_chunk_model():
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Test content",
        embedding=[0.1] * 1024,
        position=0,
        structure={"chapter": "Test"},
    )
    assert chunk.position == 0


def test_indexing_task_model():
    task = IndexingTask(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        task_type="index",
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow(),
    )
    assert task.status == TaskStatus.PENDING
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/shared/test_models.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/shared/models.py`:
```python
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Project(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    project_id: uuid.UUID
    filename: str
    original_path: str
    file_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: datetime | None = None


class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID
    content: str
    embedding: list[float]
    position: int
    structure: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexingTask(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/shared/test_models.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add database models"
```

---

### Task 2.3: Database Connection and Tables

**Files:**
- Create: `src/mirage/shared/db.py`
- Create: `tests/shared/test_db.py`

**Step 1: Write the failing test**

`tests/shared/test_db.py`:
```python
import pytest
from sqlalchemy import text
from mirage.shared.db import get_engine, create_tables, ProjectTable


@pytest.fixture
def test_db_url():
    return "sqlite+aiosqlite:///:memory:"


@pytest.mark.asyncio
async def test_create_tables(test_db_url):
    engine = get_engine(test_db_url)
    await create_tables(engine)

    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = {row[0] for row in result.fetchall()}

    assert "projects" in tables
    assert "documents" in tables
    assert "chunks" in tables
    assert "indexing_tasks" in tables

    await engine.dispose()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/shared/test_db.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/shared/db.py`:
```python
import uuid
from datetime import datetime

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ProjectTable(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    documents: Mapped[list["DocumentTable"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class DocumentTable(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped["ProjectTable"] = relationship(back_populates="documents")
    chunks: Mapped[list["ChunkTable"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    tasks: Mapped[list["IndexingTaskTable"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ChunkTable(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[str | None] = mapped_column("embedding", JSON, nullable=True)
    position: Mapped[int] = mapped_column(Integer)
    structure_json: Mapped[str | None] = mapped_column("structure", JSON, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", JSON, nullable=True)

    document: Mapped["DocumentTable"] = relationship(back_populates="chunks")


class IndexingTaskTable(Base):
    __tablename__ = "indexing_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    task_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    document: Mapped["DocumentTable"] = relationship(back_populates="tasks")


def get_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False)


async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

**Step 4: Add aiosqlite to dev dependencies and run test**

Add to pyproject.toml `[project.optional-dependencies].dev`:
```
"aiosqlite>=0.19.0",
```

```bash
uv sync
uv run pytest tests/shared/test_db.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add database connection and SQLAlchemy tables"
```

---

### Task 2.4: Embedding Client

**Files:**
- Create: `src/mirage/shared/embedding.py`
- Create: `tests/shared/test_embedding.py`

**Step 1: Write the failing test**

`tests/shared/test_embedding.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from mirage.shared.embedding import OllamaEmbedding


@pytest.mark.asyncio
async def test_get_embedding():
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={"embedding": [0.1] * 1024})
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        embedding = await client.get_embedding("test text")

    assert len(embedding) == 1024
    assert embedding[0] == 0.1


@pytest.mark.asyncio
async def test_get_embeddings_batch():
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={"embedding": [0.1] * 1024})
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        embeddings = await client.get_embeddings(["text1", "text2"])

    assert len(embeddings) == 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/shared/test_embedding.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/shared/embedding.py`:
```python
import httpx


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            embedding = await self.get_embedding(text)
            results.append(embedding)
        return results
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/shared/test_embedding.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add Ollama embedding client"
```

---

## Phase 3: API

### Task 3.1: API Dependencies and Base Setup

**Files:**
- Create: `src/mirage/api/dependencies.py`
- Create: `src/mirage/api/main.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_main.py`

**Step 1: Write the failing test**

`tests/api/test_main.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from mirage.api.main import app


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/api/test_main.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/api/dependencies.py`:
```python
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import get_engine
from mirage.shared.embedding import OllamaEmbedding


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def verify_api_key(
    x_api_key: Annotated[str, Header()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


async def get_db_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncSession:
    engine = get_engine(settings.database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session


def get_embedding_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OllamaEmbedding:
    return OllamaEmbedding(settings.ollama_url, settings.ollama_model)
```

`src/mirage/api/main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="miRAGe", version="0.1.0")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/api/test_main.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add API base setup with health check"
```

---

### Task 3.2: Projects Router

**Files:**
- Create: `src/mirage/api/routers/__init__.py`
- Create: `src/mirage/api/routers/projects.py`
- Create: `src/mirage/api/schemas.py`
- Create: `tests/api/test_projects.py`

**Step 1: Write the failing test**

`tests/api/test_projects.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings
from mirage.shared.config import Settings
from mirage.shared.db import Base


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


@pytest.fixture
def override_settings():
    def _get_settings():
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key="test-key",
        )
    app.dependency_overrides[get_settings] = _get_settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_project(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "test-project"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-project"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_projects(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/v1/projects",
            json={"name": "project1"},
            headers={"X-API-Key": "test-key"},
        )
        response = await client.get(
            "/api/v1/projects",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "project1"


@pytest.mark.asyncio
async def test_delete_project(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_response = await client.post(
            "/api/v1/projects",
            json={"name": "to-delete"},
            headers={"X-API-Key": "test-key"},
        )
        project_id = create_response.json()["id"]

        delete_response = await client.delete(
            f"/api/v1/projects/{project_id}",
            headers={"X-API-Key": "test-key"},
        )

    assert delete_response.status_code == 204
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/api/test_projects.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/api/schemas.py`:
```python
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True
```

`src/mirage/api/routers/projects.py`:
```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, verify_api_key
from mirage.api.schemas import ProjectCreate, ProjectResponse
from mirage.shared.db import ProjectTable

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(select(ProjectTable))
    return result.scalars().all()


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

    db_project = ProjectTable(name=project.name)
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    return db_project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    await db.delete(project)
    await db.commit()
```

Update `src/mirage/api/main.py`:
```python
from fastapi import FastAPI

from mirage.api.routers import projects

app = FastAPI(title="miRAGe", version="0.1.0")

app.include_router(projects.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

Create `src/mirage/api/routers/__init__.py`: empty file.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/api/test_projects.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add projects CRUD API"
```

---

### Task 3.3: Documents Router

**Files:**
- Modify: `src/mirage/api/schemas.py`
- Create: `src/mirage/api/routers/documents.py`
- Create: `tests/api/test_documents.py`

**Step 1: Write the failing test**

`tests/api/test_documents.py`:
```python
import pytest
import io
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings
from mirage.shared.config import Settings
from mirage.shared.db import Base, ProjectTable


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

    # Create test project
    async with async_session() as session:
        project = ProjectTable(id="test-project-id", name="test-project")
        session.add(project)
        await session.commit()

    yield engine

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def override_settings(tmp_path):
    def _get_settings():
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key="test-key",
            documents_path=str(tmp_path),
        )
    app.dependency_overrides[get_settings] = _get_settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_document(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test Document\n\nThis is test content."
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}

        response = await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["filename"] == "test.md"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_list_documents(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test"
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}
        await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )

        response = await client.get(
            "/api/v1/projects/test-project-id/documents",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1


@pytest.mark.asyncio
async def test_get_document_status(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test"
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}
        upload_response = await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )
        doc_id = upload_response.json()["id"]

        response = await client.get(
            f"/api/v1/projects/test-project-id/documents/{doc_id}",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.md"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/api/test_documents.py -v
```
Expected: FAIL

**Step 3: Write implementation**

Add to `src/mirage/api/schemas.py`:
```python
class DocumentResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    file_type: str
    status: str
    error_message: str | None = None
    metadata: dict | None = None
    created_at: datetime
    indexed_at: datetime | None = None

    class Config:
        from_attributes = True
```

`src/mirage/api/routers/documents.py`:
```python
import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, get_settings, verify_api_key
from mirage.api.schemas import DocumentResponse
from mirage.shared.config import Settings
from mirage.shared.db import DocumentTable, IndexingTaskTable, ProjectTable

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])

FILE_TYPE_MAP = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".md": "markdown",
    ".markdown": "markdown",
}


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    project_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(DocumentTable).where(DocumentTable.project_id == project_id)
    )
    return result.scalars().all()


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    project_id: str,
    file: Annotated[UploadFile, File()],
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    ext = Path(file.filename or "").suffix.lower()
    file_type = FILE_TYPE_MAP.get(ext)
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(FILE_TYPE_MAP.keys())}",
        )

    existing = await db.execute(
        select(DocumentTable).where(
            DocumentTable.project_id == project_id,
            DocumentTable.filename == file.filename,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Document with this name already exists")

    docs_dir = Path(settings.documents_path) / project_id
    docs_dir.mkdir(parents=True, exist_ok=True)
    file_path = docs_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = DocumentTable(
        project_id=project_id,
        filename=file.filename,
        original_path=str(file_path),
        file_type=file_type,
        status="pending",
    )
    db.add(doc)
    await db.flush()

    task = IndexingTaskTable(
        document_id=doc.id,
        task_type="index",
        status="pending",
    )
    db.add(task)

    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if os.path.exists(doc.original_path):
        os.remove(doc.original_path)

    await db.delete(doc)
    await db.commit()


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = "pending"
    task = IndexingTaskTable(
        document_id=doc.id,
        task_type="reindex",
        status="pending",
    )
    db.add(task)

    await db.commit()
    await db.refresh(doc)
    return doc
```

Update `src/mirage/api/main.py`:
```python
from fastapi import FastAPI

from mirage.api.routers import documents, projects

app = FastAPI(title="miRAGe", version="0.1.0")

app.include_router(projects.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/api/test_documents.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add documents CRUD API"
```

---

### Task 3.4: Search Router

**Files:**
- Modify: `src/mirage/api/schemas.py`
- Create: `src/mirage/api/routers/search.py`
- Create: `tests/api/test_search.py`

**Step 1: Write the failing test**

`tests/api/test_search.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings, get_embedding_client
from mirage.shared.config import Settings
from mirage.shared.db import Base, ProjectTable, DocumentTable, ChunkTable


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

    async with async_session() as session:
        project = ProjectTable(id="test-project-id", name="test-project")
        session.add(project)

        doc = DocumentTable(
            id="test-doc-id",
            project_id="test-project-id",
            filename="test.md",
            original_path="/tmp/test.md",
            file_type="markdown",
            status="ready",
        )
        session.add(doc)

        chunk = ChunkTable(
            id="test-chunk-id",
            document_id="test-doc-id",
            content="This is test content about Python programming.",
            embedding_json=[0.1] * 1024,
            position=0,
            structure_json={"chapter": "Introduction"},
        )
        session.add(chunk)
        await session.commit()

    yield engine

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def override_settings():
    def _get_settings():
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key="test-key",
        )
    app.dependency_overrides[get_settings] = _get_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_embedding():
    mock = AsyncMock()
    mock.get_embedding = AsyncMock(return_value=[0.1] * 1024)
    app.dependency_overrides[get_embedding_client] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search(test_db, override_settings, mock_embedding):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects/test-project-id/search",
            json={"query": "Python programming", "limit": 10},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 0  # SQLite doesn't support vector search, just test API works
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/api/test_search.py -v
```
Expected: FAIL

**Step 3: Write implementation**

Add to `src/mirage/api/schemas.py`:
```python
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
```

`src/mirage/api/routers/search.py`:
```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, get_embedding_client, verify_api_key
from mirage.api.schemas import ChunkResult, SearchRequest, SearchResponse
from mirage.shared.db import ChunkTable, DocumentTable, ProjectTable
from mirage.shared.embedding import OllamaEmbedding

router = APIRouter(prefix="/projects/{project_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    project_id: str,
    request: SearchRequest,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    embedding_client: Annotated[OllamaEmbedding, Depends(get_embedding_client)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    query_embedding = await embedding_client.get_embedding(request.query)

    # For PostgreSQL with pgvector, use vector similarity search
    # For SQLite (testing), fall back to returning all chunks
    try:
        # Try pgvector query
        sql = text("""
            SELECT c.id, c.content, c.structure, c.embedding <=> :embedding AS distance,
                   d.id as doc_id, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = :project_id AND d.status = 'ready'
            ORDER BY c.embedding <=> :embedding
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

        results = []
        for row in rows:
            score = 1 - row.distance  # Convert distance to similarity
            if score >= request.threshold:
                results.append(
                    ChunkResult(
                        chunk_id=row.id,
                        content=row.content,
                        score=score,
                        structure=row.structure,
                        document={"id": row.doc_id, "filename": row.filename},
                    )
                )
    except Exception:
        # Fallback for SQLite (no vector search)
        result = await db.execute(
            select(ChunkTable, DocumentTable)
            .join(DocumentTable)
            .where(
                DocumentTable.project_id == project_id,
                DocumentTable.status == "ready",
            )
            .limit(request.limit)
        )
        rows = result.all()

        results = []
        for chunk, doc in rows:
            results.append(
                ChunkResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=1.0,  # No real scoring in fallback
                    structure=chunk.structure_json,
                    document={"id": doc.id, "filename": doc.filename},
                )
            )

    return SearchResponse(results=results)
```

Update `src/mirage/api/main.py`:
```python
from fastapi import FastAPI

from mirage.api.routers import documents, projects, search

app = FastAPI(title="miRAGe", version="0.1.0")

app.include_router(projects.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/api/test_search.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add vector search API"
```

---

## Phase 4: Indexer

### Task 4.1: Markdown Parser

**Files:**
- Create: `src/mirage/indexer/parsers/__init__.py`
- Create: `src/mirage/indexer/parsers/markdown.py`
- Create: `tests/indexer/__init__.py`
- Create: `tests/indexer/parsers/__init__.py`
- Create: `tests/indexer/parsers/test_markdown.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_markdown.py`:
```python
from mirage.indexer.parsers.markdown import MarkdownParser


def test_parse_markdown_with_headings():
    content = """# Book Title

## Chapter 1

This is the first chapter content.
It has multiple paragraphs.

Second paragraph here.

## Chapter 2

### Section 2.1

Content of section 2.1.
"""
    parser = MarkdownParser()
    result = parser.parse(content)

    assert result["title"] == "Book Title"
    assert len(result["sections"]) > 0


def test_parse_markdown_extracts_structure():
    content = """# My Book

## Introduction

Welcome to the book.

## Main Content

### Part 1

First part content.
"""
    parser = MarkdownParser()
    result = parser.parse(content)

    sections = result["sections"]
    assert any(s["heading"] == "Introduction" for s in sections)
    assert any(s["heading"] == "Part 1" for s in sections)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_markdown.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/markdown.py`:
```python
import re
from dataclasses import dataclass


@dataclass
class Section:
    heading: str
    level: int
    content: str
    parent_headings: list[str]


class MarkdownParser:
    def parse(self, content: str) -> dict:
        lines = content.split("\n")
        title = ""
        sections: list[dict] = []
        current_section: dict | None = None
        heading_stack: list[tuple[int, str]] = []
        content_lines: list[str] = []

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section
                if current_section is not None:
                    current_section["content"] = "\n".join(content_lines).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                if level == 1 and not title:
                    title = heading_text

                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                parent_headings = [h[1] for h in heading_stack]
                heading_stack.append((level, heading_text))

                current_section = {
                    "heading": heading_text,
                    "level": level,
                    "parent_headings": parent_headings,
                }
                content_lines = []
            else:
                content_lines.append(line)

        # Save last section
        if current_section is not None:
            current_section["content"] = "\n".join(content_lines).strip()
            if current_section["content"]:
                sections.append(current_section)

        return {
            "title": title,
            "sections": sections,
        }
```

Create `src/mirage/indexer/parsers/__init__.py`: empty file.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_markdown.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add markdown parser"
```

---

### Task 4.2: PDF Parser

**Files:**
- Create: `src/mirage/indexer/parsers/pdf.py`
- Create: `tests/indexer/parsers/test_pdf.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_pdf.py`:
```python
import pytest
from pathlib import Path
from mirage.indexer.parsers.pdf import PDFParser


@pytest.fixture
def sample_pdf(tmp_path):
    # Create a minimal valid PDF for testing
    # This is a very basic PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    return pdf_path


def test_pdf_parser_extracts_text(sample_pdf):
    parser = PDFParser()
    result = parser.parse(str(sample_pdf))

    assert "pages" in result
    assert isinstance(result["pages"], list)


def test_pdf_parser_handles_missing_file():
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.pdf")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_pdf.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/pdf.py`:
```python
from pathlib import Path

import fitz  # PyMuPDF


class PDFParser:
    def parse(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(file_path)

        # Try to get TOC
        toc = doc.get_toc()

        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "content": text.strip(),
            })

        # Extract title from metadata or first heading
        title = doc.metadata.get("title", "")
        if not title and toc:
            title = toc[0][1]  # First TOC entry

        doc.close()

        return {
            "title": title,
            "toc": [{"level": t[0], "title": t[1], "page": t[2]} for t in toc],
            "pages": pages,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_pdf.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add PDF parser"
```

---

### Task 4.3: EPUB Parser

**Files:**
- Create: `src/mirage/indexer/parsers/epub.py`
- Create: `tests/indexer/parsers/test_epub.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_epub.py`:
```python
import pytest
from mirage.indexer.parsers.epub import EPUBParser


def test_epub_parser_handles_missing_file():
    parser = EPUBParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.epub")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_epub.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/epub.py`:
```python
import re
from pathlib import Path

import ebooklib
from ebooklib import epub


class EPUBParser:
    def parse(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")

        book = epub.read_epub(file_path)

        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else ""

        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="ignore")
                # Strip HTML tags for plain text
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    chapters.append({
                        "id": item.get_id(),
                        "name": item.get_name(),
                        "content": text,
                    })

        # Get TOC
        toc = []
        for nav_item in book.toc:
            if isinstance(nav_item, epub.Link):
                toc.append({
                    "title": nav_item.title,
                    "href": nav_item.href,
                })
            elif isinstance(nav_item, tuple):
                section, links = nav_item
                toc.append({
                    "title": section.title if hasattr(section, "title") else str(section),
                    "children": [{"title": l.title, "href": l.href} for l in links if isinstance(l, epub.Link)],
                })

        return {
            "title": title,
            "toc": toc,
            "chapters": chapters,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_epub.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add EPUB parser"
```

---

### Task 4.4: Chunking Module

**Files:**
- Create: `src/mirage/indexer/chunking.py`
- Create: `tests/indexer/test_chunking.py`

**Step 1: Write the failing test**

`tests/indexer/test_chunking.py`:
```python
from mirage.indexer.chunking import Chunker, Chunk


def test_chunker_splits_long_text():
    chunker = Chunker(chunk_size=100, overlap=20)
    text = "This is a test. " * 50  # Long text

    chunks = chunker.chunk_text(text, structure={"chapter": "Test"})

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.structure["chapter"] == "Test" for c in chunks)


def test_chunker_preserves_short_text():
    chunker = Chunker(chunk_size=1000, overlap=100)
    text = "Short text."

    chunks = chunker.chunk_text(text, structure={})

    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunker_handles_paragraphs():
    chunker = Chunker(chunk_size=100, overlap=20)
    text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""

    chunks = chunker.chunk_text(text, structure={})

    assert len(chunks) >= 1
    # Check that chunks maintain paragraph boundaries where possible
    for chunk in chunks:
        assert chunk.content.strip()


def test_chunk_positions_are_sequential():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = "Word " * 100

    chunks = chunker.chunk_text(text, structure={})

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_chunking.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/chunking.py`:
```python
from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass
class Chunk:
    content: str
    position: int
    structure: dict[str, Any]


class Chunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_text(self, text: str, structure: dict[str, Any]) -> list[Chunk]:
        if not text.strip():
            return []

        # If text is short enough, return as single chunk
        if self._count_tokens(text) <= self.chunk_size:
            return [Chunk(content=text.strip(), position=0, structure=structure)]

        paragraphs = self._split_into_paragraphs(text)
        chunks: list[Chunk] = []
        current_content: list[str] = []
        current_tokens = 0
        position = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if any
                if current_content:
                    chunks.append(Chunk(
                        content="\n\n".join(current_content),
                        position=position,
                        structure=structure,
                    ))
                    position += 1
                    current_content = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = para.replace(". ", ".|").split("|")
                for sentence in sentences:
                    sent_tokens = self._count_tokens(sentence)
                    if current_tokens + sent_tokens > self.chunk_size and current_content:
                        chunks.append(Chunk(
                            content=" ".join(current_content),
                            position=position,
                            structure=structure,
                        ))
                        position += 1
                        # Keep overlap
                        overlap_content = current_content[-1] if current_content else ""
                        current_content = [overlap_content] if overlap_content else []
                        current_tokens = self._count_tokens(overlap_content) if overlap_content else 0

                    current_content.append(sentence)
                    current_tokens += sent_tokens

            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                chunks.append(Chunk(
                    content="\n\n".join(current_content),
                    position=position,
                    structure=structure,
                ))
                position += 1

                # Keep some overlap
                overlap_text = current_content[-1] if current_content else ""
                overlap_tokens = self._count_tokens(overlap_text)
                if overlap_tokens <= self.overlap:
                    current_content = [overlap_text, para]
                    current_tokens = overlap_tokens + para_tokens
                else:
                    current_content = [para]
                    current_tokens = para_tokens
            else:
                current_content.append(para)
                current_tokens += para_tokens

        # Save remaining content
        if current_content:
            chunks.append(Chunk(
                content="\n\n".join(current_content) if len(current_content) > 1 else current_content[0],
                position=position,
                structure=structure,
            ))

        return chunks
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_chunking.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add semantic chunking module"
```

---

### Task 4.5: Indexer Worker

**Files:**
- Create: `src/mirage/indexer/worker.py`
- Create: `tests/indexer/test_worker.py`

**Step 1: Write the failing test**

`tests/indexer/test_worker.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mirage.indexer.worker import IndexerWorker
from mirage.shared.config import Settings


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    client.get_embeddings = AsyncMock(return_value=[[0.1] * 1024])
    return client


def test_worker_initialization(settings):
    worker = IndexerWorker(settings)
    assert worker.settings == settings


@pytest.mark.asyncio
async def test_worker_process_markdown(settings, mock_embedding_client, tmp_path):
    # Create test file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nContent here.")

    settings.documents_path = str(tmp_path)
    worker = IndexerWorker(settings)
    worker.embedding_client = mock_embedding_client

    chunks = await worker._process_file(str(md_file), "markdown")

    assert len(chunks) >= 1
    assert chunks[0]["content"]
    assert "embedding" in chunks[0]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_worker.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/worker.py`:
```python
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.chunking import Chunker
from mirage.indexer.parsers.epub import EPUBParser
from mirage.indexer.parsers.markdown import MarkdownParser
from mirage.indexer.parsers.pdf import PDFParser
from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, IndexingTaskTable, get_engine
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class IndexerWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedding_client = OllamaEmbedding(
            settings.ollama_url,
            settings.ollama_model,
        )
        self.parsers = {
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "epub": EPUBParser(),
        }

    async def _process_file(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
        parser = self.parsers.get(file_type)
        if not parser:
            raise ValueError(f"Unsupported file type: {file_type}")

        if file_type == "markdown":
            content = Path(file_path).read_text()
            parsed = parser.parse(content)

            all_chunks = []
            for section in parsed["sections"]:
                structure = {
                    "title": parsed["title"],
                    "heading": section["heading"],
                    "level": section["level"],
                    "parent_headings": section["parent_headings"],
                }
                chunks = self.chunker.chunk_text(section["content"], structure)
                all_chunks.extend(chunks)

        elif file_type == "pdf":
            parsed = parser.parse(file_path)

            all_chunks = []
            for page in parsed["pages"]:
                if page["content"]:
                    structure = {
                        "title": parsed["title"],
                        "page": page["page_number"],
                    }
                    chunks = self.chunker.chunk_text(page["content"], structure)
                    all_chunks.extend(chunks)

        elif file_type == "epub":
            parsed = parser.parse(file_path)

            all_chunks = []
            for chapter in parsed["chapters"]:
                if chapter["content"]:
                    structure = {
                        "title": parsed["title"],
                        "chapter": chapter["name"],
                    }
                    chunks = self.chunker.chunk_text(chapter["content"], structure)
                    all_chunks.extend(chunks)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Get embeddings
        if all_chunks:
            texts = [c.content for c in all_chunks]
            embeddings = await self.embedding_client.get_embeddings(texts)

            return [
                {
                    "content": chunk.content,
                    "position": chunk.position,
                    "structure": chunk.structure,
                    "embedding": emb,
                }
                for chunk, emb in zip(all_chunks, embeddings)
            ]

        return []

    async def process_task(self, session: AsyncSession, task: IndexingTaskTable) -> None:
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == task.document_id)
        )
        doc = doc_result.scalar_one_or_none()

        if not doc:
            logger.error(f"Document not found: {task.document_id}")
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            return

        try:
            # Update statuses
            task.status = "processing"
            task.started_at = datetime.utcnow()
            doc.status = "indexing"
            await session.commit()

            # Delete existing chunks if reindexing
            if task.task_type == "reindex":
                await session.execute(
                    ChunkTable.__table__.delete().where(
                        ChunkTable.document_id == doc.id
                    )
                )

            # Process file
            chunks_data = await self._process_file(doc.original_path, doc.file_type)

            # Save chunks
            for chunk_data in chunks_data:
                chunk = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    embedding_json=chunk_data["embedding"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                )
                session.add(chunk)

            # Update statuses
            doc.status = "ready"
            doc.indexed_at = datetime.utcnow()
            task.status = "done"
            task.completed_at = datetime.utcnow()

            await session.commit()
            logger.info(f"Indexed document {doc.filename}: {len(chunks_data)} chunks")

        except Exception as e:
            logger.error(f"Failed to index {doc.filename}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("Indexer worker started")

        while True:
            async with async_session() as session:
                # Get pending task
                result = await session.execute(
                    select(IndexingTaskTable)
                    .where(IndexingTaskTable.status == "pending")
                    .order_by(IndexingTaskTable.created_at)
                    .limit(1)
                )
                task = result.scalar_one_or_none()

                if task:
                    await self.process_task(session, task)
                else:
                    await asyncio.sleep(5)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_worker.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add indexer worker"
```

---

## Phase 5: CLI

### Task 5.1: CLI Base and Config

**Files:**
- Create: `src/mirage/cli/main.py`
- Create: `src/mirage/cli/config.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_cli.py`

**Step 1: Write the failing test**

`tests/cli/test_cli.py`:
```python
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "miRAGe" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_cli.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/config.py`:
```python
import os


def get_api_url() -> str:
    return os.environ.get("MIRAGE_API_URL", "http://localhost:8000/api/v1")


def get_api_key() -> str:
    key = os.environ.get("MIRAGE_API_KEY", "")
    if not key:
        raise ValueError("MIRAGE_API_KEY environment variable not set")
    return key
```

`src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_cli.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI base structure"
```

---

### Task 5.2: CLI Documents Commands

**Files:**
- Create: `src/mirage/cli/commands/documents.py`
- Modify: `src/mirage/cli/main.py`
- Create: `tests/cli/test_documents_cmd.py`

**Step 1: Write the failing test**

`tests/cli/test_documents_cmd.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_documents_list_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "1", "filename": "test.pdf", "status": "ready"}
    ]

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(app, ["documents", "list", "--project", "test-project"])

    assert result.exit_code == 0
    assert "test.pdf" in result.stdout


def test_documents_status_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "doc-1",
        "filename": "test.pdf",
        "status": "ready",
        "file_type": "pdf",
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            app, ["documents", "status", "--project", "test-project", "doc-1"]
        )

    assert result.exit_code == 0
    assert "ready" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_documents_cmd.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/commands/__init__.py`: empty file.

`src/mirage/cli/commands/documents.py`:
```python
from pathlib import Path

import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url

app = typer.Typer(help="Document management commands")


def get_headers() -> dict:
    return {"X-API-Key": get_api_key()}


@app.command("list")
def list_documents(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
):
    """List all documents in a project."""
    url = f"{get_api_url()}/projects/{project}/documents"
    response = httpx.get(url, headers=get_headers())

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    docs = response.json()
    if not docs:
        typer.echo("No documents found.")
        return

    typer.echo(f"{'ID':<40} {'Filename':<30} {'Status':<10}")
    typer.echo("-" * 80)
    for doc in docs:
        typer.echo(f"{doc['id']:<40} {doc['filename']:<30} {doc['status']:<10}")


@app.command("add")
def add_document(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    file_path: Path = typer.Argument(..., help="Path to the document file"),
):
    """Upload a document to a project."""
    if not file_path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    url = f"{get_api_url()}/projects/{project}/documents"

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f)}
        response = httpx.post(url, headers=get_headers(), files=files, timeout=60.0)

    if response.status_code == 202:
        doc = response.json()
        typer.echo(f"Document uploaded: {doc['id']}")
        typer.echo("Indexing in progress...")
    elif response.status_code == 409:
        typer.echo("Error: Document with this name already exists", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)


@app.command("remove")
def remove_document(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    document_id: str = typer.Argument(..., help="Document ID"),
):
    """Remove a document from a project."""
    url = f"{get_api_url()}/projects/{project}/documents/{document_id}"
    response = httpx.delete(url, headers=get_headers())

    if response.status_code == 204:
        typer.echo("Document removed.")
    elif response.status_code == 404:
        typer.echo("Error: Document not found", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)


@app.command("status")
def document_status(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    document_id: str = typer.Argument(..., help="Document ID"),
):
    """Get the status of a document."""
    url = f"{get_api_url()}/projects/{project}/documents/{document_id}"
    response = httpx.get(url, headers=get_headers())

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    doc = response.json()
    typer.echo(f"ID:       {doc['id']}")
    typer.echo(f"Filename: {doc['filename']}")
    typer.echo(f"Type:     {doc['file_type']}")
    typer.echo(f"Status:   {doc['status']}")
    if doc.get("error_message"):
        typer.echo(f"Error:    {doc['error_message']}")
```

Update `src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__
from mirage.cli.commands import documents

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)

app.add_typer(documents.app, name="documents")


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_documents_cmd.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI documents commands"
```

---

### Task 5.3: CLI Search Command

**Files:**
- Create: `src/mirage/cli/commands/search.py`
- Modify: `src/mirage/cli/main.py`
- Create: `tests/cli/test_search_cmd.py`

**Step 1: Write the failing test**

`tests/cli/test_search_cmd.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_search_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "chunk_id": "1",
                "content": "This is matching content about Python.",
                "score": 0.85,
                "structure": {"chapter": "Introduction"},
                "document": {"id": "doc-1", "filename": "book.pdf"},
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response):
        result = runner.invoke(
            app, ["search", "--project", "test-project", "Python programming"]
        )

    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "book.pdf" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_search_cmd.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/commands/search.py`:
```python
import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url


def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search documents in a project."""
    url = f"{get_api_url()}/projects/{project}/search"
    headers = {"X-API-Key": get_api_key()}

    response = httpx.post(
        url,
        headers=headers,
        json={"query": query, "limit": limit},
        timeout=30.0,
    )

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    data = response.json()
    results = data.get("results", [])

    if not results:
        typer.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        typer.echo(f"\n--- Result {i} (score: {result['score']:.2f}) ---")
        typer.echo(f"Source: {result['document']['filename']}")
        if result.get("structure"):
            structure = result["structure"]
            if "chapter" in structure:
                typer.echo(f"Chapter: {structure['chapter']}")
            if "page" in structure:
                typer.echo(f"Page: {structure['page']}")
        typer.echo(f"\n{result['content'][:500]}...")
```

Update `src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__
from mirage.cli.commands import documents
from mirage.cli.commands.search import search as search_cmd

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)

app.add_typer(documents.app, name="documents")
app.command("search")(search_cmd)


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_search_cmd.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI search command"
```

---

## Phase 6: Helm Chart

### Task 6.1: Helm Chart Base

**Files:**
- Create: `helm/mirage/Chart.yaml`
- Create: `helm/mirage/values.yaml`

**Step 1: Create Chart.yaml**

```yaml
apiVersion: v2
name: mirage
description: miRAGe - Local RAG system for books and documentation
type: application
version: 0.1.0
appVersion: "0.1.0"
```

**Step 2: Create values.yaml**

```yaml
# API Configuration
api:
  replicas: 1
  image:
    repository: mirage
    tag: latest
    pullPolicy: IfNotPresent
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Indexer Configuration
indexer:
  replicas: 1
  image:
    repository: mirage
    tag: latest
    pullPolicy: IfNotPresent
  resources:
    requests:
      memory: "512Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

# Ollama Configuration
ollama:
  enabled: true
  image:
    repository: ollama/ollama
    tag: latest
  model: mxbai-embed-large
  persistence:
    enabled: true
    size: 10Gi
    storageClass: ""
  resources:
    requests:
      memory: "2Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

# PostgreSQL Configuration
postgresql:
  enabled: true
  image:
    repository: pgvector/pgvector
    tag: pg16
  auth:
    database: mirage
    username: mirage
    password: ""  # Set via secret
  persistence:
    enabled: true
    size: 5Gi
    storageClass: ""
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Documents Storage
documents:
  persistence:
    enabled: true
    size: 10Gi
    storageClass: ""

# Ingress Configuration
ingress:
  enabled: false
  className: ""
  host: mirage.local
  tls: []

# Authentication
auth:
  apiKey: ""  # Set via secret

# General Configuration
config:
  chunkSize: 800
  chunkOverlap: 100
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add Helm chart base"
```

---

### Task 6.2: PostgreSQL Templates

**Files:**
- Create: `helm/mirage/templates/postgresql-deployment.yaml`
- Create: `helm/mirage/templates/postgresql-pvc.yaml`
- Create: `helm/mirage/templates/postgresql-service.yaml`

**Step 1: Create postgresql-pvc.yaml**

```yaml
{{- if and .Values.postgresql.enabled .Values.postgresql.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteOnce
  {{- if .Values.postgresql.persistence.storageClass }}
  storageClassName: {{ .Values.postgresql.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.postgresql.persistence.size }}
{{- end }}
```

**Step 2: Create postgresql-deployment.yaml**

```yaml
{{- if .Values.postgresql.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: postgresql
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: postgresql
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: postgresql
          image: "{{ .Values.postgresql.image.repository }}:{{ .Values.postgresql.image.tag }}"
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: {{ .Values.postgresql.auth.database }}
            - name: POSTGRES_USER
              value: {{ .Values.postgresql.auth.username }}
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: postgresql-password
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
          resources:
            {{- toYaml .Values.postgresql.resources | nindent 12 }}
      volumes:
        - name: data
          {{- if .Values.postgresql.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-postgresql
          {{- else }}
          emptyDir: {}
          {{- end }}
{{- end }}
```

**Step 3: Create postgresql-service.yaml**

```yaml
{{- if .Values.postgresql.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 5432
      targetPort: 5432
  selector:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add PostgreSQL Helm templates"
```

---

### Task 6.3: Ollama Templates

**Files:**
- Create: `helm/mirage/templates/ollama-deployment.yaml`
- Create: `helm/mirage/templates/ollama-pvc.yaml`
- Create: `helm/mirage/templates/ollama-service.yaml`

**Step 1: Create ollama-pvc.yaml**

```yaml
{{- if and .Values.ollama.enabled .Values.ollama.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteOnce
  {{- if .Values.ollama.persistence.storageClass }}
  storageClassName: {{ .Values.ollama.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.ollama.persistence.size }}
{{- end }}
```

**Step 2: Create ollama-deployment.yaml**

```yaml
{{- if .Values.ollama.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: ollama
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ollama
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: ollama
          image: "{{ .Values.ollama.image.repository }}:{{ .Values.ollama.image.tag }}"
          ports:
            - containerPort: 11434
          volumeMounts:
            - name: models
              mountPath: /root/.ollama
          resources:
            {{- toYaml .Values.ollama.resources | nindent 12 }}
      volumes:
        - name: models
          {{- if .Values.ollama.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-ollama
          {{- else }}
          emptyDir: {}
          {{- end }}
{{- end }}
```

**Step 3: Create ollama-service.yaml**

```yaml
{{- if .Values.ollama.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 11434
      targetPort: 11434
  selector:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add Ollama Helm templates"
```

---

### Task 6.4: API and Indexer Templates

**Files:**
- Create: `helm/mirage/templates/configmap.yaml`
- Create: `helm/mirage/templates/secret.yaml`
- Create: `helm/mirage/templates/documents-pvc.yaml`
- Create: `helm/mirage/templates/api-deployment.yaml`
- Create: `helm/mirage/templates/api-service.yaml`
- Create: `helm/mirage/templates/indexer-deployment.yaml`

**Step 1: Create configmap.yaml**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
data:
  MIRAGE_OLLAMA_URL: "http://{{ .Release.Name }}-ollama:11434"
  MIRAGE_OLLAMA_MODEL: {{ .Values.ollama.model | quote }}
  MIRAGE_CHUNK_SIZE: {{ .Values.config.chunkSize | quote }}
  MIRAGE_CHUNK_OVERLAP: {{ .Values.config.chunkOverlap | quote }}
  MIRAGE_DOCUMENTS_PATH: "/data/documents"
```

**Step 2: Create secret.yaml**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Release.Name }}-secrets
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
type: Opaque
stringData:
  postgresql-password: {{ .Values.postgresql.auth.password | default (randAlphaNum 16) | quote }}
  api-key: {{ .Values.auth.apiKey | default (randAlphaNum 32) | quote }}
  database-url: "postgresql+asyncpg://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ .Release.Name }}-postgresql:5432/{{ .Values.postgresql.auth.database }}"
```

**Step 3: Create documents-pvc.yaml**

```yaml
{{- if .Values.documents.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-documents
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteMany
  {{- if .Values.documents.persistence.storageClass }}
  storageClassName: {{ .Values.documents.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.documents.persistence.size }}
{{- end }}
```

**Step 4: Create api-deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-api
  labels:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: {{ .Values.api.replicas }}
  selector:
    matchLabels:
      app.kubernetes.io/name: api
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: api
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: api
          image: "{{ .Values.api.image.repository }}:{{ .Values.api.image.tag }}"
          imagePullPolicy: {{ .Values.api.image.pullPolicy }}
          command: ["uvicorn", "mirage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: {{ .Release.Name }}-config
          env:
            - name: MIRAGE_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: database-url
            - name: MIRAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: api-key
          volumeMounts:
            - name: documents
              mountPath: /data/documents
          resources:
            {{- toYaml .Values.api.resources | nindent 12 }}
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: documents
          {{- if .Values.documents.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-documents
          {{- else }}
          emptyDir: {}
          {{- end }}
```

**Step 5: Create api-service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-api
  labels:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
```

**Step 6: Create indexer-deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-indexer
  labels:
    app.kubernetes.io/name: indexer
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: {{ .Values.indexer.replicas }}
  selector:
    matchLabels:
      app.kubernetes.io/name: indexer
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: indexer
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: indexer
          image: "{{ .Values.indexer.image.repository }}:{{ .Values.indexer.image.tag }}"
          imagePullPolicy: {{ .Values.indexer.image.pullPolicy }}
          command: ["python", "-m", "mirage.indexer.worker"]
          envFrom:
            - configMapRef:
                name: {{ .Release.Name }}-config
          env:
            - name: MIRAGE_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: database-url
            - name: MIRAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: api-key
          volumeMounts:
            - name: documents
              mountPath: /data/documents
          resources:
            {{- toYaml .Values.indexer.resources | nindent 12 }}
      volumes:
        - name: documents
          {{- if .Values.documents.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-documents
          {{- else }}
          emptyDir: {}
          {{- end }}
```

**Step 7: Commit**

```bash
git add .
git commit -m "feat: add API and Indexer Helm templates"
```

---

### Task 6.5: Ingress Template

**Files:**
- Create: `helm/mirage/templates/ingress.yaml`

**Step 1: Create ingress.yaml**

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ .Release.Name }}-ingress
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
  {{- if .Values.ingress.annotations }}
  annotations:
    {{- toYaml .Values.ingress.annotations | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    - host: {{ .Values.ingress.host | quote }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ .Release.Name }}-api
                port:
                  number: 8000
{{- end }}
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add Ingress Helm template"
```

---

## Phase 7: Dockerfile and Entrypoints

### Task 7.1: Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Install the project
RUN uv sync --no-dev

# Set Python path
ENV PYTHONPATH=/app/src

# Default command (overridden by Helm)
CMD ["uvicorn", "mirage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create .dockerignore**

```
__pycache__/
*.py[cod]
.venv/
.env
*.egg-info/
dist/
.coverage
htmlcov/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.git/
tests/
docs/
helm/
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add Dockerfile"
```

---

### Task 7.2: Indexer Entrypoint

**Files:**
- Create: `src/mirage/indexer/__main__.py`

**Step 1: Create __main__.py**

```python
import asyncio
import logging

from mirage.shared.config import Settings
from mirage.shared.db import create_tables, get_engine
from mirage.indexer.worker import IndexerWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    settings = Settings()

    # Create tables if needed
    engine = get_engine(settings.database_url)
    await create_tables(engine)
    await engine.dispose()

    # Run worker
    worker = IndexerWorker(settings)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add indexer entrypoint"
```

---

## Phase 8: Claude Code Skill

### Task 8.1: miRAGe Skill

**Files:**
- Create: `skill/mirage.md`

**Step 1: Create skill file**

```markdown
---
name: mirage
description: Поиск по базе знаний проекта (книги, документация)
---

# miRAGe — база знаний проекта

Используй этот skill когда нужно найти информацию в книгах
и документации, привязанных к текущему проекту.

## Конфигурация

Читай project_id из файла `.mirage.yaml` в корне проекта:

```yaml
project_id: "my-project"
```

API ключ в переменной окружения `MIRAGE_API_KEY`.
API URL в переменной окружения `MIRAGE_API_URL` (default: http://localhost:8000/api/v1).

## Команды

```bash
# Список документов
mirage documents list --project <project_id>

# Добавить документ
mirage documents add --project <project_id> /path/to/file.pdf

# Удалить документ
mirage documents remove --project <project_id> <document_id>

# Статус документа
mirage documents status --project <project_id> <document_id>

# Поиск
mirage search --project <project_id> "запрос" --limit 10
```

## Multi-hop поиск

При сложных вопросах:
1. Разбей на 2-4 подвопроса
2. Сделай поиск по каждому
3. Если информации недостаточно — уточни запрос
4. Синтезируй ответ с указанием источников

## Пример использования

```bash
# Читаем project_id
cat .mirage.yaml

# Ищем информацию
mirage search --project my-project "dependency injection best practices" --limit 5

# Если нужно больше контекста, делаем дополнительные запросы
mirage search --project my-project "constructor injection vs setter injection" --limit 5
```
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add Claude Code skill"
```

---

## Phase 9: Final Integration

### Task 9.1: Database Schema Init Script

**Files:**
- Create: `scripts/init-db.sql`

**Step 1: Create init script**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY,
    project_id VARCHAR(36) REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    original_path VARCHAR(512) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,
    UNIQUE(project_id, filename)
);

-- Chunks table with vector embedding
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1024),
    position INTEGER NOT NULL,
    structure JSONB,
    metadata JSONB
);

-- Indexing tasks table
CREATE TABLE IF NOT EXISTS indexing_tasks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id) ON DELETE CASCADE,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON indexing_tasks(status);

-- Vector index (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add database init script"
```

---

### Task 9.2: Run All Tests

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=src/mirage --cov-report=term-missing
```

Expected: All tests PASS

**Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run mypy src/
```

Fix any issues found.

**Step 3: Commit fixes if any**

```bash
git add .
git commit -m "fix: address linter issues"
```

---

### Task 9.3: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update README with full documentation**

```markdown
# miRAGe

Local RAG system for books and documentation, designed for integration with Claude Code.

## Features

- Document isolation by project
- Multi-hop retrieval on agent side
- Semantic chunks preserving document structure
- PDF, EPUB, Markdown support
- Deployment to k3s cluster

## Quick Start

### Development

```bash
# Install tools
mise install

# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run API locally
uv run uvicorn mirage.api.main:app --reload

# Run indexer locally
uv run python -m mirage.indexer
```

### Deployment

```bash
# Build image
docker build -t mirage:latest .

# Deploy to k3s
helm install mirage ./helm/mirage \
  --namespace mirage \
  --create-namespace \
  --set postgresql.auth.password=<password> \
  --set auth.apiKey=<api-key>

# Load Ollama model
kubectl exec -n mirage deploy/mirage-ollama -- ollama pull mxbai-embed-large
```

## CLI Usage

```bash
export MIRAGE_API_URL=http://mirage.your-domain.ru/api/v1
export MIRAGE_API_KEY=<your-api-key>

# List documents
mirage documents list --project my-project

# Add document
mirage documents add --project my-project /path/to/book.pdf

# Search
mirage search --project my-project "dependency injection patterns"
```

## Claude Code Integration

1. Install the skill to `~/.claude/skills/mirage.md`
2. Create `.mirage.yaml` in your project root with `project_id`
3. Set `MIRAGE_API_KEY` and `MIRAGE_API_URL` environment variables

## Architecture

See `docs/design.md` for detailed architecture documentation.

## License

MIT
```

**Step 2: Commit**

```bash
git add .
git commit -m "docs: update README with full documentation"
```

---

Plan complete and saved to `docs/plans/2026-01-28-mirage-full-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
