# miRAGe Phase 1: Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать базовую структуру проекта и shared компоненты.

**Prerequisite:** Нет (первая фаза)

**Deliverable:** Рабочий фундамент с конфигурацией, моделями, БД и embedding клиентом. Все тесты проходят.

---

## Task 1.1: Initialize Python Project

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
    "aiosqlite>=0.19.0",
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
git add .
git commit -m "chore: initialize project structure"
```

---

## Task 1.2: Create Source Directory Structure

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

## Task 2.1: Configuration Module

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

## Task 2.2: Database Models

**Files:**
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

## Task 2.3: Database Connection and Tables

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

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/shared/test_db.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add database connection and SQLAlchemy tables"
```

---

## Task 2.4: Embedding Client

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

## Verification

После завершения всех задач:

```bash
uv run pytest tests/shared/ -v
```

Все тесты должны проходить. Готово к Phase 2 (API) или Phase 3 (Indexer).
