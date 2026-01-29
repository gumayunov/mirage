# Chunk Status Lifecycle — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the monolithic indexer worker into a pipeline of three independent workers (ChunkWorker, EmbeddingWorker, StatusWorker) with per-chunk status tracking and progress visibility in API/CLI.

**Architecture:** Three async workers poll the database independently. ChunkWorker parses documents and saves chunks without embeddings. EmbeddingWorker claims individual chunks atomically via `UPDATE ... RETURNING` and calls Ollama. StatusWorker polls every 10 seconds and computes document status from chunk statuses. Workers communicate only through the database.

**Tech Stack:** Python 3.14, SQLAlchemy + aiosqlite, FastAPI, httpx, typer

**Design document:** `docs/plans/2026-01-29-chunk-status-design.md`

---

### Task CSL-1: Add `status` column to ChunkTable

**Files:**
- Modify: `src/mirage/shared/db.py:42-53`
- Test: `tests/shared/test_db.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_db.py`:

```python
@pytest.mark.asyncio
async def test_chunks_table_has_status_column(test_db_url):
    engine = get_engine(test_db_url)
    await create_tables(engine)

    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(chunks)"))
        columns = {row[1] for row in result.fetchall()}

    assert "status" in columns
    await engine.dispose()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_db.py::test_chunks_table_has_status_column -v`
Expected: FAIL — `assert 'status' in columns`

**Step 3: Write minimal implementation**

In `src/mirage/shared/db.py`, add to `ChunkTable` class after `metadata_json`:

```python
status: Mapped[str] = mapped_column(String(50), default="pending")
```

Also add `String` is already imported — no new imports needed.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_db.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/db.py tests/shared/test_db.py
git commit -m "feat: add status column to ChunkTable"
```

---

### Task CSL-2: Return `EmbeddingResult` from embedding client

**Files:**
- Modify: `src/mirage/shared/embedding.py`
- Test: `tests/shared/test_embedding.py`

**Step 1: Write the failing test**

Replace the existing tests in `tests/shared/test_embedding.py` with:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mirage.shared.embedding import OllamaEmbedding, EmbeddingResult, MAX_PROMPT_CHARS


@pytest.mark.asyncio
async def test_get_embedding_returns_result():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding("short text")

    assert isinstance(result, EmbeddingResult)
    assert len(result.embedding) == 1024
    assert result.truncated is False


@pytest.mark.asyncio
async def test_get_embedding_truncated():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.2] * 1024}
    mock_response.raise_for_status = MagicMock()

    long_text = "x" * (MAX_PROMPT_CHARS + 100)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding(long_text)

    assert isinstance(result, EmbeddingResult)
    assert result.truncated is True
    assert len(result.embedding) == 1024


@pytest.mark.asyncio
async def test_get_embedding_ollama_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding("test text")

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_embedding.py -v`
Expected: FAIL — `ImportError: cannot import name 'EmbeddingResult'`

**Step 3: Write minimal implementation**

Replace `src/mirage/shared/embedding.py`:

```python
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# mxbai-embed-large has a 512-token context window.
# Empirically, 729 chars already exceeds the limit while 694 passes.
# Use 500 chars as a safe ceiling (~1 char per token for worst case).
MAX_PROMPT_CHARS = 500


@dataclass
class EmbeddingResult:
    embedding: list[float]
    truncated: bool


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> EmbeddingResult | None:
        logger.info("Embedding request: %d chars | %s", len(text), text[:200])
        truncated = False
        if len(text) > MAX_PROMPT_CHARS:
            logger.warning(
                "Truncating embedding input from %d to %d chars",
                len(text), MAX_PROMPT_CHARS,
            )
            text = text[:MAX_PROMPT_CHARS]
            truncated = True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60.0,
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                return EmbeddingResult(embedding=embedding, truncated=truncated)
        except Exception:
            logger.exception("Embedding request failed")
            return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_embedding.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/embedding.py tests/shared/test_embedding.py
git commit -m "feat: return EmbeddingResult from embedding client, handle errors gracefully"
```

---

### Task CSL-3: Update `chunk_size` config default

**Files:**
- Modify: `src/mirage/shared/config.py:9`
- Test: `tests/shared/test_config.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_config.py`:

```python
def test_settings_chunk_size_default(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    settings = Settings()
    assert settings.chunk_size == 400
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_config.py::test_settings_chunk_size_default -v`
Expected: FAIL — `assert 128 == 400`

**Step 3: Write minimal implementation**

In `src/mirage/shared/config.py`, change line 9:

```python
chunk_size: int = 400
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_config.py -v`
Expected: ALL PASS (check that the existing `test_settings_defaults` still expects 128 — if so, update it to 400 too)

**Step 5: Commit**

```bash
git add src/mirage/shared/config.py tests/shared/test_config.py
git commit -m "feat: increase default chunk_size to 400"
```

---

### Task CSL-4: ChunkWorker — parses documents and saves chunks

**Files:**
- Modify: `src/mirage/indexer/worker.py`
- Test: `tests/indexer/test_worker.py`

This task refactors the existing `IndexerWorker` into `ChunkWorker`. The key difference: no Ollama call. Parse file → save chunks with `status="pending"` → done.

**Step 1: Write the failing test**

Replace `tests/indexer/test_worker.py`:

```python
import pytest
from unittest.mock import AsyncMock

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, IndexingTaskTable, ProjectTable
from mirage.indexer.worker import ChunkWorker


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path=str(tmp_path),
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test")
        session.add(project)
        await session.commit()

    yield session_factory

    await engine.dispose()


def test_chunk_worker_initialization(settings):
    worker = ChunkWorker(settings)
    assert worker.settings == settings


@pytest.mark.asyncio
async def test_chunk_worker_creates_pending_chunks(settings, db_session, tmp_path):
    # Create a markdown file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nFirst paragraph content.\n\nSecond paragraph content.")

    # Create document and task in DB
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-1",
            project_id="proj-1",
            filename="test.md",
            original_path=str(md_file),
            file_type="markdown",
            status="pending",
        )
        session.add(doc)
        task = IndexingTaskTable(
            id="task-1",
            document_id="doc-1",
            task_type="index",
            status="pending",
        )
        session.add(task)
        await session.commit()

    # Process the task
    worker = ChunkWorker(settings)
    async with db_session() as session:
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1")
        )
        task = task_result.scalar_one()
        await worker.process_task(session, task)

    # Verify chunks were created with pending status and no embedding
    async with db_session() as session:
        chunks_result = await session.execute(
            select(ChunkTable).where(ChunkTable.document_id == "doc-1")
        )
        chunks = chunks_result.scalars().all()

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.status == "pending"
            assert chunk.embedding_json is None
            assert chunk.content

        # Verify document status is now "indexing"
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-1")
        )
        doc = doc_result.scalar_one()
        assert doc.status == "indexing"

        # Verify task is done
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1")
        )
        task = task_result.scalar_one()
        assert task.status == "done"


@pytest.mark.asyncio
async def test_chunk_worker_sets_error_on_parse_failure(settings, db_session, tmp_path):
    # Create document pointing to nonexistent file
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-2",
            project_id="proj-1",
            filename="missing.md",
            original_path="/nonexistent/missing.md",
            file_type="markdown",
            status="pending",
        )
        session.add(doc)
        task = IndexingTaskTable(
            id="task-2",
            document_id="doc-2",
            task_type="index",
            status="pending",
        )
        session.add(task)
        await session.commit()

    worker = ChunkWorker(settings)
    async with db_session() as session:
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-2")
        )
        task = task_result.scalar_one()
        await worker.process_task(session, task)

    # Verify error status
    async with db_session() as session:
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-2")
        )
        doc = doc_result.scalar_one()
        assert doc.status == "error"
        assert doc.error_message is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_worker.py -v`
Expected: FAIL — `ImportError: cannot import name 'ChunkWorker'`

**Step 3: Write minimal implementation**

Refactor `src/mirage/indexer/worker.py`. Replace `IndexerWorker` with `ChunkWorker`:

```python
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.chunking import Chunker
from mirage.indexer.parsers.epub import EPUBParser
from mirage.indexer.parsers.markdown import MarkdownParser
from mirage.indexer.parsers.pdf import PDFParser
from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, IndexingTaskTable, get_engine

logger = logging.getLogger(__name__)


class ChunkWorker:
    """Parses documents and saves chunks with status='pending' (no embeddings)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.parsers = {
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "epub": EPUBParser(),
        }

    def _parse_file(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
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

        return [
            {
                "content": chunk.content,
                "position": chunk.position,
                "structure": chunk.structure,
            }
            for chunk in all_chunks
        ]

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

            # Parse file and create chunks
            chunks_data = self._parse_file(doc.original_path, doc.file_type)

            for chunk_data in chunks_data:
                chunk = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                    status="pending",
                )
                session.add(chunk)

            task.status = "done"
            task.completed_at = datetime.utcnow()

            await session.commit()
            logger.info(f"Created {len(chunks_data)} chunks for {doc.filename}")

        except Exception as e:
            logger.error(f"Failed to process {doc.filename}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("ChunkWorker started")

        while True:
            async with async_session() as session:
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

Run: `uv run pytest tests/indexer/test_worker.py -v`
Expected: ALL PASS

**Step 5: Run all tests to check for regressions**

Run: `uv run pytest -v`
Expected: Some tests may break if they import `IndexerWorker`. Fix imports in any broken tests (the old class is gone, replaced by `ChunkWorker`).

**Step 6: Commit**

```bash
git add src/mirage/indexer/worker.py tests/indexer/test_worker.py
git commit -m "feat: replace IndexerWorker with ChunkWorker (parse only, no embeddings)"
```

---

### Task CSL-5: EmbeddingWorker — processes chunks individually

**Files:**
- Create: `src/mirage/indexer/embedding_worker.py`
- Test: `tests/indexer/test_embedding_worker.py`

**Step 1: Write the failing test**

Create `tests/indexer/test_embedding_worker.py`:

```python
import pytest
from unittest.mock import AsyncMock

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable
from mirage.shared.embedding import EmbeddingResult
from mirage.indexer.embedding_worker import EmbeddingWorker


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Create test data: project + document + pending chunks
    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test")
        session.add(project)
        doc = DocumentTable(
            id="doc-1",
            project_id="proj-1",
            filename="test.md",
            original_path="/tmp/test.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        chunk = ChunkTable(
            id="chunk-1",
            document_id="doc-1",
            content="Some test content",
            position=0,
            status="pending",
        )
        session.add(chunk)
        await session.commit()

    yield session_factory

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_worker_processes_chunk_ready(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(
        return_value=EmbeddingResult(embedding=[0.1] * 1024, truncated=False)
    )

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is True

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "ready"
        assert chunk.embedding_json == [0.1] * 1024


@pytest.mark.asyncio
async def test_embedding_worker_truncated_chunk(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(
        return_value=EmbeddingResult(embedding=[0.2] * 1024, truncated=True)
    )

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        await worker.process_one(session)

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "corrupted"
        assert chunk.embedding_json is not None


@pytest.mark.asyncio
async def test_embedding_worker_ollama_error(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(return_value=None)

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        await worker.process_one(session)

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "error"
        assert chunk.embedding_json is None


@pytest.mark.asyncio
async def test_embedding_worker_no_pending_chunks(settings, db_session):
    # Mark the only chunk as ready
    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        chunk.status = "ready"
        await session.commit()

    mock_client = AsyncMock()
    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_embedding_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mirage.indexer.embedding_worker'`

**Step 3: Write minimal implementation**

Create `src/mirage/indexer/embedding_worker.py`:

```python
import asyncio
import logging

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, get_engine
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Claims pending chunks and adds embeddings via Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_client = OllamaEmbedding(
            settings.ollama_url,
            settings.ollama_model,
        )

    async def _claim_chunk(self, session: AsyncSession) -> ChunkTable | None:
        """Atomically claim a pending chunk by setting status to 'processing'."""
        # SQLite doesn't support UPDATE ... RETURNING with subquery well,
        # so use SELECT + UPDATE in a transaction (SQLite serializes writes).
        result = await session.execute(
            select(ChunkTable)
            .where(ChunkTable.status == "pending")
            .limit(1)
        )
        chunk = result.scalar_one_or_none()
        if chunk:
            chunk.status = "processing"
            await session.flush()
        return chunk

    async def process_one(self, session: AsyncSession) -> bool:
        """Process a single chunk. Returns True if a chunk was processed."""
        chunk = await self._claim_chunk(session)
        if not chunk:
            return False

        result = await self.embedding_client.get_embedding(chunk.content)

        if result is None:
            chunk.status = "error"
        elif result.truncated:
            chunk.embedding_json = result.embedding
            chunk.status = "corrupted"
        else:
            chunk.embedding_json = result.embedding
            chunk.status = "ready"

        await session.commit()
        logger.info(f"Chunk {chunk.id}: status={chunk.status}")
        return True

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("EmbeddingWorker started")

        while True:
            async with async_session() as session:
                processed = await self.process_one(session)

            if not processed:
                await asyncio.sleep(2)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexer/test_embedding_worker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/indexer/embedding_worker.py tests/indexer/test_embedding_worker.py
git commit -m "feat: add EmbeddingWorker for per-chunk embedding processing"
```

---

### Task CSL-6: StatusWorker — computes document status from chunks

**Files:**
- Create: `src/mirage/indexer/status_worker.py`
- Test: `tests/indexer/test_status_worker.py`

**Step 1: Write the failing test**

Create `tests/indexer/test_status_worker.py`:

```python
import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable
from mirage.indexer.status_worker import StatusWorker


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test")
        session.add(project)
        await session.commit()

    yield session_factory

    await engine.dispose()


async def _create_doc_with_chunks(session_factory, doc_id, chunk_statuses):
    async with session_factory() as session:
        doc = DocumentTable(
            id=doc_id,
            project_id="proj-1",
            filename=f"{doc_id}.md",
            original_path=f"/tmp/{doc_id}.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        for i, status in enumerate(chunk_statuses):
            chunk = ChunkTable(
                document_id=doc_id,
                content=f"Chunk {i}",
                position=i,
                status=status,
            )
            session.add(chunk)
        await session.commit()


@pytest.mark.asyncio
async def test_status_worker_all_ready(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-1", ["ready", "ready", "ready"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-1")
        )).scalar_one()
        assert doc.status == "ready"
        assert doc.indexed_at is not None


@pytest.mark.asyncio
async def test_status_worker_partial(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-2", ["ready", "corrupted", "error"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-2")
        )).scalar_one()
        assert doc.status == "partial"
        assert doc.indexed_at is not None


@pytest.mark.asyncio
async def test_status_worker_still_processing(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-3", ["ready", "pending", "processing"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-3")
        )).scalar_one()
        assert doc.status == "indexing"  # unchanged
        assert doc.indexed_at is None


@pytest.mark.asyncio
async def test_status_worker_ignores_non_indexing_docs(settings, db_session):
    # Create a "ready" document — should not be touched
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-4",
            project_id="proj-1",
            filename="done.md",
            original_path="/tmp/done.md",
            file_type="markdown",
            status="ready",
        )
        session.add(doc)
        await session.commit()

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-4")
        )).scalar_one()
        assert doc.status == "ready"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_status_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mirage.indexer.status_worker'`

**Step 3: Write minimal implementation**

Create `src/mirage/indexer/status_worker.py`:

```python
import asyncio
import logging
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, get_engine

logger = logging.getLogger(__name__)


class StatusWorker:
    """Polls indexing documents and updates their status based on chunk statuses."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def check_documents(self, session: AsyncSession) -> None:
        result = await session.execute(
            select(DocumentTable).where(DocumentTable.status == "indexing")
        )
        docs = result.scalars().all()

        for doc in docs:
            counts = await session.execute(
                select(
                    ChunkTable.status,
                    func.count().label("cnt"),
                )
                .where(ChunkTable.document_id == doc.id)
                .group_by(ChunkTable.status)
            )
            status_counts = {row[0]: row[1] for row in counts.fetchall()}

            pending = status_counts.get("pending", 0)
            processing = status_counts.get("processing", 0)

            if pending > 0 or processing > 0:
                continue  # still working

            ready = status_counts.get("ready", 0)
            corrupted = status_counts.get("corrupted", 0)
            error = status_counts.get("error", 0)

            if corrupted == 0 and error == 0 and ready > 0:
                doc.status = "ready"
            else:
                doc.status = "partial"

            doc.indexed_at = datetime.utcnow()
            logger.info(
                f"Document {doc.filename}: status={doc.status} "
                f"(ready={ready}, corrupted={corrupted}, error={error})"
            )

        await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("StatusWorker started")

        while True:
            async with async_session() as session:
                await self.check_documents(session)
            await asyncio.sleep(10)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexer/test_status_worker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/indexer/status_worker.py tests/indexer/test_status_worker.py
git commit -m "feat: add StatusWorker for document status computation"
```

---

### Task CSL-7: API — add chunk progress to DocumentResponse

**Files:**
- Modify: `src/mirage/api/schemas.py:18-29`
- Modify: `src/mirage/api/routers/documents.py`
- Test: `tests/api/test_documents.py`

**Step 1: Write the failing test**

Add to `tests/api/test_documents.py`:

```python
@pytest.mark.asyncio
async def test_document_status_includes_chunk_counts(test_db, override_settings):
    # Create document with chunks in various statuses
    engine = test_db
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        doc = DocumentTable(
            id="doc-with-chunks",
            project_id="test-project-id",
            filename="chunked.md",
            original_path="/tmp/chunked.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        for i, status in enumerate(["ready", "ready", "pending", "processing"]):
            chunk = ChunkTable(
                document_id="doc-with-chunks",
                content=f"Chunk {i}",
                position=i,
                status=status,
            )
            session.add(chunk)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/projects/test-project-id/documents/doc-with-chunks",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_total"] == 4
    assert data["chunks_processed"] == 2  # only "ready" chunks are processed (not pending/processing)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_documents.py::test_document_status_includes_chunk_counts -v`
Expected: FAIL — `chunks_total` not in response

**Step 3: Write minimal implementation**

In `src/mirage/api/schemas.py`, add to `DocumentResponse`:

```python
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
    chunks_total: int | None = None
    chunks_processed: int | None = None
```

In `src/mirage/api/routers/documents.py`, update the `get_document` endpoint to query chunk counts:

```python
from sqlalchemy import func

from mirage.shared.db import ChunkTable, DocumentTable, IndexingTaskTable, ProjectTable


@router.get("/{document_id}")
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

    # Count chunks
    total_result = await db.execute(
        select(func.count()).select_from(ChunkTable).where(
            ChunkTable.document_id == document_id
        )
    )
    total = total_result.scalar() or 0

    processed_result = await db.execute(
        select(func.count()).select_from(ChunkTable).where(
            ChunkTable.document_id == document_id,
            ChunkTable.status.not_in(["pending", "processing"]),
        )
    )
    processed = processed_result.scalar() or 0

    return DocumentResponse(
        id=doc.id,
        project_id=doc.project_id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status,
        error_message=doc.error_message,
        metadata=doc.metadata_json,
        created_at=doc.created_at,
        indexed_at=doc.indexed_at,
        chunks_total=total if total > 0 else None,
        chunks_processed=processed if total > 0 else None,
    )
```

Update `list_documents` similarly — use a single query with LEFT JOIN and GROUP BY:

```python
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

    docs_result = await db.execute(
        select(DocumentTable).where(DocumentTable.project_id == project_id)
    )
    docs = docs_result.scalars().all()

    # Get chunk counts per document in one query
    if docs:
        doc_ids = [d.id for d in docs]
        counts_result = await db.execute(
            select(
                ChunkTable.document_id,
                func.count().label("total"),
                func.count().filter(
                    ChunkTable.status.not_in(["pending", "processing"])
                ).label("processed"),
            )
            .where(ChunkTable.document_id.in_(doc_ids))
            .group_by(ChunkTable.document_id)
        )
        counts = {row[0]: (row[1], row[2]) for row in counts_result.fetchall()}
    else:
        counts = {}

    return [
        DocumentResponse(
            id=doc.id,
            project_id=doc.project_id,
            filename=doc.filename,
            file_type=doc.file_type,
            status=doc.status,
            error_message=doc.error_message,
            metadata=doc.metadata_json,
            created_at=doc.created_at,
            indexed_at=doc.indexed_at,
            chunks_total=counts.get(doc.id, (None, None))[0],
            chunks_processed=counts.get(doc.id, (None, None))[1],
        )
        for doc in docs
    ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_documents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/api/schemas.py src/mirage/api/routers/documents.py tests/api/test_documents.py
git commit -m "feat: add chunk progress counts to document API responses"
```

---

### Task CSL-8: CLI — display chunk progress

**Files:**
- Modify: `src/mirage/cli/commands/documents.py:85-104` (status command)
- Modify: `src/mirage/cli/commands/documents.py:15-35` (list command)
- Test: `tests/cli/test_documents_cmd.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_documents_cmd.py`:

```python
def test_documents_status_shows_chunk_progress(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "doc-1",
        "filename": "test.pdf",
        "status": "indexing",
        "file_type": "pdf",
        "chunks_total": 100,
        "chunks_processed": 42,
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            app, ["documents", "status", "--project", "test-project", "doc-1"]
        )

    assert result.exit_code == 0
    assert "42/100" in result.stdout
    assert "42%" in result.stdout


def test_documents_list_shows_progress(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "doc-1",
            "filename": "book.pdf",
            "status": "indexing",
            "chunks_total": 128,
            "chunks_processed": 42,
        },
        {
            "id": "doc-2",
            "filename": "notes.md",
            "status": "ready",
            "chunks_total": 15,
            "chunks_processed": 15,
        },
    ]

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            app, ["documents", "list", "--project", "test-project"]
        )

    assert result.exit_code == 0
    assert "42/128" in result.stdout
    assert "15/15" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_documents_cmd.py::test_documents_status_shows_chunk_progress -v`
Expected: FAIL — output doesn't contain chunk info

**Step 3: Write minimal implementation**

In `src/mirage/cli/commands/documents.py`, update `document_status`:

```python
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
    if doc.get("chunks_total"):
        total = doc["chunks_total"]
        processed = doc.get("chunks_processed", 0)
        pct = round(processed / total * 100) if total > 0 else 0
        typer.echo(f"Chunks:   {processed}/{total} ({pct}%)")
    if doc.get("error_message"):
        typer.echo(f"Error:    {doc['error_message']}")
```

Update `list_documents`:

```python
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

    typer.echo(f"{'ID':<40} {'Filename':<30} {'Status':<10} {'Progress':<15}")
    typer.echo("-" * 95)
    for doc in docs:
        total = doc.get("chunks_total")
        processed = doc.get("chunks_processed", 0)
        if total:
            pct = round(processed / total * 100) if total > 0 else 0
            progress = f"{processed}/{total} ({pct}%)"
        else:
            progress = ""
        typer.echo(f"{doc['id']:<40} {doc['filename']:<30} {doc['status']:<10} {progress:<15}")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_documents_cmd.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/cli/commands/documents.py tests/cli/test_documents_cmd.py
git commit -m "feat: display chunk progress in CLI documents commands"
```

---

### Task CSL-9: Final — run all tests and fix regressions

**Step 1: Run the full test suite**

Run: `uv run pytest -v`

**Step 2: Fix any import errors**

The old `IndexerWorker` class is gone. If any test or module imports it, update to `ChunkWorker`. Check:
- `tests/indexer/test_worker.py` (already updated in Task 4)
- `src/mirage/indexer/__init__.py` (if it re-exports)
- Any `__main__` block in `worker.py` (already updated in Task 4)

**Step 3: Fix any failing tests**

- `tests/shared/test_config.py::test_settings_defaults` may still expect `chunk_size == 128` — update to `400`
- `tests/shared/test_embedding.py` was rewritten in Task 2 — verify no leftover tests reference old API

**Step 4: Run tests again**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -u
git commit -m "fix: resolve test regressions from worker pipeline refactor"
```
