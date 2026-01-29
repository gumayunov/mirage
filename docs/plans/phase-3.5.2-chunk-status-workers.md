# Phase 3.5.2: Chunk Status — Worker Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor monolithic `IndexerWorker` into three independent workers: ChunkWorker (parse), EmbeddingWorker (embed), StatusWorker (aggregate status).

**Parent design:** `docs/plans/2026-01-29-chunk-status-design.md`

**Phases:**
- **Phase 3.5.1 — Foundation:** Schema, embedding client, config
- **Phase 3.5.2 — Worker Pipeline** (this file): ChunkWorker, EmbeddingWorker, StatusWorker
- **Phase 3.5.3 — Visibility:** API and CLI progress display, regression fixes

**Depends on:** Phase 3.5.1 (status column, EmbeddingResult, chunk_size=400).

**Breaking change:** CSL-4 removes `IndexerWorker` — existing worker tests will break and must be updated within this phase.

**Tech Stack:** Python 3.14, SQLAlchemy + aiosqlite, FastAPI, httpx, typer

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
