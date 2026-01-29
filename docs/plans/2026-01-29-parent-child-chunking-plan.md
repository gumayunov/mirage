# Parent-Child Chunking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split documents into two levels of chunks — large parent chunks (context) and small child chunks (embeddings) — to improve search precision while returning full context.

**Architecture:** Single `chunks` table with self-referencing FK `parent_id`. Parent chunks have `status="parent"` and no embeddings. Child chunks have embeddings and `parent_id` pointing to their parent. Search queries find child chunks by embedding similarity, then JOIN parent chunks for context. Deduplication ensures one result per parent.

**Tech Stack:** SQLAlchemy (self-referencing FK), pgvector (unchanged), tiktoken (chunking)

---

## Phase 3.6.1: Config & Data Model

**Files:**
- Modify: `src/mirage/shared/config.py`
- Modify: `src/mirage/shared/db.py`
- Modify: `src/mirage/shared/models.py`
- Modify: `tests/shared/test_config.py`
- Modify: `tests/shared/test_models.py`

---

### Task 1: Add child chunk config fields

**Files:**
- Modify: `src/mirage/shared/config.py:9-10`
- Modify: `tests/shared/test_config.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_config.py`:

```python
def test_settings_child_chunk_defaults(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    settings = Settings()
    assert settings.chunk_size == 3000
    assert settings.chunk_overlap == 200
    assert settings.child_chunk_size == 500
    assert settings.child_chunk_overlap == 50
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_config.py::test_settings_child_chunk_defaults -v`
Expected: FAIL — `settings.chunk_size == 1500` (not 3000), and `child_chunk_size` attribute missing.

**Step 3: Write minimal implementation**

In `src/mirage/shared/config.py`, change lines 9-10 and add new fields:

```python
class Settings(BaseSettings):
    database_url: str
    api_key: str
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "nomic-embed-text"
    chunk_size: int = 3000
    chunk_overlap: int = 200
    child_chunk_size: int = 500
    child_chunk_overlap: int = 50
    documents_path: str = "/data/documents"
    log_level: str = "INFO"

    model_config = {"env_prefix": "MIRAGE_"}
```

**Step 4: Fix existing tests**

Update `tests/shared/test_config.py`:
- `test_settings_defaults`: change `assert settings.chunk_size == 1500` → `assert settings.chunk_size == 3000`
- `test_settings_chunk_size_default`: change `assert settings.chunk_size == 1500` → `assert settings.chunk_size == 3000`

**Step 5: Run all config tests**

Run: `uv run pytest tests/shared/test_config.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/mirage/shared/config.py tests/shared/test_config.py
git commit -m "feat: add child_chunk_size and child_chunk_overlap config fields"
```

---

### Task 2: Add parent_id to ChunkTable

**Files:**
- Modify: `src/mirage/shared/db.py:43-55`
- Modify: `tests/shared/test_models.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_models.py`:

```python
def test_chunk_model_with_parent():
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Child content",
        embedding=[0.1] * 768,
        position=0,
        structure={"chapter": "Test"},
        parent_id=uuid.uuid4(),
    )
    assert chunk.parent_id is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_models.py::test_chunk_model_with_parent -v`
Expected: FAIL — `parent_id` not recognized.

**Step 3: Write minimal implementation**

In `src/mirage/shared/models.py`, add `parent_id` to the `Chunk` model (line 42-49):

```python
class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID
    content: str
    embedding: list[float]
    position: int
    structure: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: uuid.UUID | None = None
```

In `src/mirage/shared/db.py`, add `parent_id` column and relationships to `ChunkTable` (lines 43-55):

```python
class ChunkTable(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(768), nullable=True)
    position: Mapped[int] = mapped_column(Integer)
    structure_json: Mapped[str | None] = mapped_column("structure", JSON, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    parent_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True)

    document: Mapped["DocumentTable"] = relationship(back_populates="chunks")
    children: Mapped[list["ChunkTable"]] = relationship(back_populates="parent", cascade="all, delete-orphan")
    parent: Mapped["ChunkTable | None"] = relationship(back_populates="children", remote_side=[id])
```

**Step 4: Run tests**

Run: `uv run pytest tests/shared/test_models.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/db.py src/mirage/shared/models.py tests/shared/test_models.py
git commit -m "feat: add parent_id self-referencing FK to ChunkTable"
```

---

## Phase 3.6.2: Chunking Pipeline

**Files:**
- Modify: `src/mirage/indexer/chunking.py`
- Modify: `tests/indexer/test_chunking.py`

---

### Task 3: Add chunk_children() method to Chunker

**Files:**
- Modify: `src/mirage/indexer/chunking.py`
- Modify: `tests/indexer/test_chunking.py`

**Step 1: Write the failing test**

Add to `tests/indexer/test_chunking.py`:

```python
def test_chunk_children_splits_parent():
    """chunk_children splits a parent chunk into smaller child chunks."""
    chunker = Chunker(chunk_size=3000, overlap=200)
    # Create a parent text long enough to produce multiple children at 500 tokens
    parent_text = "This is a sentence about programming. " * 200  # ~1000 tokens
    structure = {"chapter": "Test"}

    children = chunker.chunk_children(parent_text, structure, child_size=500, child_overlap=50)

    assert len(children) >= 2
    for child in children:
        assert child.content
        assert child.structure == structure


def test_chunk_children_short_text():
    """Short parent text produces a single child chunk."""
    chunker = Chunker(chunk_size=3000, overlap=200)
    structure = {"chapter": "Intro"}

    children = chunker.chunk_children("Short text.", structure, child_size=500, child_overlap=50)

    assert len(children) == 1
    assert children[0].content == "Short text."
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_chunking.py::test_chunk_children_splits_parent -v`
Expected: FAIL — `chunk_children` method does not exist.

**Step 3: Write minimal implementation**

In `src/mirage/indexer/chunking.py`, add the `chunk_children` method to `Chunker` class after `chunk_text`:

```python
def chunk_children(self, text: str, structure: dict[str, Any], child_size: int = 500, child_overlap: int = 50) -> list[Chunk]:
    """Split a parent chunk text into smaller child chunks."""
    if not text.strip():
        return []

    if self._count_tokens(text) <= child_size:
        return [Chunk(content=text.strip(), position=0, structure=structure)]

    child_chunker = Chunker(chunk_size=child_size, overlap=child_overlap)
    return child_chunker.chunk_text(text, structure)
```

**Step 4: Run tests**

Run: `uv run pytest tests/indexer/test_chunking.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/indexer/chunking.py tests/indexer/test_chunking.py
git commit -m "feat: add chunk_children() method for two-level chunking"
```

---

## Phase 3.6.3: Worker Pipeline

**Files:**
- Modify: `src/mirage/indexer/worker.py`
- Modify: `tests/indexer/test_worker.py` (if exists, otherwise `tests/indexer/test_embedding_worker.py`)

---

### Task 4: Update ChunkWorker for two-level chunking

**Files:**
- Modify: `src/mirage/indexer/worker.py:20-91`

**Step 1: Write the failing test**

Create or add to `tests/indexer/test_worker.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, IndexingTaskTable, ProjectTable
from mirage.indexer.worker import ChunkWorker


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
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
        doc = DocumentTable(
            id="doc-1",
            project_id="proj-1",
            filename="test.md",
            original_path="/tmp/test.md",
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

    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_chunk_worker_creates_parent_and_child_chunks(settings, db_session, tmp_path):
    # Create a markdown file long enough to produce multiple parent chunks
    content = "# Test\n\n" + ("This is a test paragraph with enough text. " * 100 + "\n\n") * 20
    md_file = tmp_path / "test.md"
    md_file.write_text(content)

    # Update doc path
    async with db_session() as session:
        doc = (await session.execute(select(DocumentTable).where(DocumentTable.id == "doc-1"))).scalar_one()
        doc.original_path = str(md_file)
        task = (await session.execute(select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1"))).scalar_one()
        await session.commit()

    worker = ChunkWorker(settings)

    async with db_session() as session:
        task = (await session.execute(select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1"))).scalar_one()
        await worker.process_task(session, task)

    async with db_session() as session:
        all_chunks = (await session.execute(select(ChunkTable))).scalars().all()

        parents = [c for c in all_chunks if c.parent_id is None]
        children = [c for c in all_chunks if c.parent_id is not None]

        assert len(parents) > 0, "Should have parent chunks"
        assert len(children) > 0, "Should have child chunks"

        # Parents should have status="parent"
        for p in parents:
            assert p.status == "parent"
            assert p.embedding is None

        # Children should have status="pending" and valid parent_id
        parent_ids = {p.id for p in parents}
        for c in children:
            assert c.status == "pending"
            assert c.parent_id in parent_ids
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_worker.py::test_chunk_worker_creates_parent_and_child_chunks -v`
Expected: FAIL — worker creates flat chunks without parent_id.

**Step 3: Write minimal implementation**

Update `src/mirage/indexer/worker.py`:

1. Constructor (lines 23-28) — add child chunker params:

```python
def __init__(self, settings: Settings):
    self.settings = settings
    self.chunker = Chunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    self.child_chunk_size = settings.child_chunk_size
    self.child_chunk_overlap = settings.child_chunk_overlap
    self.parsers = {
        "markdown": MarkdownParser(),
        "pdf": PDFParser(),
        "epub": EPUBParser(),
    }
```

2. `_parse_file` return type stays `list[dict]` but now returns parent chunks. No change needed to parsing logic itself — `chunk_text()` already produces parent-sized chunks (3000 tokens from config).

3. `process_task` method (lines 119-136) — save parents first, then children:

Replace chunk creation loop (lines 122-130) with:

```python
# Parse file — produces parent-level chunks
chunks_data = self._parse_file(doc.original_path, doc.file_type)

parent_chunks = []
for chunk_data in chunks_data:
    parent = ChunkTable(
        document_id=doc.id,
        content=chunk_data["content"],
        position=chunk_data["position"],
        structure_json=chunk_data["structure"],
        status="parent",
    )
    session.add(parent)
    parent_chunks.append((parent, chunk_data))

await session.flush()  # generate parent IDs

# Create child chunks for each parent
child_count = 0
for parent, chunk_data in parent_chunks:
    children = self.chunker.chunk_children(
        parent.content,
        chunk_data["structure"],
        child_size=self.child_chunk_size,
        child_overlap=self.child_chunk_overlap,
    )
    for child in children:
        child_row = ChunkTable(
            document_id=doc.id,
            content=child.content,
            position=child.position,
            structure_json=child.structure,
            status="pending",
            parent_id=parent.id,
        )
        session.add(child_row)
        child_count += 1

task.status = "done"
task.completed_at = datetime.utcnow()

await session.commit()
logger.info(f"Created {len(parent_chunks)} parents, {child_count} children for {doc.filename}")
```

**Step 4: Run tests**

Run: `uv run pytest tests/indexer/test_worker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/indexer/worker.py tests/indexer/test_worker.py
git commit -m "feat: ChunkWorker creates parent + child chunks"
```

---

## Phase 3.6.4: Search with Parent Context

**Files:**
- Modify: `src/mirage/api/schemas.py:41-46`
- Modify: `src/mirage/api/routers/search.py:45-79`
- Modify: `tests/api/test_search.py`

---

### Task 5: Add parent_content to ChunkResult schema

**Files:**
- Modify: `src/mirage/api/schemas.py:41-46`

**Step 1: Write minimal implementation**

In `src/mirage/api/schemas.py`, add `parent_content` field to `ChunkResult`:

```python
class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    parent_content: str | None = None
    score: float
    structure: dict | None = None
    document: dict
```

No separate test needed — this is a schema field with a default, tested via the search endpoint.

**Step 2: Commit**

```bash
git add src/mirage/api/schemas.py
git commit -m "feat: add parent_content field to ChunkResult schema"
```

---

### Task 6: Update search query to JOIN parent and deduplicate

**Files:**
- Modify: `src/mirage/api/routers/search.py:45-79`
- Modify: `tests/api/test_search.py`

**Step 1: Write the failing test**

Add to `tests/api/test_search.py`:

```python
@pytest.fixture
async def test_db_with_parent_child():
    """DB fixture with parent+child chunk structure."""
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

        parent = ChunkTable(
            id="parent-chunk-id",
            document_id="test-doc-id",
            content="Full context paragraph about Python and its ecosystem.",
            position=0,
            status="parent",
            structure_json={"chapter": "Introduction"},
        )
        session.add(parent)
        await session.flush()

        child = ChunkTable(
            id="child-chunk-id",
            document_id="test-doc-id",
            content="Python programming language.",
            embedding=[0.1] * 768,
            position=0,
            status="ready",
            structure_json={"chapter": "Introduction"},
            parent_id="parent-chunk-id",
        )
        session.add(child)
        await session.commit()

    yield engine

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_search_returns_parent_content(test_db_with_parent_child, override_settings, mock_embedding):
    """Search results include parent_content from the parent chunk."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects/test-project-id/search",
            json={"query": "Python", "limit": 10},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    # SQLite fallback won't do vector search, but test the response shape
    assert "results" in data
    for result in data["results"]:
        assert "parent_content" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_search.py::test_search_returns_parent_content -v`
Expected: FAIL — `parent_content` not in response / fixture issues.

**Step 3: Write minimal implementation**

Update `src/mirage/api/routers/search.py`:

1. pgvector SQL query (lines 45-53) — add JOIN on parent, filter child chunks only, deduplicate by parent_id:

```python
sql = text("""
    SELECT DISTINCT ON (child.parent_id)
           child.id, child.content, child.structure,
           child.embedding <=> :embedding AS distance,
           parent.content AS parent_content,
           d.id as doc_id, d.filename
    FROM chunks child
    JOIN chunks parent ON child.parent_id = parent.id
    JOIN documents d ON child.document_id = d.id
    WHERE d.project_id = :project_id
      AND d.status = 'ready'
      AND child.parent_id IS NOT NULL
    ORDER BY child.parent_id, child.embedding <=> :embedding
    LIMIT :limit
""")
```

Note: `DISTINCT ON` is PostgreSQL-specific. For the SQLite fallback, deduplication is done in Python.

2. Result mapping (lines 65-79) — add `parent_content`:

```python
results = []
for row in rows:
    score = 1 - row.distance
    if score >= request.threshold:
        results.append(
            ChunkResult(
                chunk_id=row.id,
                content=row.content,
                parent_content=row.parent_content,
                score=score,
                structure=row.structure,
                document={"id": row.doc_id, "filename": row.filename},
            )
        )
```

3. SQLite fallback (lines 86-111) — filter children and add parent_content:

```python
except Exception:
    logger.exception("pgvector query failed, falling back to SQLite")
    result = await db.execute(
        select(ChunkTable, DocumentTable)
        .join(DocumentTable)
        .where(
            DocumentTable.project_id == project_id,
            DocumentTable.status == "ready",
            ChunkTable.parent_id.is_not(None),
        )
        .limit(request.limit)
    )
    rows = result.all()
    logger.info("Fallback returned %d rows", len(rows))

    # Deduplicate by parent_id: keep first per parent
    seen_parents: set[str] = set()
    results = []
    for chunk, doc in rows:
        if chunk.parent_id in seen_parents:
            continue
        seen_parents.add(chunk.parent_id)

        # Load parent content
        parent_result = await db.execute(
            select(ChunkTable.content).where(ChunkTable.id == chunk.parent_id)
        )
        parent_content = parent_result.scalar_one_or_none()

        results.append(
            ChunkResult(
                chunk_id=chunk.id,
                content=chunk.content,
                parent_content=parent_content,
                score=1.0,
                structure=chunk.structure_json,
                document={"id": doc.id, "filename": doc.filename},
            )
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/api/test_search.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/api/routers/search.py tests/api/test_search.py
git commit -m "feat: search returns parent_content, deduplicates by parent"
```

---

## Phase 3.6.5: Document Chunk Counts & Status Worker

**Files:**
- Modify: `src/mirage/api/routers/documents.py`
- Modify: `src/mirage/indexer/status_worker.py`
- Modify: `tests/api/test_documents.py`
- Modify: `tests/indexer/test_embedding_worker.py`

---

### Task 7: Filter parent chunks from document chunk counts

**Files:**
- Modify: `src/mirage/api/routers/documents.py:44-56, 172-195`
- Modify: `tests/api/test_documents.py`

**Step 1: Write the failing test**

Add to `tests/api/test_documents.py`:

```python
@pytest.mark.asyncio
async def test_document_chunk_counts_exclude_parents(test_db, override_settings):
    """Chunk counts should only count child chunks, not parent chunks."""
    engine = test_db
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        doc = DocumentTable(
            id="doc-parent-child",
            project_id="test-project-id",
            filename="pc.md",
            original_path="/tmp/pc.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)

        parent = ChunkTable(
            id="p1",
            document_id="doc-parent-child",
            content="Parent content",
            position=0,
            status="parent",
        )
        session.add(parent)

        for i, s in enumerate(["ready", "pending"]):
            child = ChunkTable(
                document_id="doc-parent-child",
                content=f"Child {i}",
                position=i,
                status=s,
                parent_id="p1",
            )
            session.add(child)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/projects/test-project-id/documents/doc-parent-child",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    # Should count only children (2), not the parent
    assert data["chunks_total"] == 2
    assert data["chunks_processed"] == 1
    assert "parent" not in data["chunks_by_status"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_documents.py::test_document_chunk_counts_exclude_parents -v`
Expected: FAIL — counts include parent chunk (total=3).

**Step 3: Write minimal implementation**

In `src/mirage/api/routers/documents.py`:

Add `ChunkTable.parent_id.is_not(None)` filter to all chunk count queries:

For `list_documents` (lines 45-55):

```python
counts_result = await db.execute(
    select(
        ChunkTable.document_id,
        func.count().label("total"),
        func.count().filter(
            ChunkTable.status.not_in(["pending", "processing"])
        ).label("processed"),
    )
    .where(
        ChunkTable.document_id.in_(doc_ids),
        ChunkTable.parent_id.is_not(None),
    )
    .group_by(ChunkTable.document_id)
)
```

For `list_documents` status breakdown (lines 59-67):

```python
status_result = await db.execute(
    select(
        ChunkTable.document_id,
        ChunkTable.status,
        func.count().label("cnt"),
    )
    .where(
        ChunkTable.document_id.in_(doc_ids),
        ChunkTable.parent_id.is_not(None),
    )
    .group_by(ChunkTable.document_id, ChunkTable.status)
)
```

For `get_document` total count (lines 172-177):

```python
total_result = await db.execute(
    select(func.count()).select_from(ChunkTable).where(
        ChunkTable.document_id == document_id,
        ChunkTable.parent_id.is_not(None),
    )
)
```

For `get_document` processed count (lines 179-185):

```python
processed_result = await db.execute(
    select(func.count()).select_from(ChunkTable).where(
        ChunkTable.document_id == document_id,
        ChunkTable.parent_id.is_not(None),
        ChunkTable.status.not_in(["pending", "processing"]),
    )
)
```

For `get_document` status breakdown (lines 190-195):

```python
status_result = await db.execute(
    select(ChunkTable.status, func.count().label("cnt"))
    .where(
        ChunkTable.document_id == document_id,
        ChunkTable.parent_id.is_not(None),
    )
    .group_by(ChunkTable.status)
)
```

**Step 4: Run tests**

Run: `uv run pytest tests/api/test_documents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/api/routers/documents.py tests/api/test_documents.py
git commit -m "feat: chunk counts filter parent chunks (count children only)"
```

---

### Task 8: Update StatusWorker to count only child chunks

**Files:**
- Modify: `src/mirage/indexer/status_worker.py:27-35`

**Step 1: Write the failing test**

Create or add to `tests/indexer/test_status_worker.py`:

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
        documents_path="/tmp/docs",
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_status_worker_ignores_parent_chunks(settings, db_session):
    """StatusWorker should only consider child chunks for document readiness."""
    async with db_session() as session:
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

        # Parent chunk with status="parent" — should not prevent "ready"
        parent = ChunkTable(
            id="parent-1",
            document_id="doc-1",
            content="Parent",
            position=0,
            status="parent",
        )
        session.add(parent)

        # All children are ready
        for i in range(2):
            child = ChunkTable(
                document_id="doc-1",
                content=f"Child {i}",
                position=i,
                status="ready",
                parent_id="parent-1",
            )
            session.add(child)
        await session.commit()

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(select(DocumentTable).where(DocumentTable.id == "doc-1"))).scalar_one()
        assert doc.status == "ready"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexer/test_status_worker.py::test_status_worker_ignores_parent_chunks -v`
Expected: FAIL — StatusWorker sees `status="parent"` as neither ready/corrupted/error, causing it to not update document status (or stay in "indexing" because "parent" status is unrecognized).

**Step 3: Write minimal implementation**

In `src/mirage/indexer/status_worker.py`, add filter for child chunks only (lines 27-34):

```python
counts = await session.execute(
    select(
        ChunkTable.status,
        func.count().label("cnt"),
    )
    .where(
        ChunkTable.document_id == doc.id,
        ChunkTable.parent_id.is_not(None),
    )
    .group_by(ChunkTable.status)
)
```

**Step 4: Run tests**

Run: `uv run pytest tests/indexer/test_status_worker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/indexer/status_worker.py tests/indexer/test_status_worker.py
git commit -m "feat: StatusWorker counts only child chunks for document readiness"
```

---

### Task 9: Update test fixtures with parent_id

**Files:**
- Modify: `tests/indexer/test_embedding_worker.py`
- Modify: `tests/api/test_documents.py`

**Step 1: Update embedding worker fixture**

In `tests/indexer/test_embedding_worker.py`, update the fixture to create a proper parent+child structure. The chunk should have a `parent_id` so the embedding worker processes children:

```python
# In db_session fixture, after creating doc:
parent = ChunkTable(
    id="parent-1",
    document_id="doc-1",
    content="Parent context",
    position=0,
    status="parent",
)
session.add(parent)
chunk = ChunkTable(
    id="chunk-1",
    document_id="doc-1",
    content="Some test content",
    position=0,
    status="pending",
    parent_id="parent-1",
)
session.add(chunk)
```

**Step 2: Update document test fixtures**

In `tests/api/test_documents.py`, update `test_document_status_includes_chunk_counts` and `test_list_documents_includes_chunks_by_status` to create proper parent+child structures. Chunks need `parent_id`:

For `test_document_status_includes_chunk_counts`:
```python
parent = ChunkTable(
    id="parent-for-doc-with-chunks",
    document_id="doc-with-chunks",
    content="Parent",
    position=0,
    status="parent",
)
session.add(parent)
for i, status in enumerate(["ready", "ready", "pending", "processing"]):
    chunk = ChunkTable(
        document_id="doc-with-chunks",
        content=f"Chunk {i}",
        position=i,
        status=status,
        parent_id="parent-for-doc-with-chunks",
    )
    session.add(chunk)
```

For `test_list_documents_includes_chunks_by_status`:
```python
parent = ChunkTable(
    id="parent-for-list-chunks",
    document_id="doc-list-chunks",
    content="Parent",
    position=0,
    status="parent",
)
session.add(parent)
for i, chunk_status in enumerate(["ready", "ready", "ready", "error"]):
    chunk = ChunkTable(
        document_id="doc-list-chunks",
        content=f"Chunk {i}",
        position=i,
        status=chunk_status,
        parent_id="parent-for-list-chunks",
    )
    session.add(chunk)
```

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/indexer/test_embedding_worker.py tests/api/test_documents.py tests/api/test_search.py
git commit -m "fix: update test fixtures for parent-child chunk structure"
```

---

## Phase 3.6.6: Progress Tracking Update

### Task 10: Update progress.md

**Files:**
- Modify: `docs/plans/progress.md`

**Step 1: Add Phase 3.6 section to progress.md**

After "Phase 3.5.3" section, add:

```markdown
## Phase 3.6.1: Parent-Child Chunking — Config & Data Model
- [x] Task 1: Add child chunk config fields
- [x] Task 2: Add parent_id to ChunkTable

## Phase 3.6.2: Parent-Child Chunking — Chunking Pipeline
- [x] Task 3: Add chunk_children() method to Chunker

## Phase 3.6.3: Parent-Child Chunking — Worker Pipeline
- [x] Task 4: Update ChunkWorker for two-level chunking

## Phase 3.6.4: Parent-Child Chunking — Search
- [x] Task 5: Add parent_content to ChunkResult schema
- [x] Task 6: Update search query with parent JOIN and deduplication

## Phase 3.6.5: Parent-Child Chunking — Counts & Status
- [x] Task 7: Filter parent chunks from document chunk counts
- [x] Task 8: Update StatusWorker to count only child chunks
- [x] Task 9: Update test fixtures with parent_id
```

**Step 2: Commit**

```bash
git add docs/plans/progress.md
git commit -m "docs: mark parent-child chunking tasks complete"
```

---

## Summary of Changes by File

| File | Phase | Change |
|---|---|---|
| `src/mirage/shared/config.py` | 3.6.1 | `chunk_size=3000`, add `child_chunk_size=500`, `child_chunk_overlap=50` |
| `src/mirage/shared/db.py` | 3.6.1 | `parent_id` FK, `children`/`parent` relationships on ChunkTable |
| `src/mirage/shared/models.py` | 3.6.1 | `parent_id` field on Chunk pydantic model |
| `src/mirage/indexer/chunking.py` | 3.6.2 | New `chunk_children()` method |
| `src/mirage/indexer/worker.py` | 3.6.3 | Two-level chunking: parents first, then children |
| `src/mirage/api/schemas.py` | 3.6.4 | `parent_content` in ChunkResult |
| `src/mirage/api/routers/search.py` | 3.6.4 | JOIN parent, dedup by parent_id |
| `src/mirage/api/routers/documents.py` | 3.6.5 | Filter `parent_id IS NOT NULL` in counts |
| `src/mirage/indexer/status_worker.py` | 3.6.5 | Count only child chunks |
| `tests/shared/test_config.py` | 3.6.1 | New defaults |
| `tests/shared/test_models.py` | 3.6.1 | parent_id in model |
| `tests/indexer/test_chunking.py` | 3.6.2 | chunk_children tests |
| `tests/indexer/test_worker.py` | 3.6.3 | Parent+child creation tests |
| `tests/api/test_search.py` | 3.6.4 | Parent+child fixture, parent_content verification |
| `tests/api/test_documents.py` | 3.6.5 | Fixtures with parent_id |
| `tests/indexer/test_embedding_worker.py` | 3.6.5 | Fixture with parent_id |
| `tests/indexer/test_status_worker.py` | 3.6.5 | StatusWorker ignores parent chunks |
