# LLM-based Chunk Summarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured summary generation for chunks to improve search quality.

**Architecture:** SummaryWorker generates markdown summaries via Ollama mistral:7b. Both original and summary get embedded. Search deduplicates by chunk_id, returns original content.

**Tech Stack:** Ollama, SQLAlchemy, Alembic, FastAPI

---

## Phase T03-00: Database Schema

### Task T03-00-01: Add summary_text and update status to ChunkTable

**Files:**
- Modify: `src/mirage/shared/db.py:57-72`

**Step 1: Update ChunkTable model**

Add `summary_text` field and note status is now state machine:

```python
class ChunkTable(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
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

**Step 2: Commit**

```bash
git add src/mirage/shared/db.py
git commit -m "feat(db): add summary_text field to ChunkTable"
```

---

### Task T03-00-02: Add content_type to EmbeddingStatusTable

**Files:**
- Modify: `src/mirage/shared/db.py:46-55`

**Step 1: Update EmbeddingStatusTable**

Extend composite primary key with `content_type`:

```python
class EmbeddingStatusTable(Base):
    __tablename__ = "embedding_status"

    chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    content_type: Mapped[str] = mapped_column(String(50), primary_key=True, default="original")
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
```

**Step 2: Commit**

```bash
git add src/mirage/shared/db.py
git commit -m "feat(db): add content_type to EmbeddingStatusTable"
```

---

### Task T03-00-03: Update dynamic EmbeddingsTable with content_type

**Files:**
- Modify: `src/mirage/shared/db.py:93-112`

**Step 1: Update get_embeddings_table_class**

Add `content_type` to primary key:

```python
def get_embeddings_table_class(model: SupportedModel) -> Type:
    """Get or create an embeddings table class for a model."""
    table_name = get_model_table_name(model)

    if table_name in _embeddings_table_classes:
        return _embeddings_table_classes[table_name]

    class EmbeddingsTable(Base):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        chunk_id: Mapped[str] = mapped_column(
            String(36), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
        )
        content_type: Mapped[str] = mapped_column(String(50), primary_key=True, default="original")
        embedding = mapped_column(Vector(model.dimensions), nullable=False)

    EmbeddingsTable.__name__ = f"EmbeddingsTable_{model.table_alias.title()}"
    _embeddings_table_classes[table_name] = EmbeddingsTable

    return EmbeddingsTable
```

**Step 2: Commit**

```bash
git add src/mirage/shared/db.py
git commit -m "feat(db): add content_type to dynamic EmbeddingsTable"
```

---

### Task T03-00-04: Create Alembic migration

**Files:**
- Create: `src/mirage/migrations/versions/xxxx_add_summary_fields.py`

**Step 1: Generate migration**

```bash
make migration msg="add summary_text and content_type fields"
```

**Step 2: Edit migration file**

Migration should:
1. Add `summary_text` column to `chunks` table
2. Add `content_type` column to `embedding_status` table (default "original")
3. Add `content_type` column to each embeddings table
4. Update primary keys accordingly

**Step 3: Test migration locally**

```bash
make migrate
```

**Step 4: Commit**

```bash
git add src/mirage/migrations/versions/
git commit -m "feat(db): migration for summary fields"
```

---

## Phase T03-01: SummaryWorker

### Task T03-01-01: Create summary prompt template

**Files:**
- Create: `src/mirage/indexer/summary_prompt.py`

**Step 1: Create prompt template**

```python
SUMMARY_PROMPT = """Извлеки ключевую информацию из текста. Верни результат в формате markdown.

Если категория пуста — не добавляй её секцию.

Текст:
{text}

Формат ответа:
## Сущности
- имя1
- имя2

## Места
- место1

## Даты
- дата1

## Действия
- действие1

## Факты
- факт1

## Описания
- объект: описание

Ответ:"""
```

**Step 2: Commit**

```bash
git add src/mirage/indexer/summary_prompt.py
git commit -m "feat(indexer): add summary prompt template"
```

---

### Task T03-02-02: Implement Ollama client for summarization

**Files:**
- Create: `src/mirage/shared/llm_client.py`

**Step 1: Write the test**

```python
# tests/shared/test_llm_client.py
import pytest
from unittest.mock import AsyncMock, patch
from mirage.shared.llm_client import OllamaLLM

@pytest.mark.asyncio
async def test_generate_summary():
    client = OllamaLLM("http://localhost:11434", "mistral:7b")
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            json=lambda: {"message": {"content": "## Сущности\n- Алиса"}},
            raise_for_status=lambda: None
        )
        
        result = await client.generate("Test prompt")
        assert "Алиса" in result
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/shared/test_llm_client.py -v
```

**Step 3: Implement OllamaLLM client**

```python
# src/mirage/shared/llm_client.py
import httpx
import logging

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Client for Ollama chat completions."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str) -> str | None:
        """Generate text completion via Ollama API."""
        url = f"{self.base_url}/api/chat"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/shared/test_llm_client.py -v
```

**Step 5: Commit**

```bash
git add src/mirage/shared/llm_client.py tests/shared/test_llm_client.py
git commit -m "feat(shared): add OllamaLLM client for summarization"
```

---

### Task T03-01-03: Implement SummaryWorker with state machine

**Files:**
- Create: `src/mirage/indexer/summary_worker.py`
- Create: `tests/indexer/test_summary_worker.py`

**Step 1: Write the test**

```python
# tests/indexer/test_summary_worker.py
import pytest
from unittest.mock import AsyncMock, patch
from mirage.indexer.summary_worker import SummaryWorker
from mirage.shared.config import Settings


@pytest.mark.asyncio
async def test_summary_worker_processes_pending_chunk(test_db):
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
    worker = SummaryWorker(settings)
    
    # Setup: create chunk with status=pending
    
    with patch.object(worker, "_generate_summary", return_value="## Сущности\n- Алиса"):
        processed = await worker.process_one(session)
        
    assert processed is True
    # Verify chunk status changed to "summarized"
    # Verify summary_text populated
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_summary_worker.py -v
```

**Step 3: Implement SummaryWorker**

```python
# src/mirage/indexer/summary_worker.py
import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, EmbeddingStatusTable, get_engine
from mirage.shared.llm_client import OllamaLLM
from mirage.indexer.summary_prompt import SUMMARY_PROMPT
from mirage.shared.models_registry import get_all_models

logger = logging.getLogger(__name__)


@dataclass
class PendingChunk:
    chunk_id: str
    content: str
    ollama_url: str


class SummaryWorker:
    """Generates structured summaries for chunks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_model = getattr(settings, "summary_model", "mistral:7b")

    async def _generate_summary(self, content: str, ollama_url: str) -> str | None:
        """Call Ollama to generate structured summary."""
        client = OllamaLLM(ollama_url, self.llm_model)
        prompt = SUMMARY_PROMPT.format(text=content)
        return await client.generate(prompt)

    async def _claim_pending(self, session: AsyncSession) -> PendingChunk | None:
        """Find a chunk with status=pending."""
        result = await session.execute(
            select(ChunkTable)
            .where(ChunkTable.status == "pending")
            .limit(1)
        )
        chunk = result.scalar_one_or_none()
        if not chunk:
            return None

        chunk.status = "summarizing"
        await session.flush()

        # Get ollama_url from project
        from mirage.shared.db import DocumentTable, ProjectTable
        project_result = await session.execute(
            select(ProjectTable)
            .join(DocumentTable)
            .where(DocumentTable.id == chunk.document_id)
        )
        project = project_result.scalar_one_or_none()

        return PendingChunk(
            chunk_id=chunk.id,
            content=chunk.content,
            ollama_url=project.ollama_url if project else self.settings.ollama_url,
        )

    async def process_one(self, session: AsyncSession) -> bool:
        """Process a single chunk. Returns True if processed."""
        pending = await self._claim_pending(session)
        if not pending:
            return False

        chunk = await session.get(ChunkTable, pending.chunk_id)

        try:
            summary = await self._generate_summary(
                pending.content, 
                pending.ollama_url
            )
            
            if summary:
                chunk.summary_text = summary
                chunk.status = "summarized"
                
                # Create embedding_status for original and summary
                models = get_all_models()
                for model in models:
                    for content_type in ["original", "summary"]:
                        status_row = EmbeddingStatusTable(
                            chunk_id=chunk.id,
                            model_name=model.name,
                            content_type=content_type,
                            status="pending",
                        )
                        session.add(status_row)
            else:
                chunk.status = "error"
                chunk.error_message = "Summary generation failed"
                
            await session.commit()
            logger.info(f"Summarized chunk {chunk.id}")
            return True
            
        except Exception as e:
            chunk.status = "error"
            logger.error(f"Failed to summarize chunk {chunk.id}: {e}")
            await session.commit()
            return False

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("SummaryWorker started")

        while True:
            async with async_session() as session:
                processed = await self.process_one(session)

            if not processed:
                await asyncio.sleep(5)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_summary_worker.py -v
```

**Step 5: Commit**

```bash
git add src/mirage/indexer/summary_worker.py tests/indexer/test_summary_worker.py
git commit -m "feat(indexer): implement SummaryWorker"
```

---

### Task T03-01-04: Update indexer __main__.py to run SummaryWorker

**Files:**
- Modify: `src/mirage/indexer/__main__.py`

**Step 1: Add SummaryWorker to runner**

```python
import asyncio
import logging

from mirage.shared.config import Settings
from mirage.indexer.worker import ChunkWorker
from mirage.indexer.summary_worker import SummaryWorker
from mirage.indexer.embedding_worker import MultiModelEmbeddingWorker
from mirage.indexer.status_worker import StatusWorker

logger = logging.getLogger(__name__)


async def main():
    settings = Settings()
    
    workers = [
        ChunkWorker(settings),
        SummaryWorker(settings),
        MultiModelEmbeddingWorker(settings),
        StatusWorker(settings),
    ]
    
    await asyncio.gather(*[w.run() for w in workers])


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add src/mirage/indexer/__main__.py
git commit -m "feat(indexer): add SummaryWorker to indexer"
```

---

## Phase T03-02: EmbeddingWorker Updates

### Task T03-02-01: Read content_type from EmbeddingStatusTable

**Files:**
- Modify: `src/mirage/indexer/embedding_worker.py`
- Create: `tests/indexer/test_embedding_worker_content_type.py`

**Step 1: Write the test**

```python
# tests/indexer/test_embedding_worker_content_type.py
import pytest
from mirage.indexer.embedding_worker import MultiModelEmbeddingWorker


@pytest.mark.asyncio
async def test_embedding_worker_handles_content_type(test_db):
    # Setup: chunk with summary_text, embedding_status with content_type="summary"
    
    worker = MultiModelEmbeddingWorker(settings)
    pending = await worker._claim_pending(session)
    
    assert pending.content_type == "summary"
    assert pending.content == chunk.summary_text  # not chunk.content
```

**Step 2: Update PendingEmbedding dataclass**

```python
@dataclass
class PendingEmbedding:
    chunk_id: str
    model_name: str
    content_type: str
    content: str
    ollama_url: str
```

**Step 3: Update _claim_pending to read content_type**

Modify query to include content_type and select correct content:

```python
async def _claim_pending(self, session: AsyncSession) -> PendingEmbedding | None:
    """Find a pending embedding_status row."""
    result = await session.execute(
        select(EmbeddingStatusTable, ChunkTable, DocumentTable, ProjectTable)
        .join(ChunkTable, EmbeddingStatusTable.chunk_id == ChunkTable.id)
        .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
        .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
        .where(EmbeddingStatusTable.status == "pending")
        .limit(1)
    )
    row = result.first()
    if not row:
        return None

    embedding_status, chunk, document, project = row

    embedding_status.status = "processing"
    await session.flush()

    # Select content based on content_type
    content = chunk.content if embedding_status.content_type == "original" else chunk.summary_text

    return PendingEmbedding(
        chunk_id=chunk.id,
        model_name=embedding_status.model_name,
        content_type=embedding_status.content_type,
        content=content or "",
        ollama_url=project.ollama_url,
    )
```

**Step 4: Run tests**

```bash
uv run pytest tests/indexer/test_embedding_worker_content_type.py -v
```

**Step 5: Commit**

```bash
git add src/mirage/indexer/embedding_worker.py tests/indexer/
git commit -m "feat(indexer): handle content_type in EmbeddingWorker"
```

---

### Task T03-02-02: Write to embeddings table with content_type

**Files:**
- Modify: `src/mirage/indexer/embedding_worker.py:85-95`

**Step 1: Update process_one to include content_type**

```python
if result is None:
    status.status = "failed"
    status.error_message = "Embedding request failed"
else:
    TableClass = get_embeddings_table_class(model)
    embedding_row = TableClass(
        chunk_id=pending.chunk_id,
        content_type=pending.content_type,
        embedding=result.embedding,
    )
    session.add(embedding_row)
    status.status = "ready"
```

**Step 2: Run existing tests**

```bash
uv run pytest tests/indexer/ -v
```

**Step 3: Commit**

```bash
git add src/mirage/indexer/embedding_worker.py
git commit -m "feat(indexer): write content_type to embeddings table"
```

---

## Phase T03-03: ChunkWorker Cleanup

### Task T03-03-01: Remove status="parent" hack

**Files:**
- Modify: `src/mirage/indexer/worker.py:130-165`

**Step 1: Update ChunkWorker to use status="pending"**

Change parent chunk creation:

```python
parent = ChunkTable(
    document_id=doc.id,
    content=chunk_data["content"],
    position=chunk_data["position"],
    structure_json=chunk_data["structure"],
    status="pending",  # Changed from "parent"
)
```

**Step 2: Remove status="parent" from child chunks**

Child chunks also start with `status="pending"`.

**Step 3: Remove EmbeddingStatusTable creation from ChunkWorker**

SummaryWorker now handles this. Remove lines 168-186.

**Step 4: Run tests**

```bash
uv run pytest tests/indexer/test_worker.py -v
```

**Step 5: Commit**

```bash
git add src/mirage/indexer/worker.py
git commit -m "refactor(indexer): remove status=parent hack, defer to SummaryWorker"
```

---

## Phase T03-04: Search Updates

### Task T03-04-01: Query embeddings with content_type filter

**Files:**
- Modify: `src/mirage/api/routers/search.py:85-99`

**Step 1: Update search SQL to include content_type**

```python
sql = text(f"""
    SELECT DISTINCT ON (child.parent_id, e.content_type)
           child.id, child.content, child.structure,
           e.embedding <=> :embedding AS distance,
           e.content_type,
           parent.content AS parent_content,
           d.id as doc_id, d.filename
    FROM {table_name} e
    JOIN chunks child ON e.chunk_id = child.id
    JOIN chunks parent ON child.parent_id = parent.id
    JOIN documents d ON child.document_id = d.id
    WHERE d.project_id = :project_id
      AND d.status IN ('ready', 'partial')
    ORDER BY child.parent_id, e.content_type, e.embedding <=> :embedding
    LIMIT :limit
""")
```

**Step 2: Commit**

```bash
git add src/mirage/api/routers/search.py
git commit -m "feat(search): include content_type in search query"
```

---

### Task T03-04-02: Deduplicate by chunk_id across content_types

**Files:**
- Modify: `src/mirage/api/routers/search.py:126-132`

**Step 1: Update deduplication logic**

Keep minimum distance per chunk_id regardless of content_type:

```python
# Deduplicate by chunk_id, keep minimum distance
seen_chunks: dict[str, tuple] = {}
for chunk_id, content, parent_content, distance, doc_id, filename, structure in all_results:
    if chunk_id not in seen_chunks or seen_chunks[chunk_id][2] > distance:
        seen_chunks[chunk_id] = (content, parent_content, distance, doc_id, filename, structure)
```

**Step 2: Run existing search tests**

```bash
uv run pytest tests/api/test_search.py -v
```

**Step 3: Commit**

```bash
git add src/mirage/api/routers/search.py
git commit -m "feat(search): deduplicate across content_types"
```

---

## Phase T03-05: Configuration

### Task T03-05-01: Add MIRAGE_SUMMARY_MODEL setting

**Files:**
- Modify: `src/mirage/shared/config.py`

**Step 1: Add summary_model to Settings**

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MIRAGE_")

    database_url: str
    api_key: str
    ollama_url: str = "http://ollama:11434"
    summary_model: str = "mistral:7b"  # New field
    # ... rest of settings
```

**Step 2: Commit**

```bash
git add src/mirage/shared/config.py
git commit -m "feat(config): add MIRAGE_SUMMARY_MODEL setting"
```

---

### Task T03-05-02: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Add configuration to docs**

Add to Configuration table:

```markdown
| `MIRAGE_SUMMARY_MODEL` | `mistral:7b` | Ollama model for summarization |
```

**Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add MIRAGE_SUMMARY_MODEL to configuration"
```

---

## Phase T03-06: Testing

### Task T03-06-01: Unit tests for summary generation

**Files:**
- Modify: `tests/indexer/test_summary_worker.py`

**Step 1: Add comprehensive tests**

- Test empty summary handling
- Test error handling
- Test embedding_status creation for both content_types

**Step 2: Run tests**

```bash
uv run pytest tests/indexer/test_summary_worker.py -v
```

**Step 3: Commit**

```bash
git add tests/indexer/test_summary_worker.py
git commit -m "test: add comprehensive summary worker tests"
```

---

### Task T03-06-02: Unit tests for EmbeddingWorker content_type handling

**Files:**
- Modify: `tests/indexer/test_embedding_worker_content_type.py`

**Step 1: Add tests**

- Test original content_type
- Test summary content_type
- Test missing summary_text handling

**Step 2: Run tests**

```bash
uv run pytest tests/indexer/test_embedding_worker_content_type.py -v
```

**Step 3: Commit**

```bash
git add tests/indexer/test_embedding_worker_content_type.py
git commit -m "test: add content_type embedding tests"
```

---

### Task T03-06-03: Integration test for full pipeline

**Files:**
- Create: `tests/integration/test_summarization_pipeline.py`

**Step 1: Write integration test**

Test full flow: ChunkWorker → SummaryWorker → EmbeddingWorker → Search

**Step 2: Run test**

```bash
uv run pytest tests/integration/test_summarization_pipeline.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_summarization_pipeline.py
git commit -m "test: add summarization pipeline integration test"
```

---

### Task T03-06-04: Search tests with summary embeddings

**Files:**
- Modify: `tests/api/test_search.py`

**Step 1: Add test for search with summary**

- Create document with summary
- Verify search finds via both original and summary embeddings
- Verify deduplication works

**Step 2: Run tests**

```bash
uv run pytest tests/api/test_search.py -v
```

**Step 3: Commit**

```bash
git add tests/api/test_search.py
git commit -m "test: add search tests with summary embeddings"
```

---

## Summary

After completing all phases:

1. **Database**: Schema supports summary_text and content_type
2. **SummaryWorker**: Generates structured summaries via Ollama
3. **EmbeddingWorker**: Handles both original and summary content
4. **Search**: Queries both, deduplicates, returns original
5. **Config**: MIRAGE_SUMMARY_MODEL configurable
6. **Tests**: Full coverage of new functionality
