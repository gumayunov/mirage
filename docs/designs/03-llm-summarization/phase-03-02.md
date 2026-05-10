# Phase 03-02: Worker Pipeline Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the legacy `status='parent'` marker with `parent_id IS NULL` semantics, route parent embeddings through `summary_text`, and propagate parent summary failures into document `partial` status.

**Architecture:** ChunkWorker becomes the place where summarization is *enabled* per parent (sets `summary_status='pending'` when an effective model resolves; otherwise leaves it NULL). EmbeddingWorker no longer assumes `chunk.content` is the source — it picks `summary_text` for parents (`parent_id IS NULL`) and `content` for children. StatusWorker treats parents identified by `parent_id IS NULL` and counts only children for completeness, escalating to `partial` if any parent has `summary_status='error'`.

---

### Task T03-02-01: ChunkWorker — drop `status='parent'`, set `summary_status`

**Files:**
- Modify: `src/mirage/indexer/worker.py:131-167` (parent creation block)
- Modify: existing tests in `tests/indexer/test_worker.py` referencing `status="parent"`

- [ ] **Step 1: Update parent creation in `worker.py`**

Replace the parent-creation loop:

```python
            # Resolve effective summary model for this project
            project = (await session.execute(
                select(ProjectTable).where(ProjectTable.id == doc.project_id)
            )).scalar_one()
            effective_summary_model = resolve_summary_model(project, self.settings)
            initial_summary_status = "pending" if effective_summary_model else None

            parent_chunks = []
            for chunk_data in chunks_data:
                parent = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                    status="pending",
                    summary_status=initial_summary_status,
                )
                session.add(parent)
                parent_chunks.append((parent, chunk_data))
```

Add imports near the top:

```python
from mirage.indexer.summary_resolver import resolve_summary_model
from mirage.shared.db import ProjectTable
```

- [ ] **Step 2: Skip embedding_status creation for parents**

In the same `process_task`, the existing block already filters children with `parent_id.is_not(None)` — keep it. Confirm no embedding_status rows are added for parents in this task.

- [ ] **Step 3: Find/replace legacy markers in tests**

Run: `rg -l 'status="parent"' tests/`
For each hit (`tests/indexer/test_embedding_worker.py:60`, `tests/indexer/test_worker.py`, `tests/indexer/test_status_worker.py` if present), replace `status="parent"` with `status="pending"` and where appropriate add `summary_status="pending"`. Existing parent-id linkage is unchanged.

- [ ] **Step 4: Run worker tests**

Run: `uv run pytest tests/indexer/test_worker.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```
git add src/mirage/indexer/worker.py tests/indexer/test_worker.py tests/indexer/test_embedding_worker.py
git commit -m "refactor(indexer): parents identified by parent_id IS NULL (T03-02-01)"
```

---

### Task T03-02-03: EmbeddingWorker — content source switch

**Files:**
- Modify: `src/mirage/indexer/embedding_worker.py:37-99`
- Test: `tests/indexer/test_embedding_worker.py`

- [ ] **Step 1: Failing test — parent embedding uses summary_text**

Append to `tests/indexer/test_embedding_worker.py`:

```python
@pytest.mark.asyncio
async def test_parent_embedding_uses_summary_text(settings, monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        model = get_model("nomic-embed-text")
        TableClass = get_embeddings_table_class(model)
        await conn.run_sync(lambda c: TableClass.__table__.create(c, checkfirst=True))
    sf = async_sessionmaker(engine, expire_on_commit=False)

    async with sf() as s:
        s.add(ProjectTable(id="p", name="t", ollama_url="http://x"))
        s.add(DocumentTable(id="d", project_id="p", filename="f", original_path="/f", file_type="markdown", status="indexing"))
        s.add(ChunkTable(
            id="parent-1", document_id="d", content="raw parent text",
            position=0, status="pending",
            summary_text="## Сущности\n- Алиса", summary_status="done",
        ))
        s.add(EmbeddingStatusTable(chunk_id="parent-1", model_name="nomic-embed-text", status="pending"))
        await s.commit()

    captured = {}
    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def get_embedding(self, content, prefix=""):
            captured["content"] = content
            return EmbeddingResult(embedding=[0.1] * 768, truncated=False)
    monkeypatch.setattr("mirage.indexer.embedding_worker.OllamaEmbedding", FakeClient)

    async with sf() as s:
        await MultiModelEmbeddingWorker(settings).process_one(s)

    assert captured["content"] == "## Сущности\n- Алиса"
    await engine.dispose()
```

Run: `uv run pytest tests/indexer/test_embedding_worker.py::test_parent_embedding_uses_summary_text -v` → FAIL (worker still uses `chunk.content`).

- [ ] **Step 2: Implement source switch**

In `embedding_worker.py`, change the `_claim_pending` join to also load enough to decide source:

```python
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

        content = chunk.summary_text if chunk.parent_id is None else chunk.content
        if content is None:
            embedding_status.status = "failed"
            embedding_status.error_message = "no content to embed"
            await session.flush()
            return None

        return PendingEmbedding(
            chunk_id=chunk.id,
            model_name=embedding_status.model_name,
            content=content,
            ollama_url=project.ollama_url,
        )
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/indexer/test_embedding_worker.py -v
git add src/mirage/indexer/embedding_worker.py tests/indexer/test_embedding_worker.py
git commit -m "feat(indexer): EmbeddingWorker uses summary_text for parents (T03-02-03)"
```

---

### Task T03-02-04: StatusWorker — parent identification + error escalation

**Files:**
- Modify: `src/mirage/indexer/status_worker.py`
- Test: `tests/indexer/test_status_worker.py`

- [ ] **Step 1: Failing test — partial when parent summary_status='error'**

Append to `tests/indexer/test_status_worker.py`:

```python
@pytest.mark.asyncio
async def test_document_partial_when_parent_summary_error(settings, db_session):
    """All children ready + one parent in summary_error → document partial."""
    async with db_session() as s:
        # arrange: doc with one parent (summary_status='error') and one child (status='ready')
        s.add(DocumentTable(id="d2", project_id="p", filename="f2", original_path="/f", file_type="markdown", status="indexing"))
        s.add(ChunkTable(
            id="parent-2", document_id="d2", content="x", position=0,
            status="pending", summary_status="error", summary_error="boom",
        ))
        s.add(ChunkTable(
            id="child-2", document_id="d2", content="y", position=0,
            status="ready", parent_id="parent-2",
        ))
        await s.commit()

    async with db_session() as s:
        await StatusWorker(settings).check_documents(s)

    async with db_session() as s:
        from sqlalchemy import select
        doc = (await s.execute(select(DocumentTable).where(DocumentTable.id == "d2"))).scalar_one()
        assert doc.status == "partial"
```

(Re-use the existing `db_session` fixture in that file, extending it with project `"p"` if not already seeded.)

- [ ] **Step 2: Update StatusWorker logic**

In `check_documents`:

```python
        for doc in docs:
            counts = await session.execute(
                select(ChunkTable.status, func.count().label("cnt"))
                .where(
                    ChunkTable.document_id == doc.id,
                    ChunkTable.parent_id.is_not(None),
                )
                .group_by(ChunkTable.status)
            )
            status_counts = {row[0]: row[1] for row in counts.fetchall()}

            pending = status_counts.get("pending", 0)
            processing = status_counts.get("processing", 0)
            if pending > 0 or processing > 0:
                continue

            ready = status_counts.get("ready", 0)
            corrupted = status_counts.get("corrupted", 0)
            error = status_counts.get("error", 0)

            parent_error = await session.execute(
                select(func.count()).where(
                    ChunkTable.document_id == doc.id,
                    ChunkTable.parent_id.is_(None),
                    ChunkTable.summary_status == "error",
                )
            )
            parent_summary_errors = parent_error.scalar_one()

            if corrupted == 0 and error == 0 and parent_summary_errors == 0 and ready > 0:
                doc.status = "ready"
            else:
                doc.status = "partial"

            doc.indexed_at = datetime.utcnow()
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/indexer/test_status_worker.py -v
git add src/mirage/indexer/status_worker.py tests/indexer/test_status_worker.py
git commit -m "feat(indexer): StatusWorker tracks parent summary errors (T03-02-04)"
```
