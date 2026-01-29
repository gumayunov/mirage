# Phase 3.5.3: Chunk Status — Visibility

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose chunk progress in API responses and CLI output. Run a final regression-fix pass over the full test suite.

**Parent design:** `docs/plans/2026-01-29-chunk-status-design.md`

**Phases:**
- **Phase 3.5.1 — Foundation:** Schema, embedding client, config
- **Phase 3.5.2 — Worker Pipeline:** ChunkWorker, EmbeddingWorker, StatusWorker
- **Phase 3.5.3 — Visibility** (this file): API and CLI progress display, regression fixes

**Depends on:** Phase 3.5.2 (workers exist and produce chunk statuses).

**Tech Stack:** Python 3.14, SQLAlchemy + aiosqlite, FastAPI, httpx, typer

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
