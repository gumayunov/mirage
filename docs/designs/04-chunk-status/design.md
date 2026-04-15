# Chunk Status Lifecycle Design

## Problem

When embedding a chunk fails (e.g. text exceeds model context window), the entire document fails indexing. No chunks are saved. The current monolithic worker does parsing and embedding in one pass — no progress visibility, no partial results, no parallelism.

## Architecture

Three independent workers communicate through the database:

1. **ChunkWorker** — parses documents and saves chunks (no embeddings)
2. **EmbeddingWorker** — processes chunks one at a time, calls Ollama (parallelizable)
3. **StatusWorker** — polls every 10 seconds, computes document status from chunk statuses

Workers are decoupled: each polls the DB for work and processes it independently.

## Data Model

### Chunk statuses

`pending` → `processing` → `ready` | `corrupted` | `error`

- `pending` — chunk created, embedding not yet requested
- `processing` — EmbeddingWorker claimed the chunk, embedding in progress
- `ready` — embedding obtained successfully
- `corrupted` — text was truncated before embedding (exceeds model limit)
- `error` — Ollama returned an error

New field on `ChunkTable`:
```python
status: Mapped[str] = mapped_column(String(50), default="pending")
```

### Document statuses

`pending` → `indexing` → `ready` | `partial` | `error`

- `pending` — document uploaded, chunks not yet created
- `indexing` — ChunkWorker split into chunks, EmbeddingWorker processing
- `ready` — all chunks are `ready`
- `partial` — all chunks processed, but some are `corrupted`/`error`
- `error` — critical failure (file doesn't parse)

### DocumentResponse (API) — new fields

```python
chunks_total: int | None = None       # total number of chunks
chunks_processed: int | None = None   # chunks with status ready/corrupted/error
```

Fields are `None` for documents in `pending` status (no chunks yet). Percentage is computed client-side: `chunks_processed / chunks_total * 100`.

## Workers

### ChunkWorker

Polls `indexing_tasks` with `status=pending`:

1. Claims task, sets `task.status = "processing"`, `doc.status = "indexing"`
2. Parses file, splits into chunks
3. Saves all chunks to DB with `status = "pending"` (no embedding)
4. Sets `task.status = "done"`
5. On parse error: `doc.status = "error"`, `task.status = "failed"`

This is the current `IndexerWorker` minus the Ollama call.

### EmbeddingWorker

Polls `chunks` with `status=pending`:

1. Atomically claims a chunk via `UPDATE ... RETURNING` (`pending` → `processing`)
2. Calls Ollama
3. Success → writes embedding, `status = "ready"`
4. Text truncated → writes embedding, `status = "corrupted"`
5. Ollama error → `status = "error"`

Multiple instances can run in parallel — each claims its own chunk atomically.

### StatusWorker

Polls `documents` with `status=indexing`, every 10 seconds:

1. For each indexing document, counts chunks by status
2. If all chunks are processed (no `pending` or `processing`):
   - All `ready` → `doc.status = "ready"`, `doc.indexed_at = now()`
   - Some `corrupted`/`error` → `doc.status = "partial"`, `doc.indexed_at = now()`
3. If unprocessed chunks remain — does nothing, waits for next cycle

## Embedding Client

`shared/embedding.py` — `get_embedding` returns `EmbeddingResult` instead of `list[float]`:

```python
@dataclass
class EmbeddingResult:
    embedding: list[float]
    truncated: bool
```

- Text > `MAX_PROMPT_CHARS` → truncates, sets `truncated = True`
- Ollama error → returns `None` (EmbeddingWorker sets chunk `status = "error"`)

`get_embeddings` (batch method) is removed — chunks are processed individually.

## Config

`shared/config.py` — `chunk_size` back to 400. Truncation is now handled in the embedding layer, so chunks can be larger.

## API and CLI Changes

### API

**`GET /projects/{project_id}/documents/{document_id}`** — additional query to count chunks by status. Returns `chunks_total` and `chunks_processed` in `DocumentResponse`.

**`GET /projects/{project_id}/documents`** — same counts per document, via a single query with `GROUP BY` (no N+1).

### CLI

**`mirage documents status`**:
```
ID:       abc-123
Filename: book.pdf
Type:     pdf
Status:   indexing
Chunks:   42/128 (32%)
```

For `pending` (no chunks) the `Chunks` line is omitted. For `ready`: `Chunks: 128/128 (100%)`.

**`mirage documents list`**:
```
ID         Filename       Status     Progress
---------- -------------- ---------- ----------
abc-123    book.pdf       indexing   42/128 (32%)
def-456    notes.md       ready      15/15 (100%)
```

## File Changes

- `shared/db.py` — add `status` column to `ChunkTable`
- `shared/embedding.py` — return `EmbeddingResult`, handle errors gracefully, remove `get_embeddings`
- `shared/config.py` — `chunk_size = 400`
- `indexer/worker.py` — refactor into `ChunkWorker`, `EmbeddingWorker`, `StatusWorker`
- `api/schemas.py` — add `chunks_total`, `chunks_processed` to `DocumentResponse`
- `api/routers/documents.py` — add chunk count queries
- `cli/commands/documents.py` — display chunk progress in `status` and `list`
