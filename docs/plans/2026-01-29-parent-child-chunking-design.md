# Parent-Child Chunking Design

## Goal

Improve search precision by splitting documents into two levels of chunks:
- **Parent chunks** — large (3000 tokens), store full context, no embeddings
- **Child chunks** — small (500 tokens), have embeddings, used for vector search

Search finds relevant child chunks, returns them together with parent content for context.

## Data Model

Single `chunks` table with self-referencing FK:

```
ChunkTable:
  id, document_id, content, position, structure_json, metadata_json, status
  embedding       — Vector(768), NULL for parent chunks
  parent_id       — FK(chunks.id), NULL for parent chunks
```

- **Parent chunks** (`parent_id = NULL`): ~3000 tokens, `status = "parent"`, `embedding = NULL`
- **Child chunks** (`parent_id = <parent id>`): ~500 tokens, regular embedding flow (pending → processing → ready)

Chunk counts (chunks_total, chunks_processed, chunks_by_status) count only child chunks (`WHERE parent_id IS NOT NULL`).

## Chunking Pipeline

### ChunkWorker (worker.py)

1. Parse document (sections/pages/chapters) — unchanged
2. Cut text into parent chunks (3000 tokens, `chunk_size`)
3. Save parent chunks with `status = "parent"`
4. For each parent, cut child chunks (500 tokens, `child_chunk_size`, overlap `child_chunk_overlap`)
5. Save child chunks with `status = "pending"` and `parent_id`

### Chunker (chunking.py)

- `chunk_text()` — unchanged, cuts into parent chunks (large)
- `chunk_children()` — new method, cuts one parent into child chunks (small)
- Same splitting logic, different size/overlap parameters

### EmbeddingWorker — no changes

Claims `status = "pending"` chunks. Parent chunks have `status = "parent"`, so they are skipped.

## Config

```python
chunk_size: int = 3000              # parent chunk size (tokens)
chunk_overlap: int = 200            # parent overlap
child_chunk_size: int = 500         # child chunk size (tokens)
child_chunk_overlap: int = 50       # child overlap
```

## Search

### SQL Query

Search by child embeddings, JOIN parent for context:

```sql
SELECT child.id, child.content, child.structure,
       child.embedding <=> :embedding AS distance,
       parent.content AS parent_content,
       d.id AS doc_id, d.filename
FROM chunks child
JOIN chunks parent ON child.parent_id = parent.id
JOIN documents d ON child.document_id = d.id
WHERE d.project_id = :project_id
  AND d.status = 'ready'
  AND child.parent_id IS NOT NULL
ORDER BY child.embedding <=> :embedding
LIMIT :limit
```

### API Response

Add `parent_content` to `ChunkResult`:

```python
class ChunkResult(BaseModel):
    chunk_id: str
    content: str                       # child content
    parent_content: str | None = None  # parent content for context
    score: float
    structure: dict | None
    document: dict
```

### Deduplication

If multiple children of the same parent appear in top results, return only the best-scoring one per parent to avoid duplicate context.

## Files Changed

| File | Change |
|---|---|
| `shared/config.py` | `chunk_size=3000`, new `child_chunk_size=500`, `child_chunk_overlap=50` |
| `shared/db.py` | `parent_id` FK in ChunkTable, `children`/`parent` relationships |
| `indexer/chunking.py` | New `chunk_children()` method |
| `indexer/worker.py` | Two-level chunking: create parents, then children |
| `api/schemas.py` | `parent_content` in `ChunkResult` |
| `api/routers/search.py` | JOIN on parent, deduplication by parent_id |
| `api/routers/documents.py` | Filter `parent_id IS NOT NULL` in chunk counts |

### Tests

| File | Change |
|---|---|
| `tests/shared/test_config.py` | New defaults |
| `tests/shared/test_models.py` | parent_id in model |
| `tests/api/test_documents.py` | Chunks with parent_id in fixtures |
| `tests/api/test_search.py` | Parent+child chunks in fixture, verify parent_content |
| `tests/indexer/test_embedding_worker.py` | Child chunks with parent_id |

### Not Changed

- CLI display (search result rendering unchanged)
- `status_worker.py` (counts by document, not chunk type)
- Parsers (markdown, PDF, EPUB)
