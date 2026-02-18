# Simplify Multi-Model Architecture

> Design 02 — created 2026-02-18

## Problem

Current architecture allows per-project model selection via `ProjectModelTable`. This adds complexity:
- Extra database table and relationships
- Workers must JOIN with this table to determine which models to use
- API/CLI has logic for model selection at project creation
- In practice, always indexing with all models is simpler and more flexible

## Goals

- Remove `ProjectModelTable` and related code
- All projects always index with all registered models
- Model selection moves to search time only (optional filter)
- Simpler codebase with same functionality

## Non-Goals

- Adding new embedding models
- Changing embedding storage structure
- Changing search results format

## Design

### Data Model Changes

**Remove:**
- `ProjectModelTable` (entire table)
- `models` relationship from `ProjectTable`

**Unchanged:**
- `EmbeddingStatusTable` — tracks per-chunk per-model embedding status
- Embedding tables (`embeddings_nomic_768`, `embeddings_bge_m3_1024`, `embeddings_mxbai_1024`)

### API Changes

| Endpoint | Change |
|----------|--------|
| `POST /projects` | Remove `models` from request body |
| `GET /projects` | Remove `models` from response (or compute from registry) |
| `POST /projects/{id}/search` | Add optional `models: list[str]` filter |

### Worker Changes

**ChunkWorker:**
- Replace JOIN with `ProjectModelTable` → `get_all_models()` from registry
- Always create `embedding_status` rows for all models

**EmbeddingWorker:**
- Remove JOIN with `ProjectModelTable`
- Simply process any pending `embedding_status` rows

### CLI Changes

- `mirage projects create`: Remove `--model/-m` flag
- `mirage projects list`: Remove `Models` column or show static list

### Search Changes

```python
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.0
    models: list[str] | None = None  # Optional filter
```

If `models` specified — validate against registry and filter. Otherwise search all models.

### Migration

```sql
DROP TABLE IF EXISTS project_models;
```

No data migration needed (assuming no existing data to preserve).

## Alternatives Considered

1. **Keep table, always fill with all models** — simpler migration but leaves dead code
2. **Soft deprecation** — mark table as deprecated, remove later — adds tech debt
3. **Full removal (chosen)** — cleanest solution, breaks existing projects but acceptable for current stage

## Open Questions

None.
