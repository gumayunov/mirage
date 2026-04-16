# LLM-based Chunk Summarization

**Goal:** Improve search quality by generating compressed summaries of parent chunks for embedding alongside child original content.

**Architecture:** SummaryWorker generates structured markdown summaries of parent chunks via Ollama. Parent summary embeddings provide a high-level search entry point; child original embeddings provide detail-level search. Search merges both, deduplicates by parent_id, returns original content.

**Tech Stack:** Ollama (configurable model per project), existing embedding infrastructure, PostgreSQL

---

## Problem Statement

Current search uses child chunk content for embeddings. High-level queries ("о чём глава", "какие персонажи") match poorly because relevant information is spread across multiple children. Parent chunks contain the full context but aren't embedded — they serve only as context for display.

## Solution

Generate structured summaries of parent chunks, embed them, and search across both child original and parent summary embeddings. User always sees original text.

**Key decision (dec-20260416-001):** summarize only parent chunks (V1). Children are small (~400 tokens) — summarization adds little value. Parent summaries give a new search vector (compressed/structured) that children don't have. This is a stepping stone toward hierarchical summarization (chapter/part level) from the research doc.

## Database Changes

### ChunkTable — new fields

```python
summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
summary_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
summary_error: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- `summary_text` — generated markdown summary (parent only, NULL for child)
- `summary_status` — `pending | processing | done | error` (NULL for child, NULL for parent if summarization disabled)
- `summary_error` — LLM error message

### ChunkTable — unchanged

- `status` — embedding lifecycle from design 04 (pending|processing|ready|corrupted|error), applies to both parent and child
- `content` — original chunk text, untouched

### ProjectTable — new fields

```python
summary_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
language: Mapped[str] = mapped_column(String(10), default="ru")
```

- `summary_model` — per-project override, NULL → fallback to `Settings.summary_model`
- `language` — prompt template selection (ru, en)

### EmbeddingStatusTable — unchanged

No `content_type` field. Content source is determined by `chunk.parent_id IS NULL`: parent → `summary_text`, child → `content`. PK remains `(chunk_id, model_name)`.

### Embeddings tables (dynamic) — unchanged

`(chunk_id, embedding)`. Parent and child rows coexist in the same table.

## State Machines

### Parent chunk — two orthogonal state machines (B1)

**Summary lifecycle** (`summary_status`):
```
pending → processing → done
                     → error
```

**Embedding lifecycle** (`status`, same as design 04):
```
pending → processing → ready | corrupted | error
```

Embedding starts only after `summary_status = done` and `summary_text IS NOT NULL`.

### Child chunk — unchanged (design 04)

```
status: pending → processing → ready | corrupted | error
summary_status: NULL (not applicable)
```

Child embedding and parent summarization run in parallel.

## Workers

### SummaryWorker (new)

`src/mirage/indexer/summary_worker.py`

Polls: `SELECT ... FROM chunks WHERE summary_status = 'pending' LIMIT 1 FOR UPDATE SKIP LOCKED`

1. Claim parent: `summary_status = 'processing'`
2. Resolve model: `project.summary_model ?? Settings.summary_model`
3. Select prompt template by `project.language`
4. Call Ollama (`/api/chat`, stream=false)
5. Non-empty response → `summary_text = result`, `summary_status = 'done'`, create `embedding_status(parent, model)` for each model
6. Empty response → `summary_status = 'done'`, `summary_text = NULL`, NO embedding_status created
7. Error → `summary_status = 'error'`, `summary_error = str(e)`

### EmbeddingWorker — minimal changes

When claiming pending embedding_status, determine content source:

```python
content = chunk.summary_text if chunk.parent_id is None else chunk.content
```

Everything else unchanged — same `OllamaEmbedding`, same `get_embeddings_table_class`.

### ChunkWorker — changes

When creating parent chunks:
- Resolve effective summary_model for project
- If model != None → `summary_status = 'pending'`
- If model == None → `summary_status = NULL`, no embedding_status for parent
- Remove `status="parent"` hack — parent identified by `parent_id IS NULL`

When creating child chunks — unchanged (creates embedding_status rows as before).

### StatusWorker — changes

Document status considers:
- All child embedding_status processed (ready|failed) — primary criterion
- Parent `summary_status = error` → document `partial` (does not block)
- Parent without embedding_status (empty summary or disabled) — ignored

## Search Changes

Two queries per model + merge:

**Query 1 — child original (as now):**
```sql
SELECT DISTINCT ON (child.parent_id)
       child.id, child.content, child.structure,
       e.embedding <=> :embedding AS distance,
       parent.content AS parent_content,
       d.id as doc_id, d.filename
FROM {table} e
JOIN chunks child ON e.chunk_id = child.id
JOIN chunks parent ON child.parent_id = parent.id
JOIN documents d ON child.document_id = d.id
WHERE d.project_id = :project_id
  AND child.parent_id IS NOT NULL
  AND d.status IN ('ready', 'partial')
ORDER BY child.parent_id, distance
LIMIT :limit
```

**Query 2 — parent summary:**
```sql
SELECT parent.id, parent.content AS parent_content,
       e.embedding <=> :embedding AS distance,
       d.id as doc_id, d.filename
FROM {table} e
JOIN chunks parent ON e.chunk_id = parent.id
JOIN documents d ON parent.document_id = d.id
WHERE d.project_id = :project_id
  AND parent.parent_id IS NULL
  AND d.status IN ('ready', 'partial')
ORDER BY distance
LIMIT :limit
```

**Merge and deduplication:**
- Join results by parent_id (query 1: `child.parent_id`, query 2: `parent.id`)
- If same parent found via both child and summary — keep minimum distance
- When parent found via summary: `content = parent.content`, `parent_content = parent.content`
- Sort by distance, apply limit and threshold

## Summary Format (Markdown)

```markdown
## Сущности
- Алиса

## Места
- Комната

## Даты
- 15 мая 2024

## Действия
- Алиса вошла в комнату

## Факты
- В комнате было темно

## Описания
- Сундук: старый, деревянный
```

Rules:
- Empty sections not included
- If summary is empty, `summary_text = NULL`
- Prompt template selected by `project.language` (ru, en)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_SUMMARY_MODEL` | `mistral:7b` | Global default Ollama model for summarization. None = disabled. |

Per-project overrides via ProjectTable:

| Field | Default | Description |
|-------|---------|-------------|
| `summary_model` | NULL (use global) | Ollama model for summarization |
| `language` | `ru` | Prompt template language |

Model resolution: `project.summary_model ?? Settings.summary_model ?? None (disabled)`

## Edge Cases

**Empty LLM response:** `summary_status = 'done'`, `summary_text = NULL`, no embedding_status. Parent not searchable via summary; child search works.

**LLM error:** `summary_status = 'error'`, `summary_error = message`. No embedding_status. Document → `partial`. Retry: manual reset `summary_status → 'pending'`.

**Summarization disabled:** `effective_model = None` → `summary_status = NULL`. SummaryWorker ignores. Search works only via child.

**Reindex:** Deletes all chunks → parent recreated with `summary_status = 'pending'` → pipeline restarts.

## Migration

1. `chunks` — add `summary_text TEXT NULL`, `summary_status VARCHAR(20) NULL`, `summary_error TEXT NULL`
2. `projects` — add `summary_model VARCHAR(100) NULL`, `language VARCHAR(10) NOT NULL DEFAULT 'ru'`
3. Existing parent chunks (`status='parent'`) → `status='pending'`, `summary_status='pending'`
4. Embeddings tables and embedding_status — no changes

## Decision Log

| Fork | Decision | Rationale |
|------|----------|-----------|
| Scope | V1: parent only | 5-20x cheaper; child original already covers detail search; stepping stone to hierarchical summarization |
| State machine | B1: orthogonal columns | Clean observability; workers don't compete for same column; compatible with design 04 |
| Config | C2: per-project + language | Consistent with per-project ollama_url; supports multilingual corpora |
| content_type | Not introduced | Derivable from `parent_id IS NULL`; simpler schema |
| Search result (parent hit) | Parent content | User sees original text; consistent with existing UX |
