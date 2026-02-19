# LLM-based Chunk Summarization

**Goal:** Improve search quality by generating compressed summaries of chunks for embedding alongside original content.

**Architecture:** SummaryWorker generates structured markdown summaries (entities, places, dates, actions, facts, descriptions) via Ollama mistral:7b. Both original and summary content get embedded for all models. Search queries both, deduplicates by chunk_id, returns original content.

**Tech Stack:** Ollama (mistral:7b), existing embedding infrastructure, PostgreSQL

---

## Problem Statement

Current search uses original chunk content for embeddings. Long passages with rich information may not match queries well because relevant details are diluted by narrative text.

## Solution

Generate structured summaries extracting key information, embed both versions, search across both with deduplication.

## Database Changes

### ChunkTable

New fields:
- `summary_text: Text | None` — compressed markdown version
- `status` — unified state machine (replaces `status="parent"` hack)

### EmbeddingStatusTable

Extend composite primary key:
- `content_type: String(50)` — `"original"` | `"summary"`

Existing fields remain:
- `chunk_id` (FK to chunks.id)
- `model_name`
- `status`
- `error_message`

### EmbeddingsTable (dynamic)

Extend primary key:
- `content_type: String(50)` — `"original"` | `"summary"`

Existing fields remain:
- `chunk_id` (FK to chunks.id)
- `embedding` (Vector)

## Chunk State Machine

```
pending → summarizing → summarized → embedding → ready
              ↓             ↓            ↓
            error         error        error
```

| State | Description |
|-------|-------------|
| `pending` | Chunk created, awaiting summarization |
| `summarizing` | SummaryWorker processing |
| `summarized` | Summary ready, awaiting embeddings |
| `embedding` | EmbeddingWorker processing |
| `ready` | All embeddings complete |
| `error` | Failed (error_message contains details) |

Chunk type determined by `parent_id`:
- `parent_id IS NULL` → parent chunk (large context)
- `parent_id IS NOT NULL` → child chunk (search target)

## SummaryWorker

New worker in `src/mirage/indexer/summary_worker.py`.

**Responsibilities:**
1. Find chunks with `status=pending`
2. Set `status=summarizing`
3. Call Ollama mistral:7b with extraction prompt
4. Parse response, save to `summary_text`
5. Set `status=summarized`
6. Create EmbeddingStatusTable records: `(original, summary) × all_models`
7. On error: set `status=error`, populate `error_message`

**Prompt structure:** Instruct model to extract structured markdown with sections for entities, places, dates, actions, facts, descriptions. Empty sections omitted.

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
- If summary is empty, `summary_text = None`
- Compact format optimized for embedding

## EmbeddingWorker Changes

Minimal changes required:

1. When claiming pending EmbeddingStatusTable row, also read `content_type`
2. Embed `chunk.content` if `content_type="original"`, `chunk.summary_text` if `content_type="summary"`
3. Write to embeddings table with `content_type` in primary key

## Search Changes

In `src/mirage/api/routers/search.py`:

1. Query all embeddings tables (models × content_types)
2. Deduplicate by `chunk_id`, keeping minimum distance
3. Return original `content` + `parent_content`

User always sees original text, discovers it through summary embedding.

## Migration from status="parent"

Remove the hack:
- All chunks start with `status="pending"`
- ChunkWorker no longer sets `status="parent"`
- Search filters by `parent_id` as before

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_SUMMARY_MODEL` | `mistral:7b` | Ollama model for summarization |
| `MIRAGE_SUMMARY_URL` | (from project.ollama_url) | Ollama server for summaries |

## Notes

- All models always active for indexing (ProjectModelTable being removed in parallel work)
- Both parent and child chunks go through summarization
- If parent summary is comparable to child size, it gets embedded and searched too
