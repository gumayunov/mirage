# Chunk Status Lifecycle Design

## Problem

When embedding a chunk fails (e.g. text exceeds model context window), the entire document fails indexing. No chunks are saved.

## Design

### Chunk statuses

- `pending` — chunk created, embedding not yet requested
- `ready` — embedding obtained successfully
- `corrupted` — text was truncated before embedding (exceeds model limit)
- `error` — Ollama returned an error

### Document statuses

- `ready` — all chunks are `ready`
- `partial` — some chunks are `error`/`corrupted`, rest are `ready`
- `error` — critical failure (file doesn't parse, etc.)

### Processing flow

1. Parse document and split into chunks (unchanged)
2. Save all chunks to DB with status `pending`, no embedding
3. For each chunk, request embedding:
   - Success → write embedding, status `ready`
   - Truncated → write embedding, status `corrupted`
   - Ollama error → status `error`
4. Count chunk statuses to determine document status

### File changes

- `shared/db.py` — add `status` column to ChunkTable
- `shared/embedding.py` — return `EmbeddingResult(embedding, truncated)`, handle errors gracefully
- `indexer/worker.py` — save chunks before embeddings, process individually, compute final doc status
- `shared/config.py` — restore `chunk_size` to 400 (truncation handled in embedding layer)
