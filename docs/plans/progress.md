# Implementation Progress

This file tracks the implementation progress across all phases. Update checkboxes as tasks are completed.

## Phase 1: Foundation
- [x] Task 1.1: Project structure
- [x] Task 1.2: Settings module
- [x] Task 1.3: Database models
- [x] Task 1.4: Embedding client

## Phase 2: API
- [x] Task 2.1: Health endpoint
- [x] Task 2.2: Documents CRUD
- [x] Task 2.3: Search endpoint
- [x] Task 2.4: Indexing tasks endpoint

## Phase 3: Indexer
- [x] Task 4.1: Markdown parser
- [x] Task 4.2: PDF parser
- [x] Task 4.3: EPUB parser
- [x] Task 4.4: Chunking module
- [x] Task 4.5: Indexer worker

## Phase 3.5.1: Chunk Status — Foundation
- [x] Task CSL-1: Add `status` column to ChunkTable
- [x] Task CSL-2: Return `EmbeddingResult` from embedding client
- [x] Task CSL-3: Update `chunk_size` default to 400

## Phase 3.5.2: Chunk Status — Worker Pipeline
- [x] Task CSL-4: ChunkWorker (parse documents, save chunks without embeddings)
- [x] Task CSL-5: EmbeddingWorker (per-chunk embedding with atomic claim)
- [x] Task CSL-6: StatusWorker (document status from chunk counts)

## Phase 3.5.3: Chunk Status — Visibility
- [x] Task CSL-7: API — chunk progress in DocumentResponse
- [x] Task CSL-8: CLI — chunk progress in `status` and `list`
- [x] Task CSL-9: Full test suite regression fix

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

## Phase 4: CLI Integration
- [x] Task 5.1: CLI entry point
- [x] Task 5.2: Document commands
- [x] Task 5.3: Search command
- [x] Task 8.1: Claude Code skill

## Phase 5: Infrastructure
- [x] Task 7.1: Dockerfile
- [x] Task 7.2: Indexer entrypoint
- [ ] Task 7.3: Database migrations (Alembic)
- [ ] Task 6.1: Helm chart base
- [ ] Task 6.2: PostgreSQL templates
- [ ] Task 6.3: Ollama templates
- [ ] Task 6.4: API and Indexer templates
- [ ] Task 6.5: Ingress template
- [ ] Task 9.1: Database schema init script
- [ ] Task 9.2: Run all tests and linters
- [ ] Task 9.3: Update README

---

## How to use this file

When resuming work:
1. Read this file to see current progress
2. Find the first unchecked task
3. Read the corresponding plan in `docs/plans/`
4. Continue from that task

When completing a task:
1. Mark the checkbox: `- [ ]` → `- [x]`
2. Commit the progress update with your implementation
