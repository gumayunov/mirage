# Progress: LLM-based Chunk Summarization

## Phase T03-00: Database Schema

- [ ] T03-00-01: Add summary_text and update status to ChunkTable
- [ ] T03-00-02: Add content_type to EmbeddingStatusTable
- [ ] T03-00-03: Update dynamic EmbeddingsTable with content_type
- [ ] T03-00-04: Create Alembic migration

## Phase T03-01: SummaryWorker

- [ ] T03-01-01: Create summary prompt template
- [ ] T03-01-02: Implement Ollama client for summarization
- [ ] T03-01-03: Implement SummaryWorker with state machine
- [ ] T03-01-04: Update indexer __main__.py to run SummaryWorker

## Phase T03-02: EmbeddingWorker Updates

- [ ] T03-02-01: Read content_type from EmbeddingStatusTable
- [ ] T03-02-02: Write to embeddings table with content_type

## Phase T03-03: ChunkWorker Cleanup

- [ ] T03-03-01: Remove status="parent" hack, defer to SummaryWorker

## Phase T03-04: Search Updates

- [ ] T03-04-01: Query embeddings with content_type filter
- [ ] T03-04-02: Deduplicate by chunk_id across content_types

## Phase T03-05: Configuration

- [ ] T03-05-01: Add MIRAGE_SUMMARY_MODEL setting
- [ ] T03-05-02: Update AGENTS.md

## Phase T03-06: Testing

- [ ] T03-06-01: Unit tests for summary generation
- [ ] T03-06-02: Unit tests for EmbeddingWorker content_type handling
- [ ] T03-06-03: Integration test for full pipeline
- [ ] T03-06-04: Search tests with summary embeddings
