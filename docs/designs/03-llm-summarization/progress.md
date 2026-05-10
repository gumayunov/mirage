# Design 03: LLM-based Chunk Summarization — Progress

GitHub issue: https://github.com/gumayunov/mirage/issues/1

## Phase 03-00: Schema Foundation
- [ ] T03-00-01: Migration — add `chunks.summary_text/summary_status/summary_error`
- [ ] T03-00-02: Migration — add `projects.summary_model/language`
- [ ] T03-00-03: Migration — backfill existing parent chunks (`status='parent' → 'pending'`, `summary_status='pending'`)
- [ ] T03-00-04: Update SQLAlchemy models (`ChunkTable`, `ProjectTable`)
- [ ] T03-00-05: Update Pydantic schemas (`ProjectCreate`, `ProjectResponse`)

## Phase 03-01: Configuration & Summary Worker
- [ ] T03-01-01: Add `summary_model` to `Settings`
- [ ] T03-01-02: `resolve_summary_model(project, settings)` helper + tests
- [ ] T03-01-03: Ollama chat client wrapper (`shared/llm.py`)
- [ ] T03-01-04: Prompt templates (ru, en) + selector
- [ ] T03-01-05: `SummaryWorker.process_one` — happy path
- [ ] T03-01-06: `SummaryWorker` — empty response and error paths
- [ ] T03-01-07: `SummaryWorker.run` loop + `__main__` entry

## Phase 03-02: Worker Pipeline Integration
- [ ] T03-02-01: ChunkWorker — drop `status='parent'` hack, set `summary_status='pending'` for parents
- [ ] T03-02-02: ChunkWorker — skip embedding_status creation for parents
- [ ] T03-02-03: EmbeddingWorker — pick `summary_text` for parents, `content` for children
- [ ] T03-02-04: StatusWorker — count via `parent_id IS NULL` (not `status='parent'`); flag `partial` on parent `summary_status='error'`

## Phase 03-03: Search Update
- [ ] T03-03-01: Add parent-summary SQL query alongside existing child query
- [ ] T03-03-02: Merge results — dedup by `parent_id`, keep min distance, parent hits → `content = parent.content`
- [ ] T03-03-03: Search integration test — high-level query hits parent summary

## Phase 03-04: End-to-End & Docs
- [ ] T03-04-01: Full integration test — parse → chunk → summary → embed → search
- [ ] T03-04-02: API — extend project create/response with `summary_model/language`
- [ ] T03-04-03: Docs — update `docs/reference/` env vars + project config table
- [ ] T03-04-04: Run full test suite; update top-level `docs/plans/progress.md`
