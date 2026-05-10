# Phase 03-04: End-to-End & Documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove the full pipeline (parse → chunk → summary → embed → search) works on a fixture document with all workers stubbed at the network boundary, and update user-facing reference docs.

---

### Task T03-04-01: Full integration test

**Files:**
- Create: `tests/integration/test_summary_pipeline.py`

- [ ] **Step 1: Write end-to-end test**

Walk through:

1. Spin up an in-memory sqlite DB with `Base.metadata.create_all`.
2. Seed `ProjectTable` (with `summary_model='mistral:7b'`), insert a `DocumentTable` and an `IndexingTaskTable` with `task_type='index'`.
3. Stub `OllamaChat.complete` (returns canned summary) and `OllamaEmbedding.get_embedding` (returns deterministic vector).
4. Run one iteration each: `ChunkWorker.process_task`, then drain `SummaryWorker.process_one` until it returns False, then drain `MultiModelEmbeddingWorker.process_one` until False, then `StatusWorker.check_documents`.
5. Assert: document `status='ready'`, parents have `summary_text` set, parent embedding rows exist in the embeddings table, child embedding rows exist.
6. Issue a search via the FastAPI test client and assert at least one result.

```python
@pytest.mark.asyncio
async def test_summary_pipeline_end_to_end(tmp_path, monkeypatch):
    # ... setup as above
    # ... drive workers
    # ... search
    assert response.status_code == 200
    assert response.json()["results"]
```

(Use the helper patterns from `tests/integration/test_multi_model.py` for the FastAPI client + override.)

- [ ] **Step 2: Run + commit**

```
uv run pytest tests/integration/test_summary_pipeline.py -v
git add tests/integration/test_summary_pipeline.py
git commit -m "test(integration): full summary pipeline (T03-04-01)"
```

---

### Task T03-04-02: API surface for project summary fields

**Files:**
- Verify: changes from T03-00-05 already cover create + response.
- Optional: add PATCH `/projects/{id}` to update `summary_model`/`language` post-hoc.

- [ ] **Step 1: Decide scope**

If the user has not asked for live reconfiguration, skip PATCH. Re-indexing a project after changing `ProjectTable.summary_model` directly in the DB is acceptable for V1.

- [ ] **Step 2 (only if implementing PATCH): Add endpoint with test, then commit**

```python
class ProjectUpdate(BaseModel):
    summary_model: str | None = None
    language: str | None = None

@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, payload: ProjectUpdate, ...): ...
```

---

### Task T03-04-03: Reference docs

**Files:**
- Modify: `docs/reference/` — find or create the env-vars and project-config tables.

- [ ] **Step 1: Inspect what reference docs exist**

Run: `ls docs/reference/`
Pick the file that documents env vars / project fields (e.g., `configuration.md` or `api.md`). Create one if none exists and the table belongs there.

- [ ] **Step 2: Add rows**

Env variables table:

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_SUMMARY_MODEL` | _(unset)_ | Default Ollama chat model for parent-chunk summarization. Unset → summarization disabled globally. |

Project fields table:

| Field | Default | Description |
|-------|---------|-------------|
| `summary_model` | NULL | Per-project override for summary model. Falls back to `MIRAGE_SUMMARY_MODEL`. |
| `language` | `ru` | Prompt template language (`ru`, `en`). |

- [ ] **Step 3: Commit**

```
git add docs/reference/
git commit -m "docs(reference): document MIRAGE_SUMMARY_MODEL and project summary config (T03-04-03)"
```

---

### Task T03-04-04: Full suite + progress update

- [ ] **Step 1: Run all tests + linters**

```
uv run pytest
```

Expected: all green.

- [ ] **Step 2: Update progress files**

In `docs/designs/03-llm-summarization/progress.md`, mark every `- [ ]` you completed `- [x]`.
In `docs/plans/progress.md`, add a section for Design 03 if not present:

```markdown
## Design 03: LLM Summarization
- [x] Phase 03-00: Schema foundation
- [x] Phase 03-01: Configuration & Summary Worker
- [x] Phase 03-02: Worker pipeline integration
- [x] Phase 03-03: Search update
- [x] Phase 03-04: End-to-end & docs
```

- [ ] **Step 3: Commit**

```
git add docs/designs/03-llm-summarization/progress.md docs/plans/progress.md
git commit -m "docs: mark design 03 LLM summarization complete (T03-04-04)"
```
