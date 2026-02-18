# Progress — Simplify Multi-Model Architecture

> Design 02 | Last updated: 2026-02-18

## Summary

| Phase | Total | Done | Status |
|-------|-------|------|--------|
| 00    | 8     | 0    | 🔲 Not started |

## Tasks

### Phase 00: Main

- [ ] `T02-00-01` — Remove ProjectModelTable from db.py
- [ ] `T02-00-02` — Update ChunkWorker to use get_all_models()
- [ ] `T02-00-03` — Update EmbeddingWorker to remove ProjectModelTable JOIN
- [ ] `T02-00-04` — Update search router: add models filter, use registry
- [ ] `T02-00-05` — Update projects API: remove models handling
- [ ] `T02-00-06` — Update CLI: remove --model flag
- [ ] `T02-00-07` — Create Alembic migration to drop project_models table
- [ ] `T02-00-08` — Run tests and fix regressions
