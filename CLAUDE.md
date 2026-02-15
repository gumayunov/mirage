# miRAGe - Claude Code Instructions

## Project Overview

miRAGe is a local RAG (Retrieval-Augmented Generation) system for books. It indexes PDF, EPUB, and Markdown files and provides semantic search.

## Tech Stack

- Python 3.14, uv
- FastAPI (API)
- SQLAlchemy + aiosqlite (database)
- Ollama (embeddings - supports multiple models per project)
- PyMuPDF, ebooklib (document parsing)

## Development Commands

```bash
# Run tests
uv run pytest

# Run API server
uv run uvicorn mirage.api.main:app --reload

# Run indexer worker
uv run python -m mirage.indexer.worker

# Start local infrastructure (PostgreSQL, Ollama)
make dev
```

## Project Structure

```
src/mirage/
├── shared/       # Config, DB models, embedding client
├── api/          # FastAPI application
└── indexer/      # Document parsers, chunking, worker
```

## Implementation Plans

Plans are in `docs/plans/`:
- `phase-1-foundation.md` - Core modules
- `phase-2-api.md` - REST API
- `phase-3-indexer.md` - Document indexing
- `phase-4-cli-integration.md` - CLI commands
- `phase-5-infrastructure.md` - Deployment

## Progress Tracking

**Important:** Progress is tracked in `docs/plans/progress.md`.

When starting a new session:
1. Read `docs/plans/progress.md` to see what's done
2. Find the first unchecked `- [ ]` task
3. Read the corresponding phase plan
4. Continue implementation from that task

When completing a task:
1. Mark checkbox in `progress.md`: `- [ ]` → `- [x]`
2. Commit the update together with your implementation

## Project Conventions

When creating designs and plans, always use the structured-project-workflow skill for file naming and organization.

## Code Style

- Use type hints
- Follow TDD: write failing test first, then implement
- Keep imports sorted (stdlib, third-party, local)
- Async where applicable (DB operations, HTTP calls)
