# Simplify Multi-Model Architecture — Implementation Plan

> Design 02 | Plan created 2026-02-18
> **For agent:** Execute tasks sequentially using task IDs.

## Overview

Remove `ProjectModelTable` and related code. All projects will always index with all registered embedding models. Model selection moves to search time only.

## Tasks

### T02-00-01: Remove ProjectModelTable from db.py

**Files:**
- Modify: `src/mirage/shared/db.py`
- Modify: `tests/shared/test_db.py` (if exists)

**Step 1: Remove ProjectModelTable class and imports**

In `src/mirage/shared/db.py`:

1. Delete the `ProjectModelTable` class (lines 48-57)
2. Remove `models` relationship from `ProjectTable` (line 26)
3. Remove `ProjectModelTable` from imports in other files (will be done in subsequent tasks)

**Step 2: Run tests to identify breakage**

Run: `uv run pytest -x`
Expected: Tests fail due to missing `ProjectModelTable` references

**Step 3: Commit**

```bash
git add src/mirage/shared/db.py
git commit -m "refactor: remove ProjectModelTable from db.py"
```

---

### T02-00-02: Update ChunkWorker to use get_all_models()

**Files:**
- Modify: `src/mirage/indexer/worker.py`

**Step 1: Update imports**

Add import at top of file:
```python
from mirage.shared.models_registry import get_all_models
```

Remove import:
```python
from mirage.shared.db import ProjectModelTable  # remove this
```

**Step 2: Replace model query with registry call**

Replace lines 168-175 in `worker.py`:

```python
# DELETE THIS:
result = await session.execute(
    select(ProjectModelTable.model_name).where(
        ProjectModelTable.project_id == doc.project_id,
        ProjectModelTable.enabled == True,
    )
)
enabled_models = [row[0] for row in result.fetchall()]

# REPLACE WITH:
enabled_models = [m.name for m in get_all_models()]
```

**Step 3: Verify tests pass**

Run: `uv run pytest tests/indexer/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/mirage/indexer/worker.py
git commit -m "refactor: ChunkWorker uses get_all_models() instead of ProjectModelTable"
```

---

### T02-00-03: Update EmbeddingWorker to remove ProjectModelTable JOIN

**Files:**
- Modify: `src/mirage/indexer/embedding_worker.py`

**Step 1: Update imports**

Remove import:
```python
from mirage.shared.db import ProjectModelTable  # remove this
```

**Step 2: Simplify _claim_pending query**

Replace lines 40-55 in `embedding_worker.py`:

```python
# DELETE THIS:
result = await session.execute(
    select(EmbeddingStatusTable, ChunkTable, ProjectModelTable, ProjectTable)
    .join(ChunkTable, EmbeddingStatusTable.chunk_id == ChunkTable.id)
    .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
    .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
    .join(
        ProjectModelTable,
        (ProjectModelTable.project_id == ProjectTable.id)
        & (ProjectModelTable.model_name == EmbeddingStatusTable.model_name),
    )
    .where(
        EmbeddingStatusTable.status == "pending",
        ProjectModelTable.enabled == True,
    )
    .limit(1)
)
row = result.first()
if not row:
    return None

embedding_status, chunk, _, project = row

# REPLACE WITH:
result = await session.execute(
    select(EmbeddingStatusTable, ChunkTable, DocumentTable, ProjectTable)
    .join(ChunkTable, EmbeddingStatusTable.chunk_id == ChunkTable.id)
    .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
    .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
    .where(EmbeddingStatusTable.status == "pending")
    .limit(1)
)
row = result.first()
if not row:
    return None

embedding_status, chunk, document, project = row
```

**Step 3: Verify tests pass**

Run: `uv run pytest tests/indexer/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/mirage/indexer/embedding_worker.py
git commit -m "refactor: EmbeddingWorker removes ProjectModelTable JOIN"
```

---

### T02-00-04: Update search router

**Files:**
- Modify: `src/mirage/api/routers/search.py`
- Modify: `src/mirage/api/schemas.py`

**Step 1: Update schemas.py - add models to SearchRequest**

In `src/mirage/api/schemas.py`, find `SearchRequest` class and add:

```python
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.0
    models: list[str] | None = None  # NEW: optional model filter
```

**Step 2: Update search.py imports**

Remove:
```python
from mirage.shared.db import ProjectModelTable  # remove this
```

Add:
```python
from mirage.shared.models_registry import get_all_models, get_model
```

(Keep `get_model` if already imported, add `get_all_models`)

**Step 3: Update search logic**

Replace lines 37-56 in `search.py`:

```python
# DELETE THIS:
if request.models:
    models_result = await db.execute(
        select(ProjectModelTable.model_name).where(
            ProjectModelTable.project_id == project_id,
            ProjectModelTable.enabled == True,
            ProjectModelTable.model_name.in_(request.models),
        )
    )
else:
    models_result = await db.execute(
        select(ProjectModelTable.model_name).where(
            ProjectModelTable.project_id == project_id,
            ProjectModelTable.enabled == True,
        )
    )

model_names = [row[0] for row in models_result.fetchall()]
if not model_names:
    raise HTTPException(status_code=400, detail="No enabled models for search")

# REPLACE WITH:
all_model_names = [m.name for m in get_all_models()]

if request.models:
    # Validate requested models exist
    for m in request.models:
        if m not in all_model_names:
            raise HTTPException(status_code=400, detail=f"Unknown model: {m}")
    model_names = request.models
else:
    model_names = all_model_names
```

**Step 4: Verify tests pass**

Run: `uv run pytest tests/api/test_search.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/mirage/api/schemas.py src/mirage/api/routers/search.py
git commit -m "refactor: search uses registry models, adds optional models filter"
```

---

### T02-00-05: Update projects API

**Files:**
- Modify: `src/mirage/api/routers/projects.py`
- Modify: `src/mirage/api/schemas.py`

**Step 1: Update schemas.py - remove models from ProjectCreate**

In `src/mirage/api/schemas.py`:

```python
class ProjectCreate(BaseModel):
    name: str
    ollama_url: str | None = None
    # REMOVE: models: list[str] | None = None
```

For `ProjectResponse`, either remove `models` field or keep it computed:

```python
class ProjectResponse(BaseModel):
    id: str
    name: str
    ollama_url: str
    created_at: datetime
    models: list[str] = []  # Keep for backward compat, always empty or computed
```

**Step 2: Update projects.py - remove model creation logic**

In `src/mirage/api/routers/projects.py`:

Remove import:
```python
from mirage.shared.db import ProjectModelTable  # remove this
```

Remove import:
```python
from mirage.shared.models_registry import get_all_models, get_model  # remove if only used for models
```

In `create_project` function, delete lines 49-65 (model validation and creation):

```python
# DELETE THIS ENTIRE BLOCK:
if project.models:
    model_names = project.models
else:
    model_names = [m.name for m in get_all_models()]

for model_name in model_names:
    if get_model(model_name) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {model_name}",
        )
    db_model = ProjectModelTable(
        project_id=db_project.id,
        model_name=model_name,
        enabled=True,
    )
    db.add(db_model)
```

In `list_projects`, remove `selectinload(ProjectTable.models)`:

```python
# CHANGE FROM:
result = await db.execute(
    select(ProjectTable).options(selectinload(ProjectTable.models))
)

# TO:
result = await db.execute(select(ProjectTable))
```

Remove unused import:
```python
from sqlalchemy.orm import selectinload  # remove if no longer used
```

**Step 3: Verify tests pass**

Run: `uv run pytest tests/api/test_projects.py -v`
Expected: All tests pass (may need fixture updates)

**Step 4: Commit**

```bash
git add src/mirage/api/schemas.py src/mirage/api/routers/projects.py
git commit -m "refactor: projects API removes model selection"
```

---

### T02-00-06: Update CLI

**Files:**
- Modify: `src/mirage/cli/commands/projects.py`

**Step 1: Remove --model flag from create command**

In `src/mirage/cli/commands/projects.py`:

```python
# CHANGE FROM:
@app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    models: list[str] = typer.Option(
        None, "--model", "-m", help="Embedding models to use (can be specified multiple times)"
    ),
    ollama_url: str | None = typer.Option(None, "--ollama-url", help="Ollama server URL"),
):
    payload: dict = {"name": name}
    if models:
        payload["models"] = models
    if ollama_url:
        payload["ollama_url"] = ollama_url

# TO:
@app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    ollama_url: str | None = typer.Option(None, "--ollama-url", help="Ollama server URL"),
):
    payload: dict = {"name": name}
    if ollama_url:
        payload["ollama_url"] = ollama_url
```

**Step 2: Update list command - remove Models column**

```python
# CHANGE FROM:
typer.echo(f"{'ID':<40} {'Name':<30} {'Models':<25} {'Created':<20}")
typer.echo("-" * 115)
for p in projects:
    created = p["created_at"][:19].replace("T", " ")
    models = ", ".join(m["model_name"] for m in p.get("models", []))
    typer.echo(f"{p['id']:<40} {p['name']:<30} {models:<25} {created:<20}")

# TO:
typer.echo(f"{'ID':<40} {'Name':<30} {'Created':<20}")
typer.echo("-" * 90)
for p in projects:
    created = p["created_at"][:19].replace("T", " ")
    typer.echo(f"{p['id']:<40} {p['name']:<30} {created:<20}")
```

**Step 3: Verify CLI works**

Run: `uv run mirage projects --help`
Expected: Shows commands without --model flag

**Step 4: Commit**

```bash
git add src/mirage/cli/commands/projects.py
git commit -m "refactor: CLI removes --model flag from project create"
```

---

### T02-00-07: Create Alembic migration

**Files:**
- Create: `src/mirage/migrations/versions/XXXX_drop_project_models.py`

**Step 1: Generate migration**

Run: `uv run alembic revision -m "drop_project_models_table"`

**Step 2: Edit migration file**

```python
"""drop project_models table

Revision ID: <generated>
Revises: <previous>
Create Date: 2026-02-18
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '<generated>'
down_revision = '<previous_revision>'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.drop_table('project_models')

def downgrade() -> None:
    op.create_table(
        'project_models',
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('model_name', sa.String(100), primary_key=True),
        sa.Column('enabled', sa.Boolean, default=True),
    )
```

**Step 3: Commit**

```bash
git add src/mirage/migrations/versions/*drop_project_models*.py
git commit -m "feat: add migration to drop project_models table"
```

---

### T02-00-08: Run tests and fix regressions

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: Some tests may fail due to fixture references to `ProjectModelTable`

**Step 2: Fix test fixtures**

Search for and update any test files that reference:
- `ProjectModelTable`
- `project_models`
- `models` in project fixtures

Common locations:
- `tests/conftest.py`
- `tests/api/test_projects.py`
- `tests/api/test_search.py`
- `tests/indexer/test_worker.py`

**Step 3: Run tests again**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 4: Run linting**

Run: `uv run ruff check src/`
Expected: No errors (or fix them)

Run: `uv run mypy src/`
Expected: No type errors (or fix them)

**Step 5: Final commit**

```bash
git add tests/
git commit -m "fix: update tests for simplified multi-model architecture"
```

**Step 6: Update progress.md**

Mark all tasks as complete in `docs/designs/02-simplify-multi-model/progress.md`.

---

## Completion Criteria

- [ ] All tests pass (`uv run pytest`)
- [ ] No lint errors (`uv run ruff check src/`)
- [ ] No type errors (`uv run mypy src/`)
- [ ] CLI works without --model flag
- [ ] API creates projects without model selection
- [ ] Search accepts optional models filter
- [ ] Migration drops project_models table
