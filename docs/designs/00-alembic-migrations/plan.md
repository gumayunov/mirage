# Alembic Migrations — Implementation Plan

> Design 00 | Plan created 2026-01-31
> **For agent:** Execute tasks sequentially using task IDs.
>
> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Overview

Add Alembic to manage PostgreSQL schema migrations. Replace the destructive `recreate_tables()` call in API startup with a no-op (migrations run separately via `make migrate`). Create a baseline migration that produces the current schema from scratch.

**Goal:** DB schema is managed through versioned Alembic migrations; data persists across restarts.

**Architecture:** Alembic async config reads `MIRAGE_DATABASE_URL`, `env.py` imports `Base.metadata` from `db.py`. A single baseline migration creates all four tables + pgvector extension. API lifespan no longer touches the schema.

**Tech Stack:** Alembic, SQLAlchemy (async), asyncpg, pgvector

---

## Tasks

### T00-00-01: Add alembic dependency

**Files:** `pyproject.toml`

Add `alembic>=1.13.0` to the main dependencies list.

In `pyproject.toml`, add to the `dependencies` array (after the last entry):

```python
"alembic>=1.13.0",
```

**Step 1: Edit pyproject.toml**

Add `"alembic>=1.13.0",` to the `dependencies` list in `pyproject.toml` (line 22, before the closing bracket).

**Step 2: Lock dependencies**

Run: `uv lock`
Expected: lockfile updated, exit 0.

**Step 3: Sync**

Run: `uv sync --all-extras`
Expected: alembic installed, exit 0.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add alembic dependency"
```

---

### T00-00-02: Create alembic.ini

**Files:**
- Create: `alembic.ini`

Create `alembic.ini` in the project root. It should:
- Point `script_location` to `src/mirage/migrations`
- Read `sqlalchemy.url` from the `MIRAGE_DATABASE_URL` env var (we'll override this in `env.py`, but alembic.ini needs a placeholder)

```ini
[alembic]
script_location = src/mirage/migrations
prepend_sys_path = .

# Placeholder — overridden in env.py from MIRAGE_DATABASE_URL
sqlalchemy.url = postgresql+asyncpg://mirage:mirage@localhost:5433/mirage

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Verify:** `cat alembic.ini | head -3`
**Expected:** Shows `[alembic]` and `script_location = src/mirage/migrations`

**Step 2: Commit**

```bash
git add alembic.ini
git commit -m "chore: add alembic.ini config"
```

---

### T00-00-03: Create migrations env.py

**Files:**
- Create: `src/mirage/migrations/__init__.py` (empty)
- Create: `src/mirage/migrations/versions/__init__.py` (empty)
- Create: `src/mirage/migrations/env.py`

Create the directory structure and the async `env.py` that:
1. Reads `MIRAGE_DATABASE_URL` from environment
2. Imports `Base.metadata` as `target_metadata`
3. Implements `run_migrations_offline()` for SQL generation
4. Implements `run_migrations_online()` using `create_async_engine`

```python
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from mirage.shared.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

database_url = os.environ.get("MIRAGE_DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_async_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect to the database."""
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(run_async_migrations)

    await connectable.dispose()


import asyncio

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

**Step 1: Create directory structure**

```bash
mkdir -p src/mirage/migrations/versions
touch src/mirage/migrations/__init__.py
touch src/mirage/migrations/versions/__init__.py
```

**Step 2: Write env.py**

Write the file above to `src/mirage/migrations/env.py`.

**Step 3: Verify import works**

Run: `PYTHONPATH=src uv run python -c "from mirage.migrations import env; print('ok')"`
Expected: This will fail because alembic context isn't available outside of alembic — that's fine. We just want to make sure the file parses.

Actually, better verification:

Run: `PYTHONPATH=src uv run python -c "import ast; ast.parse(open('src/mirage/migrations/env.py').read()); print('syntax ok')"`
Expected: `syntax ok`

**Step 4: Commit**

```bash
git add src/mirage/migrations/
git commit -m "feat: add alembic migrations env.py (async)"
```

---

### T00-00-04: Create script.py.mako template

**Files:**
- Create: `src/mirage/migrations/script.py.mako`

This is the Alembic migration template used when generating new migrations.

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

**Verify:** File exists and contains `${message}` template variable.

**Step 2: Commit**

```bash
git add src/mirage/migrations/script.py.mako
git commit -m "chore: add alembic migration template"
```

---

### T00-00-05: Create baseline migration

**Files:**
- Create: `src/mirage/migrations/versions/001_baseline.py`

Write the baseline migration manually (not via autogenerate) so it's clear and complete. The `upgrade()` creates:
1. pgvector extension
2. `projects` table
3. `documents` table
4. `chunks` table (with `Vector(768)` column)
5. `indexing_tasks` table

The `downgrade()` drops tables in reverse order.

```python
"""Baseline: create all tables.

Revision ID: 001
Revises: None
Create Date: 2026-01-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_path", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("indexed_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("structure", sa.JSON, nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column(
            "parent_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    op.create_table(
        "indexing_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("task_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("indexing_tasks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("projects")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

**Verify:** Syntax check:

Run: `PYTHONPATH=src uv run python -c "import ast; ast.parse(open('src/mirage/migrations/versions/001_baseline.py').read()); print('syntax ok')"`
Expected: `syntax ok`

**Step 2: Commit**

```bash
git add src/mirage/migrations/versions/001_baseline.py
git commit -m "feat: add baseline migration (all four tables + pgvector)"
```

---

### T00-00-06: Remove recreate_tables from API lifespan

**Files:**
- Modify: `src/mirage/api/main.py:8,19-21`

Remove the `recreate_tables` import and the three lines that create an engine, call `recreate_tables`, and dispose the engine. The lifespan should only configure logging.

**Before** (`src/mirage/api/main.py`):

```python
from mirage.shared.db import get_engine, recreate_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure logging, recreate tables (destroys all data!)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    engine = get_engine(settings.database_url)
    await recreate_tables(engine)
    await engine.dispose()
    yield
    # Shutdown: nothing to do
```

**After:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure logging
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    yield
    # Shutdown: nothing to do
```

Changes:
1. Remove line 8: `from mirage.shared.db import get_engine, recreate_tables`
2. Update comment on line 13 to remove "recreate tables" mention
3. Remove lines 19-21 (engine creation, recreate_tables, dispose)

**Verify:** `uv run python -c "from mirage.api.main import app; print('import ok')"`
Expected: `import ok`

**Step 2: Commit**

```bash
git add src/mirage/api/main.py
git commit -m "fix: remove destructive recreate_tables from API startup"
```

---

### T00-00-07: Remove recreate_tables from db.py

**Files:**
- Modify: `src/mirage/shared/db.py:92-97`

Delete the `recreate_tables` function entirely (lines 92-97). Keep `create_tables` — it's used by the indexer and tests.

**Delete these lines:**

```python
async def recreate_tables(engine: AsyncEngine) -> None:
    """Drop all tables and recreate them. WARNING: Destroys all data!"""
    await _ensure_pgvector(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
```

**Verify:** `uv run python -c "from mirage.shared.db import create_tables; print('ok')"`
Expected: `ok`

Also verify recreate_tables is gone:

Run: `uv run python -c "from mirage.shared.db import recreate_tables" 2>&1`
Expected: ImportError

**Step 2: Commit**

```bash
git add src/mirage/shared/db.py
git commit -m "chore: remove recreate_tables function from db.py"
```

---

### T00-00-08: Remove create_tables from indexer startup

**Files:**
- Modify: `src/mirage/indexer/__main__.py:8,21-24`

The indexer currently calls `create_tables()` on startup (line 23). With Alembic managing the schema, the indexer should not touch the schema. Remove the import and the engine/create_tables/dispose block.

**Before:**

```python
from mirage.shared.db import create_tables, get_engine

    # Create tables if needed
    engine = get_engine(settings.database_url)
    await create_tables(engine)
    await engine.dispose()
```

**After:**

Remove line 8 (`from mirage.shared.db import create_tables, get_engine`) and lines 21-24 (comment + engine + create_tables + dispose).

**Verify:** `PYTHONPATH=src uv run python -c "import ast; ast.parse(open('src/mirage/indexer/__main__.py').read()); print('syntax ok')"`
Expected: `syntax ok`

**Step 2: Commit**

```bash
git add src/mirage/indexer/__main__.py
git commit -m "chore: remove create_tables from indexer startup"
```

---

### T00-00-09: Add migrate target to Makefile

**Files:**
- Modify: `Makefile:1` (add to .PHONY list)
- Modify: `Makefile` (add new target)

Add a `migrate` target that runs Alembic upgrade head inside the API container. Also add a `migration` target for generating new migrations.

Add to `.PHONY` line: `migrate migration`

Add these targets after the `db-shell` target (before `clean`):

```makefile
# Run database migrations
migrate:
	docker exec mirage-api uv run alembic upgrade head

# Generate a new migration (usage: make migration msg="add foo column")
migration:
	docker exec mirage-api uv run alembic revision --autogenerate -m "$(msg)"
```

Also update the help target to include the new commands:

Add these echo lines in the help target:
```makefile
	@echo "  make migrate      - Run database migrations"
	@echo "  make migration    - Generate new migration (msg='description')"
```

**Verify:** `make help` shows the new targets.

**Step 2: Commit**

```bash
git add Makefile
git commit -m "feat: add make migrate and make migration targets"
```

---

### T00-00-10: Add alembic.ini to Dockerfile COPY

**Files:**
- Modify: `Dockerfile:9`

The Dockerfile needs to copy `alembic.ini` into the container so `alembic` commands work inside Docker.

**Before (line 9):**

```dockerfile
COPY pyproject.toml uv.lock ./
```

**After:**

```dockerfile
COPY pyproject.toml uv.lock alembic.ini ./
```

The migrations themselves are already copied via `COPY src/ ./src/` on line 15.

**Verify:** `docker build -t mirage-test . --no-cache` (or just syntax check)

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "chore: copy alembic.ini into Docker image"
```

---

### T00-00-11: Run full test suite

**Files:** None (verification only)

Run the full test suite to make sure nothing is broken.

Run: `uv run pytest -v`
Expected: All tests pass. The tests use `create_tables()` with SQLite in fixtures — they don't depend on Alembic.

---

### T00-00-12: Integration test with Docker

**Files:** None (verification only)

End-to-end verification that migrations work in Docker:

**Step 1: Rebuild and start services**

```bash
make dev-build
make dev
```

**Step 2: Run migrations**

```bash
make migrate
```

Expected: `INFO  [alembic.runtime.migration] Running upgrade  -> 001, Baseline: create all tables.`

**Step 3: Verify tables exist**

```bash
make db-shell
# Inside psql:
\dt
# Should show: projects, documents, chunks, indexing_tasks, alembic_version
SELECT * FROM alembic_version;
# Should show: version_num = '001'
\q
```

**Step 4: Verify restart preserves data**

```bash
make dev-stop
make dev
# Tables should still exist, no data loss
make db-shell
# \dt should still show all tables
```

**Step 5: Commit progress**

```bash
git add docs/designs/00-alembic-migrations/progress.md
git commit -m "feat: alembic migrations complete — verified in Docker"
```

---

### T00-00-13: Update global progress.md

**Files:**
- Modify: `docs/plans/progress.md:67`

Mark Task 7.3 as done:

```markdown
- [x] Task 7.3: Database migrations (Alembic)
```

**Commit** with the final implementation commit or separately.
