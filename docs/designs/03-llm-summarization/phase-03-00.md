# Phase 03-00: Schema Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add database columns required for parent-chunk summarization without changing any worker behavior yet.

**Architecture:** One Alembic migration `004_summarization.py` adds three columns to `chunks` and two to `projects`. SQLAlchemy ORM models and Pydantic schemas are updated to mirror the new columns. Existing data is backfilled so parent chunks become eligible for the summary pipeline on first run after migration.

**Tech Stack:** Alembic, SQLAlchemy 2.0 async, Pydantic v2.

---

### Task T03-00-01: Migration — chunk summarization columns

**Files:**
- Create: `src/mirage/migrations/versions/004_summarization.py`
- Reference: `src/mirage/migrations/versions/003_drop_project_models.py` (style)

- [ ] **Step 1: Write the migration**

```python
"""Add summarization columns to chunks and projects.

Revision ID: 004
Revises: 003
Create Date: 2026-05-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("summary_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("summary_status", sa.String(20), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("summary_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("summary_model", sa.String(100), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column(
            "language",
            sa.String(10),
            nullable=False,
            server_default=sa.text("'ru'"),
        ),
    )

    # Backfill: existing parent chunks (status='parent') become pending for both pipelines.
    op.execute(
        "UPDATE chunks SET status='pending', summary_status='pending' "
        "WHERE status='parent'"
    )


def downgrade() -> None:
    op.drop_column("projects", "language")
    op.drop_column("projects", "summary_model")
    op.drop_column("chunks", "summary_error")
    op.drop_column("chunks", "summary_status")
    op.drop_column("chunks", "summary_text")
```

- [ ] **Step 2: Apply migration on dev DB**

Run: `uv run alembic upgrade head`
Expected: `Running upgrade 003 -> 004, Add summarization columns…`

- [ ] **Step 3: Verify schema**

Run: `uv run python -c "import asyncio; from mirage.shared.config import Settings; from mirage.shared.db import get_engine; from sqlalchemy import text; s=Settings(); e=get_engine(s.database_url); asyncio.run(e.dispose())"` (smoke import)
Then in psql or sqlite shell: `\d chunks` / `PRAGMA table_info(chunks)` — confirm three new columns exist.

- [ ] **Step 4: Verify rollback**

Run: `uv run alembic downgrade -1 && uv run alembic upgrade head`
Expected: both succeed without error.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/migrations/versions/004_summarization.py
git commit -m "feat(db): add summarization columns to chunks and projects (T03-00-01..03)"
```

---

### Task T03-00-04: Update SQLAlchemy models

**Files:**
- Modify: `src/mirage/shared/db.py:17-25` (ProjectTable), `src/mirage/shared/db.py:57-72` (ChunkTable)
- Test: `tests/shared/test_db.py`

- [ ] **Step 1: Add fields to `ProjectTable`**

Replace the body of `ProjectTable` (between `__tablename__` and `documents` relationship) with:

```python
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True)
    ollama_url: Mapped[str] = mapped_column(String(512), default="http://ollama:11434")
    summary_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="ru")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 2: Add fields to `ChunkTable`**

After the existing `status` column, add:

```python
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary_error: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- [ ] **Step 3: Write failing test**

Append to `tests/shared/test_db.py`:

```python
@pytest.mark.asyncio
async def test_chunk_summarization_columns_persist():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    async with sf() as s:
        s.add(ProjectTable(id="p", name="t", summary_model="mistral:7b", language="en"))
        s.add(DocumentTable(id="d", project_id="p", filename="f", original_path="/f", file_type="markdown"))
        s.add(ChunkTable(
            id="c", document_id="d", content="x", position=0,
            summary_text="## Сущности\n- A", summary_status="done",
        ))
        await s.commit()

    async with sf() as s:
        from sqlalchemy import select
        chunk = (await s.execute(select(ChunkTable))).scalar_one()
        project = (await s.execute(select(ProjectTable))).scalar_one()
        assert chunk.summary_status == "done"
        assert chunk.summary_text.startswith("## Сущности")
        assert chunk.summary_error is None
        assert project.summary_model == "mistral:7b"
        assert project.language == "en"
    await engine.dispose()
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/shared/test_db.py::test_chunk_summarization_columns_persist -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/shared/db.py tests/shared/test_db.py
git commit -m "feat(db): ORM fields for summarization (T03-00-04)"
```

---

### Task T03-00-05: Update Pydantic schemas

**Files:**
- Modify: `src/mirage/api/schemas.py:6-17`
- Test: `tests/api/test_projects.py`

- [ ] **Step 1: Extend `ProjectCreate` and `ProjectResponse`**

```python
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    ollama_url: str | None = None
    summary_model: str | None = None
    language: str = Field(default="ru", min_length=2, max_length=10)


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    ollama_url: str = "http://ollama:11434"
    summary_model: str | None = None
    language: str = "ru"
    created_at: datetime
```

- [ ] **Step 2: Add failing test**

Append to `tests/api/test_projects.py` (use existing `client` fixture):

```python
def test_create_project_with_summary_config(client):
    r = client.post(
        "/projects",
        headers={"X-API-Key": "test-key"},
        json={"name": "p1", "summary_model": "mistral:7b", "language": "en"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["summary_model"] == "mistral:7b"
    assert body["language"] == "en"
```

- [ ] **Step 3: Run — expect failure**

Run: `uv run pytest tests/api/test_projects.py::test_create_project_with_summary_config -v`
Expected: FAIL — `create_project` ignores new fields.

- [ ] **Step 4: Wire fields in router**

Modify `src/mirage/api/routers/projects.py` `create_project` body construction:

```python
    db_project = ProjectTable(
        name=project.name,
        ollama_url=project.ollama_url or "http://ollama:11434",
        summary_model=project.summary_model,
        language=project.language,
    )
```

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/api/test_projects.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mirage/api/schemas.py src/mirage/api/routers/projects.py tests/api/test_projects.py
git commit -m "feat(api): expose summary_model/language on Project (T03-00-05)"
```
