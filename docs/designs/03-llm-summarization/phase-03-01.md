# Phase 03-01: Configuration & Summary Worker

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve effective summary model from settings/project, call Ollama chat to produce structured summaries, and persist them on parent chunks via a new `SummaryWorker`.

**Architecture:** A new module `shared/llm.py` wraps Ollama `/api/chat` (stream=false). `shared/summary_prompt.py` exposes `build_prompt(content, language)` selecting a Russian or English template. `indexer/summary_worker.py` polls `chunks` for `summary_status='pending'`, claims one row, calls Ollama, writes `summary_text`+`summary_status='done'` (or `'error'`), and creates `embedding_status` rows for each enabled model when the summary is non-empty.

**Tech Stack:** httpx async, SQLAlchemy 2.0 async, pytest-asyncio, monkeypatch.

---

### Task T03-01-01: `Settings.summary_model`

**Files:**
- Modify: `src/mirage/shared/config.py`
- Test: `tests/shared/test_config.py`

- [ ] **Step 1: Add field**

Append after `ollama_model` in `Settings`:

```python
    summary_model: str | None = None
```

- [ ] **Step 2: Test**

Add to `tests/shared/test_config.py`:

```python
def test_summary_model_default_none(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("MIRAGE_API_KEY", "k")
    from mirage.shared.config import Settings
    assert Settings().summary_model is None


def test_summary_model_from_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("MIRAGE_API_KEY", "k")
    monkeypatch.setenv("MIRAGE_SUMMARY_MODEL", "mistral:7b")
    from mirage.shared.config import Settings
    assert Settings().summary_model == "mistral:7b"
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/shared/test_config.py -v
git add src/mirage/shared/config.py tests/shared/test_config.py
git commit -m "feat(config): MIRAGE_SUMMARY_MODEL setting (T03-01-01)"
```

---

### Task T03-01-02: `resolve_summary_model` helper

**Files:**
- Create: `src/mirage/indexer/summary_resolver.py`
- Test: `tests/indexer/test_summary_resolver.py`

- [ ] **Step 1: Failing test**

```python
from mirage.indexer.summary_resolver import resolve_summary_model

class _P:
    def __init__(self, m): self.summary_model = m

class _S:
    def __init__(self, m): self.summary_model = m

def test_project_overrides_settings():
    assert resolve_summary_model(_P("a"), _S("b")) == "a"

def test_falls_back_to_settings():
    assert resolve_summary_model(_P(None), _S("b")) == "b"

def test_disabled_when_both_none():
    assert resolve_summary_model(_P(None), _S(None)) is None
```

Run: `uv run pytest tests/indexer/test_summary_resolver.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement**

```python
def resolve_summary_model(project, settings) -> str | None:
    return project.summary_model or settings.summary_model
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/indexer/test_summary_resolver.py -v
git add src/mirage/indexer/summary_resolver.py tests/indexer/test_summary_resolver.py
git commit -m "feat(indexer): summary_model resolution (T03-01-02)"
```

---

### Task T03-01-03: Ollama chat client

**Files:**
- Create: `src/mirage/shared/llm.py`
- Test: `tests/shared/test_llm.py`

- [ ] **Step 1: Failing test**

```python
import pytest, httpx
from mirage.shared.llm import OllamaChat

@pytest.mark.asyncio
async def test_chat_returns_text(monkeypatch):
    class FakeResp:
        def __init__(self, payload): self._p = payload; self.status_code = 200
        def raise_for_status(self): pass
        def json(self): return self._p

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return FakeResp({"message": {"content": "## Сущности\n- A"}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: FakeClient())
    out = await OllamaChat("http://x", "mistral:7b").complete("hi")
    assert out == "## Сущности\n- A"


@pytest.mark.asyncio
async def test_chat_returns_none_on_error(monkeypatch):
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: FakeClient())
    out = await OllamaChat("http://x", "mistral:7b").complete("hi")
    assert out is None
```

- [ ] **Step 2: Implement**

`src/mirage/shared/llm.py`:

```python
import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaChat:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def complete(self, prompt: str) -> str | None:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
        except Exception:
            logger.exception("Ollama chat failed")
            return None
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/shared/test_llm.py -v
git add src/mirage/shared/llm.py tests/shared/test_llm.py
git commit -m "feat(shared): OllamaChat client (T03-01-03)"
```

---

### Task T03-01-04: Prompt templates

**Files:**
- Create: `src/mirage/indexer/summary_prompt.py`
- Test: `tests/indexer/test_summary_prompt.py`

- [ ] **Step 1: Failing test**

```python
from mirage.indexer.summary_prompt import build_prompt

def test_ru_template_contains_section_headers():
    p = build_prompt("Алиса вошла в комнату.", "ru")
    assert "## Сущности" in p
    assert "Алиса вошла в комнату." in p

def test_en_template_used_for_en():
    p = build_prompt("Alice entered.", "en")
    assert "## Entities" in p

def test_unknown_language_falls_back_to_ru():
    p = build_prompt("x", "fr")
    assert "## Сущности" in p
```

- [ ] **Step 2: Implement**

```python
_RU = """\
Сделай структурированную выжимку текста ниже в формате Markdown.

Используй ровно эти разделы (пропускай пустые):
## Сущности
## Места
## Даты
## Действия
## Факты
## Описания

Каждый пункт — короткая фраза. Только то, что есть в тексте.

Текст:
{content}
"""

_EN = """\
Produce a structured Markdown summary of the text below.

Use exactly these sections (skip empty ones):
## Entities
## Places
## Dates
## Actions
## Facts
## Descriptions

Each bullet is a short phrase. Only what's in the text.

Text:
{content}
"""

_TEMPLATES = {"ru": _RU, "en": _EN}


def build_prompt(content: str, language: str) -> str:
    template = _TEMPLATES.get(language, _RU)
    return template.format(content=content)
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/indexer/test_summary_prompt.py -v
git add src/mirage/indexer/summary_prompt.py tests/indexer/test_summary_prompt.py
git commit -m "feat(indexer): summary prompt templates ru/en (T03-01-04)"
```

---

### Task T03-01-05: SummaryWorker happy path

**Files:**
- Create: `src/mirage/indexer/summary_worker.py`
- Test: `tests/indexer/test_summary_worker.py`

Reference: mirror the structure of `tests/indexer/test_embedding_worker.py` for fixtures (in-memory sqlite, project/document/parent chunk seed).

- [ ] **Step 1: Failing test**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import (
    Base, ChunkTable, DocumentTable, EmbeddingStatusTable, ProjectTable,
)
from mirage.indexer.summary_worker import SummaryWorker


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="k",
        summary_model="mistral:7b",
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as s:
        s.add(ProjectTable(id="p", name="t", language="ru"))
        s.add(DocumentTable(id="d", project_id="p", filename="f", original_path="/f", file_type="markdown"))
        s.add(ChunkTable(
            id="parent-1", document_id="d", content="Алиса вошла.", position=0,
            status="pending", summary_status="pending",
        ))
        await s.commit()
    yield sf
    await engine.dispose()


@pytest.mark.asyncio
async def test_summary_worker_writes_summary_and_creates_embedding_status(
    settings, db_session, monkeypatch,
):
    chat = MagicMock()
    chat.complete = AsyncMock(return_value="## Сущности\n- Алиса")
    monkeypatch.setattr(
        "mirage.indexer.summary_worker.OllamaChat", lambda *a, **k: chat
    )

    worker = SummaryWorker(settings)
    async with db_session() as s:
        processed = await worker.process_one(s)
    assert processed is True

    async with db_session() as s:
        chunk = (await s.execute(select(ChunkTable).where(ChunkTable.id == "parent-1"))).scalar_one()
        assert chunk.summary_status == "done"
        assert "Алиса" in chunk.summary_text
        statuses = (await s.execute(
            select(EmbeddingStatusTable).where(EmbeddingStatusTable.chunk_id == "parent-1")
        )).scalars().all()
        assert len(statuses) >= 1
        assert all(es.status == "pending" for es in statuses)
```

- [ ] **Step 2: Implement worker**

`src/mirage/indexer/summary_worker.py`:

```python
import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.summary_prompt import build_prompt
from mirage.indexer.summary_resolver import resolve_summary_model
from mirage.shared.config import Settings
from mirage.shared.db import (
    ChunkTable, DocumentTable, EmbeddingStatusTable, ProjectTable, get_engine,
)
from mirage.shared.llm import OllamaChat
from mirage.shared.models_registry import get_all_models

logger = logging.getLogger(__name__)


@dataclass
class _Job:
    chunk: ChunkTable
    project: ProjectTable
    model: str


class SummaryWorker:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def _claim(self, session: AsyncSession) -> _Job | None:
        row = (await session.execute(
            select(ChunkTable, DocumentTable, ProjectTable)
            .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
            .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
            .where(ChunkTable.summary_status == "pending")
            .limit(1)
        )).first()
        if not row:
            return None
        chunk, _doc, project = row
        model = resolve_summary_model(project, self.settings)
        if model is None:
            chunk.summary_status = None
            await session.flush()
            return None
        chunk.summary_status = "processing"
        await session.flush()
        return _Job(chunk=chunk, project=project, model=model)

    async def process_one(self, session: AsyncSession) -> bool:
        job = await self._claim(session)
        if job is None:
            return False

        prompt = build_prompt(job.chunk.content, job.project.language)
        client = OllamaChat(job.project.ollama_url, job.model)
        result = await client.complete(prompt)

        if result is None:
            job.chunk.summary_status = "error"
            job.chunk.summary_error = "LLM request failed"
        else:
            text = result.strip()
            if not text:
                job.chunk.summary_status = "done"
                job.chunk.summary_text = None
            else:
                job.chunk.summary_text = text
                job.chunk.summary_status = "done"
                for m in get_all_models():
                    session.add(EmbeddingStatusTable(
                        chunk_id=job.chunk.id, model_name=m.name, status="pending",
                    ))
        await session.commit()
        return True

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        logger.info("SummaryWorker started")
        while True:
            async with sf() as s:
                processed = await self.process_one(s)
            if not processed:
                await asyncio.sleep(2)
```

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/indexer/test_summary_worker.py::test_summary_worker_writes_summary_and_creates_embedding_status -v
git add src/mirage/indexer/summary_worker.py tests/indexer/test_summary_worker.py
git commit -m "feat(indexer): SummaryWorker happy path (T03-01-05)"
```

---

### Task T03-01-06: SummaryWorker — empty + error paths

**Files:**
- Modify: `tests/indexer/test_summary_worker.py`

- [ ] **Step 1: Add empty-response test**

```python
@pytest.mark.asyncio
async def test_empty_response_marks_done_without_embedding_status(
    settings, db_session, monkeypatch,
):
    chat = MagicMock()
    chat.complete = AsyncMock(return_value="   ")
    monkeypatch.setattr("mirage.indexer.summary_worker.OllamaChat", lambda *a, **k: chat)

    async with db_session() as s:
        await SummaryWorker(settings).process_one(s)

    async with db_session() as s:
        chunk = (await s.execute(select(ChunkTable).where(ChunkTable.id == "parent-1"))).scalar_one()
        assert chunk.summary_status == "done"
        assert chunk.summary_text is None
        rows = (await s.execute(
            select(EmbeddingStatusTable).where(EmbeddingStatusTable.chunk_id == "parent-1")
        )).scalars().all()
        assert rows == []
```

- [ ] **Step 2: Add error test**

```python
@pytest.mark.asyncio
async def test_llm_error_marks_error_with_message(
    settings, db_session, monkeypatch,
):
    chat = MagicMock()
    chat.complete = AsyncMock(return_value=None)
    monkeypatch.setattr("mirage.indexer.summary_worker.OllamaChat", lambda *a, **k: chat)

    async with db_session() as s:
        await SummaryWorker(settings).process_one(s)

    async with db_session() as s:
        chunk = (await s.execute(select(ChunkTable).where(ChunkTable.id == "parent-1"))).scalar_one()
        assert chunk.summary_status == "error"
        assert chunk.summary_error
```

- [ ] **Step 3: Add disabled-model test**

```python
@pytest.mark.asyncio
async def test_disabled_summarization_clears_pending(db_session):
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:", api_key="k", summary_model=None)
    async with db_session() as s:
        await SummaryWorker(settings).process_one(s)
    async with db_session() as s:
        chunk = (await s.execute(select(ChunkTable).where(ChunkTable.id == "parent-1"))).scalar_one()
        assert chunk.summary_status is None
```

- [ ] **Step 4: Run + commit**

```
uv run pytest tests/indexer/test_summary_worker.py -v
git add tests/indexer/test_summary_worker.py
git commit -m "test(indexer): SummaryWorker empty/error/disabled paths (T03-01-06)"
```

---

### Task T03-01-07: Run loop + entry point

**Files:**
- Modify: `src/mirage/indexer/__main__.py`

- [ ] **Step 1: Spawn SummaryWorker alongside existing workers**

Open `src/mirage/indexer/__main__.py`. Add `SummaryWorker(settings).run()` to the list of `asyncio.gather` tasks (mirror how `MultiModelEmbeddingWorker` is launched).

- [ ] **Step 2: Smoke**

Run: `uv run python -m mirage.indexer --help` (or short-lived run); confirm import succeeds and worker logs `SummaryWorker started`.

- [ ] **Step 3: Commit**

```
git add src/mirage/indexer/__main__.py
git commit -m "feat(indexer): wire SummaryWorker into main loop (T03-01-07)"
```
