# Phase 3.5.1: Chunk Status — Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare the `shared/` layer for the worker pipeline: add chunk status tracking to the DB schema, return structured results from the embedding client, and adjust the default chunk size.

**Parent design:** `docs/plans/2026-01-29-chunk-status-design.md`

**Phases:**
- **Phase 3.5.1 — Foundation** (this file): Schema, embedding client, config
- **Phase 3.5.2 — Worker Pipeline:** ChunkWorker, EmbeddingWorker, StatusWorker
- **Phase 3.5.3 — Visibility:** API and CLI progress display, regression fixes

**Tech Stack:** Python 3.14, SQLAlchemy + aiosqlite, FastAPI, httpx, typer

---

### Task CSL-1: Add `status` column to ChunkTable

**Files:**
- Modify: `src/mirage/shared/db.py:42-53`
- Test: `tests/shared/test_db.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_db.py`:

```python
@pytest.mark.asyncio
async def test_chunks_table_has_status_column(test_db_url):
    engine = get_engine(test_db_url)
    await create_tables(engine)

    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(chunks)"))
        columns = {row[1] for row in result.fetchall()}

    assert "status" in columns
    await engine.dispose()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_db.py::test_chunks_table_has_status_column -v`
Expected: FAIL — `assert 'status' in columns`

**Step 3: Write minimal implementation**

In `src/mirage/shared/db.py`, add to `ChunkTable` class after `metadata_json`:

```python
status: Mapped[str] = mapped_column(String(50), default="pending")
```

Also add `String` is already imported — no new imports needed.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_db.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/db.py tests/shared/test_db.py
git commit -m "feat: add status column to ChunkTable"
```

---

### Task CSL-2: Return `EmbeddingResult` from embedding client

**Files:**
- Modify: `src/mirage/shared/embedding.py`
- Test: `tests/shared/test_embedding.py`

**Step 1: Write the failing test**

Replace the existing tests in `tests/shared/test_embedding.py` with:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mirage.shared.embedding import OllamaEmbedding, EmbeddingResult, MAX_PROMPT_CHARS


@pytest.mark.asyncio
async def test_get_embedding_returns_result():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding("short text")

    assert isinstance(result, EmbeddingResult)
    assert len(result.embedding) == 1024
    assert result.truncated is False


@pytest.mark.asyncio
async def test_get_embedding_truncated():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.2] * 1024}
    mock_response.raise_for_status = MagicMock()

    long_text = "x" * (MAX_PROMPT_CHARS + 100)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding(long_text)

    assert isinstance(result, EmbeddingResult)
    assert result.truncated is True
    assert len(result.embedding) == 1024


@pytest.mark.asyncio
async def test_get_embedding_ollama_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        result = await client.get_embedding("test text")

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_embedding.py -v`
Expected: FAIL — `ImportError: cannot import name 'EmbeddingResult'`

**Step 3: Write minimal implementation**

Replace `src/mirage/shared/embedding.py`:

```python
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# mxbai-embed-large has a 512-token context window.
# Empirically, 729 chars already exceeds the limit while 694 passes.
# Use 500 chars as a safe ceiling (~1 char per token for worst case).
MAX_PROMPT_CHARS = 500


@dataclass
class EmbeddingResult:
    embedding: list[float]
    truncated: bool


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> EmbeddingResult | None:
        logger.info("Embedding request: %d chars | %s", len(text), text[:200])
        truncated = False
        if len(text) > MAX_PROMPT_CHARS:
            logger.warning(
                "Truncating embedding input from %d to %d chars",
                len(text), MAX_PROMPT_CHARS,
            )
            text = text[:MAX_PROMPT_CHARS]
            truncated = True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60.0,
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                return EmbeddingResult(embedding=embedding, truncated=truncated)
        except Exception:
            logger.exception("Embedding request failed")
            return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_embedding.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/mirage/shared/embedding.py tests/shared/test_embedding.py
git commit -m "feat: return EmbeddingResult from embedding client, handle errors gracefully"
```

---

### Task CSL-3: Update `chunk_size` config default

**Files:**
- Modify: `src/mirage/shared/config.py:9`
- Test: `tests/shared/test_config.py`

**Step 1: Write the failing test**

Add to `tests/shared/test_config.py`:

```python
def test_settings_chunk_size_default(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    settings = Settings()
    assert settings.chunk_size == 400
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/shared/test_config.py::test_settings_chunk_size_default -v`
Expected: FAIL — `assert 128 == 400`

**Step 3: Write minimal implementation**

In `src/mirage/shared/config.py`, change line 9:

```python
chunk_size: int = 400
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/shared/test_config.py -v`
Expected: ALL PASS (check that the existing `test_settings_defaults` still expects 128 — if so, update it to 400 too)

**Step 5: Commit**

```bash
git add src/mirage/shared/config.py tests/shared/test_config.py
git commit -m "feat: increase default chunk_size to 400"
```
