import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mirage.indexer.worker import IndexerWorker
from mirage.shared.config import Settings


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    client.get_embeddings = AsyncMock(return_value=[[0.1] * 1024])
    return client


def test_worker_initialization(settings):
    worker = IndexerWorker(settings)
    assert worker.settings == settings


@pytest.mark.asyncio
async def test_worker_process_markdown(settings, mock_embedding_client, tmp_path):
    # Create test file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nContent here.")

    settings.documents_path = str(tmp_path)
    worker = IndexerWorker(settings)
    worker.embedding_client = mock_embedding_client

    chunks = await worker._process_file(str(md_file), "markdown")

    assert len(chunks) >= 1
    assert chunks[0]["content"]
    assert "embedding" in chunks[0]
