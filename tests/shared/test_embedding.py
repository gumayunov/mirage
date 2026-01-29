import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mirage.shared.embedding import OllamaEmbedding, EmbeddingResult, MAX_PROMPT_CHARS


@pytest.mark.asyncio
async def test_get_embedding_returns_result():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 768}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "nomic-embed-text")
        result = await client.get_embedding("short text")

    assert isinstance(result, EmbeddingResult)
    assert len(result.embedding) == 768
    assert result.truncated is False


@pytest.mark.asyncio
async def test_get_embedding_truncated():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.2] * 768}
    mock_response.raise_for_status = MagicMock()

    long_text = "x" * (MAX_PROMPT_CHARS + 100)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "nomic-embed-text")
        result = await client.get_embedding(long_text)

    assert isinstance(result, EmbeddingResult)
    assert result.truncated is True
    assert len(result.embedding) == 768


@pytest.mark.asyncio
async def test_get_embedding_ollama_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
        client = OllamaEmbedding("http://localhost:11434", "nomic-embed-text")
        result = await client.get_embedding("test text")

    assert result is None
