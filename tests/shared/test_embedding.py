import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mirage.shared.embedding import OllamaEmbedding


@pytest.mark.asyncio
async def test_get_embedding():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        embedding = await client.get_embedding("test text")

    assert len(embedding) == 1024
    assert embedding[0] == 0.1


@pytest.mark.asyncio
async def test_get_embeddings_batch():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaEmbedding("http://localhost:11434", "mxbai-embed-large")
        embeddings = await client.get_embeddings(["text1", "text2"])

    assert len(embeddings) == 2
