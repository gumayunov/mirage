import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_search_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "chunk_id": "1",
                "content": "This is matching content about Python.",
                "score": 0.85,
                "structure": {"chapter": "Introduction"},
                "document": {"id": "doc-1", "filename": "book.pdf"},
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response):
        result = runner.invoke(
            app, ["search", "--project", "test-project", "Python programming"]
        )

    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "book.pdf" in result.stdout
