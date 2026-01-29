import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_documents_list_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "1", "filename": "test.pdf", "status": "ready"}
    ]

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(app, ["documents", "list", "--project", "test-project"])

    assert result.exit_code == 0
    assert "test.pdf" in result.stdout


def test_documents_status_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "doc-1",
        "filename": "test.pdf",
        "status": "ready",
        "file_type": "pdf",
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            app, ["documents", "status", "--project", "test-project", "doc-1"]
        )

    assert result.exit_code == 0
    assert "ready" in result.stdout
