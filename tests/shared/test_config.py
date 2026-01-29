import os
from mirage.shared.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    monkeypatch.setenv("MIRAGE_OLLAMA_URL", "http://localhost:11434")

    settings = Settings()

    assert settings.database_url == "postgresql://test:test@localhost/test"
    assert settings.api_key == "test-key"
    assert settings.ollama_url == "http://localhost:11434"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")

    settings = Settings()

    assert settings.ollama_url == "http://ollama:11434"
    assert settings.ollama_model == "mxbai-embed-large"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
