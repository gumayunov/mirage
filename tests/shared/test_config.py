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
    assert settings.ollama_model == "nomic-embed-text"
    assert settings.chunk_size == 3000
    assert settings.chunk_overlap == 200


def test_settings_chunk_size_default(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    settings = Settings()
    assert settings.chunk_size == 3000


def test_settings_child_chunk_defaults(monkeypatch):
    monkeypatch.setenv("MIRAGE_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    settings = Settings()
    assert settings.chunk_size == 3000
    assert settings.chunk_overlap == 200
    assert settings.child_chunk_size == 500
    assert settings.child_chunk_overlap == 50
