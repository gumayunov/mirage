import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings
from mirage.shared.config import Settings
from mirage.shared.db import Base


@pytest.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db
    yield engine
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def override_settings():
    def _get_settings():
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key="test-key",
        )

    app.dependency_overrides[get_settings] = _get_settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_project_with_custom_models(test_db, override_settings):
    """Test creating project with specific models."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "custom-models-project",
                "models": ["nomic-embed-text"],
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["model_name"] == "nomic-embed-text"


@pytest.mark.asyncio
async def test_project_with_multiple_models(test_db, override_settings):
    """Test creating project with multiple models."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "multi-model-project",
                "models": ["nomic-embed-text", "bge-m3"],
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["models"]) == 2
    model_names = {m["model_name"] for m in data["models"]}
    assert model_names == {"nomic-embed-text", "bge-m3"}


@pytest.mark.asyncio
async def test_project_default_models(test_db, override_settings):
    """Test project defaults to all supported models when not specified."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "default-models-project"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["models"]) == 3


@pytest.mark.asyncio
async def test_project_with_custom_ollama_url(test_db, override_settings):
    """Test creating project with custom Ollama URL."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "custom-ollama-project",
                "models": ["nomic-embed-text"],
                "ollama_url": "http://custom-ollama:11434",
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["ollama_url"] == "http://custom-ollama:11434"
