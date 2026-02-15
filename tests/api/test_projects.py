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
async def test_create_project(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "test-project"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-project"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_projects(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/v1/projects",
            json={"name": "project1"},
            headers={"X-API-Key": "test-key"},
        )
        response = await client.get(
            "/api/v1/projects",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "project1"


@pytest.mark.asyncio
async def test_delete_project(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_response = await client.post(
            "/api/v1/projects",
            json={"name": "to-delete"},
            headers={"X-API-Key": "test-key"},
        )
        project_id = create_response.json()["id"]

        delete_response = await client.delete(
            f"/api/v1/projects/{project_id}",
            headers={"X-API-Key": "test-key"},
        )

    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_create_project_with_models(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "multi-model-project",
                "models": ["nomic-embed-text", "bge-m3"],
                "ollama_url": "http://custom-ollama:11434",
            },
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "multi-model-project"
    assert data["ollama_url"] == "http://custom-ollama:11434"
    assert set(m["model_name"] for m in data["models"]) == {"nomic-embed-text", "bge-m3"}


@pytest.mark.asyncio
async def test_create_project_default_models(test_db, override_settings):
    """If models not specified, all supported models should be enabled."""
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
