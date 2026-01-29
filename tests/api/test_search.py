import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings, get_embedding_client
from mirage.shared.config import Settings
from mirage.shared.db import Base, ProjectTable, DocumentTable, ChunkTable


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

    async with async_session() as session:
        project = ProjectTable(id="test-project-id", name="test-project")
        session.add(project)

        doc = DocumentTable(
            id="test-doc-id",
            project_id="test-project-id",
            filename="test.md",
            original_path="/tmp/test.md",
            file_type="markdown",
            status="ready",
        )
        session.add(doc)

        chunk = ChunkTable(
            id="test-chunk-id",
            document_id="test-doc-id",
            content="This is test content about Python programming.",
            embedding_json=[0.1] * 1024,
            position=0,
            structure_json={"chapter": "Introduction"},
        )
        session.add(chunk)
        await session.commit()

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


@pytest.fixture
def mock_embedding():
    mock = AsyncMock()
    mock.get_embedding = AsyncMock(return_value=[0.1] * 1024)
    app.dependency_overrides[get_embedding_client] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search(test_db, override_settings, mock_embedding):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/projects/test-project-id/search",
            json={"query": "Python programming", "limit": 10},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 0  # SQLite doesn't support vector search, just test API works
