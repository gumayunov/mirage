import pytest
import io
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.api.main import app
from mirage.api.dependencies import get_db_session, get_settings
from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable


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

    # Create test project
    async with async_session() as session:
        project = ProjectTable(id="test-project-id", name="test-project")
        session.add(project)
        await session.commit()

    yield engine

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def override_settings(tmp_path):
    def _get_settings():
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key="test-key",
            documents_path=str(tmp_path),
        )
    app.dependency_overrides[get_settings] = _get_settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_document(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test Document\n\nThis is test content."
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}

        response = await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["filename"] == "test.md"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_list_documents(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test"
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}
        await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )

        response = await client.get(
            "/api/v1/projects/test-project-id/documents",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1


@pytest.mark.asyncio
async def test_get_document_status(test_db, override_settings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"# Test"
        files = {"file": ("test.md", io.BytesIO(file_content), "text/markdown")}
        upload_response = await client.post(
            "/api/v1/projects/test-project-id/documents",
            files=files,
            headers={"X-API-Key": "test-key"},
        )
        doc_id = upload_response.json()["id"]

        response = await client.get(
            f"/api/v1/projects/test-project-id/documents/{doc_id}",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.md"


@pytest.mark.asyncio
async def test_document_status_includes_chunk_counts(test_db, override_settings):
    # Create document with chunks in various statuses
    engine = test_db
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        doc = DocumentTable(
            id="doc-with-chunks",
            project_id="test-project-id",
            filename="chunked.md",
            original_path="/tmp/chunked.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        parent = ChunkTable(
            id="parent-for-doc-with-chunks",
            document_id="doc-with-chunks",
            content="Parent",
            position=0,
            status="parent",
        )
        session.add(parent)
        for i, status in enumerate(["ready", "ready", "pending", "processing"]):
            chunk = ChunkTable(
                document_id="doc-with-chunks",
                content=f"Chunk {i}",
                position=i,
                status=status,
                parent_id="parent-for-doc-with-chunks",
            )
            session.add(chunk)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/projects/test-project-id/documents/doc-with-chunks",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_total"] == 4
    assert data["chunks_processed"] == 2  # only "ready" chunks are processed (not pending/processing)
    assert data["chunks_by_status"] == {"ready": 2, "pending": 1, "processing": 1}


@pytest.mark.asyncio
async def test_list_documents_includes_chunks_by_status(test_db, override_settings):
    engine = test_db
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        doc = DocumentTable(
            id="doc-list-chunks",
            project_id="test-project-id",
            filename="listed.md",
            original_path="/tmp/listed.md",
            file_type="markdown",
            status="indexed",
        )
        session.add(doc)
        parent = ChunkTable(
            id="parent-for-list-chunks",
            document_id="doc-list-chunks",
            content="Parent",
            position=0,
            status="parent",
        )
        session.add(parent)
        for i, chunk_status in enumerate(["ready", "ready", "ready", "error"]):
            chunk = ChunkTable(
                document_id="doc-list-chunks",
                content=f"Chunk {i}",
                position=i,
                status=chunk_status,
                parent_id="parent-for-list-chunks",
            )
            session.add(chunk)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/projects/test-project-id/documents",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    doc_data = next(d for d in data if d["id"] == "doc-list-chunks")
    assert doc_data["chunks_by_status"] == {"ready": 3, "error": 1}


@pytest.mark.asyncio
async def test_document_chunk_counts_exclude_parents(test_db, override_settings):
    """Chunk counts should only count child chunks, not parent chunks."""
    engine = test_db
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        doc = DocumentTable(
            id="doc-parent-child",
            project_id="test-project-id",
            filename="pc.md",
            original_path="/tmp/pc.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)

        parent = ChunkTable(
            id="p1",
            document_id="doc-parent-child",
            content="Parent content",
            position=0,
            status="parent",
        )
        session.add(parent)

        for i, s in enumerate(["ready", "pending"]):
            child = ChunkTable(
                document_id="doc-parent-child",
                content=f"Child {i}",
                position=i,
                status=s,
                parent_id="p1",
            )
            session.add(child)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/projects/test-project-id/documents/doc-parent-child",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    # Should count only children (2), not the parent
    assert data["chunks_total"] == 2
    assert data["chunks_processed"] == 1
    assert "parent" not in data["chunks_by_status"]
