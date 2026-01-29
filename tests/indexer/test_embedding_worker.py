import pytest
from unittest.mock import AsyncMock

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable
from mirage.shared.embedding import EmbeddingResult
from mirage.indexer.embedding_worker import EmbeddingWorker


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Create test data: project + document + pending chunks
    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test")
        session.add(project)
        doc = DocumentTable(
            id="doc-1",
            project_id="proj-1",
            filename="test.md",
            original_path="/tmp/test.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        parent = ChunkTable(
            id="parent-1",
            document_id="doc-1",
            content="Parent context",
            position=0,
            status="parent",
        )
        session.add(parent)
        chunk = ChunkTable(
            id="chunk-1",
            document_id="doc-1",
            content="Some test content",
            position=0,
            status="pending",
            parent_id="parent-1",
        )
        session.add(chunk)
        await session.commit()

    yield session_factory

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_worker_processes_chunk_ready(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(
        return_value=EmbeddingResult(embedding=[0.1] * 768, truncated=False)
    )

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is True

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "ready"
        assert list(chunk.embedding) == [0.1] * 768


@pytest.mark.asyncio
async def test_embedding_worker_truncated_chunk(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(
        return_value=EmbeddingResult(embedding=[0.2] * 768, truncated=True)
    )

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        await worker.process_one(session)

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "corrupted"
        assert chunk.embedding is not None


@pytest.mark.asyncio
async def test_embedding_worker_ollama_error(settings, db_session):
    mock_client = AsyncMock()
    mock_client.get_embedding = AsyncMock(return_value=None)

    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        await worker.process_one(session)

    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        assert chunk.status == "error"
        assert chunk.embedding is None


@pytest.mark.asyncio
async def test_embedding_worker_no_pending_chunks(settings, db_session):
    # Mark the only chunk as ready
    async with db_session() as session:
        chunk = (await session.execute(
            select(ChunkTable).where(ChunkTable.id == "chunk-1")
        )).scalar_one()
        chunk.status = "ready"
        await session.commit()

    mock_client = AsyncMock()
    worker = EmbeddingWorker(settings)
    worker.embedding_client = mock_client

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is False
