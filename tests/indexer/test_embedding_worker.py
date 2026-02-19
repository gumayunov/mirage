import pytest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import (
    Base,
    ChunkTable,
    DocumentTable,
    EmbeddingStatusTable,
    ProjectTable,
    get_embeddings_table_class,
)
from mirage.shared.embedding import EmbeddingResult
from mirage.shared.models_registry import get_model
from mirage.indexer.embedding_worker import MultiModelEmbeddingWorker


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
        
        model = get_model("nomic-embed-text")
        TableClass = get_embeddings_table_class(model)
        await conn.run_sync(lambda sync_conn: TableClass.__table__.create(sync_conn, checkfirst=True))

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test", ollama_url="http://localhost:11434")
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
        embedding_status = EmbeddingStatusTable(
            chunk_id="chunk-1",
            model_name="nomic-embed-text",
            status="pending",
        )
        session.add(embedding_status)
        await session.commit()

    yield session_factory

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_worker_processes_chunk_ready(settings, db_session, monkeypatch):
    mock_client = MagicMock()
    mock_client.get_embedding = AsyncMock(
        return_value=EmbeddingResult(embedding=[0.1] * 768, truncated=False)
    )

    worker = MultiModelEmbeddingWorker(settings)

    def mock_create_client(url, model):
        return mock_client

    monkeypatch.setattr(
        "mirage.indexer.embedding_worker.OllamaEmbedding", mock_create_client
    )

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is True

    async with db_session() as session:
        status = (
            (
                await session.execute(
                    select(EmbeddingStatusTable).where(
                        EmbeddingStatusTable.chunk_id == "chunk-1"
                    )
                )
            )
            .scalar_one()
        )
        assert status.status == "ready"


@pytest.mark.asyncio
async def test_embedding_worker_ollama_error(settings, db_session, monkeypatch):
    mock_client = MagicMock()
    mock_client.get_embedding = AsyncMock(return_value=None)

    worker = MultiModelEmbeddingWorker(settings)

    def mock_create_client(url, model):
        return mock_client

    monkeypatch.setattr(
        "mirage.indexer.embedding_worker.OllamaEmbedding", mock_create_client
    )

    async with db_session() as session:
        await worker.process_one(session)

    async with db_session() as session:
        status = (
            (
                await session.execute(
                    select(EmbeddingStatusTable).where(
                        EmbeddingStatusTable.chunk_id == "chunk-1"
                    )
                )
            )
            .scalar_one()
        )
        assert status.status == "failed"
        assert status.error_message is not None


@pytest.mark.asyncio
async def test_embedding_worker_no_pending(settings, db_session):
    async with db_session() as session:
        status = (
            (
                await session.execute(
                    select(EmbeddingStatusTable).where(
                        EmbeddingStatusTable.chunk_id == "chunk-1"
                    )
                )
            )
            .scalar_one()
        )
        status.status = "ready"
        await session.commit()

    worker = MultiModelEmbeddingWorker(settings)

    async with db_session() as session:
        processed = await worker.process_one(session)

    assert processed is False
