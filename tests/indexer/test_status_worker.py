import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, ProjectTable
from mirage.indexer.status_worker import StatusWorker


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

    async with session_factory() as session:
        project = ProjectTable(id="proj-1", name="test")
        session.add(project)
        await session.commit()

    yield session_factory

    await engine.dispose()


async def _create_doc_with_chunks(session_factory, doc_id, chunk_statuses):
    async with session_factory() as session:
        doc = DocumentTable(
            id=doc_id,
            project_id="proj-1",
            filename=f"{doc_id}.md",
            original_path=f"/tmp/{doc_id}.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)
        parent_id = f"parent-{doc_id}"
        parent = ChunkTable(
            id=parent_id,
            document_id=doc_id,
            content="Parent chunk",
            position=0,
            status="parent",
        )
        session.add(parent)
        for i, status in enumerate(chunk_statuses):
            chunk = ChunkTable(
                document_id=doc_id,
                content=f"Chunk {i}",
                position=i,
                status=status,
                parent_id=parent_id,
            )
            session.add(chunk)
        await session.commit()


@pytest.mark.asyncio
async def test_status_worker_all_ready(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-1", ["ready", "ready", "ready"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-1")
        )).scalar_one()
        assert doc.status == "ready"
        assert doc.indexed_at is not None


@pytest.mark.asyncio
async def test_status_worker_partial(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-2", ["ready", "corrupted", "error"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-2")
        )).scalar_one()
        assert doc.status == "partial"
        assert doc.indexed_at is not None


@pytest.mark.asyncio
async def test_status_worker_still_processing(settings, db_session):
    await _create_doc_with_chunks(db_session, "doc-3", ["ready", "pending", "processing"])

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-3")
        )).scalar_one()
        assert doc.status == "indexing"  # unchanged
        assert doc.indexed_at is None


@pytest.mark.asyncio
async def test_status_worker_ignores_non_indexing_docs(settings, db_session):
    # Create a "ready" document — should not be touched
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-4",
            project_id="proj-1",
            filename="done.md",
            original_path="/tmp/done.md",
            file_type="markdown",
            status="ready",
        )
        session.add(doc)
        await session.commit()

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-4")
        )).scalar_one()
        assert doc.status == "ready"


@pytest.mark.asyncio
async def test_status_worker_ignores_parent_chunks(settings, db_session):
    """StatusWorker should only consider child chunks for document readiness."""
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-5",
            project_id="proj-1",
            filename="parent-child.md",
            original_path="/tmp/parent-child.md",
            file_type="markdown",
            status="indexing",
        )
        session.add(doc)

        # Parent chunk with status="parent" — should not prevent "ready"
        parent = ChunkTable(
            id="parent-1",
            document_id="doc-5",
            content="Parent",
            position=0,
            status="parent",
        )
        session.add(parent)

        # All children are ready
        for i in range(2):
            child = ChunkTable(
                document_id="doc-5",
                content=f"Child {i}",
                position=i,
                status="ready",
                parent_id="parent-1",
            )
            session.add(child)
        await session.commit()

    worker = StatusWorker(settings)
    async with db_session() as session:
        await worker.check_documents(session)

    async with db_session() as session:
        doc = (await session.execute(select(DocumentTable).where(DocumentTable.id == "doc-5"))).scalar_one()
        assert doc.status == "ready"
