import pytest
from unittest.mock import AsyncMock

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import Base, ChunkTable, DocumentTable, IndexingTaskTable, ProjectTable
from mirage.indexer.worker import ChunkWorker


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path=str(tmp_path),
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


def test_chunk_worker_initialization(settings):
    worker = ChunkWorker(settings)
    assert worker.settings == settings


@pytest.mark.asyncio
async def test_chunk_worker_creates_pending_chunks(settings, db_session, tmp_path):
    # Create a markdown file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nFirst paragraph content.\n\nSecond paragraph content.")

    # Create document and task in DB
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-1",
            project_id="proj-1",
            filename="test.md",
            original_path=str(md_file),
            file_type="markdown",
            status="pending",
        )
        session.add(doc)
        task = IndexingTaskTable(
            id="task-1",
            document_id="doc-1",
            task_type="index",
            status="pending",
        )
        session.add(task)
        await session.commit()

    # Process the task
    worker = ChunkWorker(settings)
    async with db_session() as session:
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1")
        )
        task = task_result.scalar_one()
        await worker.process_task(session, task)

    # Verify parent and child chunks were created
    async with db_session() as session:
        chunks_result = await session.execute(
            select(ChunkTable).where(ChunkTable.document_id == "doc-1")
        )
        chunks = chunks_result.scalars().all()

        assert len(chunks) >= 2  # at least 1 parent + 1 child
        parents = [c for c in chunks if c.parent_id is None]
        children = [c for c in chunks if c.parent_id is not None]
        assert len(parents) >= 1
        assert len(children) >= 1
        for p in parents:
            assert p.status == "parent"
            assert p.embedding is None
        for c in children:
            assert c.status == "pending"
            assert c.embedding is None
            assert c.content

        # Verify document status is now "indexing"
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-1")
        )
        doc = doc_result.scalar_one()
        assert doc.status == "indexing"

        # Verify task is done
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-1")
        )
        task = task_result.scalar_one()
        assert task.status == "done"


@pytest.mark.asyncio
async def test_chunk_worker_sets_error_on_parse_failure(settings, db_session, tmp_path):
    # Create document pointing to nonexistent file
    async with db_session() as session:
        doc = DocumentTable(
            id="doc-2",
            project_id="proj-1",
            filename="missing.md",
            original_path="/nonexistent/missing.md",
            file_type="markdown",
            status="pending",
        )
        session.add(doc)
        task = IndexingTaskTable(
            id="task-2",
            document_id="doc-2",
            task_type="index",
            status="pending",
        )
        session.add(task)
        await session.commit()

    worker = ChunkWorker(settings)
    async with db_session() as session:
        task_result = await session.execute(
            select(IndexingTaskTable).where(IndexingTaskTable.id == "task-2")
        )
        task = task_result.scalar_one()
        await worker.process_task(session, task)

    # Verify error status
    async with db_session() as session:
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == "doc-2")
        )
        doc = doc_result.scalar_one()
        assert doc.status == "error"
        assert doc.error_message is not None


@pytest.mark.asyncio
async def test_chunk_worker_creates_parent_and_child_chunks(settings, db_session, tmp_path):
    # Create a markdown file long enough to produce multiple parent chunks
    content = "# Test\n\n" + ("This is a test paragraph with enough text. " * 100 + "\n\n") * 20
    md_file = tmp_path / "test_pc.md"
    md_file.write_text(content)

    async with db_session() as session:
        doc = DocumentTable(
            id="doc-pc",
            project_id="proj-1",
            filename="test_pc.md",
            original_path=str(md_file),
            file_type="markdown",
            status="pending",
        )
        session.add(doc)
        task = IndexingTaskTable(
            id="task-pc",
            document_id="doc-pc",
            task_type="index",
            status="pending",
        )
        session.add(task)
        await session.commit()

    worker = ChunkWorker(settings)

    async with db_session() as session:
        task = (await session.execute(select(IndexingTaskTable).where(IndexingTaskTable.id == "task-pc"))).scalar_one()
        await worker.process_task(session, task)

    async with db_session() as session:
        all_chunks = (await session.execute(select(ChunkTable).where(ChunkTable.document_id == "doc-pc"))).scalars().all()

        parents = [c for c in all_chunks if c.parent_id is None]
        children = [c for c in all_chunks if c.parent_id is not None]

        assert len(parents) > 0, "Should have parent chunks"
        assert len(children) > 0, "Should have child chunks"

        # Parents should have status="parent"
        for p in parents:
            assert p.status == "parent"
            assert p.embedding is None

        # Children should have status="pending" and valid parent_id
        parent_ids = {p.id for p in parents}
        for c in children:
            assert c.status == "pending"
            assert c.parent_id in parent_ids
