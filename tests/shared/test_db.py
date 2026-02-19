import pytest
from sqlalchemy import text
from mirage.shared.db import (
    get_engine,
    create_tables,
    ProjectTable,
    EmbeddingStatusTable,
    get_embeddings_table_class,
)
from mirage.shared.models_registry import get_model


@pytest.fixture
def test_db_url():
    return "sqlite+aiosqlite:///:memory:"


@pytest.mark.asyncio
async def test_create_tables(test_db_url):
    engine = get_engine(test_db_url)
    await create_tables(engine)

    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = {row[0] for row in result.fetchall()}

    assert "projects" in tables
    assert "documents" in tables
    assert "chunks" in tables
    assert "indexing_tasks" in tables

    await engine.dispose()


@pytest.mark.asyncio
async def test_chunks_table_has_status_column(test_db_url):
    engine = get_engine(test_db_url)
    await create_tables(engine)

    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(chunks)"))
        columns = {row[1] for row in result.fetchall()}

    assert "status" in columns
    await engine.dispose()


def test_embedding_status_table():
    """Test EmbeddingStatusTable can be instantiated."""
    es = EmbeddingStatusTable(
        chunk_id="test-chunk-id",
        model_name="nomic-embed-text",
        status="pending",
    )
    assert es.model_name == "nomic-embed-text"
    assert es.status == "pending"


def test_get_embeddings_table_class_nomic():
    """Test dynamic embeddings table class for nomic."""
    model = get_model("nomic-embed-text")
    TableClass = get_embeddings_table_class(model)
    assert TableClass.__tablename__ == "embeddings_nomic_768"


def test_get_embeddings_table_class_bge_m3():
    """Test dynamic embeddings table class for bge-m3."""
    model = get_model("bge-m3")
    TableClass = get_embeddings_table_class(model)
    assert TableClass.__tablename__ == "embeddings_bge_m3_1024"
