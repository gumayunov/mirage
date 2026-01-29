import pytest
from sqlalchemy import text
from mirage.shared.db import get_engine, create_tables, ProjectTable


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
