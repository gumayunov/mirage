import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, get_engine
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Claims pending chunks and adds embeddings via Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_client = OllamaEmbedding(
            settings.ollama_url,
            settings.ollama_model,
        )

    async def _claim_chunk(self, session: AsyncSession) -> ChunkTable | None:
        """Atomically claim a pending chunk by setting status to 'processing'."""
        # SQLite doesn't support UPDATE ... RETURNING with subquery well,
        # so use SELECT + UPDATE in a transaction (SQLite serializes writes).
        result = await session.execute(
            select(ChunkTable)
            .where(ChunkTable.status == "pending")
            .limit(1)
        )
        chunk = result.scalar_one_or_none()
        if chunk:
            chunk.status = "processing"
            await session.flush()
        return chunk

    async def process_one(self, session: AsyncSession) -> bool:
        """Process a single chunk. Returns True if a chunk was processed."""
        chunk = await self._claim_chunk(session)
        if not chunk:
            return False

        result = await self.embedding_client.get_embedding(chunk.content, prefix="search_document: ")

        if result is None:
            chunk.status = "error"
        elif result.truncated:
            chunk.embedding = result.embedding
            chunk.status = "corrupted"
        else:
            chunk.embedding = result.embedding
            chunk.status = "ready"

        await session.commit()
        logger.info(f"Chunk {chunk.id}: status={chunk.status}")
        return True

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("EmbeddingWorker started")

        while True:
            async with async_session() as session:
                processed = await self.process_one(session)

            if not processed:
                await asyncio.sleep(2)
