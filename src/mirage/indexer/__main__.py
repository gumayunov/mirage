import asyncio
import logging

from mirage.indexer.embedding_worker import EmbeddingWorker
from mirage.indexer.status_worker import StatusWorker
from mirage.indexer.worker import ChunkWorker
from mirage.shared.config import Settings
from mirage.shared.db import create_tables, get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    settings = Settings()

    # Create tables if needed
    engine = get_engine(settings.database_url)
    await create_tables(engine)
    await engine.dispose()

    # Run all workers concurrently
    chunk_worker = ChunkWorker(settings)
    embedding_worker = EmbeddingWorker(settings)
    status_worker = StatusWorker(settings)

    logger.info("Starting all workers")
    await asyncio.gather(
        chunk_worker.run(),
        embedding_worker.run(),
        status_worker.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
