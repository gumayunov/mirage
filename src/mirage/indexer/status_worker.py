import asyncio
import logging
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, get_engine

logger = logging.getLogger(__name__)


class StatusWorker:
    """Polls indexing documents and updates their status based on chunk statuses."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def check_documents(self, session: AsyncSession) -> None:
        result = await session.execute(
            select(DocumentTable).where(DocumentTable.status.in_(["indexing", "partial"]))
        )
        docs = result.scalars().all()

        for doc in docs:
            counts = await session.execute(
                select(
                    ChunkTable.status,
                    func.count().label("cnt"),
                )
                .where(
                    ChunkTable.document_id == doc.id,
                    ChunkTable.parent_id.is_not(None),
                )
                .group_by(ChunkTable.status)
            )
            status_counts = {row[0]: row[1] for row in counts.fetchall()}

            pending = status_counts.get("pending", 0)
            processing = status_counts.get("processing", 0)

            if pending > 0 or processing > 0:
                continue  # still working

            ready = status_counts.get("ready", 0)
            corrupted = status_counts.get("corrupted", 0)
            error = status_counts.get("error", 0)

            if corrupted == 0 and error == 0 and ready > 0:
                doc.status = "ready"
            else:
                doc.status = "partial"

            doc.indexed_at = datetime.utcnow()
            logger.info(
                f"Document {doc.filename}: status={doc.status} "
                f"(ready={ready}, corrupted={corrupted}, error={error})"
            )

        await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("StatusWorker started")

        while True:
            async with async_session() as session:
                await self.check_documents(session)
            await asyncio.sleep(10)
