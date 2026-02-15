import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import (
    ChunkTable,
    DocumentTable,
    EmbeddingStatusTable,
    ProjectModelTable,
    ProjectTable,
    get_embeddings_table_class,
    get_engine,
)
from mirage.shared.embedding import OllamaEmbedding
from mirage.shared.models_registry import get_model

logger = logging.getLogger(__name__)


@dataclass
class PendingEmbedding:
    chunk_id: str
    model_name: str
    content: str
    ollama_url: str


class MultiModelEmbeddingWorker:
    """Claims pending embeddings and processes them via Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _claim_pending(self, session: AsyncSession) -> PendingEmbedding | None:
        """Find a pending embedding_status row with project's enabled model."""
        result = await session.execute(
            select(EmbeddingStatusTable, ChunkTable, ProjectModelTable, ProjectTable)
            .join(ChunkTable, EmbeddingStatusTable.chunk_id == ChunkTable.id)
            .join(DocumentTable, ChunkTable.document_id == DocumentTable.id)
            .join(ProjectTable, DocumentTable.project_id == ProjectTable.id)
            .join(
                ProjectModelTable,
                (ProjectModelTable.project_id == ProjectTable.id)
                & (ProjectModelTable.model_name == EmbeddingStatusTable.model_name),
            )
            .where(
                EmbeddingStatusTable.status == "pending",
                ProjectModelTable.enabled == True,
            )
            .limit(1)
        )
        row = result.first()
        if not row:
            return None

        embedding_status, chunk, _, project = row

        embedding_status.status = "processing"
        await session.flush()

        return PendingEmbedding(
            chunk_id=chunk.id,
            model_name=embedding_status.model_name,
            content=chunk.content,
            ollama_url=project.ollama_url,
        )

    async def process_one(self, session: AsyncSession) -> bool:
        """Process a single embedding. Returns True if processed."""
        pending = await self._claim_pending(session)
        if not pending:
            return False

        model = get_model(pending.model_name)
        if not model:
            logger.error(f"Unknown model: {pending.model_name}")
            return False

        client = OllamaEmbedding(pending.ollama_url, model.ollama_name)
        result = await client.get_embedding(pending.content, prefix="search_document: ")

        status_record = await session.execute(
            select(EmbeddingStatusTable).where(
                EmbeddingStatusTable.chunk_id == pending.chunk_id,
                EmbeddingStatusTable.model_name == pending.model_name,
            )
        )
        status = status_record.scalar_one()

        if result is None:
            status.status = "failed"
            status.error_message = "Embedding request failed"
        else:
            TableClass = get_embeddings_table_class(model)
            embedding_row = TableClass(
                chunk_id=pending.chunk_id,
                embedding=result.embedding,
            )
            session.add(embedding_row)
            status.status = "ready"

        await session.commit()
        logger.info(f"Embedded chunk {pending.chunk_id} with {pending.model_name}")
        return True

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("MultiModelEmbeddingWorker started")

        while True:
            async with async_session() as session:
                processed = await self.process_one(session)

            if not processed:
                await asyncio.sleep(2)
