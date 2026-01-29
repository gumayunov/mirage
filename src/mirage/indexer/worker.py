import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.chunking import Chunker
from mirage.indexer.parsers.epub import EPUBParser
from mirage.indexer.parsers.markdown import MarkdownParser
from mirage.indexer.parsers.pdf import PDFParser
from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, IndexingTaskTable, get_engine
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class IndexerWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedding_client = OllamaEmbedding(
            settings.ollama_url,
            settings.ollama_model,
        )
        self.parsers = {
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "epub": EPUBParser(),
        }

    async def _process_file(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
        parser = self.parsers.get(file_type)
        if not parser:
            raise ValueError(f"Unsupported file type: {file_type}")

        if file_type == "markdown":
            content = Path(file_path).read_text()
            parsed = parser.parse(content)

            all_chunks = []
            for section in parsed["sections"]:
                structure = {
                    "title": parsed["title"],
                    "heading": section["heading"],
                    "level": section["level"],
                    "parent_headings": section["parent_headings"],
                }
                chunks = self.chunker.chunk_text(section["content"], structure)
                all_chunks.extend(chunks)

        elif file_type == "pdf":
            parsed = parser.parse(file_path)

            all_chunks = []
            for page in parsed["pages"]:
                if page["content"]:
                    structure = {
                        "title": parsed["title"],
                        "page": page["page_number"],
                    }
                    chunks = self.chunker.chunk_text(page["content"], structure)
                    all_chunks.extend(chunks)

        elif file_type == "epub":
            parsed = parser.parse(file_path)

            all_chunks = []
            for chapter in parsed["chapters"]:
                if chapter["content"]:
                    structure = {
                        "title": parsed["title"],
                        "chapter": chapter["name"],
                    }
                    chunks = self.chunker.chunk_text(chapter["content"], structure)
                    all_chunks.extend(chunks)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Get embeddings
        if all_chunks:
            texts = [c.content for c in all_chunks]
            embeddings = await self.embedding_client.get_embeddings(texts)

            return [
                {
                    "content": chunk.content,
                    "position": chunk.position,
                    "structure": chunk.structure,
                    "embedding": emb,
                }
                for chunk, emb in zip(all_chunks, embeddings)
            ]

        return []

    async def process_task(self, session: AsyncSession, task: IndexingTaskTable) -> None:
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == task.document_id)
        )
        doc = doc_result.scalar_one_or_none()

        if not doc:
            logger.error(f"Document not found: {task.document_id}")
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            return

        try:
            # Update statuses
            task.status = "processing"
            task.started_at = datetime.utcnow()
            doc.status = "indexing"
            await session.commit()

            # Delete existing chunks if reindexing
            if task.task_type == "reindex":
                await session.execute(
                    ChunkTable.__table__.delete().where(
                        ChunkTable.document_id == doc.id
                    )
                )

            # Process file
            chunks_data = await self._process_file(doc.original_path, doc.file_type)

            # Save chunks
            for chunk_data in chunks_data:
                chunk = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    embedding_json=chunk_data["embedding"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                )
                session.add(chunk)

            # Update statuses
            doc.status = "ready"
            doc.indexed_at = datetime.utcnow()
            task.status = "done"
            task.completed_at = datetime.utcnow()

            await session.commit()
            logger.info(f"Indexed document {doc.filename}: {len(chunks_data)} chunks")

        except Exception as e:
            logger.error(f"Failed to index {doc.filename}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("Indexer worker started")

        while True:
            async with async_session() as session:
                # Get pending task
                result = await session.execute(
                    select(IndexingTaskTable)
                    .where(IndexingTaskTable.status == "pending")
                    .order_by(IndexingTaskTable.created_at)
                    .limit(1)
                )
                task = result.scalar_one_or_none()

                if task:
                    await self.process_task(session, task)
                else:
                    await asyncio.sleep(5)
