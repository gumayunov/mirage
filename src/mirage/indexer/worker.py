import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.chunking import Chunker
from mirage.indexer.parsers.epub import EPUBParser
from mirage.indexer.parsers.markdown import MarkdownParser
from mirage.indexer.parsers.pdf import PDFParser
from mirage.shared.config import Settings
from mirage.shared.db import (
    ChunkTable,
    DocumentTable,
    EmbeddingStatusTable,
    IndexingTaskTable,
    ProjectModelTable,
    get_engine,
)

logger = logging.getLogger(__name__)


class ChunkWorker:
    """Parses documents and saves chunks with status='pending' (no embeddings)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.child_chunk_size = settings.child_chunk_size
        self.child_chunk_overlap = settings.child_chunk_overlap
        self.parsers = {
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "epub": EPUBParser(),
        }

    def _parse_file(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
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

        return [
            {
                "content": chunk.content,
                "position": chunk.position,
                "structure": chunk.structure,
            }
            for chunk in all_chunks
        ]

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

            # Parse file — produces parent-level chunks
            chunks_data = self._parse_file(doc.original_path, doc.file_type)

            parent_chunks = []
            for chunk_data in chunks_data:
                parent = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                    status="parent",
                )
                session.add(parent)
                parent_chunks.append((parent, chunk_data))

            await session.flush()  # generate parent IDs

            # Create child chunks for each parent
            child_count = 0
            for parent, chunk_data in parent_chunks:
                children = self.chunker.chunk_children(
                    parent.content,
                    chunk_data["structure"],
                    child_size=self.child_chunk_size,
                    child_overlap=self.child_chunk_overlap,
                )
                for child in children:
                    child_row = ChunkTable(
                        document_id=doc.id,
                        content=child.content,
                        position=child.position,
                        structure_json=child.structure,
                        status="pending",
                        parent_id=parent.id,
                    )
                    session.add(child_row)
                    child_count += 1

            await session.flush()  # generate child IDs

            # Create embedding_status rows for all child chunks
            result = await session.execute(
                select(ProjectModelTable.model_name).where(
                    ProjectModelTable.project_id == doc.project_id,
                    ProjectModelTable.enabled == True,
                )
            )
            enabled_models = [row[0] for row in result.fetchall()]

            child_result = await session.execute(
                select(ChunkTable.id).where(
                    ChunkTable.document_id == doc.id,
                    ChunkTable.parent_id.is_not(None),
                )
            )
            child_ids = [row[0] for row in child_result.fetchall()]

            for child_id in child_ids:
                for model_name in enabled_models:
                    status_row = EmbeddingStatusTable(
                        chunk_id=child_id,
                        model_name=model_name,
                        status="pending",
                    )
                    session.add(status_row)

            task.status = "done"
            task.completed_at = datetime.utcnow()

            await session.commit()
            embedding_count = child_count * len(enabled_models) if enabled_models else 0
            logger.info(f"Created {len(parent_chunks)} parents, {child_count} children, {embedding_count} embedding statuses for {doc.filename}")

        except Exception as e:
            logger.error(f"Failed to process {doc.filename}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("ChunkWorker started")

        while True:
            async with async_session() as session:
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
