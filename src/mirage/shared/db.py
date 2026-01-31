import uuid
from datetime import datetime

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, JSON, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class ProjectTable(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    documents: Mapped[list["DocumentTable"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class DocumentTable(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped["ProjectTable"] = relationship(back_populates="documents")
    chunks: Mapped[list["ChunkTable"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    tasks: Mapped[list["IndexingTaskTable"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ChunkTable(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(768), nullable=True)
    position: Mapped[int] = mapped_column(Integer)
    structure_json: Mapped[str | None] = mapped_column("structure", JSON, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    parent_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True)

    document: Mapped["DocumentTable"] = relationship(back_populates="chunks")
    children: Mapped[list["ChunkTable"]] = relationship(back_populates="parent", cascade="all, delete-orphan")
    parent: Mapped["ChunkTable | None"] = relationship(back_populates="children", remote_side=[id])


class IndexingTaskTable(Base):
    __tablename__ = "indexing_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    task_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    document: Mapped["DocumentTable"] = relationship(back_populates="tasks")


def get_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False)


async def _ensure_pgvector(engine: AsyncEngine) -> None:
    if engine.dialect.name != "postgresql":
        return
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def create_tables(engine: AsyncEngine) -> None:
    await _ensure_pgvector(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
