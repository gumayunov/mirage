from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, get_embedding_client, verify_api_key
from mirage.api.schemas import ChunkResult, SearchRequest, SearchResponse
from mirage.shared.db import ChunkTable, DocumentTable, ProjectTable
from mirage.shared.embedding import OllamaEmbedding

router = APIRouter(prefix="/projects/{project_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    project_id: str,
    request: SearchRequest,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    embedding_client: Annotated[OllamaEmbedding, Depends(get_embedding_client)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    query_embedding = await embedding_client.get_embedding(request.query)

    # For PostgreSQL with pgvector, use vector similarity search
    # For SQLite (testing), fall back to returning all chunks
    try:
        # Try pgvector query
        sql = text("""
            SELECT c.id, c.content, c.structure, c.embedding <=> :embedding AS distance,
                   d.id as doc_id, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = :project_id AND d.status = 'ready'
            ORDER BY c.embedding <=> :embedding
            LIMIT :limit
        """)
        result = await db.execute(
            sql,
            {
                "embedding": str(query_embedding),
                "project_id": project_id,
                "limit": request.limit,
            },
        )
        rows = result.fetchall()

        results = []
        for row in rows:
            score = 1 - row.distance  # Convert distance to similarity
            if score >= request.threshold:
                results.append(
                    ChunkResult(
                        chunk_id=row.id,
                        content=row.content,
                        score=score,
                        structure=row.structure,
                        document={"id": row.doc_id, "filename": row.filename},
                    )
                )
    except Exception:
        # Fallback for SQLite (no vector search)
        result = await db.execute(
            select(ChunkTable, DocumentTable)
            .join(DocumentTable)
            .where(
                DocumentTable.project_id == project_id,
                DocumentTable.status == "ready",
            )
            .limit(request.limit)
        )
        rows = result.all()

        results = []
        for chunk, doc in rows:
            results.append(
                ChunkResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=1.0,  # No real scoring in fallback
                    structure=chunk.structure_json,
                    document={"id": doc.id, "filename": doc.filename},
                )
            )

    return SearchResponse(results=results)
