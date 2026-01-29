import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, get_embedding_client, verify_api_key
from mirage.api.schemas import ChunkResult, SearchRequest, SearchResponse
from mirage.shared.db import ChunkTable, DocumentTable, ProjectTable
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)

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

    logger.info("Search request: project=%s query=%r limit=%d threshold=%.2f",
                project_id, request.query, request.limit, request.threshold)

    embedding_result = await embedding_client.get_embedding(request.query)
    if embedding_result is None:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    query_embedding = embedding_result.embedding
    logger.info("Query embedding: dims=%d", len(query_embedding))

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
        logger.info("pgvector returned %d rows", len(rows))

        results = []
        for row in rows:
            score = 1 - row.distance  # Convert distance to similarity
            logger.debug("Chunk %s: distance=%.4f score=%.4f (threshold=%.2f) doc=%s",
                         row.id, row.distance, score, request.threshold, row.filename)
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

        if rows and not results:
            best = min(rows, key=lambda r: r.distance)
            logger.warning("All %d rows filtered by threshold %.2f; best score=%.4f",
                           len(rows), request.threshold, 1 - best.distance)

    except Exception:
        logger.exception("pgvector query failed, falling back to SQLite")
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
        logger.info("Fallback returned %d rows", len(rows))

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

    logger.info("Returning %d results", len(results))
    return SearchResponse(results=results)
