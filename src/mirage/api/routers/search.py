import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, verify_api_key
from mirage.api.schemas import ChunkResult, SearchRequest, SearchResponse
from mirage.shared.db import ProjectTable
from mirage.shared.embedding import OllamaEmbedding
from mirage.shared.models_registry import get_all_models, get_model, get_model_table_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    project_id: str,
    request: SearchRequest,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    logger.info("Search request: project=%s query=%r limit=%d threshold=%.2f",
                project_id, request.query, request.limit, request.threshold)

    # Get models (optionally filtered by request.models)
    all_model_names = [m.name for m in get_all_models()]

    if request.models:
        # Validate requested models exist
        for m in request.models:
            if m not in all_model_names:
                raise HTTPException(status_code=400, detail=f"Unknown model: {m}")
        model_names = request.models
    else:
        model_names = all_model_names

    # Embed query with each model (parallel)
    async def embed_query(model_name: str) -> tuple[str, list[float] | None]:
        model = get_model(model_name)
        if not model:
            return model_name, None
        client = OllamaEmbedding(project.ollama_url, model.ollama_name)
        result = await client.get_embedding(request.query, prefix="search_query: ")
        return model_name, result.embedding if result else None

    embedding_tasks = [embed_query(m) for m in model_names]
    embeddings = await asyncio.gather(*embedding_tasks)

    # Search each model's embeddings table
    all_results: list[tuple[str, str, str, float, str, str, dict]] = []

    for model_name, query_embedding in embeddings:
        if query_embedding is None:
            logger.warning(f"Failed to embed query with model {model_name}")
            continue

        model = get_model(model_name)
        if not model:
            continue

        table_name = get_model_table_name(model)

        try:
            sql = text(f"""
                SELECT DISTINCT ON (child.parent_id)
                       child.id, child.content, child.structure,
                       e.embedding <=> :embedding AS distance,
                       parent.content AS parent_content,
                       d.id as doc_id, d.filename
                FROM {table_name} e
                JOIN chunks child ON e.chunk_id = child.id
                JOIN chunks parent ON child.parent_id = parent.id
                JOIN documents d ON child.document_id = d.id
                WHERE d.project_id = :project_id
                  AND d.status IN ('ready', 'partial')
                ORDER BY child.parent_id, e.embedding <=> :embedding
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
            logger.info(f"Model {model_name} returned {len(rows)} results")

            for row in rows:
                all_results.append((
                    row.id,
                    row.content,
                    row.parent_content,
                    row.distance,
                    row.doc_id,
                    row.filename,
                    row.structure,
                ))
        except Exception as e:
            logger.warning(f"Search failed for model {model_name}: {e}")
            continue

    # Deduplicate by chunk_id, keep minimum distance
    seen_chunks: dict[str, tuple] = {}
    for chunk_id, content, parent_content, distance, doc_id, filename, structure in all_results:
        if chunk_id not in seen_chunks or seen_chunks[chunk_id][2] > distance:
            seen_chunks[chunk_id] = (content, parent_content, distance, doc_id, filename, structure)

    # Sort by distance and limit
    sorted_results = sorted(seen_chunks.items(), key=lambda x: x[1][2])[:request.limit]

    # Build response
    results = []
    for chunk_id, (content, parent_content, distance, doc_id, filename, structure) in sorted_results:
        score = 1 - distance
        if score >= request.threshold:
            results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    content=content,
                    parent_content=parent_content,
                    score=score,
                    structure=structure,
                    document={"id": doc_id, "filename": filename},
                )
            )

    logger.info("Returning %d results", len(results))
    return SearchResponse(results=results)
