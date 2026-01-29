import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mirage.api.dependencies import get_db_session, get_settings, verify_api_key
from mirage.api.schemas import DocumentResponse
from mirage.shared.config import Settings
from mirage.shared.db import DocumentTable, IndexingTaskTable, ProjectTable

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])

FILE_TYPE_MAP = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".md": "markdown",
    ".markdown": "markdown",
}


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    project_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(DocumentTable).where(DocumentTable.project_id == project_id)
    )
    return result.scalars().all()


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    project_id: str,
    file: Annotated[UploadFile, File()],
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    ext = Path(file.filename or "").suffix.lower()
    file_type = FILE_TYPE_MAP.get(ext)
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(FILE_TYPE_MAP.keys())}",
        )

    existing = await db.execute(
        select(DocumentTable).where(
            DocumentTable.project_id == project_id,
            DocumentTable.filename == file.filename,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Document with this name already exists")

    docs_dir = Path(settings.documents_path) / project_id
    docs_dir.mkdir(parents=True, exist_ok=True)
    file_path = docs_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = DocumentTable(
        project_id=project_id,
        filename=file.filename,
        original_path=str(file_path),
        file_type=file_type,
        status="pending",
    )
    db.add(doc)
    await db.flush()

    task = IndexingTaskTable(
        document_id=doc.id,
        task_type="index",
        status="pending",
    )
    db.add(task)

    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if os.path.exists(doc.original_path):
        os.remove(doc.original_path)

    await db.delete(doc)
    await db.commit()


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    project_id: str,
    document_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(DocumentTable).where(
            DocumentTable.id == document_id,
            DocumentTable.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = "pending"
    task = IndexingTaskTable(
        document_id=doc.id,
        task_type="reindex",
        status="pending",
    )
    db.add(task)

    await db.commit()
    await db.refresh(doc)
    return doc
