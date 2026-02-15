from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from mirage.api.dependencies import get_db_session, verify_api_key
from mirage.api.schemas import ProjectCreate, ProjectResponse
from mirage.shared.db import ProjectTable, ProjectModelTable
from mirage.shared.models_registry import get_all_models, get_model

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).options(selectinload(ProjectTable.models))
    )
    return result.scalars().all()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    existing = await db.execute(
        select(ProjectTable).where(ProjectTable.name == project.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project with this name already exists",
        )

    db_project = ProjectTable(
        name=project.name,
        ollama_url=project.ollama_url or "http://ollama:11434",
    )
    db.add(db_project)
    await db.flush()

    if project.models:
        model_names = project.models
    else:
        model_names = [m.name for m in get_all_models()]

    for model_name in model_names:
        if get_model(model_name) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model: {model_name}",
            )
        db_model = ProjectModelTable(
            project_id=db_project.id,
            model_name=model_name,
            enabled=True,
        )
        db.add(db_model)

    await db.commit()

    result = await db.execute(
        select(ProjectTable)
        .where(ProjectTable.id == db_project.id)
        .options(selectinload(ProjectTable.models))
    )
    return result.scalar_one()


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(ProjectTable).where(ProjectTable.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    await db.delete(project)
    await db.commit()
