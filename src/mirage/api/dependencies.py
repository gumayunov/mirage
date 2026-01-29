from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.shared.config import Settings
from mirage.shared.db import get_engine
from mirage.shared.embedding import OllamaEmbedding


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def verify_api_key(
    x_api_key: Annotated[str, Header()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


async def get_db_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncSession:
    engine = get_engine(settings.database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session


def get_embedding_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OllamaEmbedding:
    return OllamaEmbedding(settings.ollama_url, settings.ollama_model)
