import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# nomic-embed-text has an 8192-token context window.
# At ~4 chars/token for English, 8000 chars is a safe ceiling.
MAX_PROMPT_CHARS = 8000


@dataclass
class EmbeddingResult:
    embedding: list[float]
    truncated: bool


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> EmbeddingResult | None:
        logger.info("Embedding request: %d chars | %s", len(text), text[:200])
        truncated = False
        if len(text) > MAX_PROMPT_CHARS:
            logger.warning(
                "Truncating embedding input from %d to %d chars",
                len(text), MAX_PROMPT_CHARS,
            )
            text = text[:MAX_PROMPT_CHARS]
            truncated = True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60.0,
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                return EmbeddingResult(embedding=embedding, truncated=truncated)
        except Exception:
            logger.exception("Embedding request failed")
            return None
