import logging

import httpx

logger = logging.getLogger(__name__)

# mxbai-embed-large has a 512-token context window.
# Empirically, 729 chars already exceeds the limit while 694 passes.
# Use 500 chars as a safe ceiling (~1 char per token for worst case).
MAX_PROMPT_CHARS = 500


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> list[float]:
        logger.info("Embedding request: %d chars | %s", len(text), text[:200])
        if len(text) > MAX_PROMPT_CHARS:
            logger.warning(
                "Truncating embedding input from %d to %d chars",
                len(text), MAX_PROMPT_CHARS,
            )
            text = text[:MAX_PROMPT_CHARS]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            embedding = await self.get_embedding(text)
            results.append(embedding)
        return results
