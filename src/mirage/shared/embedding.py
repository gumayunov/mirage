import httpx


class OllamaEmbedding:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def get_embedding(self, text: str) -> list[float]:
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
