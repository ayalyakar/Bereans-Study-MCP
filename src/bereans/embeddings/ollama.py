"""Ollama API client for generating embeddings."""

import httpx


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.host)
                response.raise_for_status()
                return True
        except Exception:
            return False
