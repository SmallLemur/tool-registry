"""OpenAI-compatible embedding provider.

Calls any OpenAI-compatible /v1/embeddings endpoint.
Works with: OpenAI, vLLM, LiteLLM, OpenRouter, local llama.cpp servers.

Configuration:
  OPENAI_EMBEDDING_URL: base URL (default: https://api.openai.com/v1)
  OPENAI_API_KEY:       API key
  EMBEDDING_MODEL:      model name (e.g. text-embedding-3-small, text-embedding-ada-002)
  EMBEDDING_DIM:        expected output dimension
"""

import logging

import httpx

from app.config import settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# How many texts to embed per API call
BATCH_SIZE = 64


class OpenAIProvider(EmbeddingProvider):
    """OpenAI-compatible HTTP embedding provider."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self._model_name = model_name or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._base_url = settings.openai_embedding_url.rstrip("/")
        self._api_key = settings.openai_api_key
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )
        logger.info(
            "OpenAIProvider ready: %s, model=%s", self._base_url, self._model_name
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        logger.info("OpenAIProvider shut down")

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            raise RuntimeError("OpenAIProvider not started")

        all_embeddings: list[list[float]] = []

        # Process in batches to avoid request size limits
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = await self._client.post(
                "/embeddings",
                json={"model": self._model_name, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to maintain input order
            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            for item in sorted_items:
                embedding = item["embedding"]
                all_embeddings.append(embedding[: self._dim])

        return all_embeddings

    def dimension(self) -> int:
        return self._dim
