"""Ollama embedding provider.

Calls Ollama's /api/embeddings HTTP endpoint.
Requires an Ollama instance running and the model already pulled.

Configuration:
  OLLAMA_URL:      base URL of Ollama (default: http://ollama:11434)
  EMBEDDING_MODEL: model name (e.g. nomic-embed-text, mxbai-embed-large)
  EMBEDDING_DIM:   expected output dimension (must match the model)
"""

import logging

import httpx

from app.config import settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaProvider(EmbeddingProvider):
    """Ollama HTTP embedding provider."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self._model_name = model_name or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._base_url = settings.ollama_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._ready = False

    async def startup(self) -> None:
        """Create HTTP client and verify Ollama is reachable."""
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            self._ready = True
            logger.info(
                "OllamaProvider connected: %s, model=%s",
                self._base_url,
                self._model_name,
            )
        except Exception:
            logger.warning(
                "OllamaProvider: could not reach Ollama at %s — will retry on use",
                self._base_url,
            )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        self._ready = False
        logger.info("OllamaProvider shut down")

    async def embed(self, text: str) -> list[float]:
        if not self._client:
            raise RuntimeError("OllamaProvider not started")
        resp = await self._client.post(
            "/api/embeddings",
            json={"model": self._model_name, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data["embedding"]
        return embedding[: self._dim]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Ollama has no native batch endpoint — call sequentially."""
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results

    def dimension(self) -> int:
        return self._dim

    @property
    def is_ready(self) -> bool:
        return self._ready
