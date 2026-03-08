"""Sentence-transformers embedding provider.

Loads a model locally using the sentence-transformers library.
Same library used by Mneme — model files can share the NFS volume.

Configuration:
  EMBEDDING_MODEL: model name from HuggingFace (default: all-MiniLM-L6-v2)
  EMBEDDING_DIM:   output dimension — if less than model native dim, truncates
  MODELS_DIR:      local cache directory for downloaded models

Supported models and their native dimensions:
  all-MiniLM-L6-v2    → 384d  (fast, good quality)
  all-mpnet-base-v2   → 768d  (slower, higher quality)
  nomic-embed-text-v1 → 768d  (requires trust_remote_code=True)
"""

import asyncio
import logging

from app.config import settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self._model_name = model_name or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._model = None
        self._loaded = False

    async def startup(self) -> None:
        """Load the model in a thread pool (blocking I/O)."""
        logger.info(
            "Loading sentence-transformers model: %s (dim=%d, cache=%s)",
            self._model_name,
            self._dim,
            settings.models_dir,
        )
        await asyncio.to_thread(self._load_model)
        logger.info(
            "Sentence-transformers model loaded: %s (native_dim=%d, output_dim=%d)",
            self._model_name,
            self._native_dim,
            self._dim,
        )

    def _load_model(self) -> None:
        """Synchronous model load — called in thread pool."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=settings.models_dir,
        )
        # Detect native dimension
        self._native_dim: int = self._model.get_sentence_embedding_dimension()
        self._loaded = True

    async def shutdown(self) -> None:
        """Release model from memory."""
        self._model = None
        self._loaded = False
        logger.info("SentenceTransformersProvider shut down")

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call startup() first")
        result = await asyncio.to_thread(self._encode_single, text)
        return result

    def _encode_single(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding[: self._dim].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once (batch is more efficient)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call startup() first")
        results = await asyncio.to_thread(self._encode_batch, texts)
        return results

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [emb[: self._dim].tolist() for emb in embeddings]

    def dimension(self) -> int:
        return self._dim

    @property
    def is_ready(self) -> bool:
        return self._loaded
