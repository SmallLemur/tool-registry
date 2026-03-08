"""Embedding provider factory.

Reads EMBEDDING_PROVIDER from config and returns the appropriate
EmbeddingProvider instance. All providers are configured via env vars.

Usage:
    provider = create_embedding_provider()
    await provider.startup()
    embedding = await provider.embed("some text")
"""

import logging

from app.config import settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Registry of known provider names
PROVIDERS = {
    "sentence_transformers",
    "ollama",
    "openai",
}


def create_embedding_provider() -> EmbeddingProvider:
    """Instantiate the embedding provider selected by EMBEDDING_PROVIDER env var.

    Raises ValueError for unknown provider names.
    """
    provider_name = settings.embedding_provider.lower().strip()

    if provider_name == "sentence_transformers":
        from app.embeddings.sentence_transformers import SentenceTransformersProvider

        logger.info(
            "Using sentence_transformers provider (model=%s, dim=%d)",
            settings.embedding_model,
            settings.embedding_dim,
        )
        return SentenceTransformersProvider()

    elif provider_name == "ollama":
        from app.embeddings.ollama import OllamaProvider

        logger.info(
            "Using Ollama provider (url=%s, model=%s, dim=%d)",
            settings.ollama_url,
            settings.embedding_model,
            settings.embedding_dim,
        )
        return OllamaProvider()

    elif provider_name == "openai":
        from app.embeddings.openai import OpenAIProvider

        logger.info(
            "Using OpenAI-compatible provider (url=%s, model=%s, dim=%d)",
            settings.openai_embedding_url,
            settings.embedding_model,
            settings.embedding_dim,
        )
        return OpenAIProvider()

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider_name}'. "
            f"Valid options: {', '.join(sorted(PROVIDERS))}"
        )
