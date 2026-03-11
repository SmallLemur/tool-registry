"""Reranker factory.

Reads RERANKER_PROVIDER from config and returns the appropriate
Reranker instance. Mirrors the embedding provider factory pattern.

Usage (in main.py lifespan):
    reranker = create_reranker()
    # reranker is None if RERANKER_ENABLED=false
    # reranker is a Reranker instance otherwise
"""

import logging

from app.config import settings
from app.reranker.base import Reranker

logger = logging.getLogger(__name__)

# Registry of known provider names
PROVIDERS = {
    "openai_compatible",
    "ollama",
}


def create_reranker() -> Reranker | None:
    """Instantiate the reranker selected by RERANKER_PROVIDER env var.

    Returns None if RERANKER_ENABLED=false.
    Raises ValueError for unknown provider names.
    """
    if not settings.reranker_enabled:
        return None

    provider_name = settings.reranker_provider.lower().strip()

    if provider_name == "openai_compatible":
        from app.reranker.openai_compatible import OpenAICompatibleReranker

        logger.info(
            "Using OpenAI-compatible reranker (url=%s, model=%s)",
            settings.reranker_llm_url,
            settings.reranker_model,
        )
        return OpenAICompatibleReranker()

    elif provider_name == "ollama":
        from app.reranker.ollama import OllamaReranker

        logger.info(
            "Using Ollama reranker (url=%s, model=%s)",
            settings.reranker_ollama_url,
            settings.reranker_model,
        )
        return OllamaReranker()

    else:
        raise ValueError(
            f"Unknown RERANKER_PROVIDER: '{provider_name}'. "
            f"Valid options: {', '.join(sorted(PROVIDERS))}"
        )
