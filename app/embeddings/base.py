"""Abstract embedding provider interface.

All embedding backends implement this ABC so the registry manager
can swap providers without changing business logic.

Data flow:
  config.EMBEDDING_PROVIDER → factory.py → EmbeddingProvider instance
  RegistryManager.register() → provider.embed_batch() → list[list[float]]
  RegistryManager.search()   → provider.embed()       → list[float]
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding backends."""

    @abstractmethod
    async def startup(self) -> None:
        """Initialize the provider (load models, connect to service, etc.)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns a list of floats with length == self.dimension().
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call (more efficient than looping).

        Returns a list of embeddings, same order as input texts.
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the output embedding dimension for this provider+model."""
        ...

    @property
    def is_ready(self) -> bool:
        """True if the provider is ready to embed. Override if needed."""
        return True
