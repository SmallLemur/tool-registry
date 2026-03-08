"""Abstract reranker interface.

Rerankers take the top-K vector search results and reorder them
based on a more expensive scoring function (e.g. LLM-based relevance scoring).

The vector search score alone can miss semantic nuances — an LLM reranker
reads both the query and the full capability description to judge relevance more precisely.
"""

from abc import ABC, abstractmethod
from typing import Any


class Reranker(ABC):
    """Abstract base class for capability rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by relevance to query.

        Args:
            query:      Original user intent / search query
            candidates: Top-K results from vector search (each has service, action, description, score, ...)
            limit:      How many results to return after reranking

        Returns:
            Reordered list (up to limit), with rerank_score added to each result.
        """
        ...
