"""Abstract reranker interface and shared utilities.

Rerankers take the top-K vector search results and reorder them
based on a more expensive scoring function (e.g. LLM-based relevance scoring).

The vector search score alone can miss semantic nuances — an LLM reranker
reads both the query and the full capability description to judge relevance more precisely.

Shared utility:
  _apply_ranking(candidates, ranked_indices, limit) — converts a list of
  LLM-returned indices into a reordered candidate list with rerank_score set.
  Used by all Reranker implementations to avoid duplicating this logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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


def _apply_ranking(
    candidates: list[dict[str, Any]],
    ranked_indices: list[int],
    limit: int,
) -> list[dict[str, Any]]:
    """Apply a ranked index list to a candidate list.

    Builds the reordered result, adds a position-based rerank_score to each
    entry, appends any candidates the LLM missed (at score 0.0), and truncates
    to limit.

    Args:
        candidates:     Original candidate list from vector search.
        ranked_indices: Integer indices returned by the LLM, most-relevant first.
        limit:          Maximum number of results to return.

    Returns:
        Reordered candidates (up to limit) with rerank_score added.
    """
    if not isinstance(ranked_indices, list):
        logger.warning(
            "Reranker returned unexpected type %s — returning original order",
            type(ranked_indices),
        )
        return candidates[:limit]

    reranked: list[dict[str, Any]] = []
    seen: set[int] = set()

    for idx in ranked_indices:
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        candidate = dict(candidates[idx])
        # Position-based score: 1.0 for rank-0, decreasing linearly
        candidate["rerank_score"] = round(
            1.0 - (len(reranked) / max(len(candidates), 1)), 3
        )
        reranked.append(candidate)

    # Append any candidates the LLM omitted (safety net)
    for i, cap in enumerate(candidates):
        if i not in seen:
            candidate = dict(cap)
            candidate["rerank_score"] = 0.0
            reranked.append(candidate)

    return reranked[:limit]
