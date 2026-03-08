"""LLM-based capability reranker.

Takes top-K vector search results and asks a cheap LLM to rerank them
by relevance to the user's intent. Improves precision when semantic similarity
alone isn't sufficient (e.g. query is ambiguous, or descriptions are similar).

The LLM receives:
  - The user's original query
  - Each candidate's service name, action name, and description
  - Instructions to return a ranked list of indices

Uses the cheap model (same as Cortex's rumination/chain decisions) via
any OpenRouter-compatible endpoint. Configured via env vars.

Only active when RERANKER_ENABLED=true.
"""

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.reranker.base import Reranker

logger = logging.getLogger(__name__)

RERANK_PROMPT_TEMPLATE = """\
You are a tool-selection assistant. A user has expressed an intent, and a list of available tool capabilities has been retrieved. Your job is to rerank these capabilities by how relevant and useful they are for the user's intent.

User intent: "{query}"

Available capabilities (0-indexed):
{capabilities_text}

Instructions:
- Return ONLY a JSON array of indices, ordered from most to least relevant.
- Include ALL indices (even irrelevant ones, just put them last).
- Example: [2, 0, 1, 3] means capability 2 is most relevant.
- Respond with ONLY the JSON array, no explanation.

Ranked indices:"""


class LLMReranker(Reranker):
    """Reranks capabilities using a cheap LLM via OpenRouter-compatible API."""

    def __init__(self):
        self._base_url = settings.reranker_llm_url.rstrip("/")
        self._model = settings.reranker_model
        self._api_key = settings.reranker_api_key

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using LLM relevance scoring."""
        if not candidates:
            return candidates

        # Build capability descriptions for the prompt
        cap_lines = []
        for i, cap in enumerate(candidates):
            svc = cap.get("service", "")
            action = cap.get("action", "")
            desc = cap.get("description", "")
            cap_lines.append(f"[{i}] {svc}.{action}: {desc}")
        capabilities_text = "\n".join(cap_lines)

        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            capabilities_text=capabilities_text,
        )

        try:
            ranked_indices = await self._call_llm(prompt)
        except Exception:
            logger.warning(
                "LLM reranker call failed — returning original order", exc_info=True
            )
            return candidates[:limit]

        # Parse ranked indices and build reordered list
        try:
            if not isinstance(ranked_indices, list):
                raise ValueError(f"Expected list, got: {type(ranked_indices)}")

            reranked = []
            seen = set()
            for idx in ranked_indices:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= len(candidates):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                candidate = dict(candidates[idx])
                # Add rerank_score: position-based (1.0 for first, decreasing)
                candidate["rerank_score"] = round(
                    1.0 - (len(reranked) / max(len(candidates), 1)), 3
                )
                reranked.append(candidate)

            # Add any candidates missed by the LLM at the end
            for i, cap in enumerate(candidates):
                if i not in seen:
                    candidate = dict(cap)
                    candidate["rerank_score"] = 0.0
                    reranked.append(candidate)

            return reranked[:limit]

        except Exception:
            logger.warning(
                "Failed to parse reranker response — returning original order",
                exc_info=True,
            )
            return candidates[:limit]

    async def _call_llm(self, prompt: str) -> list[int]:
        """Call the cheap LLM and parse a JSON array of indices."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 128,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code blocks if LLM wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)
