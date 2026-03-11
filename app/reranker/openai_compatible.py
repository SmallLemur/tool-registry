"""OpenAI-compatible LLM reranker.

Reranks candidates via any OpenAI-compatible /chat/completions endpoint —
OpenRouter, vLLM, local llama.cpp server, etc.

Uses structured output (response_format json_schema) to get a guaranteed-valid
JSON response without any markdown extraction. Requires a model that supports
the json_schema response format.

Configuration (via env vars):
  RERANKER_LLM_URL   — base URL of the API (default: https://openrouter.ai/api/v1)
  RERANKER_MODEL     — model name (default: google/gemini-2.0-flash-001)
  RERANKER_API_KEY   — API key (Bearer token)
"""

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.reranker.base import Reranker, _apply_ranking

logger = logging.getLogger(__name__)

# JSON schema for the structured output response.
# The LLM must return {"ranked_indices": [2, 0, 1, 3, ...]}
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Capability indices ordered from most to least relevant",
        }
    },
    "required": ["ranked_indices"],
    "additionalProperties": False,
}

RERANK_SYSTEM_PROMPT = (
    "You are a tool-selection assistant. Given a user intent and a list of tool "
    "capabilities, rank the capabilities by relevance. Return ALL indices "
    "(even irrelevant ones, ranked last) in the ranked_indices array."
)


class OpenAICompatibleReranker(Reranker):
    """Reranks capabilities using any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._base_url = (base_url or settings.reranker_llm_url).rstrip("/")
        self._model = model or settings.reranker_model
        self._api_key = api_key or settings.reranker_api_key

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using structured LLM output."""
        if not candidates:
            return candidates

        cap_lines = []
        for i, cap in enumerate(candidates):
            svc = cap.get("service", "")
            action = cap.get("action", "")
            desc = cap.get("description", "")
            cap_lines.append(f"[{i}] {svc}.{action}: {desc}")

        user_message = (
            f'User intent: "{query}"\n\n'
            f"Available capabilities (0-indexed):\n" + "\n".join(cap_lines)
        )

        try:
            ranked_indices = await self._call_llm(user_message)
        except Exception:
            logger.warning(
                "LLM reranker call failed — returning original order", exc_info=True
            )
            return candidates[:limit]

        return _apply_ranking(candidates, ranked_indices, limit)

    async def _call_llm(self, user_message: str) -> list[int]:
        """Call the LLM with structured output and return ranked indices."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rerank_result",
                            "strict": True,
                            "schema": _RESPONSE_SCHEMA,
                        },
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"]
        return json.loads(raw)["ranked_indices"]
