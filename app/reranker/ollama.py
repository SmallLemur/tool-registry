"""Ollama LLM reranker.

Reranks candidates using a local Ollama instance via /api/chat.
Uses Ollama's structured output (format parameter with JSON schema) to get
a guaranteed-valid response without markdown extraction.

Requires a model that supports structured output (e.g. llama3.2, mistral-nemo,
qwen2.5). Check Ollama docs for model-specific support.

Configuration (via env vars):
  RERANKER_OLLAMA_URL — base URL of Ollama (default: http://ollama:11434)
  RERANKER_MODEL      — model name (e.g. llama3.2, qwen2.5:3b)
"""

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.reranker.base import Reranker, _apply_ranking

logger = logging.getLogger(__name__)

# JSON schema for the structured output response.
# Same schema as OpenAI-compatible: {"ranked_indices": [2, 0, 1, ...]}
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
}

RERANK_SYSTEM_PROMPT = (
    "You are a tool-selection assistant. Given a user intent and a list of tool "
    "capabilities, rank the capabilities by relevance. Return ALL indices "
    "(even irrelevant ones, ranked last) in the ranked_indices array."
)


class OllamaReranker(Reranker):
    """Reranks capabilities using a local Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._base_url = (base_url or settings.reranker_ollama_url).rstrip("/")
        self._model = model or settings.reranker_model

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using local Ollama with structured output."""
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
            ranked_indices = await self._call_ollama(user_message)
        except Exception:
            logger.warning(
                "Ollama reranker call failed — returning original order", exc_info=True
            )
            return candidates[:limit]

        return _apply_ranking(candidates, ranked_indices, limit)

    async def _call_ollama(self, user_message: str) -> list[int]:
        """Call Ollama /api/chat with structured output and return ranked indices."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "format": _RESPONSE_SCHEMA,
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["message"]["content"]
        return json.loads(raw)["ranked_indices"]
