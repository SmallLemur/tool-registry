"""Search endpoint.

POST /api/v1/search — semantic search over indexed capabilities.

This is the primary endpoint called by Cortex during prompt assembly.
It takes a user intent and returns the top-N most relevant capabilities,
fully specified with schemas so Cortex can inject them into the LLM prompt.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


class SearchRequest(BaseModel):
    query: str = Field(..., description="User intent or action description")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Max results (default: SEARCH_DEFAULT_LIMIT from config)",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min cosine similarity score (default: SEARCH_DEFAULT_THRESHOLD)",
    )
    rerank: bool = Field(
        default=False,
        description="Apply LLM reranking pass (requires RERANKER_ENABLED=true)",
    )
    filter_services: list[str] | None = Field(
        default=None,
        description="If set, only return capabilities from these services",
    )
    exclude_services: list[str] | None = Field(
        default=None,
        description="If set, exclude capabilities from these services",
    )
    include_stale: bool = Field(
        default=False,
        description="Include capabilities from services with no recent heartbeat",
    )


class SearchResult(BaseModel):
    service: str
    action: str
    description: str
    input_schema: dict
    output_schema: dict
    risk_level: float
    timeout_seconds: int
    tags: list[str]
    score: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total_capabilities: int


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    """Semantic search over indexed tool capabilities.

    Embeds the query, searches Milvus by cosine similarity, post-filters
    by score threshold and service allow/deny lists, then optionally reranks
    using an LLM for higher precision.

    Example request:
        {"query": "I need to track my workout", "limit": 5}

    Example result entry:
        {"service": "sparky_fitness", "action": "log_workout",
         "description": "Log a completed workout session", "score": 0.89, ...}
    """
    registry = getattr(request.app.state, "registry_manager", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not available")

    reranker = getattr(request.app.state, "reranker", None)

    try:
        results = await registry.search(
            query=body.query,
            limit=body.limit,
            threshold=body.threshold,
            filter_services=body.filter_services,
            exclude_services=body.exclude_services,
            include_stale=body.include_stale,
        )
    except Exception:
        logger.exception("Search failed for query: %.80s", body.query)
        raise HTTPException(status_code=500, detail="Search failed")

    # Optional LLM reranking pass
    if body.rerank and reranker is not None and results:
        try:
            effective_limit = body.limit or 5
            results = await reranker.rerank(
                query=body.query,
                candidates=results,
                limit=effective_limit,
            )
        except Exception:
            logger.warning(
                "Reranking failed — returning vector results: %.80s",
                body.query,
                exc_info=True,
            )

    # Get total count for response metadata
    total = 0
    store = getattr(request.app.state, "capability_store", None)
    if store and store.is_connected:
        try:
            total = await store.count()
        except Exception:
            pass

    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query=body.query,
        total_capabilities=total,
    )
