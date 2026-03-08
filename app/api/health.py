"""Health endpoint.

GET /api/v1/health — liveness and readiness probe.
Reports Milvus connectivity and embedding provider status.
"""

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health(request: Request) -> dict:
    """Service health check.

    Returns overall status plus component-level details.
    Status is 'ok' only when all required components are healthy.
    """
    store = getattr(request.app.state, "capability_store", None)
    embedder = getattr(request.app.state, "embedding_provider", None)
    registry = getattr(request.app.state, "registry_manager", None)

    milvus_status = "connected" if (store and store.is_connected) else "disconnected"
    embedding_status = "ready" if (embedder and embedder.is_ready) else "not_ready"

    healthy_services = 0
    total_services = 0
    if registry:
        for h in registry.get_service_health():
            total_services += 1
            if h.get("healthy"):
                healthy_services += 1

    overall = (
        "ok"
        if milvus_status == "connected" and embedding_status == "ready"
        else "degraded"
    )

    return {
        "status": overall,
        "service": "tool-registry",
        "version": "0.1.0",
        "components": {
            "milvus": milvus_status,
            "embedding": embedding_status,
        },
        "services": {
            "total": total_services,
            "healthy": healthy_services,
        },
    }
