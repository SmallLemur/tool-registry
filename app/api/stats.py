"""Stats endpoint.

GET /api/v1/stats — registry statistics and service inventory.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.get("/stats")
async def stats(request: Request) -> dict:
    """Registry statistics.

    Returns total capability count, per-service breakdown,
    and service health summary.
    """
    registry = getattr(request.app.state, "registry_manager", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not available")

    try:
        data = await registry.get_stats()
        data["services"] = registry.get_service_health()
        return data
    except Exception:
        logger.exception("Stats failed")
        raise HTTPException(status_code=500, detail="Stats unavailable")
