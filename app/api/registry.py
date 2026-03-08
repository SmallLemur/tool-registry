"""Registry endpoints.

POST   /api/v1/register            — register or update a service manifest
DELETE /api/v1/deregister/{service} — remove all capabilities for a service
GET    /api/v1/services            — list all registered services with health
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# -- Pydantic models (mirrors pensante-service-base ServiceManifest) ----
# Declared locally so tool-registry stays standalone (no pensante-common dep).


class ActionDef(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    risk_level: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=30, ge=1)
    tags: list[str] = Field(default_factory=list)


class StimulusDef(BaseModel):
    name: str
    description: str = ""
    payload_schema: dict[str, Any] = Field(default_factory=dict)


class ServiceManifest(BaseModel):
    name: str
    version: str = "0.0.0"
    description: str = ""
    service_type: str = "skill"
    actions: list[ActionDef] = Field(default_factory=list)
    stimuli: list[StimulusDef] = Field(default_factory=list)
    suggested_permission: str = "delegated"


# -- Endpoints ----


@router.post("/register")
async def register(manifest: ServiceManifest, request: Request) -> dict:
    """Register or update a service manifest.

    Decomposes the manifest into individual capability entries,
    embeds each action's search_text, and upserts into Milvus.

    On heartbeat (same manifest fingerprint), skips embedding and only
    updates last_seen — making this endpoint safe to call every 60 seconds.

    Returns: {"service": name, "registered": N, "action": "indexed"|"heartbeat"}
    """
    registry = _require_registry(request)
    try:
        result = await registry.register(manifest.model_dump())
        logger.info(
            "Register: %s v%s → %s (%d capabilities)",
            manifest.name,
            manifest.version,
            result["action"],
            result["registered"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Register failed for service: %s", manifest.name)
        raise HTTPException(status_code=500, detail="Registration failed")


@router.delete("/deregister/{service_name}")
async def deregister(service_name: str, request: Request) -> dict:
    """Remove all capabilities for a service from the registry.

    Returns: {"service": name, "removed": N}
    """
    registry = _require_registry(request)
    try:
        result = await registry.deregister(service_name)
        logger.info(
            "Deregistered service: %s (%d capabilities removed)",
            service_name,
            result["removed"],
        )
        return result
    except Exception:
        logger.exception("Deregister failed for service: %s", service_name)
        raise HTTPException(status_code=500, detail="Deregistration failed")


@router.get("/services")
async def list_services(request: Request) -> dict:
    """List all registered services with health status.

    Returns services from in-memory health tracking.
    Services are visible here even after a registry restart (their capabilities
    remain in Milvus), but will show as 'healthy: false' until they re-announce.
    """
    registry = _require_registry(request)
    services = registry.get_service_health()
    return {
        "services": services,
        "total": len(services),
        "healthy": sum(1 for s in services if s.get("healthy")),
    }


def _require_registry(request: Request):
    registry = getattr(request.app.state, "registry_manager", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not available")
    return registry
