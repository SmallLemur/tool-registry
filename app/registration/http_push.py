"""HTTP Push registration plugin.

Provides zero-dependency auto-discovery without RabbitMQ.

Two complementary mechanisms work together:

1. Service-initiated registration/heartbeat (inbound)
   Services POST their manifest to /api/v1/register on startup and on a
   regular interval. The RegistryManager's fingerprint deduplication treats
   an unchanged manifest as a heartbeat — no re-embedding, just last_seen update.
   This already works via app/api/registry.py — no extra code needed here.

2. Registry-initiated health polling (outbound)
   This plugin runs a background task that periodically calls /api/v1/health
   on every registered service. If a service's health endpoint is unreachable
   (or returns non-2xx), the plugin calls on_deregister() so the RegistryManager
   can mark it stale. When the service recovers and POSTs its manifest again,
   it becomes healthy again automatically.

   The health URL is derived from the service manifest's 'base_url' field, or
   constructed as http://<service_name>:<default_port>/api/v1/health if absent.
   Services can also include a 'health_url' field in their manifest to override.

Configuration (via env vars):
  HTTP_HEARTBEAT_INTERVAL_S  — seconds between polling rounds (default: 60)
  HTTP_HEARTBEAT_TIMEOUT_S   — per-request timeout in seconds (default: 10)
"""

import asyncio
import logging
from typing import Callable

import httpx

from app.config import settings
from app.registration.base import RegistrationPlugin

logger = logging.getLogger(__name__)


class HttpPushPlugin(RegistrationPlugin):
    """Zero-dependency registration: inbound HTTP POST + outbound health polling."""

    def __init__(self):
        self._on_register: Callable | None = None
        self._on_deregister: Callable | None = None
        self._poll_task: asyncio.Task | None = None
        # service_name → health_url (populated when services register)
        self._service_health_urls: dict[str, str] = {}

    async def start(
        self,
        on_register: Callable,
        on_deregister: Callable,
    ) -> None:
        """Start the plugin — launch the outbound health polling loop."""
        self._on_register = on_register
        self._on_deregister = on_deregister
        self._poll_task = asyncio.create_task(self._polling_loop())
        logger.info(
            "HttpPushPlugin active — services must POST to /api/v1/register. "
            "Polling registered services every %ds.",
            settings.http_heartbeat_interval_s,
        )

    async def stop(self) -> None:
        """Cancel the polling loop."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("HttpPushPlugin stopped")

    def register_service_url(self, service_name: str, health_url: str) -> None:
        """Record the health URL for a service so the poller can check it.

        Called by the registry API after a successful registration so we know
        where to poll. The health_url is derived from the manifest.
        """
        self._service_health_urls[service_name] = health_url
        logger.debug(
            "HttpPushPlugin: tracking health URL for %s → %s", service_name, health_url
        )

    def deregister_service_url(self, service_name: str) -> None:
        """Remove a service from health polling."""
        self._service_health_urls.pop(service_name, None)

    # -- Polling loop ----

    async def _polling_loop(self) -> None:
        """Periodically poll /api/v1/health on all registered services."""
        while True:
            await asyncio.sleep(settings.http_heartbeat_interval_s)
            if not self._service_health_urls:
                continue
            await self._poll_all()

    async def _poll_all(self) -> None:
        """Poll all tracked services concurrently."""
        # Snapshot the dict to avoid mutation-during-iteration issues
        snapshot = dict(self._service_health_urls)
        tasks = [self._poll_one(name, url) for name, url in snapshot.items()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_one(self, service_name: str, health_url: str) -> None:
        """Poll a single service. On failure, call on_deregister to mark it stale."""
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_heartbeat_timeout_s
            ) as client:
                resp = await client.get(health_url)
                resp.raise_for_status()
            logger.debug("HttpPushPlugin: %s healthy (%s)", service_name, health_url)
        except Exception as e:
            logger.warning(
                "HttpPushPlugin: %s unreachable at %s (%s) — marking stale",
                service_name,
                health_url,
                e,
            )
            if self._on_deregister:
                try:
                    await self._on_deregister(service_name)
                except Exception:
                    logger.exception(
                        "HttpPushPlugin: deregister callback failed for %s",
                        service_name,
                    )


def derive_health_url(manifest: dict) -> str | None:
    """Derive a health check URL from a service manifest.

    Priority:
      1. manifest['health_url']  — explicit override
      2. manifest['base_url'] + '/api/v1/health'  — base URL provided
      3. None — not enough info to derive a URL

    Args:
        manifest: Service manifest dict as POSTed to /api/v1/register.

    Returns:
        A health check URL string, or None if one cannot be derived.
    """
    if health_url := manifest.get("health_url"):
        return health_url
    if base_url := manifest.get("base_url"):
        return base_url.rstrip("/") + "/api/v1/health"
    return None
