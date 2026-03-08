"""HTTP Push registration plugin.

The default plugin — services explicitly POST to /api/v1/register.
This plugin is a no-op: the REST endpoints in app/api/registry.py
already call registry_manager.register() directly.

Exists to satisfy the RegistrationPlugin interface when
REGISTRATION_PLUGIN=http_push (or as a fallback).
"""

import logging
from typing import Callable

from app.registration.base import RegistrationPlugin

logger = logging.getLogger(__name__)


class HttpPushPlugin(RegistrationPlugin):
    """No-op plugin — services register via POST /api/v1/register."""

    async def start(
        self,
        on_register: Callable,
        on_deregister: Callable,
    ) -> None:
        logger.info(
            "HttpPushPlugin active — services must POST to /api/v1/register to register"
        )

    async def stop(self) -> None:
        pass
