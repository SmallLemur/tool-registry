"""RabbitMQ listener registration plugin.

Listens on the same 'pensante.announce' fanout exchange that Sensorium uses.
When a service announces its manifest, auto-registers it with the registry.

This is the recommended plugin for the Pensante stack — it provides zero-config
auto-discovery: services don't need to know the registry URL, they just announce
via RabbitMQ as they always did.

Both Sensorium AND the registry can consume from the same exchange simultaneously
(fanout = all subscribers get every message).

Exchange topology (matches pensante-service-base/pensante_service/connection.py):
  Exchange: pensante.announce (FANOUT, durable)
  Queue:    tool-registry.announce (durable, exclusive to this consumer)

Data flow:
  Service starts → publishes ServiceManifest JSON to pensante.announce
  ↓
  RabbitMQListenerPlugin receives message
  ↓
  Calls on_register(manifest_dict) → RegistryManager.register()
  ↓
  Fingerprint check: heartbeat (skip) or changed (re-index)
"""

import asyncio
import json
import logging
from typing import Callable

import aio_pika
from aio_pika import ExchangeType

from app.config import settings
from app.registration.base import RegistrationPlugin

logger = logging.getLogger(__name__)

EXCHANGE_ANNOUNCE = "pensante.announce"
QUEUE_NAME = "tool-registry.announce"


class RabbitMQListenerPlugin(RegistrationPlugin):
    """Auto-discovers services via the pensante.announce RabbitMQ exchange."""

    def __init__(self):
        self._on_register: Callable | None = None
        self._on_deregister: Callable | None = None
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.Channel | None = None
        self._consume_task: asyncio.Task | None = None

    async def start(
        self,
        on_register: Callable,
        on_deregister: Callable,
    ) -> None:
        """Connect to RabbitMQ and start consuming announce messages."""
        self._on_register = on_register
        self._on_deregister = on_deregister

        logger.info("RabbitMQListenerPlugin connecting to %s", settings.rabbitmq_url)
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()

        # Declare the announce exchange (must match the one services use)
        exchange = await self._channel.declare_exchange(
            EXCHANGE_ANNOUNCE,
            ExchangeType.FANOUT,
            durable=True,
        )

        # Declare our dedicated queue
        queue = await self._channel.declare_queue(
            QUEUE_NAME,
            durable=True,
        )
        await queue.bind(exchange)

        await queue.consume(self._on_announce)
        logger.info(
            "RabbitMQListenerPlugin listening for service announcements on '%s'",
            EXCHANGE_ANNOUNCE,
        )

    async def stop(self) -> None:
        """Close RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("RabbitMQListenerPlugin stopped")

    async def _on_announce(self, message: aio_pika.IncomingMessage) -> None:
        """Handle an incoming service announcement."""
        async with message.process():
            try:
                manifest = json.loads(message.body.decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("Invalid announce message body: %s", e)
                return

            service_name = manifest.get("name")
            if not service_name:
                logger.warning("Announce message missing 'name' field — skipping")
                return

            logger.debug(
                "Received announce from %s v%s",
                service_name,
                manifest.get("version", "?"),
            )

            if self._on_register:
                try:
                    result = await self._on_register(manifest)
                    if result.get("action") != "heartbeat":
                        logger.info(
                            "Auto-registered: %s v%s (%d capabilities, action=%s)",
                            service_name,
                            manifest.get("version", "?"),
                            result.get("registered", 0),
                            result.get("action", "?"),
                        )
                except Exception:
                    logger.exception(
                        "Failed to register service from announce: %s", service_name
                    )
