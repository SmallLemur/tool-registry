"""Tool Registry — Semantic capability index for microservices.

FastAPI application with lifespan-managed startup/shutdown.

Architecture:
  - EmbeddingProvider: loads model (sentence_transformers / ollama / openai)
  - CapabilityStore: Milvus collection adapter (IVF_FLAT / COSINE)
  - RegistryManager: orchestrates registration, fingerprinting, search
  - RegistrationPlugin: auto-discovery via RabbitMQ or HTTP push

Auto-discovery workflow:
  Services announce their manifest via RabbitMQ fanout exchange (or HTTP POST).
  RabbitMQListenerPlugin receives manifests and calls registry.register().
  Fingerprint deduplication skips re-embedding on heartbeat (unchanged manifest).
  Changed manifest (service redeployed) triggers full re-index.
  Milvus data persists across registry restarts.
  Service health is ephemeral — rebuilt from heartbeats within 60s.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, registry, search, stats
from app.config import settings
from app.core.capability_store import CapabilityStore
from app.core.registry_manager import RegistryManager
from app.embeddings.factory import create_embedding_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Tool Registry v%s starting …", VERSION)

    # -- Embedding provider ----
    embedding_provider = create_embedding_provider()
    try:
        await embedding_provider.startup()
        logger.info(
            "Embedding provider ready: %s (dim=%d)",
            settings.embedding_provider,
            embedding_provider.dimension(),
        )
    except Exception:
        logger.exception(
            "Embedding provider failed to start — registry will be degraded"
        )
    app.state.embedding_provider = embedding_provider

    # -- Milvus capability store ----
    capability_store = CapabilityStore(embedding_dim=embedding_provider.dimension())
    try:
        await capability_store.connect()
        # connect() runs migrations — collection is created/validated there
    except RuntimeError as e:
        # Dimension mismatch or fatal migration error — log clearly and let container restart
        logger.critical("FATAL: %s", e)
        raise
    except Exception:
        logger.exception(
            "Milvus failed to connect — search/register will be unavailable"
        )
    app.state.capability_store = capability_store

    # -- Registry manager ----
    registry_manager = RegistryManager(
        embedding_provider=embedding_provider,
        capability_store=capability_store,
    )
    app.state.registry_manager = registry_manager

    # -- LLM reranker (optional) ----
    reranker = None
    if settings.reranker_enabled:
        try:
            from app.reranker.factory import create_reranker

            reranker = create_reranker()
            logger.info(
                "Reranker enabled: provider=%s model=%s",
                settings.reranker_provider,
                settings.reranker_model,
            )
        except Exception:
            logger.warning(
                "Reranker failed to initialize — reranking disabled", exc_info=True
            )
    app.state.reranker = reranker

    # -- Registration plugin ----
    registration_plugin = None
    try:
        plugin_name = settings.registration_plugin.lower()
        if plugin_name == "rabbitmq_listener":
            from app.registration.rabbitmq_listener import RabbitMQListenerPlugin

            registration_plugin = RabbitMQListenerPlugin()
        elif plugin_name == "http_push":
            from app.registration.http_push import HttpPushPlugin

            registration_plugin = HttpPushPlugin()
        else:
            logger.warning(
                "Unknown REGISTRATION_PLUGIN: '%s' — using http_push fallback",
                plugin_name,
            )
            from app.registration.http_push import HttpPushPlugin

            registration_plugin = HttpPushPlugin()

        await registration_plugin.start(
            on_register=registry_manager.register,
            on_deregister=registry_manager.deregister,
        )
        logger.info("Registration plugin started: %s", plugin_name)
    except Exception:
        logger.exception(
            "Registration plugin failed to start — auto-discovery disabled, "
            "use POST /api/v1/register for manual registration"
        )
    app.state.registration_plugin = registration_plugin

    logger.info("Tool Registry v%s ready", VERSION)
    yield

    # -- Shutdown ----
    logger.info("Tool Registry shutting down …")
    if registration_plugin:
        try:
            await registration_plugin.stop()
        except Exception:
            logger.warning("Registration plugin shutdown error", exc_info=True)
    await embedding_provider.shutdown()
    await capability_store.close()
    logger.info("Tool Registry shut down")


app = FastAPI(
    title="Tool Registry",
    description=(
        "Semantic capability index for microservices. "
        "Services register their capabilities; clients search by natural-language intent."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(registry.router)
app.include_router(search.router)
app.include_router(stats.router)
