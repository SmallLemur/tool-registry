"""Registry Manager — core orchestration for tool capability registration.

This is the brain of the registry. It:
  1. Receives ServiceManifest dicts (from HTTP POST or RabbitMQ announcements)
  2. Fingerprints the manifest — skips re-embedding on heartbeat (unchanged)
  3. Builds rich search_text for each action (service context + action + params)
  4. Embeds each action's search_text via the pluggable EmbeddingProvider
  5. Atomically replaces capabilities in Milvus (delete old + insert new)
  6. Tracks service health in memory (last_seen, fingerprint, version)

Service health is ephemeral — rebuilt from heartbeats after restart.
Milvus data is permanent — survives registry restarts.
After a restart, all capabilities are searchable but marked 'stale' until
services re-announce (within 60 seconds, since services heartbeat every 60s).

Auto-discovery workflow:
  Service starts → announces via RabbitMQ pensante.announce fanout
    ↓ (first time or changed manifest)
    compute SHA-256 fingerprint → fingerprint differs → FULL RE-INDEX
      → for each action: build search_text → embed → upsert_batch in Milvus
    ↓ (heartbeat: same manifest)
    fingerprint matches → UPDATE last_seen only (no Milvus write)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.core.capability_store import CapabilityStore
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """In-memory health record for a registered service."""

    service_name: str
    version: str
    fingerprint: str  # SHA-256 of canonical manifest JSON
    last_seen: datetime  # Updated on every announce/heartbeat
    capability_count: int  # Number of actions currently indexed

    def is_healthy(self, timeout_s: int | None = None) -> bool:
        """True if a heartbeat was received within the timeout window."""
        timeout = timeout_s or settings.service_heartbeat_timeout_s
        elapsed = (datetime.now(timezone.utc) - self.last_seen).total_seconds()
        return elapsed < timeout

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "version": self.version,
            "last_seen": self.last_seen.isoformat(),
            "capability_count": self.capability_count,
            "healthy": self.is_healthy(),
        }


class RegistryManager:
    """Orchestrates service registration, heartbeat deduplication, and search."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        capability_store: CapabilityStore,
    ):
        self._embedder = embedding_provider
        self._store = capability_store
        # In-memory health tracking — ephemeral, rebuilt from heartbeats
        self._health: dict[str, ServiceHealth] = {}

    # -- Registration ----

    async def register(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Register or update a service manifest.

        If the manifest fingerprint matches the stored one (heartbeat),
        only updates last_seen. Otherwise performs a full re-index.

        Returns: {"service": name, "registered": N, "action": "indexed"|"heartbeat"}
        """
        service_name = manifest.get("name", "")
        version = manifest.get("version", "0.0.0")

        if not service_name:
            raise ValueError("Manifest missing 'name' field")

        # Compute fingerprint of canonical manifest JSON
        fingerprint = _fingerprint(manifest)

        existing = self._health.get(service_name)

        # Heartbeat path: same fingerprint → just update last_seen
        if existing and existing.fingerprint == fingerprint:
            existing.last_seen = datetime.now(timezone.utc)
            logger.debug(
                "Heartbeat from %s v%s (%d capabilities)",
                service_name,
                version,
                existing.capability_count,
            )
            return {
                "service": service_name,
                "registered": existing.capability_count,
                "action": "heartbeat",
            }

        # New or changed manifest → full re-index
        action_word = "re-indexed" if existing else "indexed"
        logger.info(
            "Indexing %s v%s (%s — fingerprint changed)",
            service_name,
            version,
            action_word,
        )

        actions = manifest.get("actions", [])
        service_description = manifest.get("description", "")

        # Build (id, search_text, metadata) tuples for each action
        entries_to_embed: list[tuple[str, str, dict[str, Any]]] = []
        for action in actions:
            action_name = action.get("name", "")
            if not action_name:
                continue

            capability_id = _capability_id(service_name, action_name)
            search_text = _build_search_text(
                service_name=service_name,
                service_description=service_description,
                action=action,
            )
            metadata = _build_metadata(
                service_name=service_name,
                version=version,
                action=action,
            )
            entries_to_embed.append((capability_id, search_text, metadata))

        # Embed all search_texts in one batch call
        if entries_to_embed:
            texts = [e[1] for e in entries_to_embed]
            try:
                embeddings = await self._embedder.embed_batch(texts)
            except Exception:
                logger.exception(
                    "Embedding failed for %s — registration aborted", service_name
                )
                raise

            # Build final (id, embedding, metadata) tuples for Milvus
            milvus_entries = [
                (entries_to_embed[i][0], embeddings[i], entries_to_embed[i][2])
                for i in range(len(entries_to_embed))
            ]
        else:
            milvus_entries = []

        # Atomically replace all capabilities for this service in Milvus
        await self._store.upsert_batch(milvus_entries, service_name)

        # Update in-memory health record
        self._health[service_name] = ServiceHealth(
            service_name=service_name,
            version=version,
            fingerprint=fingerprint,
            last_seen=datetime.now(timezone.utc),
            capability_count=len(milvus_entries),
        )

        logger.info(
            "Service %s v%s: %d capabilities %s",
            service_name,
            version,
            len(milvus_entries),
            action_word,
        )
        return {
            "service": service_name,
            "registered": len(milvus_entries),
            "action": action_word,
        }

    async def deregister(self, service_name: str) -> dict[str, Any]:
        """Remove all capabilities for a service from Milvus and health dict."""
        await self._store.delete_by_service(service_name)
        removed = self._health.pop(service_name, None)
        count = removed.capability_count if removed else 0
        logger.info(
            "Deregistered service: %s (%d capabilities removed)", service_name, count
        )
        return {"service": service_name, "removed": count}

    # -- Search ----

    async def search(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
        filter_services: list[str] | None = None,
        exclude_services: list[str] | None = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search over indexed capabilities.

        Args:
            query:           User intent or action description to search for
            limit:           Max results to return (default: SEARCH_DEFAULT_LIMIT)
            threshold:       Min cosine similarity score (default: SEARCH_DEFAULT_THRESHOLD)
            filter_services: If set, only return capabilities from these services
            exclude_services: If set, exclude these services from results
            include_stale:   If False (default), exclude services with no recent heartbeat

        Returns:
            List of capability dicts ordered by score descending.
        """
        effective_limit = limit or settings.search_default_limit
        effective_threshold = (
            threshold if threshold is not None else settings.search_default_threshold
        )

        # Embed the query
        try:
            query_embedding = await self._embedder.embed(query)
        except Exception:
            logger.exception("Query embedding failed")
            raise

        # Determine which services are healthy (unless include_stale=True)
        healthy_services: set[str] | None = None
        if not include_stale:
            healthy_services = {
                name for name, h in self._health.items() if h.is_healthy()
            }

        # Build Milvus filter expression for service filtering
        # We over-fetch and post-filter by service and score in Python,
        # because Milvus JSON path filtering with IN is verbose to construct.
        fetch_limit = effective_limit * 4  # over-fetch to allow for post-filtering

        raw_hits = await self._store.search(
            query_embedding=query_embedding,
            limit=fetch_limit,
        )

        # Post-filter: score threshold, service allow/deny lists, staleness
        results = []
        for hit in raw_hits:
            score = hit["score"]
            if score < effective_threshold:
                continue

            meta = hit.get("metadata", {})
            svc = meta.get("service_name", "")

            # Staleness filter
            if healthy_services is not None and svc not in healthy_services:
                continue

            # Service allow-list
            if filter_services and svc not in filter_services:
                continue

            # Service deny-list
            if exclude_services and svc in exclude_services:
                continue

            results.append(
                {
                    "service": svc,
                    "action": meta.get("action_name", ""),
                    "description": meta.get("description", ""),
                    "input_schema": meta.get("input_schema", {}),
                    "output_schema": meta.get("output_schema", {}),
                    "risk_level": meta.get("risk_level", 0.0),
                    "timeout_seconds": meta.get("timeout_seconds", 30),
                    "tags": meta.get("tags", []),
                    "score": round(score, 4),
                }
            )

            if len(results) >= effective_limit:
                break

        return results

    # -- Stats & Health ----

    def get_service_health(self) -> list[dict[str, Any]]:
        """Return health records for all known services."""
        return [h.to_dict() for h in self._health.values()]

    def is_service_healthy(self, service_name: str) -> bool:
        h = self._health.get(service_name)
        return h.is_healthy() if h else False

    async def get_stats(self) -> dict[str, Any]:
        """Return registry statistics."""
        total = await self._store.count()
        by_service = await self._store.count_by_service()
        healthy_count = sum(1 for h in self._health.values() if h.is_healthy())

        return {
            "total_capabilities": total,
            "total_services": len(self._health),
            "healthy_services": healthy_count,
            "by_service": by_service,
        }


# -- Helpers ----


def _fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a stable SHA-256 fingerprint of a manifest dict.

    Uses canonical JSON (sorted keys, no whitespace) for determinism.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _capability_id(service_name: str, action_name: str) -> str:
    """Generate a stable, unique ID for a (service, action) pair.

    Uses a short SHA-256 prefix to keep IDs under Milvus's VARCHAR limit.
    """
    raw = f"{service_name}::{action_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _build_search_text(
    *,
    service_name: str,
    service_description: str,
    action: dict[str, Any],
) -> str:
    """Build rich text for embedding — includes service, action, and parameter context.

    This enriched text ensures semantic search can match both high-level intent
    ("track my workout") and specific action descriptions ("log_workout action").

    Format:
        Service: web_search - Search the web using SearXNG
        Action: search - Search the web for relevant information on a topic
        Parameters: query, num_results, language
    """
    action_name = action.get("name", "")
    action_desc = action.get("description", "")
    input_schema = action.get("input_schema", {})
    props = input_schema.get("properties", {})
    param_names = list(props.keys())

    lines = [
        f"Service: {service_name} - {service_description}",
        f"Action: {action_name} - {action_desc}",
    ]
    if param_names:
        lines.append(f"Parameters: {', '.join(param_names)}")

    return "\n".join(lines)


def _build_metadata(
    *,
    service_name: str,
    version: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Build the metadata dict stored in Milvus alongside the vector.

    This is everything needed to reconstruct a full capability entry
    without a separate database lookup.
    """
    return {
        "service_name": service_name,
        "service_version": version,
        "action_name": action.get("name", ""),
        "description": action.get("description", ""),
        "input_schema": action.get("input_schema", {}),
        "output_schema": action.get("output_schema", {}),
        "risk_level": float(action.get("risk_level", 0.0)),
        "timeout_seconds": int(action.get("timeout_seconds", 30)),
        "tags": action.get("tags", []),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
