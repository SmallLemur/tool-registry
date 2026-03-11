"""Milvus capability store.

Manages the 'tool_capabilities' collection in Milvus.
Each entry represents ONE action from ONE service, stored as a vector.

Collection schema:
  id        VARCHAR PK  — 32-char hex (SHA-256 prefix of service::action)
  embedding FLOAT_VECTOR — dim from EMBEDDING_DIM config
  metadata  JSON        — full capability entry fields

Index: IVF_FLAT / COSINE (same as Mneme's collections).

Uses pymilvus AsyncMilvusClient for native async I/O — no asyncio.to_thread()
wrappers needed.

Schema migrations are run on connect() via app.core.migrations.run_migrations().
Each migration is idempotent — running on an already-migrated instance is a no-op.
"""

import logging
from typing import Any

from pymilvus import AsyncMilvusClient

from app.config import settings
from app.core.migrations import run_migrations

logger = logging.getLogger(__name__)

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},
}


class CapabilityStore:
    """Async Milvus adapter for the tool_capabilities collection."""

    def __init__(self, embedding_dim: int | None = None):
        self._dim = embedding_dim or settings.embedding_dim
        self._collection = settings.milvus_collection
        self._client: AsyncMilvusClient | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        """Connect to Milvus and run schema migrations."""
        host = settings.milvus_host
        port = settings.milvus_port
        uri = f"http://{host}:{port}"
        logger.info("Connecting to Milvus at %s", uri)

        self._client = AsyncMilvusClient(uri=uri)
        logger.info("Connected to Milvus")

        # Run idempotent schema migrations (creates collection if needed,
        # validates dimension, raises on mismatch)
        await run_migrations(
            client=self._client,
            collection_name=self._collection,
            dim=self._dim,
        )
        logger.info("Milvus ready: collection=%s (dim=%d)", self._collection, self._dim)

    async def close(self) -> None:
        """Disconnect from Milvus."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Milvus")

    # -- Write operations ----

    async def upsert(
        self,
        capability_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or replace a single capability entry."""
        client = self._require_client()
        await client.upsert(
            collection_name=self._collection,
            data=[{"id": capability_id, "embedding": embedding, "metadata": metadata}],
        )
        logger.debug("Upserted capability %s", capability_id)

    async def upsert_batch(
        self,
        entries: list[tuple[str, list[float], dict[str, Any]]],
        service_name: str,
    ) -> None:
        """Atomically replace all capabilities for a service.

        Deletes all existing entries for the service, then inserts all new ones.

        Args:
            entries:      List of (capability_id, embedding, metadata) tuples.
            service_name: Service name used to delete existing entries first.
        """
        client = self._require_client()

        # Delete all existing entries for this service
        try:
            await client.delete(
                collection_name=self._collection,
                filter=f'metadata["service_name"] == "{service_name}"',
            )
            logger.debug("Deleted existing entries for service: %s", service_name)
        except Exception:
            logger.warning(
                "Could not delete existing entries for %s — inserting anyway",
                service_name,
                exc_info=True,
            )

        if not entries:
            return

        data = [
            {"id": entry[0], "embedding": entry[1], "metadata": entry[2]}
            for entry in entries
        ]
        await client.insert(collection_name=self._collection, data=data)
        logger.info(
            "Upserted %d capabilities for service: %s", len(entries), service_name
        )

    async def delete_by_service(self, service_name: str) -> None:
        """Remove all capability entries for a service."""
        client = self._require_client()
        await client.delete(
            collection_name=self._collection,
            filter=f'metadata["service_name"] == "{service_name}"',
        )
        logger.info("Deleted all capabilities for service: %s", service_name)

    # -- Search operations ----

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Returns list of dicts with: id, score (cosine similarity), metadata.
        """
        client = self._require_client()

        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "data": [query_embedding],
            "anns_field": "embedding",
            "search_params": SEARCH_PARAMS,
            "limit": limit,
            "output_fields": ["metadata"],
        }
        if filter_expr:
            kwargs["filter"] = filter_expr

        results = await client.search(**kwargs)

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                try:
                    metadata = hit.get("entity", {}).get("metadata") or {}
                except Exception:
                    metadata = {}
                hits.append(
                    {
                        "id": hit.get("id"),
                        "score": float(hit.get("distance", 0.0)),
                        "metadata": metadata,
                    }
                )
        return hits

    # -- Count operations ----

    async def count(self) -> int:
        """Total number of capabilities in the collection."""
        client = self._require_client()
        result = await client.query(
            collection_name=self._collection,
            filter="id != ''",
            output_fields=["count(*)"],
        )
        if result:
            return result[0].get("count(*)", 0)
        return 0

    async def count_by_service(self) -> dict[str, int]:
        """Return per-service capability counts.

        Note: Milvus doesn't support GROUP BY — we query all metadata and
        count in Python. Fine for a registry with hundreds of entries.
        """
        client = self._require_client()
        query_limit = 16384
        results = await client.query(
            collection_name=self._collection,
            filter="id != ''",
            output_fields=["metadata"],
            limit=query_limit,
        )
        if len(results) >= query_limit:
            logger.warning(
                "count_by_service hit query limit (%d) — counts may be truncated. "
                "Consider raising the limit if the registry has grown beyond %d entries.",
                query_limit,
                query_limit,
            )
        counts: dict[str, int] = {}
        for r in results:
            svc = (r.get("metadata") or {}).get("service_name", "unknown")
            counts[svc] = counts.get(svc, 0) + 1
        return counts

    # -- Internal ----

    def _require_client(self) -> AsyncMilvusClient:
        if self._client is None:
            raise RuntimeError("CapabilityStore not connected — call connect() first")
        return self._client
