"""Milvus capability store.

Manages the 'tool_capabilities' collection in Milvus.
Each entry represents ONE action from ONE service, stored as a vector.

Collection schema (consistent with Mneme's pattern):
  id        VARCHAR PK  — UUID (service_name + action_name hash)
  embedding FLOAT_VECTOR — dim from EMBEDDING_DIM config
  metadata  JSON        — full capability entry fields

Index: IVF_FLAT / COSINE (same as Mneme's collections).

Dimension mismatch protection:
  If the collection already exists with a different vector dimension,
  startup() raises RuntimeError with a clear message — prevents silent
  data corruption when switching embedding models.

All pymilvus calls are synchronous and wrapped in asyncio.to_thread()
for non-blocking use in async FastAPI handlers.
"""

import asyncio
import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.milvus_collection

INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},
}


class CapabilityStore:
    """Thin async adapter over pymilvus for the tool_capabilities collection."""

    def __init__(self, embedding_dim: int | None = None):
        self._dim = embedding_dim or settings.embedding_dim
        self._collection: Collection | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to Milvus and ensure the collection exists with correct schema."""
        await asyncio.to_thread(self._connect_sync)

    def _connect_sync(self) -> None:
        host = settings.milvus_host
        port = settings.milvus_port
        logger.info("Connecting to Milvus at %s:%s", host, port)

        connections.connect(alias="default", host=host, port=str(port))
        self._connected = True
        logger.info("Connected to Milvus")

        self._collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        """Create or load the tool_capabilities collection.

        Raises RuntimeError if collection exists with mismatched dimension.
        """
        name = COLLECTION_NAME

        if utility.has_collection(name):
            col = Collection(name)
            # -- Dimension mismatch protection ----
            # Read the schema to get the stored vector dimension
            schema = col.schema
            vec_field = next(
                (f for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR),
                None,
            )
            if vec_field is not None:
                stored_dim = vec_field.params.get("dim")
                if stored_dim != self._dim:
                    raise RuntimeError(
                        f"Milvus collection '{name}' already exists with "
                        f"vector dimension {stored_dim}, but EMBEDDING_DIM={self._dim}. "
                        f"To switch dimensions, either drop the collection manually "
                        f"or set MILVUS_COLLECTION to a new name."
                    )
            logger.info("Collection %s already exists (dim=%d)", name, self._dim)
        else:
            logger.info("Creating collection %s (dim=%d)", name, self._dim)
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=64,  # service_name + action_name hash
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._dim,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                ),
            ]
            schema = CollectionSchema(
                fields=fields,
                description=f"Tool Registry capabilities ({self._dim}d)",
            )
            col = Collection(name=name, schema=schema)
            col.create_index(
                field_name="embedding",
                index_params=INDEX_PARAMS,
            )
            logger.info("Created index on %s", name)

        col.load()
        logger.info("Collection %s loaded (entities: %d)", name, col.num_entities)
        return col

    async def close(self) -> None:
        """Disconnect from Milvus."""
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            self._collection = None
            logger.info("Disconnected from Milvus")

    # -- Write operations ----

    async def upsert(
        self,
        capability_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or replace a single capability entry.

        Milvus doesn't have true upsert — we delete then insert.
        For batch operations, prefer upsert_batch() which deletes by service first.
        """
        await asyncio.to_thread(self._upsert_sync, capability_id, embedding, metadata)

    def _upsert_sync(
        self,
        capability_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        col = self._require_collection()
        # Delete if exists
        col.delete(expr=f'id == "{capability_id}"')
        col.insert([[capability_id], [embedding], [metadata]])
        logger.debug("Upserted capability %s", capability_id)

    async def upsert_batch(
        self,
        entries: list[tuple[str, list[float], dict[str, Any]]],
        service_name: str,
    ) -> None:
        """Atomically replace all capabilities for a service.

        Deletes all existing entries for the service, then inserts all new ones.
        entries: list of (capability_id, embedding, metadata)
        """
        await asyncio.to_thread(self._upsert_batch_sync, entries, service_name)

    def _upsert_batch_sync(
        self,
        entries: list[tuple[str, list[float], dict[str, Any]]],
        service_name: str,
    ) -> None:
        col = self._require_collection()

        # Delete all existing entries for this service using JSON path filter
        # Milvus JSON filter syntax: metadata["service_name"] == "web_search"
        try:
            col.delete(expr=f'metadata["service_name"] == "{service_name}"')
            logger.debug("Deleted existing entries for service: %s", service_name)
        except Exception:
            logger.warning(
                "Could not delete existing entries for %s — inserting anyway",
                service_name,
                exc_info=True,
            )

        if not entries:
            return

        ids = [e[0] for e in entries]
        embeddings = [e[1] for e in entries]
        metadatas = [e[2] for e in entries]

        col.insert([ids, embeddings, metadatas])
        col.flush()
        logger.info(
            "Upserted %d capabilities for service: %s", len(entries), service_name
        )

    async def delete_by_service(self, service_name: str) -> None:
        """Remove all capability entries for a service."""
        await asyncio.to_thread(self._delete_by_service_sync, service_name)

    def _delete_by_service_sync(self, service_name: str) -> None:
        col = self._require_collection()
        col.delete(expr=f'metadata["service_name"] == "{service_name}"')
        col.flush()
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
        return await asyncio.to_thread(
            self._search_sync, query_embedding, limit, filter_expr
        )

    def _search_sync(
        self,
        query_embedding: list[float],
        limit: int,
        filter_expr: str | None,
    ) -> list[dict[str, Any]]:
        col = self._require_collection()
        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=limit,
            expr=filter_expr,
            output_fields=["metadata"],
        )

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                try:
                    metadata = hit.entity.get("metadata") or {}
                except Exception:
                    metadata = {}
                hits.append(
                    {
                        "id": hit.id,
                        "score": float(hit.distance),
                        "metadata": metadata,
                    }
                )

        return hits

    # -- Count operations ----

    async def count(self) -> int:
        """Total number of capabilities in the collection."""
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        col = self._require_collection()
        col.flush()
        return col.num_entities

    async def count_by_service(self) -> dict[str, int]:
        """Return per-service capability counts by scanning metadata.

        Note: Milvus doesn't support GROUP BY — we query all IDs + metadata
        and count in Python. Fine for a registry with hundreds of entries.
        """
        return await asyncio.to_thread(self._count_by_service_sync)

    def _count_by_service_sync(self) -> dict[str, int]:
        col = self._require_collection()
        # Query all entries — just need the service_name from metadata.
        # Limit is set well above any realistic registry size; log a warning
        # if we're approaching it so an operator knows to raise it.
        query_limit = 16384
        results = col.query(
            expr="id != ''",
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

    def _require_collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("CapabilityStore not connected — call connect() first")
        return self._collection
