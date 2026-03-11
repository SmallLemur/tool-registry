"""Milvus schema migrations for the Tool Registry.

Simple sequential migration runner — no framework, no external state.
Each migration is an idempotent async callable that self-checks whether
its work is already done before applying changes.

Pattern:
  MIGRATIONS = [
      ("001_create_tool_capabilities", _m001_create_tool_capabilities),
      ("002_add_something_new",        _m002_add_something_new),
  ]

Each migration function receives the AsyncMilvusClient and the configured
embedding dimension. Migrations run in order on every startup. Because they
are idempotent, re-running them on an already-migrated instance is a no-op.

Adding a new migration:
  1. Write a new async function _mNNN_<description>(client, dim)
  2. Append it to MIGRATIONS
  3. Deploy — it runs once and is a no-op on subsequent restarts

Dimension change workflow:
  If you need to change EMBEDDING_DIM, write a new migration that drops the
  old collection and recreates it. This makes the migration explicit and
  auditable rather than silently failing at startup.
"""

import logging
from typing import Callable

from pymilvus import AsyncMilvusClient, DataType, MilvusException

logger = logging.getLogger(__name__)

# -- Index / search parameters ----

_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}


# -- Migration implementations ----


async def _m001_create_tool_capabilities(
    client: AsyncMilvusClient,
    collection_name: str,
    dim: int,
) -> None:
    """Create the tool_capabilities collection if it does not exist.

    Schema:
      id        VARCHAR PK  — 32-char hex (SHA-256 prefix of service::action)
      embedding FLOAT_VECTOR — dimension from EMBEDDING_DIM config
      metadata  JSON        — full capability fields (service, action, schemas, …)

    Index: IVF_FLAT / COSINE — same as Mneme's collections.

    Dimension mismatch protection: if the collection already exists with a
    different vector dimension, raises RuntimeError with a clear message.
    Operators must either drop the collection manually or point to a new
    MILVUS_COLLECTION name. This is intentional — silent data corruption
    (mismatched dims silently truncated) is worse than a loud startup failure.
    """
    exists = await client.has_collection(collection_name)
    if exists:
        # Validate that the stored dimension matches what we expect.
        desc = await client.describe_collection(collection_name)
        vec_field = next(
            (f for f in desc["fields"] if f["type"] == DataType.FLOAT_VECTOR),
            None,
        )
        if vec_field is not None:
            stored_dim = vec_field.get("params", {}).get("dim")
            if stored_dim is not None and stored_dim != dim:
                raise RuntimeError(
                    f"Milvus collection '{collection_name}' already exists with "
                    f"vector dimension {stored_dim}, but EMBEDDING_DIM={dim}. "
                    f"To switch dimensions, drop the collection manually or "
                    f"set MILVUS_COLLECTION to a new name."
                )
        logger.info(
            "Migration 001: collection '%s' already exists (dim=%d) — skipping",
            collection_name,
            dim,
        )
        return

    logger.info(
        "Migration 001: creating collection '%s' (dim=%d)", collection_name, dim
    )
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description=f"Tool Registry capabilities ({dim}d)",
    )
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("metadata", DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )

    await client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Migration 001: collection '%s' created", collection_name)


# -- Migration registry ----
# Add new migrations by appending to this list. Order matters.

MigrationFn = Callable[[AsyncMilvusClient, str, int], None]

MIGRATIONS: list[tuple[str, MigrationFn]] = [
    ("001_create_tool_capabilities", _m001_create_tool_capabilities),
]


# -- Runner ----


async def run_migrations(
    client: AsyncMilvusClient,
    collection_name: str,
    dim: int,
) -> None:
    """Run all registered migrations in order.

    Each migration is idempotent — running it on an already-migrated instance
    is a safe no-op. No external state is required to track which migrations
    have been applied.

    Args:
        client:          Connected AsyncMilvusClient.
        collection_name: Target Milvus collection name.
        dim:             Embedding dimension from config.

    Raises:
        RuntimeError: If a migration detects an unrecoverable state
                      (e.g. dimension mismatch).
        MilvusException: If Milvus returns an error during a migration.
    """
    logger.info("Running %d Milvus migration(s)…", len(MIGRATIONS))
    for name, fn in MIGRATIONS:
        logger.debug("Applying migration: %s", name)
        try:
            await fn(client, collection_name, dim)
        except RuntimeError:
            # Dimension mismatch and similar — re-raise immediately, these are fatal
            raise
        except MilvusException as e:
            logger.error("Migration '%s' failed with Milvus error: %s", name, e)
            raise
        except Exception as e:
            logger.error("Migration '%s' failed unexpectedly: %s", name, e)
            raise
    logger.info("All migrations complete")
