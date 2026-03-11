# Tool Registry

A semantic capability index for microservices. Services register their
capabilities; clients search by natural-language intent.

Services describe what they can do (actions, schemas, descriptions). The
registry embeds those descriptions as vectors in Milvus. At query time,
Cortex — or any client — sends a natural-language intent and gets back the
top-N most relevant capabilities, fully specified and ready to use.

## How it works

```
Service starts
    → POSTs ServiceManifest to /api/v1/register (or announces via RabbitMQ)
    → Registry embeds each action's description
    → Vectors stored in Milvus

Client queries
    → POST /api/v1/search {"query": "search the web for recent news"}
    → Registry embeds query, searches Milvus by cosine similarity
    → Returns top-N capabilities with full schemas
```

Heartbeat deduplication: if a service re-registers with the same manifest
(same SHA-256 fingerprint), the registry skips re-embedding and just updates
`last_seen`. Services can safely call `/api/v1/register` on a schedule
without causing unnecessary Milvus writes.

## Stack

| Component | Role |
|-----------|------|
| [FastAPI](https://fastapi.tiangolo.com) | HTTP API |
| [Milvus](https://milvus.io) | Vector store for capability embeddings |
| [sentence-transformers](https://www.sbert.net) | Default local embedding model |
| [Ollama](https://ollama.com) | Optional local embeddings or LLM reranking |
| RabbitMQ | Optional auto-discovery via fanout exchange |

## Quick start

```bash
# 1. Start Milvus (standalone mode)
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:latest standalone

# 2. Run the registry
docker run -d --name tool-registry \
  -p 8014:8014 \
  -e MILVUS_HOST=host.docker.internal \
  -e REGISTRATION_PLUGIN=http_push \
  registry.gitlab.com/pensante1/tool-registry:latest

# 3. Register a service
curl -X POST http://localhost:8014/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "web_search",
    "version": "1.0.0",
    "description": "Search the web using SearXNG",
    "actions": [{
      "name": "search",
      "description": "Search the web for information on a topic",
      "input_schema": {
        "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}
      }
    }]
  }'

# 4. Search
curl -X POST http://localhost:8014/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "find recent news about AI", "limit": 3}'
```

## Configuration

All configuration is via environment variables. No `.env` file loading in
application code — set these in your Docker Compose `environment` block.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `REGISTRY_HOST` | `0.0.0.0` | Bind address |
| `REGISTRY_PORT` | `8014` | Listen port |

### Milvus

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `milvus` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `tool_capabilities` | Collection name |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` \| `ollama` \| `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name (provider-specific) |
| `EMBEDDING_DIM` | `384` | Vector dimension — must match the model |
| `MODELS_DIR` | `/data/models` | Local model cache (sentence_transformers only) |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama base URL (ollama provider only) |
| `OPENAI_EMBEDDING_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (openai provider only) |
| `OPENAI_API_KEY` | _(empty)_ | API key (openai provider only) |

**Switching embedding models:** change `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`,
and `EMBEDDING_DIM` together. If `MILVUS_COLLECTION` already exists with a
different dimension, startup will fail with a clear error — either drop the
collection manually or point to a new collection name.

### Registration

| Variable | Default | Description |
|----------|---------|-------------|
| `REGISTRATION_PLUGIN` | `rabbitmq_listener` | `rabbitmq_listener` \| `http_push` |
| `RABBITMQ_URL` | `amqp://guest:guest@rabbitmq:5672/` | RabbitMQ connection (rabbitmq_listener only) |
| `RABBITMQ_ANNOUNCE_EXCHANGE` | `tool-registry.announce` | Fanout exchange services publish manifests to |
| `HTTP_HEARTBEAT_INTERVAL_S` | `60` | How often to poll service health endpoints (http_push only) |
| `HTTP_HEARTBEAT_TIMEOUT_S` | `10` | Per-request timeout for health polls (http_push only) |

#### `rabbitmq_listener` (default)

Services publish their manifest to a RabbitMQ fanout exchange on startup.
The registry subscribes to that exchange and auto-registers services as they
appear. No configuration needed on the service side beyond their existing
RabbitMQ announce behaviour.

#### `http_push`

No RabbitMQ required. Services POST their manifest directly to
`/api/v1/register` on startup and periodically for heartbeats. The registry
also polls each service's `/api/v1/health` endpoint to detect failures.

For health polling to work, include `base_url` in the manifest:

```json
{
  "name": "my_service",
  "base_url": "http://my-service:8020",
  "actions": [...]
}
```

Or provide an explicit `health_url` to override the derived URL.

### Search

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_DEFAULT_LIMIT` | `5` | Default number of results |
| `SEARCH_DEFAULT_THRESHOLD` | `0.5` | Minimum cosine similarity score |
| `SERVICE_HEARTBEAT_TIMEOUT_S` | `90` | Seconds before a service is considered stale |

### LLM reranker (optional)

Disabled by default. When enabled, a second LLM pass reranks vector search
results by relevance, improving precision for ambiguous queries.

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `false` | Enable the reranker |
| `RERANKER_PROVIDER` | `openai_compatible` | `openai_compatible` \| `ollama` |
| `RERANKER_MODEL` | `google/gemini-2.0-flash-001` | Model name |
| `RERANKER_LLM_URL` | `https://openrouter.ai/api/v1` | API base URL (openai_compatible only) |
| `RERANKER_API_KEY` | _(empty)_ | Bearer token (openai_compatible only) |
| `RERANKER_OLLAMA_URL` | `http://ollama:11434` | Ollama base URL (ollama only) |

The reranker requires a model that supports structured output (`response_format`
with JSON schema for OpenAI-compatible, `format` parameter for Ollama). Use a
small fast model — it only needs to return a ranked list of integers.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/register` | Register or heartbeat a service |
| `DELETE` | `/api/v1/deregister/{name}` | Remove a service |
| `GET` | `/api/v1/services` | List registered services and health |
| `POST` | `/api/v1/search` | Semantic search over capabilities |
| `GET` | `/api/v1/stats` | Registry statistics |
| `GET` | `/api/v1/health` | Health check |

Interactive docs available at `http://localhost:8014/docs`.

### ServiceManifest schema

```json
{
  "name": "my_service",
  "version": "1.2.0",
  "description": "What this service does overall",
  "service_type": "skill",
  "base_url": "http://my-service:8020",
  "actions": [
    {
      "name": "do_thing",
      "description": "Detailed description used for semantic search",
      "input_schema": {
        "type": "object",
        "properties": {
          "param1": {"type": "string", "description": "..."}
        }
      },
      "output_schema": {},
      "risk_level": 0.0,
      "timeout_seconds": 30,
      "tags": ["category"]
    }
  ]
}
```

The `description` field on each action is the primary text embedded for
search. Write it as a natural-language sentence describing what the action
does and when you'd use it — not just the action name.

## Building

```bash
# Install dependencies
uv pip install .

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8014 --reload

# Build Docker image
docker build -t tool-registry .

# Build and push to your registry
TOOL_REGISTRY_IMAGE=registry.example.com/you/tool-registry ./build-and-push.sh
```

## License

MIT
