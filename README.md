# Tool Registry

A semantic capability index for microservices — Tool RAG in a box.

Services register their capabilities; clients search by natural-language
intent and get back only the relevant tools, fully specified and ready to use.

## Background: why Tool RAG?

LLM-based agents that integrate tools typically inject all tool definitions
into the system prompt. This works with 5–10 tools but breaks down as the
number grows. A service ecosystem with 100+ capabilities consumes thousands
of tokens of context on every call — most of them irrelevant to the current
request. Selection accuracy degrades, latency increases, and inference cost
scales linearly with tool count.

Tool RAG applies the same principle as retrieval-augmented generation to tool
discovery: index capabilities as vectors, retrieve only the semantically
relevant ones for a given intent, and inject only those into the model context.
The cost and context usage stay constant regardless of how many tools exist.

**The AWS benchmark on this pattern is striking:** with 422 tools, vector-based
selection achieved 82.3% accuracy versus 75.8% when all tools were included —
giving the model *fewer* tools made it *more* accurate. It also reduced latency
by 21% and cut inference cost by 92%.

This happens because LLMs are better at choosing from a curated shortlist than
searching a haystack. Irrelevant tool definitions create noise that degrades
decision quality.

## How it works

```
Service starts
    → POSTs ServiceManifest to /api/v1/register (or announces via RabbitMQ)
    → Registry embeds each action's description as a vector
    → Vectors stored in Milvus

Client queries
    → POST /api/v1/search {"query": "search the web for recent news"}
    → Registry embeds query, searches Milvus by cosine similarity
    → Returns top-N capabilities with full schemas
    → (Optional) LLM reranker refines the ranking for higher precision
```

**Heartbeat deduplication:** if a service re-registers with an unchanged
manifest (same SHA-256 fingerprint), the registry skips re-embedding and just
updates `last_seen`. Services can call `/api/v1/register` on a schedule without
causing unnecessary Milvus writes.

**Enriched embeddings:** the registry builds composite search text from the
service context, action description, and parameter names — not just the action
name. This significantly improves retrieval quality for queries like "I'm
hungry" that don't literally match "order food delivery".

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
The registry subscribes and auto-registers services as they appear. No
configuration needed on the service side beyond their existing announce
behaviour.

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

Disabled by default. When enabled, a second LLM pass reranks the vector
search results for improved precision — vector similarity captures semantic
proximity but can miss functional relevance. The reranker uses structured
output (JSON schema) to guarantee a valid response without fragile text
extraction.

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `false` | Enable the reranker |
| `RERANKER_PROVIDER` | `openai_compatible` | `openai_compatible` \| `ollama` |
| `RERANKER_MODEL` | `google/gemini-2.0-flash-001` | Model name |
| `RERANKER_LLM_URL` | `https://openrouter.ai/api/v1` | API base URL (openai_compatible only) |
| `RERANKER_API_KEY` | _(empty)_ | Bearer token (openai_compatible only) |
| `RERANKER_OLLAMA_URL` | `http://ollama:11434` | Ollama base URL (ollama only) |

The reranker requires a model with structured output support
(`response_format` with JSON schema for OpenAI-compatible endpoints,
`format` parameter for Ollama). A small, fast model is sufficient — it
only needs to return a ranked list of integers.

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
does and when you'd use it — not just the action name. The quality of
retrieval is directly proportional to the quality of these descriptions.

## Research

The Tool RAG pattern is well-supported by recent research:

- **AWS S3 Vectors + Bedrock** (2025) — Production benchmark with 422 tools:
  vector selection achieved 82.3% accuracy, 91.9% recall, 21% lower latency,
  and 92% cost reduction versus the all-tools baseline.
  [AWS Blog](https://aws.amazon.com/blogs/storage/optimize-agent-tool-selection-using-s3-vectors-and-bedrock-knowledge-bases/)

- **ToolScope** (Ma et al., Oct 2025) — Addresses redundant and overlapping
  tool descriptions with automatic merging and context-aware filtering.
  Demonstrated 8–39% gains in tool selection accuracy across multiple LLMs.
  [ResearchGate](https://www.researchgate.net/publication/396848010)

- **Tool-to-Agent Retrieval** (Li et al., Nov 2025) — Embeds tools and their
  parent agents in a shared vector space. 19.4% improvement in Recall@5 on
  LiveMCPBench.
  [arXiv:2511.01854](https://arxiv.org/abs/2511.01854)

- **ToolReAGt** (ACL KnowLLM Workshop, Aug 2025) — ReAct-style prompting for
  iterative tool retrieval across complex multi-step tasks from large tool pools.
  [ACL Anthology](https://aclanthology.org/2025.knowllm-1.7/)

- **Tool RAG** (Red Hat, Nov 2025) — Enterprise framework showing intelligent
  tool retrieval can triple invocation accuracy while halving prompt length.
  [Red Hat Emerging Technologies](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/)

## Building

```bash
# Install dependencies
uv pip install .

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8014 --reload

# Build Docker image
docker build -t tool-registry .
```

## License

MIT
