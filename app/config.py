"""Tool Registry configuration.

All settings come from environment variables — no .env file loading in app code.
Set via Docker Compose environment section at deployment time.
"""

from pydantic_settings import BaseSettings


class RegistrySettings(BaseSettings):
    """Tool Registry service configuration."""

    # -- Service ----
    registry_host: str = "0.0.0.0"
    registry_port: int = 8014

    # -- Milvus vector store ----
    milvus_host: str = "pensante-milvus"
    milvus_port: int = 19530
    milvus_collection: str = "tool_capabilities"

    # -- Embeddings ----
    # Which provider to use: sentence_transformers | ollama | openai
    embedding_provider: str = "sentence_transformers"
    # Provider-specific model name (e.g. all-MiniLM-L6-v2, nomic-embed-text)
    embedding_model: str = "all-MiniLM-L6-v2"
    # MUST match the model's output dimension. Validated against Milvus on startup.
    embedding_dim: int = 384

    # sentence_transformers: local model cache directory
    models_dir: str = "/data/models"

    # ollama: base URL of Ollama instance
    ollama_url: str = "http://ollama:11434"

    # openai-compatible: endpoint + key
    openai_embedding_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # -- Registration plugin ----
    # Which plugin handles service discovery: http_push | rabbitmq_listener
    registration_plugin: str = "rabbitmq_listener"

    # RabbitMQ (used by rabbitmq_listener plugin)
    rabbitmq_url: str = "amqp://pensante:pensante@pensante-rabbitmq:5672/"

    # HTTP push plugin — outbound health polling
    http_heartbeat_interval_s: int = 60  # seconds between polling rounds
    http_heartbeat_timeout_s: int = 10  # per-request timeout

    # -- Search tuning ----
    search_default_limit: int = 5
    search_default_threshold: float = 0.5

    # -- Health tracking ----
    # Seconds before a service is considered stale (no heartbeat received)
    service_heartbeat_timeout_s: int = 90

    # -- LLM Reranker (optional) ----
    reranker_enabled: bool = False
    # Which reranker backend: openai_compatible | ollama
    reranker_provider: str = "openai_compatible"
    # OpenAI-compatible backend (OpenRouter, vLLM, etc.)
    reranker_llm_url: str = "https://openrouter.ai/api/v1"
    reranker_api_key: str = ""
    # Ollama backend
    reranker_ollama_url: str = "http://ollama:11434"
    # Model name used by whichever provider is active
    reranker_model: str = "google/gemini-2.0-flash-001"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = RegistrySettings()
