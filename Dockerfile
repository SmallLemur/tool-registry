# Tool Registry — Semantic capability index for Pensante services
# Indexes service capabilities as vectors in Milvus for semantic search.
# Services auto-discover via RabbitMQ announce exchange.

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app/ ./app/

USER app

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8014/api/v1/health || exit 1

EXPOSE 8014
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8014"]
