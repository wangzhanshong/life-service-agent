# Multi-stage build
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY pyproject.toml .
RUN pip install --no-cache-dir --user -e ".[faiss]"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Copy installed packages from builder
COPY --from=builder /root/.local /home/agent/.local

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent data/ ./data/
COPY --chown=agent:agent pyproject.toml .

# Create workspace
RUN mkdir -p /app/workspace /app/data/documents /app/data/faiss_index && \
    chown -R agent:agent /app/workspace /app/data

# Make installed packages available
ENV PATH="/home/agent/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
