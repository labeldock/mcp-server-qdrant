FROM python:3.13-slim

WORKDIR /app

# Copy the local source code and metadata
COPY pyproject.toml README.md ./
COPY src ./src

# Install from local source code (includes update/delete features)
RUN pip install --no-cache-dir .

# Pre-download embedding model to avoid cold start delays
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Expose the default port for StreamableHTTP transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
# COLLECTION_NAME accepts a whitespace-separated list of "name[:perms]" directives,
# e.g. "travel:ro place:rw pin:rwd". perms are any of r/w/d; no suffix means full
# access (rwd). A single bare name keeps the classic default-collection behaviour.
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV QDRANT_READ_ONLY="false"
# Shared-secret gate. When non-empty, clients must send
# "Authorization: Bearer <MCP_PASSWORD>"; the /health route stays public.
ENV MCP_PASSWORD=""

# FastMCP settings - required for StreamableHTTP
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8000"

# Unbuffer stdout/stderr so the startup collection/permission summary and other
# logs appear immediately in container/serverless logs.
ENV PYTHONUNBUFFERED=1

# Health check - use the custom health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run the server with StreamableHTTP transport (required for Lobe Chat)
CMD ["mcp-server-qdrant", "--transport", "streamable-http"]
