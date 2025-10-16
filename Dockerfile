FROM python:3.13-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the mcp-server-qdrant package
RUN uv pip install --system --no-cache-dir mcp-server-qdrant

# Expose the default port for StreamableHTTP transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV QDRANT_READ_ONLY="false"

# FastMCP settings - required for StreamableHTTP
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8000"

# Run the server with StreamableHTTP transport (required for Lobe Chat)
CMD uvx mcp-server-qdrant --transport streamable-http
