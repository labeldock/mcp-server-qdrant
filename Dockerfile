FROM python:3.13-slim

WORKDIR /app

# Copy the local source code and metadata
COPY pyproject.toml README.md ./
COPY src ./src

# Install from local source code (includes update/delete features)
RUN pip install --no-cache-dir .

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
CMD ["mcp-server-qdrant", "--transport", "streamable-http"]
