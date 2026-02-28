# Multi-stage build for QA MCP Server
FROM python:alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apk update && apk add --no-cache \
    build-base \
    git

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apk update && apk add --no-cache sqlite

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

RUN addgroup -g 1001 mcp && adduser -u 1001 -G mcp -s /bin/sh -D mcp
WORKDIR /app

COPY app/ ./app/
COPY mcp_server.py .

RUN mkdir -p /data && chown -R mcp:mcp /app /data

USER mcp

# Default symbol index path — override via SYMBOL_INDEX_PATH env var or .env
ENV SYMBOL_INDEX_PATH=/data/symbol_index.db

# TCP healthcheck — reads MCP_PORT from env so it tracks the configured port
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import socket,os; socket.create_connection(('localhost',int(os.getenv('MCP_PORT','8765'))),timeout=5)" || exit 1

# Run the MCP server in streamable-HTTP mode (port from MCP_PORT env, default 8765)
CMD ["python", "mcp_server.py", "--transport", "http"]

LABEL org.opencontainers.image.title="QA MCP Server"
LABEL org.opencontainers.image.description="MCP server for QA MySQL navigation and Grails symbol index"
LABEL mcp.server.name="qa-mcp-server"
LABEL mcp.server.type="database+symbol"
