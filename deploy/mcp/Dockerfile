# Jubilee MCP (stdio) — same dependency set as the API image, different entrypoint.
# Build from repository root: docker build -f deploy/mcp/Dockerfile .

FROM python:3.14-slim AS builder

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt


FROM python:3.14-slim AS runtime

WORKDIR /app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN chown -R appuser:appuser /app

USER appuser

# Stdio MCP does not listen on a port; Railway can still run the process.
CMD ["python", "-m", "backend.mcp.server"]
