# API Server Dockerfile for sejm-whiz REST API
FROM python:3.12-slim-bookworm AS base

# Set timezone to UTC to avoid interactive prompts
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /bin/uv

FROM base AS builder

# Set uv configuration
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app

# Install Python 3.12 using uv
RUN uv python install 3.12

# Copy dependency files
COPY uv.lock pyproject.toml /app/

# Install dependencies using locked versions
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY . /app

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

FROM base

# Install Python 3.12 using uv (needed in final stage)
RUN uv python install 3.12

# Copy the application from builder
COPY --from=builder /app /app

# Set PATH to include uv-managed Python and virtual environment
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Expose port
EXPOSE 8000

# Run the API server using uvicorn for better production performance
CMD ["uv", "run", "uvicorn", "projects.api_server.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
