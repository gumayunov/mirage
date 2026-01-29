FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Default command (overridden in docker-compose)
CMD ["uv", "run", "uvicorn", "mirage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
