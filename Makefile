.PHONY: help setup dev dev-stop dev-logs api indexer test clean ollama-pull db-shell

# Default target
help:
	@echo "miRAGe Development Commands"
	@echo ""
	@echo "  make setup        - Install Python dependencies"
	@echo "  make dev          - Start all services (PostgreSQL + Ollama)"
	@echo "  make dev-stop     - Stop all services"
	@echo "  make dev-logs     - Show service logs"
	@echo "  make api          - Run API server (requires 'make dev' first)"
	@echo "  make indexer      - Run indexer worker (requires 'make dev' first)"
	@echo "  make test         - Run tests"
	@echo "  make ollama-pull  - Download embedding model"
	@echo "  make db-shell     - Open PostgreSQL shell"
	@echo "  make clean        - Stop services and remove volumes"

# Install dependencies
setup:
	uv sync --all-extras

# Start development services
dev:
	docker compose up -d
	@echo ""
	@echo "Services started. Run 'make ollama-pull' to download the embedding model."
	@echo "Then run 'make api' to start the API server."

# Stop development services
dev-stop:
	docker compose stop

# Show logs
dev-logs:
	docker compose logs -f

# Run API server
api:
	MIRAGE_DATABASE_URL="postgresql+asyncpg://mirage:mirage@localhost:5433/mirage" \
	MIRAGE_API_KEY="dev-api-key" \
	MIRAGE_OLLAMA_URL="http://localhost:11434" \
	uv run uvicorn mirage.api.main:app --reload --host 0.0.0.0 --port 8000

# Run indexer worker
indexer:
	MIRAGE_DATABASE_URL="postgresql+asyncpg://mirage:mirage@localhost:5433/mirage" \
	MIRAGE_API_KEY="dev-api-key" \
	MIRAGE_OLLAMA_URL="http://localhost:11434" \
	uv run python -m mirage.indexer.worker

# Run tests
test:
	uv run pytest

# Download embedding model
ollama-pull:
	docker exec mirage-ollama ollama pull mxbai-embed-large

# Open PostgreSQL shell
db-shell:
	docker exec -it mirage-db psql -U mirage -d mirage

# Clean up everything
clean:
	docker compose down -v
	rm -f mirage.db
