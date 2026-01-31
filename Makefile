.PHONY: help setup dev dev-stop dev-logs dev-build test clean ollama-pull db-shell migrate migration

# Default target
help:
	@echo "miRAGe Development Commands"
	@echo ""
	@echo "  make setup        - Install Python dependencies (for local testing)"
	@echo "  make dev          - Start all services (DB, Ollama, API, Indexer)"
	@echo "  make dev-stop     - Stop all services"
	@echo "  make dev-logs     - Show service logs"
	@echo "  make dev-build    - Rebuild Docker images"
	@echo "  make test         - Run tests"
	@echo "  make ollama-pull  - Download embedding model"
	@echo "  make db-shell     - Open PostgreSQL shell"
	@echo "  make migrate      - Run database migrations"
	@echo "  make migration    - Generate new migration (msg='description')"
	@echo "  make clean        - Stop services and remove volumes"

# Install dependencies (for local testing)
setup:
	uv sync --all-extras

# Start all development services
dev:
	docker compose up -d --build
	@echo ""
	@echo "Services starting..."
	@echo "  - API:     http://localhost:8000"
	@echo "  - Ollama:  http://localhost:11434"
	@echo "  - DB:      localhost:5433"
	@echo ""
	@echo "Run 'make ollama-pull' to download the embedding model (first time only)."
	@echo "Run 'make dev-logs' to see logs."

# Stop development services
dev-stop:
	docker compose stop

# Show logs
dev-logs:
	docker compose logs -f

# Rebuild images
dev-build:
	docker compose build

# Run tests
test:
	uv run pytest

# Download embedding model
ollama-pull:
	docker exec mirage-ollama ollama pull nomic-embed-text

# Open PostgreSQL shell
db-shell:
	docker exec -it mirage-db psql -U mirage -d mirage

# Run database migrations
migrate:
	docker exec mirage-api uv run alembic upgrade head

# Generate a new migration (usage: make migration msg="add foo column")
migration:
	docker exec mirage-api uv run alembic revision --autogenerate -m "$(msg)"

# Clean up everything
clean:
	docker compose down -v
	rm -f mirage.db
