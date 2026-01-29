# miRAGe

Local RAG system for books and documentation. Supports PDF, EPUB, and Markdown files with vector search powered by pgvector and Ollama embeddings.

## Features

- REST API for managing projects and documents
- Vector search using pgvector (PostgreSQL)
- Local embeddings via Ollama (mxbai-embed-large)
- Supports PDF, EPUB, Markdown

## Quick Start

### Prerequisites

- Docker and Docker Compose
- [uv](https://github.com/astral-sh/uv) (optional, for running tests locally)

### Setup

```bash
# Start all services (DB, Ollama, API, Indexer)
make dev

# Download embedding model (first time only, ~670MB)
make ollama-pull
```

API will be available at http://localhost:8000

Live reload is enabled — edit `src/` and changes apply automatically.

### Verify Setup

```bash
# Check services are running
docker compose ps

# Check Ollama model is loaded
docker exec mirage-ollama ollama list

# Test embedding model
curl http://localhost:11434/api/embeddings -d '{
  "model": "mxbai-embed-large",
  "prompt": "Hello world"
}'

# Test API health
curl http://localhost:8000/health
```

### Verify Indexing

```bash
# 1. Create a project
curl -X POST http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "test-project"}'
# Save the project ID from response

# 2. Create a test markdown file
echo -e "# Test Book\n\n## Chapter 1\n\nThis is test content about machine learning." > /tmp/test.md

# 3. Upload document
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/documents \
  -H "X-API-Key: dev-api-key" \
  -F "file=@/tmp/test.md"
# Save the document ID from response

# 4. Check document status (should change from "pending" to "ready")
curl http://localhost:8000/api/v1/projects/{project_id}/documents/{document_id} \
  -H "X-API-Key: dev-api-key"

# 5. Search for content
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/search \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 5}'
```

## API Usage

All API endpoints require `X-API-Key` header (default: `dev-api-key`).

### Projects

```bash
# Create project
curl -X POST http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs"}'

# List projects
curl http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key"

# Delete project
curl -X DELETE http://localhost:8000/api/v1/projects/{project_id} \
  -H "X-API-Key: dev-api-key"
```

### Documents

```bash
# Upload document
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/documents \
  -H "X-API-Key: dev-api-key" \
  -F "file=@/path/to/document.pdf"

# List documents
curl http://localhost:8000/api/v1/projects/{project_id}/documents \
  -H "X-API-Key: dev-api-key"

# Get document status
curl http://localhost:8000/api/v1/projects/{project_id}/documents/{document_id} \
  -H "X-API-Key: dev-api-key"
```

### Search

```bash
# Search in project (requires indexed documents)
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/search \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "limit": 10}'
```

## Development Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start all services (DB, Ollama, API, Indexer) |
| `make dev-stop` | Stop all services |
| `make dev-logs` | Show service logs |
| `make dev-build` | Rebuild Docker images |
| `make test` | Run tests locally |
| `make setup` | Install Python dependencies (for local testing) |
| `make ollama-pull` | Download embedding model |
| `make db-shell` | Open PostgreSQL shell |
| `make clean` | Stop services and remove volumes |

## Configuration

Environment variables (set in docker-compose.yml):

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_DATABASE_URL` | - | PostgreSQL connection string |
| `MIRAGE_API_KEY` | - | API authentication key |
| `MIRAGE_OLLAMA_URL` | `http://ollama:11434` | Ollama server URL |
| `MIRAGE_OLLAMA_MODEL` | `mxbai-embed-large` | Embedding model |
| `MIRAGE_CHUNK_SIZE` | `800` | Text chunk size |
| `MIRAGE_CHUNK_OVERLAP` | `100` | Chunk overlap |
| `MIRAGE_DOCUMENTS_PATH` | `/data/documents` | Document storage path |

## Architecture

```
miRAGe
├── API (FastAPI)           - REST endpoints
│   ├── /health
│   ├── /api/v1/projects
│   ├── /api/v1/projects/{id}/documents
│   └── /api/v1/projects/{id}/search
├── Indexer (Worker)        - Document processing (Phase 3)
│   ├── Parsers (MD, PDF, EPUB)
│   └── Chunking + Embeddings
├── Shared
│   ├── Config (pydantic-settings)
│   ├── DB (SQLAlchemy + pgvector)
│   └── Embedding Client (Ollama)
└── Infrastructure
    ├── Docker Compose (local dev)
    └── Helm Chart (k8s deployment)
```

## Project Status

- [x] Phase 1: Foundation (config, db, embedding client)
- [x] Phase 2: API (projects, documents, search endpoints)
- [x] Phase 3: Indexer (document parsing, chunking, embedding)
- [ ] Phase 4: CLI integration
- [ ] Phase 5: Infrastructure (Docker, Helm, migrations)

## License

MIT
