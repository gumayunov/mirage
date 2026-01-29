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
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [mise](https://mise.jdx.dev/) (optional, for version management)

### Setup

```bash
# Install dependencies
uv sync

# Start PostgreSQL and Ollama
make dev

# Download embedding model (first time only, ~670MB)
make ollama-pull

# Run API server
make api
```

API will be available at http://localhost:8000

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
| `make dev` | Start PostgreSQL + Ollama containers |
| `make dev-stop` | Stop containers |
| `make dev-logs` | Show container logs |
| `make api` | Run API server with hot reload |
| `make test` | Run tests |
| `make ollama-pull` | Download embedding model |
| `make db-shell` | Open PostgreSQL shell |
| `make clean` | Stop containers and delete volumes |

**Why `make dev` and `make api` are separate?**

- `make dev` starts infrastructure (PostgreSQL, Ollama) in Docker containers running in background
- `make api` runs the API server locally with hot reload — code changes apply instantly without restart
- This separation enables faster development cycle and easier debugging (logs in terminal, debugger support)

## Configuration

Environment variables (set automatically by `make api`):

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
- [ ] Phase 3: Indexer (document parsing, chunking, embedding)
- [ ] Phase 4: CLI integration
- [ ] Phase 5: Infrastructure (Docker, Helm, migrations)

## License

MIT
