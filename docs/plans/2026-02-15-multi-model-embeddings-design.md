# Multi-Model Embeddings Design

## Context

Current miRAGe uses a single embedding model (`nomic-embed-text`, 768 dimensions) stored directly in the `chunks` table. This limits flexibility:
- Cannot use different models for different projects
- Cannot compare search quality across models
- Cannot leverage model-specific strengths (multilingual vs. fast vs. accurate)

The original idea of using CozoDB was explored but rejected due to migration complexity.

## Decision: Separate Embeddings Tables Per Model

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Project Config                          │
│  models: [nomic-embed-text, bge-m3, mxbai-embed-large]      │
│  ollama_url: http://ollama:11434                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Database Schema                          │
│                                                              │
│  chunks (id, document_id, content, position, ...)           │
│       │                                                      │
│       ├── embeddings_nomic_768 (chunk_id, vector)           │
│       ├── embeddings_bge_m3_1024 (chunk_id, vector)         │
│       └── embeddings_mxbai_1024 (chunk_id, vector)          │
│                                                              │
│  embedding_status (chunk_id, model_name, status)            │
│  project_models (project_id, model_name, enabled)           │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Separate embeddings tables** — one table per model with fixed dimension
2. **Predefined model set** — 3 models hardcoded in the service (see `docs/supported_models.md`)
3. **Project-level model selection** — each project specifies which models to use
4. **Default: all models** — if not specified, project uses all available models
5. **Parallel indexing** — all enabled models index simultaneously
6. **Union search** — search across all enabled models, deduplicate by chunk_id

### Database Schema

```sql
-- Projects (add ollama_url)
projects (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    ollama_url VARCHAR(512) DEFAULT 'http://ollama:11434',
    created_at TIMESTAMP
)

-- Project models configuration
project_models (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    model_name VARCHAR(100),
    enabled BOOLEAN DEFAULT true,
    PRIMARY KEY (project_id, model_name)
)

-- Documents (unchanged)
documents (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    original_path VARCHAR(512),
    file_type VARCHAR(50),
    status VARCHAR(50),
    error_message TEXT,
    metadata JSON,
    created_at TIMESTAMP,
    indexed_at TIMESTAMP
)

-- Chunks (remove embedding column)
chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT,
    position INT,
    parent_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    structure JSON,
    metadata JSON,
    status VARCHAR(50) DEFAULT 'pending'
)

-- Embedding status per chunk per model
embedding_status (
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    model_name VARCHAR(100),
    status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, ready, failed
    error_message TEXT,
    PRIMARY KEY (chunk_id, model_name)
)

-- Predefined embeddings tables
embeddings_nomic_768 (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector VECTOR(768)
)

embeddings_bge_m3_1024 (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector VECTOR(1024)
)

embeddings_mxbai_1024 (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector VECTOR(1024)
)

-- Vector indexes (one per table)
CREATE INDEX ix_embeddings_nomic_768 ON embeddings_nomic_768
    USING hnsw (vector vector_cosine_ops);

CREATE INDEX ix_embeddings_bge_m3_1024 ON embeddings_bge_m3_1024
    USING hnsw (vector vector_cosine_ops);

CREATE INDEX ix_embeddings_mxbai_1024 ON embeddings_mxbai_1024
    USING hnsw (vector vector_cosine_ops);
```

### Indexing Flow

```
1. ChunkWorker:
   - Parse document → Create chunks (status='pending')
   - Create embedding_status rows for each enabled model

2. EmbeddingWorker (parallel per model):
   - SELECT chunks WHERE embedding_status.status = 'pending'
                   AND project_models.enabled = true
   - Embed via Ollama for specific model
   - INSERT INTO embeddings_{model}
   - UPDATE embedding_status SET status = 'ready'

3. StatusWorker:
   - Check if all enabled models have ready embeddings
   - Update chunk status
```

### Search Flow

```
1. Get enabled models for project
2. Embed query with each enabled model (parallel)
3. For each model:
   - SELECT chunk_id, vector <=> query_vector as distance
     FROM embeddings_{model}
     ORDER BY distance LIMIT k
4. Union all results
5. Deduplicate by chunk_id (keep minimum distance)
6. Sort by distance, limit to k
7. Join with chunks to get content and parent_content
8. Return ranked results
```

### API

#### Create Project

```http
POST /projects
{
    "name": "my-project",
    "ollama_url": "http://ollama:11434",  // optional
    "models": ["nomic-embed-text", "bge-m3"]  // optional, default: all
}
```

#### Search (backward compatible)

```http
GET /projects/{id}/search?q=query&k=10
# Uses all enabled models

GET /projects/{id}/search?q=query&k=10&models=nomic-embed-text,bge-m3
# Uses only specified models (subset of enabled)
```

Response (unchanged):

```json
{
    "results": [
        {
            "chunk_id": "...",
            "content": "...",
            "parent_content": "...",
            "distance": 0.15,
            "document_id": "...",
            "metadata": {}
        }
    ]
}
```

### Configuration

#### Environment Variables

```bash
# Default Ollama URL (can be overridden per-project)
MIRAGE_OLLAMA_URL=http://ollama:11434
```

#### Project Config (.mirage.yaml)

```yaml
models:
  - nomic-embed-text
  - bge-m3
ollama_url: http://ollama:11434
```

### Migration Plan

1. Create new tables: `project_models`, `embedding_status`, `embeddings_*`
2. Create vector indexes
3. Remove `embedding` column from `chunks` (keep for rollback)
4. Populate `project_models` for existing projects (all models enabled)
5. Create `embedding_status` rows for existing chunks
6. Reindex all documents with all models

### Supported Models

See `docs/supported_models.md` for the list of supported models with their characteristics.

### Implementation Steps

1. [ ] Create `docs/supported_models.md` with model descriptions
2. [ ] Update database schema (`src/mirage/shared/db.py`)
3. [ ] Create Alembic migration for new tables
4. [ ] Update `ProjectTable` with `ollama_url`
5. [ ] Create `ProjectModelTable`, `EmbeddingStatusTable`, `EmbeddingsTable_*`
6. [ ] Update embedding worker to handle multiple models
7. [ ] Update search to query multiple tables
8. [ ] Update API endpoints for project creation with models
9. [ ] Update API search endpoint with `models` parameter
10. [ ] Update CLI commands
11. [ ] Write tests
12. [ ] Update documentation

### Rollback Plan

1. Keep `chunks.embedding` column during migration
2. If issues arise, revert to single-model code
3. Drop new tables, restore from backup if needed
