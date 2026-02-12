# Embedding Model Selection Design

## Context

Current miRAGe configuration uses `nomic-embed-text` (768 dimensions) for all document types. The project needs:
- Conceptual search (architecture patterns, best practices, development guides)
- Multi-language content support (RU/EN and others)
- Technical documentation focus (not code-specific, but conceptual)

## Problem

Single model approach may not be optimal for the intended use case. Domain-specific models provide better quality for specific content types.

### Constraints

- Database migration complexity (Vector dimension change requires migration)
- User prefers low-to-medium complexity solution
- Embedding dimension is not a major concern for storage

## Decision: Single Project-Level Model

### Selected Model: bge-m3

**Rationale:**
- **Multi-language:** Optimized for multi-language content (RU/EN)
- **Long context:** Supports 8192 tokens vs 8192 in current nomic-embed-text
- **Conceptual search:** High quality for semantic search of technical concepts
- **Size:** 1024 dimensions (increase from 768)

### Architecture

```
Project Configuration (.mirage.yaml)
├── embedding_model: "bge-m3" (default)
└── embedding_dimensions: 1024

Database Schema
└── chunks.embedding: Vector(1024)

Indexing Flow
└── All documents → single embedding model → same dimensionality

Search Flow
└── Query → same embedding model → vector search
```

### Why Not Other Approaches

- **Multiple models for different documents:** Too complex for the benefit
- **Zero-padding:** Degraded search quality, semantic mismatch
- **Fusion search:** Overkill for the project scope

## Migration Plan

### Phase 1: Database Migration

1. Update `src/mirage/shared/db.py`:
   ```python
   embedding = mapped_column(Vector(1024), nullable=True)
   ```

2. Create new Alembic migration:
   ```bash
   uv run alembic revision -m "update embedding dimension to 1024"
   ```
   - Change column type from `Vector(768)` to `Vector(1024)`
   - **NOTE:** This will DROP and RECREATE the column, existing embeddings lost
   - Alternative: Create new column `embedding_1024`, migrate data gradually

### Phase 2: Configuration Update

1. Update `src/mirage/shared/config.py`:
   ```python
   ollama_model: str = "bge-m3"
   ```

2. Add optional configuration in `.mirage.yaml`:
   ```yaml
   embedding_model: "bge-m3"  # Override default
   ```

### Phase 3: Reindexing

1. Reset all chunks' embedding status to `pending`:
   ```python
   UPDATE chunks SET embedding = NULL, status = 'pending'
   ```

2. Restart embedding worker:
   ```bash
   uv run python -m mirage.indexer.worker
   ```

### Phase 4: Testing

1. Test search with multi-language queries
2. Compare quality before/after migration on sample documents
3. Verify performance (embedding generation time, search latency)

## Implementation Steps

1. [ ] Update database model (`src/mirage/shared/db.py`)
2. [ ] Create Alembic migration
3. [ ] Update default config (`src/mirage/shared/config.py`)
4. [ ] Run migration
5. [ ] Download bge-m3 model in Ollama:
    ```bash
    ollama pull bge-m3
    ```
6. [ ] Reset chunks to pending state
7. [ ] Run reindexing
8. [ ] Test search quality
9. [ ] Update documentation

## Rollback Plan

If migration fails or quality is worse:

1. Keep backup of existing embeddings before migration
2. Revert database migration:
   ```bash
   alembic downgrade
   ```
3. Restore old model configuration
4. Revert to nomic-embed-text model

## Testing Strategy

**Quality Metrics:**
- Manual comparison of search results before/after
- Multi-language query testing (RU/EN)
- Conceptual search test cases

**Performance Metrics:**
- Embedding generation time per chunk
- Search latency for typical queries
- Memory usage during reindexing

## Open Questions

1. **Zero-downtime migration:** How to handle search during reindexing? (fallback to old embeddings?)
2. **Gradual migration:** Should we support both models temporarily during transition?
