# Raw API Usage

All API endpoints require `X-API-Key` header (default: `dev-api-key`).

## Verify Indexing

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

## Projects

```bash
# Create project (uses all available models by default)
curl -X POST http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs"}'

# Create project with specific embedding models
curl -X POST http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs", "models": ["nomic-embed-text", "bge-m3"]}'

# Create project with custom Ollama URL
curl -X POST http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs", "ollama_url": "http://custom-ollama:11434"}'

# List projects
curl http://localhost:8000/api/v1/projects \
  -H "X-API-Key: dev-api-key"

# Delete project
curl -X DELETE http://localhost:8000/api/v1/projects/{project_id} \
  -H "X-API-Key: dev-api-key"
```

## Documents

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

## Search

```bash
# Search in project (requires indexed documents)
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/search \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "limit": 10}'

# Search with threshold (default: 0.3)
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/search \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "threshold": 0.5}'

# Search filtering by specific models
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/search \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "models": ["nomic-embed-text", "bge-m3"]}'
```
