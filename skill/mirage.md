---
name: mirage
description: Search project knowledge base (books, documentation)
---

# miRAGe - Project Knowledge Base

Use this skill when you need to find information in books
and documentation attached to the current project.

## Configuration

Read project_id from `.mirage.yaml` file in project root:

```yaml
project_id: "my-project"
```

API key in environment variable `MIRAGE_API_KEY`.
API URL in environment variable `MIRAGE_API_URL` (default: http://localhost:8000/api/v1).

## Commands

```bash
# List documents
mirage documents list --project <project_id>

# Add document
mirage documents add --project <project_id> /path/to/file.pdf

# Remove document
mirage documents remove --project <project_id> <document_id>

# Document status
mirage documents status --project <project_id> <document_id>

# Search (supports multiple embedding models)
mirage search --project <project_id> "query" --limit 10 --threshold 0.3 --model nomic-embed-text --model bge-m3
```

## Search

By default, use multi-model search (without `--model` flag) for better results. Only specify `--model` when the user explicitly requests a particular embedding model.

Multi-model search queries all models enabled for the project in parallel and deduplicates results, providing more comprehensive coverage.

## Multi-hop Search

For complex questions:
1. Break into 2-4 sub-questions
2. Search for each
3. If information is insufficient - refine query
4. Synthesize answer with source citations

## Usage Example

```bash
# Read project_id
cat .mirage.yaml

# Search for information
mirage search --project my-project "dependency injection best practices" --limit 5

# Search with specific embedding model
mirage search --project my-project "constructor injection vs setter injection" --model bge-m3
```
