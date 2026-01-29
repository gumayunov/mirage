# miRAGe Phase 4: CLI + Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать CLI для управления системой, Claude Code skill, финальная интеграция.

**Prerequisite:** Phase 1-3 завершены. Существуют: API, Indexer, все shared компоненты.

**Deliverable:** Полностью интегрированное приложение с CLI и Claude Code skill. Все тесты проходят.

---

## Task 5.1: CLI Base and Config

**Files:**
- Create: `src/mirage/cli/main.py`
- Create: `src/mirage/cli/config.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_cli.py`

**Step 1: Write the failing test**

`tests/cli/test_cli.py`:
```python
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "miRAGe" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_cli.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/config.py`:
```python
import os


def get_api_url() -> str:
    return os.environ.get("MIRAGE_API_URL", "http://localhost:8000/api/v1")


def get_api_key() -> str:
    key = os.environ.get("MIRAGE_API_KEY", "")
    if not key:
        raise ValueError("MIRAGE_API_KEY environment variable not set")
    return key
```

`src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

Create `tests/cli/__init__.py`: empty file.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_cli.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI base structure"
```

---

## Task 5.2: CLI Documents Commands

**Files:**
- Create: `src/mirage/cli/commands/__init__.py`
- Create: `src/mirage/cli/commands/documents.py`
- Modify: `src/mirage/cli/main.py`
- Create: `tests/cli/test_documents_cmd.py`

**Step 1: Write the failing test**

`tests/cli/test_documents_cmd.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_documents_list_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "1", "filename": "test.pdf", "status": "ready"}
    ]

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(app, ["documents", "list", "--project", "test-project"])

    assert result.exit_code == 0
    assert "test.pdf" in result.stdout


def test_documents_status_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "doc-1",
        "filename": "test.pdf",
        "status": "ready",
        "file_type": "pdf",
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            app, ["documents", "status", "--project", "test-project", "doc-1"]
        )

    assert result.exit_code == 0
    assert "ready" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_documents_cmd.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/commands/__init__.py`: empty file.

`src/mirage/cli/commands/documents.py`:
```python
from pathlib import Path

import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url

app = typer.Typer(help="Document management commands")


def get_headers() -> dict:
    return {"X-API-Key": get_api_key()}


@app.command("list")
def list_documents(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
):
    """List all documents in a project."""
    url = f"{get_api_url()}/projects/{project}/documents"
    response = httpx.get(url, headers=get_headers())

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    docs = response.json()
    if not docs:
        typer.echo("No documents found.")
        return

    typer.echo(f"{'ID':<40} {'Filename':<30} {'Status':<10}")
    typer.echo("-" * 80)
    for doc in docs:
        typer.echo(f"{doc['id']:<40} {doc['filename']:<30} {doc['status']:<10}")


@app.command("add")
def add_document(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    file_path: Path = typer.Argument(..., help="Path to the document file"),
):
    """Upload a document to a project."""
    if not file_path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    url = f"{get_api_url()}/projects/{project}/documents"

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f)}
        response = httpx.post(url, headers=get_headers(), files=files, timeout=60.0)

    if response.status_code == 202:
        doc = response.json()
        typer.echo(f"Document uploaded: {doc['id']}")
        typer.echo("Indexing in progress...")
    elif response.status_code == 409:
        typer.echo("Error: Document with this name already exists", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)


@app.command("remove")
def remove_document(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    document_id: str = typer.Argument(..., help="Document ID"),
):
    """Remove a document from a project."""
    url = f"{get_api_url()}/projects/{project}/documents/{document_id}"
    response = httpx.delete(url, headers=get_headers())

    if response.status_code == 204:
        typer.echo("Document removed.")
    elif response.status_code == 404:
        typer.echo("Error: Document not found", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)


@app.command("status")
def document_status(
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    document_id: str = typer.Argument(..., help="Document ID"),
):
    """Get the status of a document."""
    url = f"{get_api_url()}/projects/{project}/documents/{document_id}"
    response = httpx.get(url, headers=get_headers())

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    doc = response.json()
    typer.echo(f"ID:       {doc['id']}")
    typer.echo(f"Filename: {doc['filename']}")
    typer.echo(f"Type:     {doc['file_type']}")
    typer.echo(f"Status:   {doc['status']}")
    if doc.get("error_message"):
        typer.echo(f"Error:    {doc['error_message']}")
```

Update `src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__
from mirage.cli.commands import documents

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)

app.add_typer(documents.app, name="documents")


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_documents_cmd.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI documents commands"
```

---

## Task 5.3: CLI Search Command

**Files:**
- Create: `src/mirage/cli/commands/search.py`
- Modify: `src/mirage/cli/main.py`
- Create: `tests/cli/test_search_cmd.py`

**Step 1: Write the failing test**

`tests/cli/test_search_cmd.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mirage.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MIRAGE_API_URL", "http://test:8000/api/v1")
    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")


def test_search_command(mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "chunk_id": "1",
                "content": "This is matching content about Python.",
                "score": 0.85,
                "structure": {"chapter": "Introduction"},
                "document": {"id": "doc-1", "filename": "book.pdf"},
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response):
        result = runner.invoke(
            app, ["search", "--project", "test-project", "Python programming"]
        )

    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "book.pdf" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/cli/test_search_cmd.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/cli/commands/search.py`:
```python
import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url


def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search documents in a project."""
    url = f"{get_api_url()}/projects/{project}/search"
    headers = {"X-API-Key": get_api_key()}

    response = httpx.post(
        url,
        headers=headers,
        json={"query": query, "limit": limit},
        timeout=30.0,
    )

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    data = response.json()
    results = data.get("results", [])

    if not results:
        typer.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        typer.echo(f"\n--- Result {i} (score: {result['score']:.2f}) ---")
        typer.echo(f"Source: {result['document']['filename']}")
        if result.get("structure"):
            structure = result["structure"]
            if "chapter" in structure:
                typer.echo(f"Chapter: {structure['chapter']}")
            if "page" in structure:
                typer.echo(f"Page: {structure['page']}")
        typer.echo(f"\n{result['content'][:500]}...")
```

Update `src/mirage/cli/main.py`:
```python
from typing import Optional

import typer

from mirage import __version__
from mirage.cli.commands import documents
from mirage.cli.commands.search import search as search_cmd

app = typer.Typer(
    name="mirage",
    help="miRAGe - Local RAG system for books and documentation",
)

app.add_typer(documents.app, name="documents")
app.command("search")(search_cmd)


def version_callback(value: bool):
    if value:
        print(f"miRAGe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/cli/test_search_cmd.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add CLI search command"
```

---

## Task 8.1: miRAGe Skill

**Files:**
- Create: `skill/mirage.md`

**Step 1: Create skill file**

`skill/mirage.md`:
```markdown
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

# Search
mirage search --project <project_id> "query" --limit 10
```

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

# If more context needed, make additional queries
mirage search --project my-project "constructor injection vs setter injection" --limit 5
```
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add Claude Code skill"
```

---

## Verification

После завершения всех задач:

```bash
uv run pytest tests/ -v
```

Все тесты должны проходить. CLI и skill готовы к использованию. Можно переходить к Phase 5 (Infrastructure).
