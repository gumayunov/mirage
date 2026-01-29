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

    typer.echo(f"{'ID':<40} {'Filename':<30} {'Status':<10} {'Progress':<15}")
    typer.echo("-" * 95)
    for doc in docs:
        total = doc.get("chunks_total")
        processed = doc.get("chunks_processed", 0)
        if total:
            pct = round(processed / total * 100) if total > 0 else 0
            progress = f"{processed}/{total} ({pct}%)"
        else:
            progress = ""
        typer.echo(f"{doc['id']:<40} {doc['filename']:<30} {doc['status']:<10} {progress:<15}")


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
    if doc.get("chunks_total"):
        total = doc["chunks_total"]
        processed = doc.get("chunks_processed", 0)
        pct = round(processed / total * 100) if total > 0 else 0
        typer.echo(f"Chunks:   {processed}/{total} ({pct}%)")
        if doc.get("chunks_by_status"):
            breakdown = ", ".join(
                f"{s}: {c}" for s, c in doc["chunks_by_status"].items()
            )
            typer.echo(f"          {breakdown}")
    if doc.get("error_message"):
        typer.echo(f"Error:    {doc['error_message']}")
