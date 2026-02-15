import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url


def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project ID"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Minimum similarity score (0.0-1.0)"),
    models: list[str] = typer.Option(
        None, "--model", "-m", help="Filter by embedding models (can be specified multiple times)"
    ),
):
    """Search documents in a project."""
    url = f"{get_api_url()}/projects/{project}/search"
    headers = {"X-API-Key": get_api_key()}

    payload = {"query": query, "limit": limit, "threshold": threshold}
    if models:
        payload["models"] = models

    response = httpx.post(
        url,
        headers=headers,
        json=payload,
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
        typer.echo(f"\n{result['content']}")
