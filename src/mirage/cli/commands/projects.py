import httpx
import typer

from mirage.cli.config import get_api_key, get_api_url

app = typer.Typer(help="Project management commands")


def get_headers() -> dict:
    return {"X-API-Key": get_api_key()}


@app.command("list")
def list_projects():
    """List all projects."""
    url = f"{get_api_url()}/projects"
    response = httpx.get(url, headers=get_headers())

    if response.status_code != 200:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)

    projects = response.json()
    if not projects:
        typer.echo("No projects found.")
        return

    typer.echo(f"{'ID':<40} {'Name':<30} {'Created':<20}")
    typer.echo("-" * 90)
    for p in projects:
        created = p["created_at"][:19].replace("T", " ")
        typer.echo(f"{p['id']:<40} {p['name']:<30} {created:<20}")


@app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    ollama_url: str | None = typer.Option(None, "--ollama-url", help="Ollama server URL"),
):
    """Create a new project."""
    payload: dict = {"name": name}
    if ollama_url:
        payload["ollama_url"] = ollama_url

    url = f"{get_api_url()}/projects"
    response = httpx.post(url, headers=get_headers(), json=payload)

    if response.status_code == 201:
        project = response.json()
        typer.echo(f"Project created: {project['id']}")
    elif response.status_code == 409:
        typer.echo("Error: Project with this name already exists", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)


@app.command("delete")
def delete_project(
    project_id: str = typer.Argument(..., help="Project ID"),
):
    """Delete a project and all its documents."""
    url = f"{get_api_url()}/projects/{project_id}"
    response = httpx.delete(url, headers=get_headers())

    if response.status_code == 204:
        typer.echo("Project deleted.")
    elif response.status_code == 404:
        typer.echo("Error: Project not found", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Error: {response.text}", err=True)
        raise typer.Exit(1)
