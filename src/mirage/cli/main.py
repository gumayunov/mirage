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
