from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from myrag import pipeline
from myrag.config import (
    ALBERT_BASE_URL,
    default_home,
    docs_dir,
    index_dir,
    load_config,
    project_dir,
)
from myrag.store import index_stats, open_collection

app = typer.Typer(
    help="my-rag-toolkit — local-first RAG with sovereign generation (Albert by default)",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _home() -> Path:
    return default_home()


@app.command()
def init(
    name: str,
    description: str = typer.Option("", "--description", "-d", help="short description of the corpus"),
):
    """Create a new RAG project."""
    _load_env()
    try:
        project = pipeline.create_project(_home(), name, description)
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"created project: {project}", fg=typer.colors.GREEN)
    typer.echo(f'next: rag add {name} <files-or-dir>  →  rag build {name}  →  rag ask {name} "question"')


@app.command()
def add(name: str, paths: list[str] = typer.Argument(..., help="files or directories to copy in")):
    """Copy documents (pdf, docx, md, txt) into the project."""
    _load_env()
    try:
        copied = pipeline.add_documents(_home(), name, paths)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(
        f"copied {len(copied)} file(s) into {docs_dir(project_dir(_home(), name))}",
        fg=typer.colors.GREEN,
    )
    for p in copied:
        typer.echo(f"  + {p.name}")


@app.command()
def build(
    name: str,
    no_reset: bool = typer.Option(
        False, "--no-reset", help="add to the existing index instead of rebuilding it"
    ),
):
    """Parse, chunk, embed (locally) and index the project's documents."""
    _load_env()
    try:
        report = pipeline.build_project(_home(), name, reset=not no_reset, log=typer.echo)
    except (FileNotFoundError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    table = Table(title=f"build report — {name}")
    table.add_column("file")
    table.add_column("status")
    table.add_column("pages", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("note")
    for f in report.files:
        table.add_row(f.file, f.status, str(f.pages) if f.pages else "-", str(f.chunks), f.note)
    console.print(table)
    for f in report.skipped_scanned:
        typer.secho(
            f"warning: {f.file} looks like a scan and was NOT indexed "
            f"(OCR is a planned phase-2 feature)",
            fg=typer.colors.YELLOW,
        )
    typer.secho(
        f"done: {report.total_chunks} chunks from {report.ok_files} file(s) "
        f"in {report.elapsed_s:.1f}s — embeddings computed locally",
        fg=typer.colors.GREEN,
    )


@app.command()
def ask(
    name: str,
    question: list[str] = typer.Argument(..., help="question (quote it)"),
    top_k: int = typer.Option(None, "--top-k", help="number of excerpts to retrieve"),
    show_context: bool = typer.Option(False, "--show-context", help="print the retrieved excerpts"),
    profile: str = typer.Option(
        "", "--profile", help="generation profile from the project config (alternative endpoint/model)"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="retrieval only — nothing is sent to any API"
    ),
):
    """Retrieve + answer a question (only top-k excerpts + question leave the machine)."""
    _load_env()
    q = " ".join(question)
    try:
        result = pipeline.query_project(
            _home(), name, q, use_llm=not no_llm, top_k=top_k, profile=profile or None
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    if not result.contexts:
        typer.secho("no matching excerpts found", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    _print_contexts(result.contexts, show_context or no_llm)
    if not result.generated:
        typer.secho("retrieval-only mode: nothing was sent to any API", fg=typer.colors.CYAN)
        return
    console.print(Markdown(result.answer))
    typer.echo("")
    typer.secho(
        f"[privacy] sent {len(result.contexts)} excerpt(s) (~{result.context_tokens} tokens) "
        f"+ question to {result.model} @ {result.base_url}",
        fg=typer.colors.CYAN,
    )


@app.command()
def search(
    name: str,
    query: list[str] = typer.Argument(..., help="search query (quote it)"),
    top_k: int = typer.Option(None, "--top-k", help="number of excerpts to retrieve"),
):
    """Retrieve top-k chunks without any LLM call (fully local)."""
    _load_env()
    q = " ".join(query)
    try:
        result = pipeline.query_project(_home(), name, q, use_llm=False, top_k=top_k)
    except (FileNotFoundError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    if not result.contexts:
        typer.secho("no matching excerpts found", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    _print_contexts(result.contexts, True)
    typer.secho("retrieval-only mode: nothing was sent to any API", fg=typer.colors.CYAN)


def _print_contexts(contexts, show_text: bool) -> None:
    table = Table(title="retrieved excerpts")
    table.add_column("n", justify="right")
    table.add_column("score", justify="right")
    table.add_column("doc")
    table.add_column("loc")
    if show_text:
        table.add_column("text")
    for c in contexts:
        loc = f"page {c.page}" if c.page and c.page > 0 else (c.section or "-")
        row = [f"[{c.index}]", f"{c.score:.3f}", c.doc_id, loc]
        if show_text:
            preview = c.text[:200].replace("\n", " ")
            row.append(preview + ("…" if len(c.text) > 200 else ""))
        table.add_row(*row)
    console.print(table)
    typer.echo("")


@app.command("list")
def list_projects():
    """List projects with their index sizes."""
    _load_env()
    home = _home()
    names = pipeline.discover_projects(home)
    if not names:
        typer.echo(f"no projects yet in {home} — create one with `rag init <name>`")
        return
    table = Table(title=f"projects in {home}")
    table.add_column("name")
    table.add_column("description")
    table.add_column("docs", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("generation model")
    for n in names:
        project = project_dir(home, n)
        cfg = load_config(project)
        docs = chunks = "-"
        try:
            stats = index_stats(open_collection(n, index_dir(project)))
            docs = str(len(stats["documents"]))
            chunks = str(stats["chunks"])
        except Exception:
            pass
        table.add_row(n, cfg.description or "-", docs, chunks, cfg.generate.model)
    console.print(table)


@app.command()
def status(name: str):
    """Show a project's configuration and index state."""
    _load_env()
    project = project_dir(_home(), name)
    if not project.exists():
        typer.secho(f"unknown project: {name}", fg=typer.colors.RED)
        raise typer.Exit(1)
    cfg = load_config(project)
    typer.echo(f"project:   {project}")
    typer.echo(f"description: {cfg.description or '-'}")
    typer.echo(f"embed:     {cfg.embed.model} (local, device={cfg.embed.device})")
    typer.echo(
        f"chunk:     max={cfg.chunk.max_tokens} tok, overlap={cfg.chunk.overlap_ratio:.0%}"
    )
    typer.echo(
        f"retrieve:  top_k={cfg.retrieve.top_k}, threshold={cfg.retrieve.score_threshold}, "
        f"rerank={'on (' + cfg.retrieve.rerank.model + ')' if cfg.retrieve.rerank.enabled else 'off'}"
    )
    typer.echo(f"generate:  {cfg.generate.model} @ {cfg.generate.base_url}")
    typer.echo(f"profiles:  {', '.join(sorted(cfg.profiles)) or '-'}")
    try:
        stats = index_stats(open_collection(name, index_dir(project)))
        typer.echo(f"index:     {stats['chunks']} chunks from {len(stats['documents'])} doc(s)")
        for doc_id, n in sorted(stats["doc_chunks"].items()):
            typer.echo(f"  - {doc_id}: {n} chunks")
    except Exception as exc:
        typer.echo(f"index:     not readable ({exc})")
    usage_path = project / "usage.jsonl"
    if usage_path.exists():
        lines = usage_path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            last = lines[-1]
            typer.echo(f"last call: {last[:160]}")


@app.command()
def remove(
    name: str,
    force: bool = typer.Option(False, "--force", help="do not ask for confirmation"),
):
    """Delete a project (documents + index)."""
    _load_env()
    if not force:
        typer.confirm(f"delete project '{name}' and its index?", abort=True)
    try:
        pipeline.remove_project(_home(), name)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"deleted project: {name}", fg=typer.colors.GREEN)


@app.command()
def ui(
    project: str = typer.Option("", "--project", help="project to open by default"),
):
    """Launch the local Streamlit web UI."""
    _load_env()
    ui_file = Path(__file__).resolve().parents[2] / "app" / "ui.py"
    if not ui_file.exists():
        ui_file = Path(os.environ.get("MYRAG_UI", "app/ui.py")).resolve()
    env = os.environ.copy()
    if project:
        env["MYRAG_UI_PROJECT"] = project
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ui_file)], env=env))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
