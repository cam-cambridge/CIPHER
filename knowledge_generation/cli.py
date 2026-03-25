"""CLI entry point — orchestrates all pipeline stages."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import typer
import openai
from dotenv import load_dotenv
from rich.panel import Panel

load_dotenv()

from knowledge_generation.utils import console, stage_header, elapsed, display_tree
from knowledge_generation.stages import (
    generate_hierarchy,
    generate_facts,
    generate_embeddings_openai,
    generate_embeddings_local,
)
from knowledge_generation.stages.visualization import visualize, print_summary

app = typer.Typer(
    name="knowledge-generation",
    help=(
        "Manufacturing Knowledge Pipeline — "
        "Topic → Hierarchy → Facts → Embeddings → Visualization"
    ),
    rich_markup_mode="rich",
    add_completion=False,
)


@app.command()
def run(
    topic: str = typer.Argument(
        ...,
        help='Manufacturing method, e.g. "Additive Manufacturing", "CNC Machining"',
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory (default: ./output/<topic_slug>)",
    ),
    model: str = typer.Option(
        "gpt-4",
        "--model", "-m",
        help="OpenAI chat model for hierarchy & fact generation",
    ),
    embedding_provider: str = typer.Option(
        "openai",
        "--embedding-provider", "-e",
        help="Embedding backend: 'openai' or 'local' (sentence-transformers)",
    ),
    embedding_model: str = typer.Option(
        "sentence-t5-large",
        "--embedding-model",
        help="Sentence-transformers model (only with --embedding-provider local)",
    ),
    skip_to: int = typer.Option(
        1,
        "--skip-to", "-s",
        help="Skip to stage N (1-4).  Earlier artifacts must already exist.",
    ),
):
    """
    Run the full Manufacturing Knowledge Pipeline.

    Generates a structured knowledge base from a manufacturing topic,
    producing a hierarchy, facts, embeddings, and visualizations.
    """

    t_start = time.time()

    # ── API key ──
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY environment variable "
            "is not set.\n  export OPENAI_API_KEY='sk-…'",
        )
        raise typer.Exit(1)
    openai.api_key = api_key

    # ── Output directory ──
    pkg_root = Path(__file__).resolve().parent
    if output_dir is None:
        slug = topic.lower().replace(" ", "_").replace("-", "_")
        out = pkg_root / "output" / slug
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hierarchy_path = out / "hierarchies.txt"
    facts_path = out / "hierarchy_with_facts.json"
    embeddings_path = out / "processed_facts.json"

    # ── Banner ──
    console.print(Panel(
        f"[bold white]Manufacturing Knowledge Pipeline[/bold white]\n\n"
        f"  Topic              [cyan]{topic}[/cyan]\n"
        f"  Chat model         [cyan]{model}[/cyan]\n"
        f"  Embedding provider [cyan]{embedding_provider}[/cyan]\n"
        f"  Output directory   [cyan]{out}[/cyan]\n"
        f"  Starting at stage  [cyan]{skip_to}[/cyan]",
        title="[bold blue]Pipeline Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 3),
    ))

    # ────────────────────────────────────────────────
    #  Stage 1 — Hierarchy
    # ────────────────────────────────────────────────
    if skip_to <= 1:
        stage_header(1)
        t1 = time.time()

        hierarchy_text = generate_hierarchy(topic, model=model)
        hierarchy_path.write_text(hierarchy_text)

        n_lines = len([l for l in hierarchy_text.splitlines() if l.strip()])
        console.log(
            f"  Saved [green]{n_lines}[/green] hierarchy nodes → "
            f"[green]{hierarchy_path}[/green]"
        )
        console.print()
        display_tree(hierarchy_text, topic)
        console.log(f"  [dim]Stage 1 completed in {elapsed(t1)}[/dim]")
    else:
        if not hierarchy_path.exists():
            console.print(f"[red]Cannot skip — {hierarchy_path} does not exist[/red]")
            raise typer.Exit(1)
        console.log(f"[dim]Skipping Stage 1 — using existing {hierarchy_path.name}[/dim]")

    # ────────────────────────────────────────────────
    #  Stage 2 — Facts
    # ────────────────────────────────────────────────
    if skip_to <= 2:
        stage_header(2)
        t2 = time.time()

        generate_facts(topic, hierarchy_path, facts_path, model=model)

        console.log(f"  Saved → [green]{facts_path}[/green]")
        console.log(f"  [dim]Stage 2 completed in {elapsed(t2)}[/dim]")
    else:
        if not facts_path.exists():
            console.print(f"[red]Cannot skip — {facts_path} does not exist[/red]")
            raise typer.Exit(1)
        console.log(f"[dim]Skipping Stage 2 — using existing {facts_path.name}[/dim]")

    # ────────────────────────────────────────────────
    #  Stage 3 — Embeddings
    # ────────────────────────────────────────────────
    if skip_to <= 3:
        stage_header(3)
        t3 = time.time()

        if embedding_provider == "openai":
            processed = generate_embeddings_openai(facts_path, embeddings_path)
        elif embedding_provider == "local":
            processed = generate_embeddings_local(
                facts_path, embeddings_path, model_name=embedding_model,
            )
        else:
            console.print(
                f"[red]Unknown embedding provider: {embedding_provider}[/red]"
            )
            raise typer.Exit(1)

        console.log(
            f"  Saved [green]{len(processed)}[/green] embedded facts → "
            f"[green]{embeddings_path}[/green]"
        )
        console.log(f"  [dim]Stage 3 completed in {elapsed(t3)}[/dim]")
    else:
        if not embeddings_path.exists():
            console.print(
                f"[red]Cannot skip — {embeddings_path} does not exist[/red]"
            )
            raise typer.Exit(1)
        console.log(
            f"[dim]Skipping Stage 3 — using existing {embeddings_path.name}[/dim]"
        )

    # ────────────────────────────────────────────────
    #  Stage 4 — Visualization
    # ────────────────────────────────────────────────
    stage_header(4)
    t4 = time.time()

    if not embeddings_path.exists():
        console.print(
            f"[red]Cannot visualize — {embeddings_path} does not exist[/red]"
        )
        raise typer.Exit(1)

    visualize(embeddings_path, out, topic)
    console.log(f"  [dim]Stage 4 completed in {elapsed(t4)}[/dim]")

    # ────────────────────────────────────────────────
    #  Summary
    # ────────────────────────────────────────────────
    processed = json.loads(embeddings_path.read_text())
    hierarchy_lines = len([
        l for l in hierarchy_path.read_text().splitlines() if l.strip()
    ])

    console.print()
    console.rule("[bold green]Pipeline Complete[/bold green]", style="green")
    print_summary(topic, out, hierarchy_lines, processed, t_start)

    console.print(
        f"\n[bold green]✓[/bold green] All outputs saved to [bold]{out}/[/bold]\n"
    )
