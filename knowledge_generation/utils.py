"""Shared utilities — console, retry logic, display helpers."""

import time

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.tree import Tree

console = Console()

STAGE_NAMES = {
    1: "Hierarchy Generation",
    2: "Fact Generation",
    3: "Embedding Generation",
    4: "Visualization",
}


def api_call_with_retry(func, max_retries=5, base_delay=2.0):
    """Execute an API call with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            console.log(
                f"[yellow]API error (attempt {attempt + 1}/{max_retries}): {e}  "
                f"— retrying in {delay:.0f}s …[/yellow]"
            )
            time.sleep(delay)


def make_progress_bar():
    """Create a consistently styled Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def stage_header(stage_num: int):
    """Print a styled stage header."""
    name = STAGE_NAMES[stage_num]
    console.print()
    console.rule(f"[bold cyan]Stage {stage_num} · {name}[/bold cyan]", style="cyan")
    console.print()


def elapsed(start: float) -> str:
    """Format elapsed time since *start* as a human-readable string."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}min"


def display_tree(hierarchy_text: str, topic: str, max_lines: int = 40):
    """Render a hierarchy preview as a Rich tree in the terminal."""
    tree = Tree(f"[bold green]{topic}[/bold green]")
    node_map: dict[str, Tree] = {topic: tree}

    for line in hierarchy_text.strip().splitlines()[:max_lines]:
        parts = [p.strip() for p in line.split(">")]
        for depth, part in enumerate(parts):
            full_key = " > ".join([topic] + parts[: depth + 1])
            if full_key not in node_map:
                parent_key = " > ".join([topic] + parts[:depth])
                parent = node_map.get(parent_key, tree)
                node_map[full_key] = parent.add(f"[white]{part}[/white]")

    total_lines = hierarchy_text.strip().count("\n") + 1
    if total_lines > max_lines:
        tree.add(f"[dim]… and {total_lines - max_lines} more nodes[/dim]")

    console.print(tree)
