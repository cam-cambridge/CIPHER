"""Stage 4 — Visualization & Summary.

Produces three visual outputs from the embedded facts:
  * ``embedding_visualization.png``  — static t-SNE scatter (matplotlib)
  * ``embedding_visualization.html`` — interactive scatter (plotly)
  * ``category_distribution.png``    — horizontal bar chart of fact counts

Also contains :func:`print_summary` for the end-of-run report.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from rich.table import Table
from rich import box

from knowledge_generation.utils import console, elapsed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def visualize(
    processed_path: Path,
    output_dir: Path,
    topic: str,
) -> dict[str, Path]:
    """Create static + interactive t-SNE plots and a distribution chart.

    Returns a dict mapping output type (``"png"``, ``"html"``,
    ``"distribution"``) to its file path.
    """

    data = json.loads(processed_path.read_text())

    embeddings = np.array([e["embedding"] for e in data])
    category_map: dict[str, int] = {}
    categories_idx: list[int] = []
    for entry in data:
        cat = entry["category"]
        if cat not in category_map:
            category_map[cat] = len(category_map)
        categories_idx.append(category_map[cat])

    console.log(
        f"  Points: [cyan]{len(data)}[/cyan]  |  "
        f"Dimensions: [cyan]{embeddings.shape[1]}[/cyan]  |  "
        f"Categories: [cyan]{len(category_map)}[/cyan]"
    )
    console.log("  Running t-SNE (this may take a moment) …")

    perplexity = min(30, max(5, len(data) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    reduced = tsne.fit_transform(embeddings)
    outputs: dict[str, Path] = {}

    n_cats = len(category_map)
    cmap = cm.get_cmap("tab20", max(n_cats, 20))

    _plot_static(reduced, categories_idx, category_map, cmap, topic, output_dir, outputs)
    _plot_interactive(reduced, categories_idx, category_map, data, topic, output_dir, outputs)
    _plot_distribution(data, category_map, cmap, topic, output_dir, outputs)

    return outputs


# ── Matplotlib static scatter ────────────────────────


def _plot_static(reduced, categories_idx, category_map, cmap, topic, output_dir, outputs):
    colors = [cmap(categories_idx[i] % 20) for i in range(len(categories_idx))]

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=colors, alpha=0.75, edgecolors="white", linewidth=0.4, s=28,
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color=cmap(i % 20),
            label=label, markersize=7, linestyle="",
            markeredgecolor="white", markeredgewidth=0.4,
        )
        for label, i in category_map.items()
    ]
    legend = ax.legend(
        handles=handles, title="Categories",
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, title_fontsize=9, frameon=True,
        fancybox=True, shadow=True,
    )
    legend.get_frame().set_alpha(0.9)

    ax.set_title(f"Embedding Space — {topic}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15, linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)

    png_path = output_dir / "embedding_visualization.png"
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs["png"] = png_path
    console.log(f"  Static plot  → [green]{png_path}[/green]")


# ── Plotly interactive scatter ───────────────────────


def _plot_interactive(reduced, categories_idx, category_map, data, topic, output_dir, outputs):
    try:
        import plotly.graph_objects as go
    except ImportError:
        console.log("[yellow]  Plotly not installed — skipping interactive plot[/yellow]")
        return

    hover_texts = [
        f"<b>{e['category']}</b><br>"
        f"{e['original_fact'][:150]}{'…' if len(e['original_fact']) > 150 else ''}"
        for e in data
    ]

    fig = go.Figure()
    for label, idx in category_map.items():
        mask = [i for i, c in enumerate(categories_idx) if c == idx]
        fig.add_trace(go.Scatter(
            x=reduced[mask, 0],
            y=reduced[mask, 1],
            mode="markers",
            name=label,
            text=[hover_texts[i] for i in mask],
            hoverinfo="text",
            marker=dict(size=5, opacity=0.75, line=dict(width=0.3, color="white")),
        ))

    fig.update_layout(
        title=dict(text=f"Embedding Space — {topic}", font=dict(size=20)),
        template="plotly_white",
        width=1300, height=850,
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend=dict(font=dict(size=9)),
        hoverlabel=dict(font_size=11),
    )

    html_path = output_dir / "embedding_visualization.html"
    fig.write_html(str(html_path))
    outputs["html"] = html_path
    console.log(f"  Interactive  → [green]{html_path}[/green]")


# ── Category distribution bar chart ──────────────────


def _plot_distribution(data, category_map, cmap, topic, output_dir, outputs):
    cat_counts: dict[str, int] = defaultdict(int)
    for e in data:
        cat_counts[e["category"]] += 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_cats)

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.35)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bar_colors = [cmap(category_map[l] % 20) for l in labels]
    bars = ax.barh(range(len(labels)), counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Facts", fontsize=10)
    ax.set_title(f"Fact Distribution by Category — {topic}", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle="--")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=7, color="#555",
        )

    dist_path = output_dir / "category_distribution.png"
    fig.savefig(str(dist_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs["distribution"] = dist_path
    console.log(f"  Distribution → [green]{dist_path}[/green]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Summary Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_summary(
    topic: str,
    output_dir: Path,
    hierarchy_lines: int,
    facts_data: list[dict],
    total_elapsed: float,
):
    """Print a polished summary table of the pipeline run."""
    categorized: dict[str, int] = defaultdict(int)
    for entry in facts_data:
        categorized[entry["category"]] += 1

    # ── Overview table ──
    table = Table(
        title=f"Pipeline Summary — {topic}",
        box=box.ROUNDED,
        title_style="bold white on blue",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Topic", topic)
    table.add_row("Hierarchy nodes", str(hierarchy_lines))
    table.add_row("Top-level categories", str(len(categorized)))
    table.add_row("Total facts embedded", str(len(facts_data)))
    table.add_row(
        "Embedding dimensions",
        str(len(facts_data[0]["embedding"])) if facts_data else "—",
    )
    table.add_row("Total runtime", elapsed(total_elapsed))
    table.add_row("Output directory", str(output_dir))

    console.print()
    console.print(table)

    # ── Per-category breakdown ──
    cat_table = Table(
        title="Facts per Category",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
    )
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Facts", justify="right", style="green")

    for cat, count in sorted(categorized.items(), key=lambda x: x[1], reverse=True):
        cat_table.add_row(cat, str(count))

    console.print(cat_table)

    # ── Output files ──
    files_table = Table(title="Output Files", box=box.SIMPLE_HEAVY, header_style="bold")
    files_table.add_column("File", style="green")
    files_table.add_column("Size", justify="right")

    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            files_table.add_row(f.name, size_str)

    console.print(files_table)
