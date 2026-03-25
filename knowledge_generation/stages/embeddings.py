"""Stage 3 — Embedding Generation.

Extracts all facts from the hierarchy JSON and encodes each into a vector
using either OpenAI (``text-embedding-ada-002``) or a local
sentence-transformers model.
"""

import json
from pathlib import Path

import openai

from knowledge_generation.utils import api_call_with_retry, console, make_progress_bar


# ── Fact extraction ──────────────────────────────────


def extract_categorized_facts(data, category=None, categorized_facts=None):
    """Recursively collect facts grouped by their top-level category."""
    if categorized_facts is None:
        categorized_facts = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                extract_categorized_facts(
                    value,
                    key if category is None else category,
                    categorized_facts,
                )
            elif "Fact" in key and isinstance(value, str):
                categorized_facts.setdefault(category, []).append(value)
    elif isinstance(data, list):
        for item in data:
            extract_categorized_facts(item, category, categorized_facts)

    return categorized_facts


# ── OpenAI embeddings ────────────────────────────────


def generate_embeddings_openai(
    facts_path: Path,
    output_path: Path,
) -> list[dict]:
    """Embed every fact with OpenAI ``text-embedding-ada-002``."""

    data = json.loads(facts_path.read_text())
    categorized = extract_categorized_facts(data)
    total = sum(len(v) for v in categorized.values())

    console.log(
        f"  Categories: [cyan]{len(categorized)}[/cyan]  |  "
        f"Total facts: [cyan]{total}[/cyan]"
    )

    processed: list[dict] = []
    fact_id = 1

    with make_progress_bar() as progress:
        task = progress.add_task("Embedding facts", total=total)

        for category, facts in categorized.items():
            for fact in facts:
                try:
                    resp = api_call_with_retry(lambda: openai.Embedding.create(
                        input=fact,
                        model="text-embedding-ada-002",
                    ))
                    embedding = resp["data"][0]["embedding"]
                    processed.append({
                        "id": fact_id,
                        "category": category,
                        "original_fact": fact,
                        "embedding": embedding,
                    })
                    fact_id += 1
                except Exception as e:
                    console.log(f"[red]Embedding error: {e}[/red]")
                progress.update(task, advance=1, description=f"[cyan]{category}[/cyan]")

    output_path.write_text(json.dumps(processed, indent=4))
    return processed


# ── Local (sentence-transformers) embeddings ─────────


def generate_embeddings_local(
    facts_path: Path,
    output_path: Path,
    model_name: str = "sentence-t5-large",
) -> list[dict]:
    """Embed every fact with a local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    console.log(f"  Loading model [cyan]{model_name}[/cyan] …")
    model = SentenceTransformer(model_name)

    data = json.loads(facts_path.read_text())
    categorized = extract_categorized_facts(data)
    total = sum(len(v) for v in categorized.values())

    console.log(
        f"  Categories: [cyan]{len(categorized)}[/cyan]  |  "
        f"Total facts: [cyan]{total}[/cyan]"
    )

    processed: list[dict] = []
    fact_id = 1

    with make_progress_bar() as progress:
        task = progress.add_task("Embedding facts", total=total)

        for category, facts in categorized.items():
            for fact in facts:
                embedding = model.encode(fact).tolist()
                processed.append({
                    "id": fact_id,
                    "category": category,
                    "original_fact": fact,
                    "embedding": embedding,
                })
                fact_id += 1
                progress.update(task, advance=1, description=f"[cyan]{category}[/cyan]")

    output_path.write_text(json.dumps(processed, indent=4))
    return processed
