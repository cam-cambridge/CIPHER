"""Stage 2 — Fact Generation.

Walks every node in a hierarchy file and generates 10 structured facts
per node using an LLM.  Progress is checkpointed after every node so
interrupted runs resume where they left off.
"""

import json
import time
from pathlib import Path

import openai

from knowledge_generation.utils import api_call_with_retry, console, make_progress_bar


def generate_facts(
    topic: str,
    hierarchy_path: Path,
    facts_path: Path,
    model: str = "gpt-4",
) -> dict:
    """Return the nested hierarchy dict populated with facts.

    Parameters
    ----------
    topic : str
        Manufacturing method name (prepended to every prompt for context).
    hierarchy_path : Path
        Path to ``hierarchies.txt``.
    facts_path : Path
        Path to write / resume ``hierarchy_with_facts.json``.
    model : str
        OpenAI chat model to use.
    """

    lines = [l for l in hierarchy_path.read_text().splitlines() if l.strip()]

    if facts_path.exists():
        hierarchy = json.loads(facts_path.read_text())
        console.log(f"[dim]Loaded existing progress from {facts_path.name}[/dim]")
    else:
        hierarchy = {}

    # ── Count work remaining ──
    already_done = 0
    to_do = 0
    for line in lines:
        parts = [p.strip() for p in line.split(">")]
        d = hierarchy
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        if parts[-1] in d and "Facts" in d.get(parts[-1], {}):
            already_done += 1
        else:
            to_do += 1

    console.log(
        f"  Hierarchy nodes: [cyan]{len(lines)}[/cyan]  |  "
        f"Already done: [green]{already_done}[/green]  |  "
        f"Remaining: [yellow]{to_do}[/yellow]"
    )

    # ── Generate facts for each node ──
    with make_progress_bar() as progress:
        task = progress.add_task("Generating facts", total=to_do)

        for line in lines:
            parts = [p.strip() for p in line.split(">")]

            d = hierarchy
            for p in parts[:-1]:
                d = d.setdefault(p, {})

            if parts[-1] in d and "Facts" in d.get(parts[-1], {}):
                continue

            full_topic = " > ".join(parts)
            progress.update(task, description=f"[cyan]{parts[-1]}[/cyan]")

            try:
                facts = _call_for_facts(topic, full_topic, model)
                d[parts[-1]] = {"Facts": facts}
            except Exception as e:
                console.log(f"[red]Error for {full_topic}: {e}[/red]")
                d[parts[-1]] = {"Facts": {"Error": str(e)}}

            facts_path.write_text(json.dumps(hierarchy, indent=4))
            progress.update(task, advance=1)
            time.sleep(0.2)

    return hierarchy


def _call_for_facts(topic: str, full_path: str, model: str) -> dict:
    """Call the LLM to produce 10 facts for a single hierarchy node."""
    prompt = (
        f"Generate 10 unique, informative facts specifically about "
        f"'{topic} > {full_path}', ensuring each fact introduces new "
        f"information not covered by the others. The facts should provide "
        f"valuable details or insights, focusing only on this topic. "
        f'Output valid JSON: {{"Fact 1": "<content>", "Fact 2": "<content>", …}}.'
    )

    resp = api_call_with_retry(lambda: openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a manufacturing-domain expert. "
                "Provide structured, detailed, and accurate information."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    ))

    raw = resp["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)
