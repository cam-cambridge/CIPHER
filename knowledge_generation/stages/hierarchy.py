"""Stage 1 — Hierarchy Generation.

Uses an LLM to produce a multi-level knowledge taxonomy for a given
manufacturing topic.  Generation is split into two passes for reliability:
  1. Top-level categories are identified first.
  2. Each category is expanded into a 3–4 level deep subtree.
"""

import time

import openai

from knowledge_generation.utils import api_call_with_retry, console, make_progress_bar


def generate_hierarchy(topic: str, model: str = "gpt-4") -> str:
    """Return the full hierarchy text (one node-path per line).

    Parameters
    ----------
    topic : str
        Manufacturing method, e.g. "Additive Manufacturing".
    model : str
        OpenAI chat model to use.
    """

    # ── Pass 1: top-level categories ──
    console.log("[bold]Step 1/2[/bold]  Generating top-level categories …")

    cat_prompt = (
        f'You are an expert in manufacturing engineering. '
        f'Generate top-level knowledge categories and their immediate '
        f'subcategories for the manufacturing method: "{topic}".\n\n'
        f'Use this exact line format (one path per line):\n'
        f'Category\nCategory > Subcategory\n\n'
        f'Cover areas such as: general overview, applications, history, '
        f'hardware/equipment, process parameters, materials, physics, '
        f'chemistry, geometry/design, common issues & defects, advanced '
        f'topics, control systems, and post-processing.\n\n'
        f'Output ONLY the hierarchy lines — no explanations, no numbering.'
    )

    resp = api_call_with_retry(lambda: openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are an expert manufacturing engineer. "
                "Output only the requested format with no extra text."
            )},
            {"role": "user", "content": cat_prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    ))

    top_text = resp["choices"][0]["message"]["content"].strip()

    categories = [
        line.strip()
        for line in top_text.splitlines()
        if line.strip() and ">" not in line
    ]

    console.log(f"  Found [cyan]{len(categories)}[/cyan] top-level categories")

    # ── Pass 2: expand each category into a subtree ──
    console.log("[bold]Step 2/2[/bold]  Expanding each category …")

    all_blocks: list[str] = []

    with make_progress_bar() as progress:
        task = progress.add_task("Expanding categories", total=len(categories))

        for cat in categories:
            expand_prompt = (
                f'You are an expert in "{topic}" (a manufacturing method). '
                f'Generate a detailed knowledge hierarchy for the category '
                f'"{cat}" within "{topic}".\n\n'
                f'Rules:\n'
                f'- Use 3–4 levels of depth.\n'
                f'- Use the ">" separator:  {cat} > Sub > Detail\n'
                f'- Include the root line "{cat}" as the first line.\n'
                f'- Generate 25–40 lines covering this category '
                f'comprehensively.\n'
                f'- Output ONLY hierarchy lines — no explanations or '
                f'numbering.'
            )

            resp = api_call_with_retry(lambda: openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        f"You are an expert in {topic}. "
                        f"Output only the hierarchy format."
                    )},
                    {"role": "user", "content": expand_prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            ))

            block = resp["choices"][0]["message"]["content"].strip()
            all_blocks.append(block)
            progress.update(task, advance=1, description=f"[cyan]{cat}[/cyan]")
            time.sleep(0.3)

    return "\n".join(all_blocks)
