"""Pipeline stages — hierarchy, facts, embeddings, visualization."""

from knowledge_generation.stages.hierarchy import generate_hierarchy
from knowledge_generation.stages.facts import generate_facts
from knowledge_generation.stages.embeddings import (
    generate_embeddings_openai,
    generate_embeddings_local,
    extract_categorized_facts,
)
from knowledge_generation.stages.visualization import visualize, print_summary
