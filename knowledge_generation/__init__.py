"""
Knowledge Generation Pipeline
==============================
End-to-end framework that transforms a manufacturing topic into
a structured knowledge base with embeddings and visualizations.

Stages:
  1. Hierarchy Generation   — LLM creates a comprehensive topic taxonomy
  2. Fact Generation         — LLM generates facts for every hierarchy node
  3. Embedding Generation    — Encode all facts into a vector space
  4. Visualization           — t-SNE scatter plots (static + interactive)

Usage:
    python -m knowledge_generation "Additive Manufacturing"
    python -m knowledge_generation "CNC Machining" --output-dir ./output/cnc
    python -m knowledge_generation "Injection Molding" --embedding-provider local
"""

__version__ = "1.0.0"
