#!/usr/bin/env python3
"""
Demo AST retriever with semantic query expansion.

Enhances retrieval recall by expanding query roots with semantic neighbors.

Usage:
    python scripts/demo_ast_retriever_enhanced.py                    # Example queries
    python scripts/demo_ast_retriever_enhanced.py -i                  # Interactive
    python scripts/demo_ast_retriever.py "Kio estas Esperanto?"      # Single query
    python scripts/demo_ast_retriever.py --no-expansion "Kio..."     # Baseline
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.kuzu_inverted_index import FallbackMode
from klareco.rag.query_expansion import SemanticQueryExpander, expand_ast_roots
from klareco.parser import parse


def run_query_with_expansion(retriever, query: str, expander=None, top_k: int = 5):
    """Run query with optional semantic expansion."""
    from klareco.parser import parse

    print(f"\nQuery: {query}")
    print("-" * 60)

    # Parse query
    ast = parse(query)

    # Expand if expander provided
    if expander:
        print("\n[Semantic Expansion Enabled]")
        from klareco.rag.query_expansion import extract_all_roots_from_ast, expand_ast_roots

        # Show original roots
        original_roots = extract_all_roots_from_ast(ast, include_expansions=False)
        print(f"Original roots: {', '.join(sorted(original_roots))}")

        # Expand
        expanded_ast = expand_ast_roots(ast, expander)
        expanded_roots = extract_all_roots_from_ast(expanded_ast, include_expansions=True)

        print(f"Expanded roots: {', '.join(sorted(expanded_roots - set(extract_all_roots_from_ast(ast))))}")
        print()

    # Continue with normal retrieval using expanded AST...
