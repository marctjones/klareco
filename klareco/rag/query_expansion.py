"""
Query expansion using semantic root embeddings.

Expands query roots with semantically similar roots to improve retrieval recall.
"""

import torch
from typing import List, Tuple, Dict, Set
from pathlib import Path


class SemanticQueryExpander:
    """Expand query roots with semantic neighbors using root embeddings."""

    def __init__(self, embeddings_path: Path, top_k: int = 5, threshold: float = 0.5):
        """
        Initialize query expander.

        Args:
            embeddings_path: Path to trained root embeddings model
            top_k: Number of similar roots to retrieve per query root
            threshold: Minimum similarity to include (0-1)
        """
        self.top_k = top_k
        self.threshold = threshold

        # Load embeddings
        checkpoint = torch.load(embeddings_path, map_location='cpu')
        self.embedding_dim = checkpoint['embedding_dim']
        self.root_to_idx = checkpoint['root_to_idx']
        self.idx_to_root = checkpoint['idx_to_root']

        # Create embedding layer
        self.embedding = torch.nn.Embedding(
            checkpoint['vocab_size'],
            self.embedding_dim
        )
        self.embedding.weight.data = checkpoint['model_state_dict']['embeddings.weight']

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        return (a @ b) / (torch.norm(a) * torch.norm(b))

    def find_similar_roots(self, root: str) -> List[Tuple[str, float]]:
        """
        Find similar roots to given root.

        Args:
            root: Root to expand

        Returns:
            List of (similar_root, similarity) tuples
        """
        if root not in self.root_to_idx:
            return []

        query_idx = self.root_to_idx[root]
        query_vec = self.embedding(torch.tensor(query_idx))

        # Compute similarities
        similarities = []
        for other_root, other_idx in self.root_to_idx.items():
            if other_root == root:
                continue

            other_vec = self.embedding(torch.tensor(other_idx))
            sim = self.cosine_similarity(query_vec, other_vec).item()

            if sim >= self.threshold:
                similarities.append((other_root, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:self.top_k]

    def expand_roots(self, roots: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Expand a list of roots with semantic neighbors.

        Args:
            roots: List of roots to expand

        Returns:
            Dict mapping original_root -> [(similar_root, similarity), ...]
        """
        expansion = {}
        for root in roots:
            similar = self.find_similar_roots(root)
            if similar:
                expansion[root] = similar
        return expansion

    def get_expanded_roots(self, roots: List[str]) -> Set[str]:
        """
        Get set of all roots (original + expansions).

        Args:
            roots: Original query roots

        Returns:
            Set of original roots + similar roots
        """
        expanded = set(roots)
        expansion = self.expand_roots(roots)

        for root, similar_list in expansion.items():
            for similar_root, _ in similar_list:
                expanded.add(similar_root)

        return expanded


def expand_ast_roots(ast: dict, expander: SemanticQueryExpander) -> dict:
    """
    Expand roots in an AST with semantic neighbors.

    Creates a modified AST where each root has additional "expansion" field
    containing similar roots. The retriever can then match on either original
    or expanded roots.

    Args:
        ast: Parsed AST
        expander: Query expander instance

    Returns:
        Modified AST with expansion annotations
    """
    import copy
    expanded_ast = copy.deepcopy(ast)

    def add_expansions(node):
        """Recursively add expansions to nodes with roots."""
        if node is None or not isinstance(node, dict):
            return

        # If node has a root, expand it
        if 'radiko' in node:
            root = node['radiko']
            similar = expander.find_similar_roots(root)
            if similar:
                node['semantic_expansion'] = similar

        # Recurse into children
        if node.get('tipo') == 'vortgrupo':
            add_expansions(node.get('kerno'))
            for priskr in node.get('priskriboj', []):
                add_expansions(priskr)

        if node.get('tipo') == 'frazo':
            add_expansions(node.get('subjekto'))
            add_expansions(node.get('verbo'))
            add_expansions(node.get('objekto'))
            for alian in node.get('aliaj', []):
                add_expansions(alian)

    add_expansions(expanded_ast)
    return expanded_ast


def extract_all_roots_from_ast(ast: dict, include_expansions: bool = False) -> Set[str]:
    """
    Extract all roots from AST, optionally including semantic expansions.

    Args:
        ast: Parsed AST
        include_expansions: Include roots from semantic_expansion fields

    Returns:
        Set of all roots found
    """
    roots = set()

    def extract(node):
        if node is None or not isinstance(node, dict):
            return

        # Original root
        if 'radiko' in node:
            roots.add(node['radiko'])

        # Expanded roots
        if include_expansions and 'semantic_expansion' in node:
            for expanded_root, _ in node['semantic_expansion']:
                roots.add(expanded_root)

        # Recurse
        if node.get('tipo') == 'vortgrupo':
            extract(node.get('kerno'))
            for priskr in node.get('priskriboj', []):
                extract(priskr)

        if node.get('tipo') == 'frazo':
            extract(node.get('subjekto'))
            extract(node.get('verbo'))
            extract(node.get('objekto'))
            for alian in node.get('aliaj', []):
                extract(alian)

    extract(ast)
    return roots
