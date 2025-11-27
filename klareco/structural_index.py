"""
Helpers for structural metadata used during indexing.

These functions rely on the canonicalizer to derive deterministic signatures
and grammar-aware tokens so retrieval can filter by structure before any neural
reranking step.
"""
from __future__ import annotations

from typing import Dict, List

from .canonicalizer import canonicalize_sentence, signature_string, tokens_for_sentence


def build_structural_metadata(ast: Dict[str, object]) -> Dict[str, object]:
    """
    Build structural fields for an indexed sentence.

    Returns:
        dict with:
            - signature: slot-based canonical signature string
            - grammar_tokens: list of grammar-aware tokens
            - slot_roots: mapping role->root for quick filtering
    """
    slots = canonicalize_sentence(ast)
    slot_roots = {role: slot.root for role, slot in slots.items() if slot and slot.root}

    return {
        "signature": signature_string(ast),
        "grammar_tokens": tokens_for_sentence(ast),
        "slot_roots": slot_roots,
    }


def rank_candidates_by_slot_overlap(
    query_slot_roots: Dict[str, str],
    metadata: List[Dict[str, object]],
    limit: int = 500,
) -> List[int]:
    """
    Rank candidate indices by overlap between query slot roots and indexed slot roots.

    Args:
        query_slot_roots: mapping of role -> root from the query AST
        metadata: list of indexed metadata dicts containing 'slot_roots'
        limit: maximum number of candidates to return

    Returns:
        List of metadata indices ordered by descending overlap score.
    """
    if not query_slot_roots:
        return []

    query_set = set(query_slot_roots.values())
    scored: List[tuple[int, int]] = []

    for idx, meta in enumerate(metadata):
        slot_roots = meta.get("slot_roots") or {}
        if not slot_roots:
            continue
        overlap = len(query_set.intersection(set(slot_roots.values())))
        if overlap > 0:
            scored.append((idx, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:limit]]
