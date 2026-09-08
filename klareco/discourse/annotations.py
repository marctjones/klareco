"""One shared annotation-layer shape for every cross-sentence ambiguity type.

Each resolver in this package (coreference, proper-noun classification, and
whatever comes next) produces the SAME kind of finding: a token in one
sentence has N grammatically/lexically possible resolutions drawn from
elsewhere in the document, and either exactly one survives (resolved), more
than one does (ambiguous -- recorded with every surviving possibility, not
guessed), or none does (unresolved -- a genuine residue, not silently
dropped). This module is the one place that shape gets turned into a real
`annotation_layers` entry, so a new ambiguity type never has to re-invent
the envelope.
"""
from __future__ import annotations

from typing import Any, Optional

STATUSES = ("resolved", "ambiguous", "unresolved")


def make_finding(status: str, candidates: list[dict]) -> dict:
    """The shared result shape every resolver in this package returns."""
    if status not in STATUSES:
        raise ValueError(f"Unknown resolution status: {status!r}")
    if status == "unresolved" and candidates:
        raise ValueError("'unresolved' must carry no candidates")
    if status == "resolved" and len(candidates) != 1:
        raise ValueError("'resolved' must carry exactly one candidate")
    if status == "ambiguous" and len(candidates) < 2:
        raise ValueError("'ambiguous' must carry 2 or more candidates")
    return {"status": status, "candidates": candidates}


def make_annotation_layer(
    *,
    kind: str,
    mention_ast: dict,
    mention_sid: int,
    token_id: int,
    finding: dict,
    value_extra: Optional[dict[str, Any]] = None,
    producer_name: str,
    producer_version: str,
    artifact_hashes: Optional[dict[str, str]] = None,
) -> dict:
    """Build one annotation_layers entry per klareco.ast_annotations's
    contract, for ANY cross-sentence ambiguity `kind` ('coreference',
    'proper_noun_classification', ...) -- ready for
    `with_annotation_layer(mention_ast, layer)`.

    `mention_ast` is the ambiguous token's OWN sentence's ast (used only to
    compute the required source/tokenization basis hashes); `mention_sid`
    is that sentence's store id, used only to make the layer/annotation id
    unique -- it plays no role in validation. A candidate whose evidence
    lives in a DIFFERENT sentence carries that sentence's own sid inside
    the free-form payload -- the envelope's TokenTarget can only address
    token ids within the annotated ast's own tokenization, so a genuinely
    cross-sentence reference has to live in `value`, not the structural
    `target`.
    """
    from klareco.ast_annotations import annotation_basis

    value = {"kind": kind, "resolution_status": finding["status"],
             "candidates": finding["candidates"]}
    if value_extra:
        value.update(value_extra)
    return {
        "version": 1,
        "id": f"{kind}:{mention_sid}:{token_id}",
        "schema": "urn:klareco:cross-sentence-ambiguity:1",
        "status": "predicted",
        "producer": {
            "name": producer_name,
            "version": producer_version,
            "method": "rule",
            "artifacts": artifact_hashes or {},
        },
        "basis": annotation_basis(mention_ast, tokens=True),
        "annotations": [{
            "id": f"{kind}:{token_id}",
            "target": {"kind": "tokens", "ids": [token_id]},
            "value": value,
        }],
    }
