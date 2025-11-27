"""
Extractive responder that uses a retriever to surface an answer from context.

This keeps the response deterministic and minimal: pick the top retrieved
sentence, return it with sources and a confidence derived from the score.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from klareco.deparser import deparse
from klareco.parser import parse


class ExtractiveResponder:
    def __init__(self, retriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    def execute(self, query_ast: Dict[str, Any], original_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve supporting sentences and return the top one as the answer.
        """
        text = original_text or deparse(query_ast)
        try:
            results = self.retriever.retrieve(text, k=self.top_k, return_scores=True)
        except Exception as exc:
            return {
                "answer": "Mi ne trovis respondon pro eraro en serĉo.",
                "sources": [],
                "confidence": 0.0,
                "error": str(exc),
            }

        if not results:
            return {
                "answer": "Mi ne trovis respondon en la korpuso.",
                "sources": [],
                "confidence": 0.0,
            }

        best = results[0]
        answer = best.get("text", "")
        confidence = float(best.get("score", 0.0)) if best.get("score") is not None else 0.5

        sources: List[Dict[str, Any]] = []
        for r in results:
            sources.append({
                "text": r.get("text"),
                "source": r.get("source_name") or r.get("source"),
                "score": float(r.get("score", 0.0)) if r.get("score") is not None else None,
            })

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }


def create_extractive_responder(retriever, top_k: int = 3) -> ExtractiveResponder:
    return ExtractiveResponder(retriever=retriever, top_k=top_k)
