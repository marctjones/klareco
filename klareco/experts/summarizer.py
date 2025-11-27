"""
Simple extractive summarizer built on top of the retriever.

Keeps things deterministic and small: take top-k retrieved sentences and stitch
them into a concise summary. This is a placeholder until a trained AST-aware
seq2seq model is available.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from klareco.deparser import deparse


class ExtractiveSummarizer:
    def __init__(self, retriever, top_k: int = 3, max_chars: int = 400):
        self.retriever = retriever
        self.top_k = top_k
        self.max_chars = max_chars

    def execute(self, query_ast: Dict[str, Any], original_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve supporting sentences and concatenate the top few as a summary.
        """
        text = original_text or deparse(query_ast)
        try:
            results = self.retriever.retrieve(text, k=self.top_k, return_scores=True)
        except Exception as exc:
            return {
                "summary": "Mi ne povis krei resumon pro serĉa eraro.",
                "sources": [],
                "confidence": 0.0,
                "error": str(exc),
            }

        if not results:
            return {
                "summary": "Mi ne trovis enhavon por resumi.",
                "sources": [],
                "confidence": 0.0,
            }

        sentences: List[str] = []
        for r in results:
            if len(" ".join(sentences)) >= self.max_chars:
                break
            sentences.append(r.get("text", ""))

        summary = " ".join(sentences)[: self.max_chars].strip()
        avg_score = sum(float(r.get("score", 0.0) or 0.0) for r in results) / len(results)

        sources: List[Dict[str, Any]] = []
        for r in results:
            sources.append(
                {
                    "text": r.get("text"),
                    "source": r.get("source_name") or r.get("source"),
                    "score": float(r.get("score", 0.0)) if r.get("score") is not None else None,
                }
            )

        return {
            "summary": summary,
            "sources": sources,
            "confidence": avg_score,
        }


def create_extractive_summarizer(retriever, top_k: int = 3, max_chars: int = 400) -> ExtractiveSummarizer:
    return ExtractiveSummarizer(retriever=retriever, top_k=top_k, max_chars=max_chars)
