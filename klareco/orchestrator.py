"""
Minimal orchestrator for routing intents to simple experts.

This version is intentionally lightweight to unblock the CLI pipeline.
It uses the symbolic GatingNetwork and a handful of built-in handlers
that return deterministic placeholder responses.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any

from .gating_network import GatingNetwork
from .deparser import deparse
from .experts.extractive import ExtractiveResponder
from .experts.summarizer import ExtractiveSummarizer

logger = logging.getLogger(__name__)


class Orchestrator:
    """Routes parsed ASTs to lightweight expert handlers."""

    def __init__(self, gating_network: GatingNetwork | None = None, retriever=None):
        self.gating_network = gating_network or GatingNetwork(mode="symbolic")
        self.retriever = retriever
        self.extractive_responder = ExtractiveResponder(retriever) if retriever else None
        self.summarizer = ExtractiveSummarizer(retriever) if retriever else None
        self.experts: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "general_query": self._echo_response,
            "factoid_question": self._factoid_response,
            "calculation_request": self._calculation_response,
            "temporal_query": self._temporal_response,
            "grammar_query": self._grammar_response,
            "dictionary_query": self._dictionary_response,
            "summarization_request": self._summarization_response,
            "command_intent": self._command_response,
        }

    def route(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify intent and dispatch to the matching expert.

        Returns a response dict suitable for trace logging.
        """
        gating_result = self.gating_network.classify(ast)
        intent = gating_result.get("intent", "general_query")
        expert = self.experts.get(intent, self._echo_response)

        try:
            answer = expert(ast)
            error = None
            confidence = 1.0
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Expert failed for intent %s", intent)
            answer = "Eraro dum respondo generacio."
            error = str(exc)
            confidence = 0.0

        return {
            "intent": intent,
            "intent_confidence": gating_result.get("confidence", 1.0),
            "expert": expert.__name__,
            "confidence": confidence,
            "answer": answer,
            "full_response": {"answer": answer},
            "error": error,
        }

    # --- Expert handlers (minimal placeholders) ---

    def _echo_response(self, ast: Dict[str, Any]) -> str:
        """Echo the parsed sentence back to the user."""
        try:
            text = deparse(ast)
        except Exception:
            text = "via frazo"
        return f"Mi ricevis: {text}"

    def _factoid_response(self, ast: Dict[str, Any]) -> str:
        if self.extractive_responder:
            result = self.extractive_responder.execute(ast, original_text=deparse(ast))
            return result.get("answer", "Neniu respondo.")
        return "Mi bezonas RAG-datumojn por respondi tiun faktan demandon."

    def _calculation_response(self, ast: Dict[str, Any]) -> str:
        return "Mi detektis kalkulan peton sed kalkula motoro ankoraŭ ne estas ligita."

    def _temporal_response(self, ast: Dict[str, Any]) -> str:
        return "Tempa demando detektita; dato/horo eksperto ankoraŭ ne estas ligita."

    def _grammar_response(self, ast: Dict[str, Any]) -> str:
        return "Vi petas klarigon pri gramatiko; plena gramatika eksperto venos poste."

    def _dictionary_response(self, ast: Dict[str, Any]) -> str:
        return "Vorta klarigo petita; vortara eksperto ankoraŭ ne disponeblas."

    def _summarization_response(self, ast: Dict[str, Any]) -> str:
        if self.summarizer:
            result = self.summarizer.execute(ast, original_text=deparse(ast))
            return result.get("summary", "Neniu resumo.")
        return "Resuma peto ricevita; resumiga eksperto estas estonta laboro."

    def _command_response(self, ast: Dict[str, Any]) -> str:
        return "Komando detektita; pliaj iloj ne estas ankoraŭ konektitaj."


def create_orchestrator_with_experts(retriever=None) -> Orchestrator:
    """Convenience factory used by the pipeline."""
    return Orchestrator(retriever=retriever)
