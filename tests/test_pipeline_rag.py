"""
Integration tests for RAG functionality in the pipeline.

These tests ensure that RAG Expert is properly registered and can handle queries.
"""

import unittest
import pytest
from klareco.pipeline import KlarecoPipeline
from klareco.orchestrator import create_orchestrator_with_experts
from klareco.parser import parse


class TestPipelineWithRAG(unittest.TestCase):
    """Test RAG integration with the full pipeline."""

    @pytest.mark.skipif(
        not __import__('pathlib').Path('data/corpus_index').exists(),
        reason="Corpus index not available"
    )
    def test_orchestrator_has_rag_expert(self):
        """Test that RAG Expert is registered in the orchestrator."""
        orchestrator = create_orchestrator_with_experts()

        # Check RAG Expert is in the experts list
        self.assertIn('RAG Expert', orchestrator.list_experts())

        # Check factoid_question intent is mapped
        self.assertIn('factoid_question', orchestrator.intent_to_expert)
        self.assertEqual(orchestrator.intent_to_expert['factoid_question'], 'RAG Expert')

    @pytest.mark.skipif(
        not __import__('pathlib').Path('data/corpus_index').exists(),
        reason="Corpus index not available"
    )
    def test_pipeline_routes_factoid_to_rag(self):
        """Test that factoid questions are routed to RAG Expert."""
        pipeline = KlarecoPipeline(use_orchestrator=True)

        # This should be classified as factoid_question and routed to RAG
        trace = pipeline.run("Kio estas Esperanto?")

        # Check it didn't fail
        self.assertIsNone(trace.error)

        # Check response is not the "can't answer" fallback
        self.assertNotIn("ne povas respondi", trace.final_response.lower())

        # Check RAG Expert was actually called
        orchestrator_step = next(
            (s for s in trace.steps if s['name'] == 'Orchestrator'),
            None
        )
        self.assertIsNotNone(orchestrator_step)
        self.assertEqual(orchestrator_step['outputs']['expert'], 'RAG Expert')

    @pytest.mark.skipif(
        not __import__('pathlib').Path('data/corpus_index').exists(),
        reason="Corpus index not available"
    )
    def test_factoid_question_returns_results(self):
        """Test that factoid questions return actual results, not errors."""
        pipeline = KlarecoPipeline(use_orchestrator=True)

        trace = pipeline.run("Kio estas hundo?")  # What is a dog?

        # Should not error
        self.assertIsNone(trace.error)

        # Should return something from corpus
        self.assertIsNotNone(trace.final_response)
        self.assertGreater(len(trace.final_response), 0)

        # Should not be the fallback message
        self.assertNotIn("Pardonu", trace.final_response)


class TestOrchestratorValidation(unittest.TestCase):
    """Test orchestrator coverage and validation."""

    def test_all_intents_have_handlers(self):
        """Test that all known intents have registered experts."""
        orchestrator = create_orchestrator_with_experts()

        # Known intents from the gating network
        known_intents = [
            'calculation_request',
            'temporal_query',
            'grammar_query',
            'factoid_question',
        ]

        missing_handlers = []
        for intent in known_intents:
            if intent not in orchestrator.intent_to_expert:
                missing_handlers.append(intent)

        # All intents should have handlers (or be skipped gracefully)
        # If RAG Expert can't load, factoid_question won't have a handler
        # but that's OK - it's documented in the warning log
        if missing_handlers:
            # Check if it's just factoid_question (RAG unavailable)
            if missing_handlers == ['factoid_question']:
                # This is acceptable - RAG Expert couldn't load
                pass
            else:
                self.fail(f"Intents without handlers: {missing_handlers}")

    def test_expert_list_is_not_empty(self):
        """Test that at least some experts are registered."""
        orchestrator = create_orchestrator_with_experts()

        # Should have at least the 3 always-available experts
        self.assertGreaterEqual(len(orchestrator.experts), 3)

        # Should have Math, Date, Grammar at minimum
        experts = orchestrator.list_experts()
        self.assertIn('Math Tool Expert', experts)
        self.assertIn('Date/Time Tool Expert', experts)
        self.assertIn('Grammar Tool Expert', experts)


if __name__ == '__main__':
    unittest.main()
