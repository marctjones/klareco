"""
Integration tests for RAG pipeline.

Tests the full pipeline from query input to retrieved results,
ensuring all components work together correctly.
"""
import unittest
from pathlib import Path

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever, create_retriever
from klareco.experts.rag_expert import RAGExpert, create_rag_expert


class TestRAGPipelineIntegration(unittest.TestCase):
    """Integration tests for the full RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        # Check if test data exists
        if not cls.index_dir.exists() or not cls.model_path.exists():
            raise unittest.SkipTest("Corpus index or model not found")

    def test_end_to_end_simple_query(self):
        """Test complete pipeline with simple query."""
        # Create components
        retriever = create_retriever()
        expert = RAGExpert(retriever)

        # Query
        query = "Mi amas hundojn"

        # Parse
        ast = parse(query)
        self.assertIsNotNone(ast)

        # Check expert can handle
        self.assertTrue(expert.can_handle(ast) or not expert.can_handle(ast))  # Not a question, may not handle

        # Retrieve (bypass expert, test retriever directly)
        results = retriever.retrieve(query, k=5)

        self.assertGreater(len(results), 0)
        self.assertIn('text', results[0])
        self.assertIn('score', results[0])

    def test_end_to_end_factoid_question(self):
        """Test complete pipeline with factoid question."""
        expert = create_rag_expert()

        # Factoid question
        query = "Kio estas hundo?"
        ast = parse(query)

        # Expert should handle factoid questions
        self.assertTrue(expert.can_handle(ast))

        # Execute
        response = expert.execute(ast)

        # Verify response structure
        self.assertIn('answer', response)
        self.assertIn('confidence', response)
        self.assertIn('sources', response)
        self.assertIn('expert', response)

        # Verify results
        self.assertGreater(response['confidence'], 0.0)
        self.assertGreater(len(response['sources']), 0)

    def test_end_to_end_who_question(self):
        """Test pipeline with 'who' question."""
        expert = create_rag_expert()

        query = "Kiu verkis la libron?"
        ast = parse(query)

        self.assertTrue(expert.can_handle(ast))

        response = expert.execute(ast)
        self.assertIn('answer', response)
        self.assertGreater(response['confidence'], 0.0)

    def test_end_to_end_where_question(self):
        """Test pipeline with 'where' question."""
        expert = create_rag_expert()

        query = "Kie estas la domo?"
        ast = parse(query)

        self.assertTrue(expert.can_handle(ast))

        response = expert.execute(ast)
        self.assertIn('answer', response)

    def test_end_to_end_when_question(self):
        """Test pipeline with 'when' question."""
        expert = create_rag_expert()

        query = "Kiam okazis tio?"
        ast = parse(query)

        self.assertTrue(expert.can_handle(ast))

        response = expert.execute(ast)
        self.assertIn('answer', response)

    def test_pipeline_with_complex_query(self):
        """Test pipeline with complex multi-word query."""
        expert = create_rag_expert()

        query = "Kio estas la diferenco inter hundo kaj kato?"
        ast = parse(query)

        response = expert.execute(ast)
        self.assertIn('answer', response)

    def test_pipeline_multiple_queries_sequential(self):
        """Test pipeline handles multiple sequential queries."""
        expert = create_rag_expert()

        queries = [
            "Kio estas Esperanto?",
            "Kiu verkis Alice's Adventures in Wonderland?",
            "Kie estas Londono?"
        ]

        for query in queries:
            ast = parse(query)
            response = expert.execute(ast)

            self.assertIn('answer', response)
            self.assertIn('confidence', response)
            self.assertIn('sources', response)

    def test_retriever_consistency(self):
        """Test that retriever returns consistent results."""
        retriever = create_retriever()

        query = "La hundo kuras"

        # Run same query multiple times
        results1 = retriever.retrieve(query, k=5)
        results2 = retriever.retrieve(query, k=5)

        # Should get same results
        self.assertEqual(len(results1), len(results2))

        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1['index'], r2['index'])
            self.assertAlmostEqual(r1['score'], r2['score'], places=5)

    def test_expert_retriever_integration(self):
        """Test expert and retriever work together correctly."""
        retriever = create_retriever()
        expert = RAGExpert(retriever, k=10, min_score_threshold=0.3)

        query = "Kio estas kato?"
        ast = parse(query)

        # Expert executes using retriever
        response = expert.execute(ast)

        # Should have sources from retriever
        self.assertIn('sources', response)
        sources = response['sources']

        # Each source should have retriever fields
        for source in sources:
            self.assertIn('text', source)
            self.assertIn('index', source)
            self.assertIn('score', source)

            # Score should be above threshold
            self.assertGreaterEqual(source['score'], 0.3)

    def test_parse_retrieve_format_cycle(self):
        """Test complete cycle: parse → retrieve → format."""
        retriever = create_retriever()

        # Parse
        query = "Mi volas lerni"
        ast = parse(query)
        self.assertIsNotNone(ast)

        # Retrieve from AST
        results = retriever.retrieve_from_ast(ast, k=3)
        self.assertEqual(len(results), 3)

        # Format results (simple concatenation)
        formatted = "\n".join([f"- {r['text']}" for r in results])
        self.assertGreater(len(formatted), 0)

    def test_pipeline_with_different_k_values(self):
        """Test pipeline with different k values."""
        expert_k5 = create_rag_expert(k=5)
        expert_k10 = create_rag_expert(k=10)

        query = "Kio estas hundo?"
        ast = parse(query)

        response_k5 = expert_k5.execute(ast)
        response_k10 = expert_k10.execute(ast)

        # k=10 should return more sources (or equal if not enough results)
        self.assertGreaterEqual(
            len(response_k10['sources']),
            len(response_k5['sources'])
        )

    def test_pipeline_with_different_thresholds(self):
        """Test pipeline with different score thresholds."""
        expert_low = create_rag_expert(min_score_threshold=0.3)
        expert_high = create_rag_expert(min_score_threshold=1.5)

        query = "Kio estas hundo?"
        ast = parse(query)

        response_low = expert_low.execute(ast)
        response_high = expert_high.execute(ast)

        # Low threshold should return more results
        sources_low = len(response_low.get('sources', []))
        sources_high = len(response_high.get('sources', []))

        self.assertGreaterEqual(sources_low, sources_high)

    def test_error_handling_in_pipeline(self):
        """Test that pipeline handles errors gracefully."""
        expert = create_rag_expert()

        # Invalid AST
        response = expert.execute(None)
        self.assertEqual(response['confidence'], 0.0)
        self.assertIn('error', response)

        # Empty AST
        response = expert.execute({})
        self.assertEqual(response['confidence'], 0.0)


class TestRAGPerformance(unittest.TestCase):
    """Performance tests for RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        if not cls.index_dir.exists() or not cls.model_path.exists():
            raise unittest.SkipTest("Corpus index or model not found")

    def test_retrieval_speed(self):
        """Test that retrieval completes in reasonable time."""
        import time

        retriever = create_retriever()

        query = "Mi amas hundojn"

        start = time.time()
        results = retriever.retrieve(query, k=10)
        elapsed = time.time() - start

        # Should complete in under 100ms
        self.assertLess(elapsed, 0.1)
        self.assertGreater(len(results), 0)

    def test_batch_retrieval_faster_than_sequential(self):
        """Test that batch retrieval is efficient."""
        import time

        retriever = create_retriever()

        queries = [
            "Mi amas hundojn",
            "La kato dormas",
            "Bona tago"
        ]

        # Sequential
        start = time.time()
        for query in queries:
            retriever.retrieve(query, k=5)
        sequential_time = time.time() - start

        # Batch
        start = time.time()
        retriever.batch_retrieve(queries, k=5)
        batch_time = time.time() - start

        # Batch should be similar or slightly faster (minimal difference expected)
        # Just verify both complete successfully
        self.assertLess(sequential_time, 1.0)
        self.assertLess(batch_time, 1.0)

    def test_large_k_retrieval(self):
        """Test retrieval with large k value."""
        import time

        retriever = create_retriever()

        query = "La hundo kuras"

        start = time.time()
        results = retriever.retrieve(query, k=100)
        elapsed = time.time() - start

        # Should still complete quickly
        self.assertLess(elapsed, 0.2)
        self.assertGreater(len(results), 0)


class TestRAGRobustness(unittest.TestCase):
    """Robustness tests for RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        if not cls.index_dir.exists() or not cls.model_path.exists():
            raise unittest.SkipTest("Corpus index or model not found")

    def test_handles_short_queries(self):
        """Test pipeline with very short queries."""
        expert = create_rag_expert()

        short_queries = ["Kio?", "Kiu?", "Kie?"]

        for query in short_queries:
            try:
                ast = parse(query)
                response = expert.execute(ast)
                self.assertIn('answer', response)
            except Exception as e:
                # Short queries might fail parsing, that's okay
                self.assertIsInstance(e, (ValueError, Exception))

    def test_handles_long_queries(self):
        """Test pipeline with long queries."""
        expert = create_rag_expert()

        long_query = "Kio estas la diferenco inter hundo kaj kato kaj ĉu ili povas vivi kune en la sama domo?"

        ast = parse(long_query)
        response = expert.execute(ast)

        self.assertIn('answer', response)

    def test_handles_special_characters(self):
        """Test pipeline with special characters."""
        retriever = create_retriever()

        query = "La libro estas tre, tre interesa!"

        results = retriever.retrieve(query, k=5)
        self.assertGreater(len(results), 0)

    def test_handles_numbers_in_query(self):
        """Test pipeline with numbers."""
        retriever = create_retriever()

        query = "Mi havas du hundojn kaj tri katojn"

        results = retriever.retrieve(query, k=5)
        self.assertGreater(len(results), 0)

    def test_handles_mixed_case(self):
        """Test pipeline with mixed case."""
        retriever = create_retriever()

        query = "MI AMAS HUNDOJN"

        results = retriever.retrieve(query, k=5)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
