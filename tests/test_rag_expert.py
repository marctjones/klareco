"""
Tests for RAG Expert.
"""
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from klareco.experts.rag_expert import RAGExpert, create_rag_expert
from klareco.parser import parse


class TestRAGExpert(unittest.TestCase):
    """Test suite for RAGExpert."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        # Check if test data exists
        if not cls.index_dir.exists() or not cls.model_path.exists():
            raise unittest.SkipTest("Corpus index or model not found")

    def setUp(self):
        """Set up expert for each test."""
        self.expert = create_rag_expert()

    def test_expert_initialization(self):
        """Test expert initialization."""
        self.assertEqual(self.expert.name, "RAG Expert")
        self.assertIsNotNone(self.expert.retriever)
        self.assertEqual(self.expert.k, 5)
        self.assertEqual(self.expert.min_score_threshold, 0.5)

    def test_can_handle_factoid_questions(self):
        """Test that expert handles factoid questions."""
        # "What is" questions
        ast = parse("Kio estas Esperanto?")
        self.assertTrue(self.expert.can_handle(ast))

        # "Who" questions
        ast = parse("Kiu verkis la libron?")
        self.assertTrue(self.expert.can_handle(ast))

        # "Where" questions
        ast = parse("Kie estas la hundo?")
        self.assertTrue(self.expert.can_handle(ast))

        # "When" questions
        ast = parse("Kiam okazis tio?")
        self.assertTrue(self.expert.can_handle(ast))

    def test_cannot_handle_non_questions(self):
        """Test that expert doesn't handle non-questions."""
        # Statement
        ast = parse("La hundo kuras.")
        self.assertFalse(self.expert.can_handle(ast))

        # Command
        ast = parse("Kuru!")
        self.assertFalse(self.expert.can_handle(ast))

    def test_estimate_confidence_what_is(self):
        """Test confidence estimation for 'what is' questions."""
        ast = parse("Kio estas Esperanto?")
        confidence = self.expert.estimate_confidence(ast)
        self.assertGreaterEqual(confidence, 0.90)

    def test_estimate_confidence_who_is(self):
        """Test confidence estimation for 'who is' questions."""
        ast = parse("Kiu estas la prezidanto?")
        confidence = self.expert.estimate_confidence(ast)
        self.assertGreaterEqual(confidence, 0.90)

    def test_estimate_confidence_where(self):
        """Test confidence estimation for 'where' questions."""
        ast = parse("Kie estas la domo?")
        confidence = self.expert.estimate_confidence(ast)
        self.assertGreaterEqual(confidence, 0.85)

    def test_estimate_confidence_when(self):
        """Test confidence estimation for 'when' questions."""
        ast = parse("Kiam okazis tio?")
        confidence = self.expert.estimate_confidence(ast)
        self.assertGreaterEqual(confidence, 0.85)

    def test_estimate_confidence_non_question(self):
        """Test confidence for non-questions is zero."""
        ast = parse("La hundo kuras.")
        confidence = self.expert.estimate_confidence(ast)
        self.assertEqual(confidence, 0.0)

    def test_execute_simple_query(self):
        """Test executing a simple factoid query."""
        ast = parse("Kio estas hundo?")
        response = self.expert.execute(ast)

        # Check response structure
        self.assertIn('answer', response)
        self.assertIn('confidence', response)
        self.assertIn('expert', response)
        self.assertIn('sources', response)
        self.assertIn('retrieved_count', response)

        # Check types
        self.assertIsInstance(response['answer'], str)
        self.assertIsInstance(response['confidence'], float)
        self.assertEqual(response['expert'], "RAG Expert")
        self.assertIsInstance(response['sources'], list)
        self.assertIsInstance(response['retrieved_count'], int)

        # Check values
        self.assertGreater(response['retrieved_count'], 0)
        self.assertGreater(response['confidence'], 0.0)

    def test_execute_returns_sources(self):
        """Test that execute returns source documents."""
        ast = parse("Kio estas kato?")
        response = self.expert.execute(ast)

        sources = response.get('sources', [])
        self.assertGreater(len(sources), 0)

        for source in sources:
            self.assertIn('text', source)
            self.assertIn('index', source)
            self.assertIn('score', source)

            self.assertIsInstance(source['text'], str)
            self.assertIsInstance(source['index'], int)
            self.assertIsInstance(source['score'], float)

    def test_execute_with_invalid_ast(self):
        """Test execute with invalid AST."""
        # Empty AST
        response = self.expert.execute(None)
        self.assertEqual(response['confidence'], 0.0)
        self.assertIn('error', response)

        # Wrong type
        response = self.expert.execute({'tipo': 'wrong'})
        self.assertEqual(response['confidence'], 0.0)

    def test_answer_formatting(self):
        """Test that answers are properly formatted."""
        ast = parse("Kio estas Esperanto?")
        response = self.expert.execute(ast)

        answer = response['answer']

        # Should contain header
        self.assertIn("Jen kion mi trovis:", answer)

        # Should have numbered results
        self.assertIn("1.", answer)

    def test_confidence_computation(self):
        """Test confidence computation from retrieval scores."""
        ast = parse("Kio estas hundo?")
        response = self.expert.execute(ast)

        confidence = response['confidence']

        # Confidence should be reasonable
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Should correlate with retrieved count
        if response['retrieved_count'] > 0:
            self.assertGreater(confidence, 0.0)

    def test_score_threshold_filtering(self):
        """Test that low-score results are filtered."""
        # Create expert with high threshold
        from klareco.rag.retriever import KlarecoRetriever

        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        expert = RAGExpert(
            retriever=retriever,
            k=10,
            min_score_threshold=1.5  # Very high threshold
        )

        ast = parse("Kio estas hundo?")
        response = expert.execute(ast)

        # With high threshold, fewer results expected
        self.assertIsInstance(response['retrieved_count'], int)

    def test_multiple_queries(self):
        """Test expert handles multiple queries correctly."""
        queries = [
            "Kio estas hundo?",
            "Kiu estas la aŭtoro?",
            "Kie estas la libro?"
        ]

        for query in queries:
            ast = parse(query)
            self.assertTrue(self.expert.can_handle(ast))

            response = self.expert.execute(ast)
            self.assertIn('answer', response)
            self.assertGreater(response['confidence'], 0.0)


class TestRAGExpertWithMock(unittest.TestCase):
    """Test RAG Expert with mocked retriever."""

    def setUp(self):
        """Set up mocked expert."""
        self.mock_retriever = Mock()
        self.expert = RAGExpert(
            retriever=self.mock_retriever,
            k=5,
            min_score_threshold=0.5
        )

    def test_execute_with_mock_results(self):
        """Test execute with mocked retrieval results."""
        # Mock retrieval results
        self.mock_retriever.retrieve_from_ast.return_value = [
            {'text': 'Test sentence 1', 'index': 0, 'score': 1.5},
            {'text': 'Test sentence 2', 'index': 1, 'score': 1.3},
            {'text': 'Test sentence 3', 'index': 2, 'score': 1.1},
        ]

        ast = parse("Kio estas testo?")
        response = self.expert.execute(ast)

        # Verify retriever was called
        self.mock_retriever.retrieve_from_ast.assert_called_once()

        # Check response
        self.assertEqual(response['retrieved_count'], 3)
        self.assertGreater(response['confidence'], 0.0)
        self.assertEqual(len(response['sources']), 3)

    def test_execute_with_no_results(self):
        """Test execute when retrieval returns no results."""
        self.mock_retriever.retrieve_from_ast.return_value = []

        ast = parse("Kio estas nenio?")
        response = self.expert.execute(ast)

        self.assertEqual(response['retrieved_count'], 0)
        self.assertEqual(response['confidence'], 0.0)
        self.assertIn("ne trovis", response['answer'].lower())

    def test_execute_with_low_score_results(self):
        """Test execute filters out low-score results."""
        # Mock results with low scores
        self.mock_retriever.retrieve_from_ast.return_value = [
            {'text': 'Low score result', 'index': 0, 'score': 0.3},
            {'text': 'Another low score', 'index': 1, 'score': 0.2},
        ]

        ast = parse("Kio estas io?")
        response = self.expert.execute(ast)

        # All results below threshold should be filtered
        self.assertEqual(response['retrieved_count'], 0)
        self.assertEqual(response['confidence'], 0.0)

    def test_execute_handles_retriever_exception(self):
        """Test execute handles retriever exceptions gracefully."""
        # Mock retriever to raise exception
        self.mock_retriever.retrieve_from_ast.side_effect = Exception("Test error")

        ast = parse("Kio estas eraro?")
        response = self.expert.execute(ast)

        self.assertEqual(response['confidence'], 0.0)
        self.assertIn('error', response)
        self.assertIn('Test error', response['error'])


class TestCreateRAGExpert(unittest.TestCase):
    """Test create_rag_expert convenience function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        if not cls.index_dir.exists() or not cls.model_path.exists():
            raise unittest.SkipTest("Corpus index or model not found")

    def test_create_rag_expert_defaults(self):
        """Test creating expert with default parameters."""
        expert = create_rag_expert()

        self.assertIsInstance(expert, RAGExpert)
        self.assertEqual(expert.name, "RAG Expert")
        self.assertEqual(expert.k, 5)
        self.assertEqual(expert.min_score_threshold, 0.5)

    def test_create_rag_expert_custom_params(self):
        """Test creating expert with custom parameters."""
        expert = create_rag_expert(
            k=10,
            min_score_threshold=0.7
        )

        self.assertEqual(expert.k, 10)
        self.assertEqual(expert.min_score_threshold, 0.7)


if __name__ == '__main__':
    unittest.main()
