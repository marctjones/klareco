"""
Tests for RAG Retriever.
"""
import unittest
from pathlib import Path
import numpy as np
import tempfile
import json
import faiss

from klareco.rag.retriever import KlarecoRetriever, create_retriever
from klareco.parser import parse


class TestKlarecoRetriever(unittest.TestCase):
    """Test suite for KlarecoRetriever."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        # Check if test data exists
        if not cls.index_dir.exists():
            raise unittest.SkipTest(f"Corpus index not found at {cls.index_dir}")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model not found at {cls.model_path}")

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Check components loaded
        self.assertIsNotNone(retriever.index)
        self.assertIsNotNone(retriever.metadata)
        self.assertIsNotNone(retriever.encoder)
        self.assertIsNotNone(retriever.converter)

        # Check index size
        self.assertGreater(retriever.index.ntotal, 0)
        self.assertEqual(len(retriever.metadata), retriever.index.ntotal)

    def test_retrieve_simple_query(self):
        """Test retrieval with simple query."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Simple query
        query = "Mi amas hundojn"
        results = retriever.retrieve(query, k=5, return_scores=True)

        # Check results structure
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)

        for i, result in enumerate(results):
            self.assertIn('text', result)
            self.assertIn('index', result)
            self.assertIn('rank', result)
            self.assertIn('score', result)

            # Check types
            self.assertIsInstance(result['text'], str)
            self.assertIsInstance(result['index'], int)
            self.assertIsInstance(result['rank'], int)
            self.assertIsInstance(result['score'], float)

            # Check ranks are sequential
            self.assertEqual(result['rank'], i + 1)

    def test_retrieve_without_scores(self):
        """Test retrieval without scores."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        results = retriever.retrieve("La hundo kuras", k=3, return_scores=False)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('text', result)
            self.assertIn('index', result)
            self.assertIn('rank', result)
            self.assertNotIn('score', result)

    def test_retrieve_from_ast(self):
        """Test retrieval from pre-parsed AST."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Parse query manually
        query = "La kato dormas"
        ast = parse(query)

        # Retrieve from AST
        results = retriever.retrieve_from_ast(ast, k=3)

        self.assertEqual(len(results), 3)
        self.assertIn('text', results[0])

    def test_batch_retrieve(self):
        """Test batch retrieval."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        queries = [
            "Mi amas hundojn",
            "La kato dormas",
            "Bona tago"
        ]

        results = retriever.batch_retrieve(queries, k=2)

        # Check batch results
        self.assertEqual(len(results), 3)
        for query_results in results:
            self.assertEqual(len(query_results), 2)

    def test_retrieve_with_unknown_words(self):
        """Test retrieval with unknown words (gracefully handled)."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Unknown Esperanto word - parser marks as foreign word but doesn't fail
        results = retriever.retrieve("xyzabc", k=5)

        # Parser gracefully handles unknown words, so retrieval should still work
        self.assertEqual(len(results), 5)
        self.assertIn('text', results[0])

    def test_retrieve_k_larger_than_corpus(self):
        """Test retrieval with k larger than corpus size."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        corpus_size = retriever.index.ntotal

        # Request more results than corpus has
        results = retriever.retrieve("La hundo", k=corpus_size + 1000)

        # Should return at most corpus_size results
        self.assertLessEqual(len(results), corpus_size)

    def test_create_retriever_convenience(self):
        """Test create_retriever convenience function."""
        retriever = create_retriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        self.assertIsInstance(retriever, KlarecoRetriever)
        self.assertEqual(retriever.mode, 'tree_lstm')

    def test_retriever_mode_validation(self):
        """Test that invalid mode raises error."""
        with self.assertRaises(ValueError):
            KlarecoRetriever(
                index_dir=str(self.index_dir),
                model_path=str(self.model_path),
                mode='invalid_mode',
                device='cpu'
            )

    def test_baseline_mode_not_implemented(self):
        """Test that baseline mode raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            KlarecoRetriever(
                index_dir=str(self.index_dir),
                model_path=str(self.model_path),
                mode='baseline',
                device='cpu'
            )

    def test_missing_index_raises_error(self):
        """Test that missing FAISS index raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                KlarecoRetriever(
                    index_dir=tmpdir,
                    model_path=str(self.model_path),
                    mode='tree_lstm',
                    device='cpu'
                )

    def test_missing_metadata_raises_error(self):
        """Test that missing metadata raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index but no metadata
            index = faiss.IndexFlatIP(512)
            faiss.write_index(index, str(Path(tmpdir) / "faiss_index.bin"))

            with self.assertRaises(FileNotFoundError):
                KlarecoRetriever(
                    index_dir=tmpdir,
                    model_path=str(self.model_path),
                    mode='tree_lstm',
                    device='cpu'
                )

    def test_missing_model_raises_error(self):
        """Test that missing model checkpoint raises error."""
        with self.assertRaises(FileNotFoundError):
            KlarecoRetriever(
                index_dir=str(self.index_dir),
                model_path="/nonexistent/model.pt",
                mode='tree_lstm',
                device='cpu'
            )


class TestRetrieverIntegration(unittest.TestCase):
    """Integration tests for retriever."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.index_dir = Path("data/corpus_index")
        cls.model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

        # Check if test data exists
        if not cls.index_dir.exists():
            raise unittest.SkipTest(f"Corpus index not found at {cls.index_dir}")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model not found at {cls.model_path}")

    def test_semantic_similarity(self):
        """Test that semantically similar queries return similar results."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Similar queries
        query1 = "Mi amas hundojn"
        query2 = "Mi amas katojn"

        results1 = retriever.retrieve(query1, k=10)
        results2 = retriever.retrieve(query2, k=10)

        # Check that results are not empty
        self.assertGreater(len(results1), 0)
        self.assertGreater(len(results2), 0)

        # Both should have reasonable scores
        self.assertGreater(results1[0]['score'], 0.3)
        self.assertGreater(results2[0]['score'], 0.3)

    def test_different_queries_different_results(self):
        """Test that different queries return different results."""
        retriever = KlarecoRetriever(
            index_dir=str(self.index_dir),
            model_path=str(self.model_path),
            mode='tree_lstm',
            device='cpu'
        )

        # Very different queries
        query1 = "La hundo kuras"
        query2 = "Du plus tri"

        results1 = retriever.retrieve(query1, k=3)
        results2 = retriever.retrieve(query2, k=3)

        # Top results should be different
        indices1 = [r['index'] for r in results1]
        indices2 = [r['index'] for r in results2]

        # At least some difference in top 3
        self.assertNotEqual(indices1, indices2)


if __name__ == '__main__':
    unittest.main()
