"""
Integration tests for corpus management CLI.

Tests the full workflow: add, list, validate, remove texts.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klareco.corpus_manager import CorpusManager, TextValidator


class TestCorpusManagementCLI(unittest.TestCase):
    """Test corpus management CLI workflows."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_add_valid_text(self):
        """Test adding a valid Esperanto text."""
        # Create valid test file
        test_file = self.test_path / "test_book.txt"
        long_esperanto_text = """
La rapida bruna vulpo saltas super la mallaborema hundo.
Esperanto estas la internacia lingvo. Ĝi estas tre facila por lerni.
La suno brilas kaj la birdoj kantas en la arboj.
""" * 35
        test_file.write_text(long_esperanto_text, encoding='utf-8')

        # Add to corpus
        with CorpusManager(self.test_path) as manager:
            success, message, text_id = manager.add_text_from_file(
                test_file,
                title="Test Book",
                source_type="literature"
            )

            self.assertTrue(success)
            self.assertIsNotNone(text_id)
            self.assertIn("Test Book", message)

            # Verify it's in database
            text = manager.db.get_text(text_id)
            self.assertIsNotNone(text)
            self.assertEqual(text['title'], "Test Book")
            self.assertEqual(text['source_type'], "literature")
            self.assertEqual(text['validation_status'], "valid")
            self.assertGreater(text['validation_score'], 0.7)

    def test_add_invalid_text(self):
        """Test that invalid text is rejected."""
        # Create invalid test file (not Esperanto)
        test_file = self.test_path / "invalid.txt"
        test_file.write_text(
            "This is English text.\n" * 50 +
            "Not Esperanto at all.\n" * 50,
            encoding='utf-8'
        )

        # Try to add to corpus
        with CorpusManager(self.test_path) as manager:
            success, message, text_id = manager.add_text_from_file(
                test_file,
                title="Invalid Book"
            )

            self.assertFalse(success)
            self.assertIn("Validation failed", message)

    def test_add_duplicate_text(self):
        """Test that duplicate filenames are rejected."""
        # Create test file
        test_file = self.test_path / "test_book.txt"
        long_esperanto_text = """
La rapida bruna vulpo saltas super la mallaborema hundo.
Esperanto estas la internacia lingvo. Ĝi estas tre facila por lerni.
La suno brilas kaj la birdoj kantas en la arboj.
""" * 35
        test_file.write_text(long_esperanto_text, encoding='utf-8')

        with CorpusManager(self.test_path) as manager:
            # Add first time - should succeed
            success1, message1, text_id1 = manager.add_text_from_file(
                test_file,
                title="Test Book"
            )
            self.assertTrue(success1, msg=message1)

            # Add second time - should fail
            success2, message2, text_id2 = manager.add_text_from_file(
                test_file,
                title="Test Book"
            )
            self.assertFalse(success2)
            self.assertIn("already exists", message2)

    def test_list_texts(self):
        """Test listing texts in corpus."""
        # Create and add multiple test files
        for i in range(3):
            test_file = self.test_path / f"book{i}.txt"
            test_file.write_text(
                f"La libro numero {i}.\n" * 100,
                encoding='utf-8'
            )

            with CorpusManager(self.test_path) as manager:
                manager.add_text_from_file(
                    test_file,
                    title=f"Book {i}"
                )

        # List all texts
        with CorpusManager(self.test_path) as manager:
            texts = manager.list_texts()

            self.assertEqual(len(texts), 3)
            titles = [t['title'] for t in texts]
            self.assertIn("Book 0", titles)
            self.assertIn("Book 1", titles)
            self.assertIn("Book 2", titles)

    def test_list_indexed_only(self):
        """Test listing only indexed texts."""
        # Create test files
        test_file = self.test_path / "book.txt"
        test_file.write_text(
            "La libro.\n" * 100,
            encoding='utf-8'
        )

        with CorpusManager(self.test_path) as manager:
            # Add text
            _, _, text_id = manager.add_text_from_file(
                test_file,
                title="Test Book"
            )

            # Mark as indexed
            manager.db.mark_indexed(
                text_id,
                sentence_count=100,
                sentence_data=[(f"Sentence {i}", i, i) for i in range(100)]
            )

            # List all texts
            all_texts = manager.list_texts(indexed_only=False)
            self.assertEqual(len(all_texts), 1)

            # List indexed only
            indexed_texts = manager.list_texts(indexed_only=True)
            self.assertEqual(len(indexed_texts), 1)
            self.assertEqual(indexed_texts[0]['is_indexed'], 1)

    def test_remove_unindexed_text(self):
        """Test removing a text that is not indexed."""
        # Create and add test file
        test_file = self.test_path / "book.txt"
        test_file.write_text(
            "La libro.\n" * 100,
            encoding='utf-8'
        )

        with CorpusManager(self.test_path) as manager:
            # Add text
            _, _, text_id = manager.add_text_from_file(
                test_file,
                title="Test Book"
            )

            # Remove it
            success, message = manager.remove_text(text_id)

            self.assertTrue(success)
            self.assertIn("Removed", message)

            # Verify it's gone
            text = manager.db.get_text(text_id)
            self.assertIsNone(text)

    def test_remove_indexed_text_fails(self):
        """Test that removing indexed text requires rebuild."""
        # Create and add test file
        test_file = self.test_path / "book.txt"
        test_file.write_text(
            "La libro.\n" * 100,
            encoding='utf-8'
        )

        with CorpusManager(self.test_path) as manager:
            # Add text
            _, _, text_id = manager.add_text_from_file(
                test_file,
                title="Test Book"
            )

            # Mark as indexed
            manager.db.mark_indexed(
                text_id,
                sentence_count=100,
                sentence_data=[(f"Sentence {i}", i, i) for i in range(100)]
            )

            # Try to remove - should fail
            success, message = manager.remove_text(text_id)

            self.assertFalse(success)
            self.assertIn("indexed", message)
            self.assertIn("Rebuild", message)

    def test_get_stats(self):
        """Test getting corpus statistics."""
        # Create and add test files
        for i in range(3):
            test_file = self.test_path / f"book{i}.txt"
            test_file.write_text(
                f"La libro numero {i}.\n" * 100,
                encoding='utf-8'
            )

            with CorpusManager(self.test_path) as manager:
                manager.add_text_from_file(
                    test_file,
                    title=f"Book {i}"
                )

        # Get stats
        with CorpusManager(self.test_path) as manager:
            stats = manager.get_stats()

            self.assertEqual(stats['total_texts'], 3)
            self.assertEqual(stats['indexed_texts'], 0)
            self.assertEqual(stats['total_sentences'], 0)  # None indexed yet


class TestTextValidator(unittest.TestCase):
    """Test text validation functionality."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.validator = TextValidator()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_validate_valid_esperanto(self):
        """Test validating a valid Esperanto file."""
        test_file = self.test_path / "valid.txt"
        long_esperanto_text = """
La rapida bruna vulpo saltas super la mallaborema hundo.
Esperanto estas la internacia lingvo. Ĝi estas tre facila por lerni.
La suno brilas kaj la birdoj kantas en la arboj.
""" * 20
        test_file.write_text(long_esperanto_text, encoding='utf-8')

        is_valid, score, message = self.validator.validate_file(test_file)

        self.assertTrue(is_valid)
        self.assertGreater(score, 0.4)  # At least acceptable
        self.assertIn("Esperanto", message)

    def test_validate_too_small_file(self):
        """Test that very small files are rejected."""
        test_file = self.test_path / "tiny.txt"
        test_file.write_text("La", encoding='utf-8')

        is_valid, score, message = self.validator.validate_file(test_file)

        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        self.assertIn("too small", message)

    def test_validate_missing_file(self):
        """Test validating a non-existent file."""
        test_file = self.test_path / "missing.txt"

        is_valid, score, message = self.validator.validate_file(test_file)

        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        self.assertIn("does not exist", message)

    def test_validate_english_text(self):
        """Test that English text is rejected."""
        test_file = self.test_path / "english.txt"
        test_file.write_text(
            "This is English text.\n" * 50 +
            "Not Esperanto at all.\n" * 50,
            encoding='utf-8'
        )

        is_valid, score, message = self.validator.validate_file(test_file)

        self.assertFalse(is_valid)
        self.assertLess(score, 0.4)
        # Language detection might fail, but parse rate should be low


class TestCorpusDatabase(unittest.TestCase):
    """Test corpus database operations."""

    def setUp(self):
        """Create temporary database."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        from klareco.corpus_manager import CorpusDatabase
        self.db = CorpusDatabase(self.test_path / "test.db")

    def tearDown(self):
        """Clean up."""
        self.db.close()
        import shutil
        shutil.rmtree(self.test_dir)

    def test_add_and_get_text(self):
        """Test adding and retrieving a text."""
        text_id = self.db.add_text(
            filename="test.txt",
            title="Test Book",
            source_type="literature"
        )

        text = self.db.get_text(text_id)

        self.assertIsNotNone(text)
        self.assertEqual(text['filename'], "test.txt")
        self.assertEqual(text['title'], "Test Book")
        self.assertEqual(text['source_type'], "literature")
        self.assertEqual(text['validation_status'], "pending")

    def test_get_text_by_filename(self):
        """Test retrieving text by filename."""
        self.db.add_text(
            filename="test.txt",
            title="Test Book"
        )

        text = self.db.get_text_by_filename("test.txt")

        self.assertIsNotNone(text)
        self.assertEqual(text['filename'], "test.txt")

    def test_update_validation(self):
        """Test updating validation status."""
        text_id = self.db.add_text(
            filename="test.txt",
            title="Test Book"
        )

        self.db.update_validation(text_id, "valid", 0.85)

        text = self.db.get_text(text_id)
        self.assertEqual(text['validation_status'], "valid")
        self.assertEqual(text['validation_score'], 0.85)

    def test_mark_indexed(self):
        """Test marking a text as indexed."""
        text_id = self.db.add_text(
            filename="test.txt",
            title="Test Book"
        )

        sentence_data = [
            ("Sentence 1", 1, 0),
            ("Sentence 2", 2, 1),
            ("Sentence 3", 3, 2)
        ]

        self.db.mark_indexed(text_id, 3, sentence_data)

        text = self.db.get_text(text_id)
        self.assertEqual(text['is_indexed'], 1)
        self.assertEqual(text['sentence_count'], 3)

        # Verify sentences were stored
        sentences = self.db.get_indexed_sentences(text_id)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0]['sentence'], "Sentence 1")
        self.assertEqual(sentences[0]['embedding_idx'], 0)

    def test_mark_unindexed(self):
        """Test marking a text as not indexed."""
        text_id = self.db.add_text(
            filename="test.txt",
            title="Test Book"
        )

        # First mark as indexed
        sentence_data = [("Sentence 1", 1, 0)]
        self.db.mark_indexed(text_id, 1, sentence_data)

        # Then mark as unindexed
        self.db.mark_unindexed(text_id)

        text = self.db.get_text(text_id)
        self.assertEqual(text['is_indexed'], 0)
        self.assertEqual(text['sentence_count'], 0)

        # Verify sentences were removed
        sentences = self.db.get_indexed_sentences(text_id)
        self.assertEqual(len(sentences), 0)

    def test_remove_text(self):
        """Test removing a text."""
        text_id = self.db.add_text(
            filename="test.txt",
            title="Test Book"
        )

        self.db.remove_text(text_id)

        text = self.db.get_text(text_id)
        self.assertIsNone(text)

    def test_list_texts(self):
        """Test listing all texts."""
        self.db.add_text(filename="book1.txt", title="Book 1")
        self.db.add_text(filename="book2.txt", title="Book 2")
        self.db.add_text(filename="book3.txt", title="Book 3")

        texts = self.db.list_texts()

        self.assertEqual(len(texts), 3)
        titles = [t['title'] for t in texts]
        self.assertIn("Book 1", titles)
        self.assertIn("Book 2", titles)
        self.assertIn("Book 3", titles)

    def test_get_stats(self):
        """Test getting database statistics."""
        # Add texts
        text_id1 = self.db.add_text(filename="book1.txt", title="Book 1")
        text_id2 = self.db.add_text(filename="book2.txt", title="Book 2")

        # Mark one as indexed
        self.db.mark_indexed(text_id1, 100, [(f"S{i}", i, i) for i in range(100)])

        stats = self.db.get_stats()

        self.assertEqual(stats['total_texts'], 2)
        self.assertEqual(stats['indexed_texts'], 1)
        self.assertEqual(stats['total_sentences'], 100)


if __name__ == '__main__':
    unittest.main()
