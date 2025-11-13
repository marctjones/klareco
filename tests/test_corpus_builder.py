"""
Unit tests for corpus building with source attribution.

Tests the build_corpus_with_sources.py script functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from build_corpus_with_sources import build_corpus


class TestCorpusBuilder(unittest.TestCase):
    """Test corpus building functionality."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_basic_corpus_building(self):
        """Test basic corpus building from simple text files."""
        # Create test file
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "La hundo vidas la katon.\n"
            "Mi amas Esperanton.\n"
            "La suno brilas.\n",
            encoding='utf-8'
        )

        # Build corpus
        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=10
        )

        # Verify
        self.assertEqual(count, 3)
        self.assertTrue(output_file.exists())

        # Read and verify JSONL format
        with output_file.open('r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

            # Check first entry
            entry = json.loads(lines[0])
            self.assertIn('text', entry)
            self.assertIn('source', entry)
            self.assertIn('source_name', entry)
            self.assertIn('line', entry)
            self.assertEqual(entry['text'], "La hundo vidas la katon.")
            self.assertEqual(entry['source'], "test")
            self.assertEqual(entry['source_name'], "Test Book")
            self.assertEqual(entry['line'], 1)

    def test_min_length_filtering(self):
        """Test that min_length parameter filters short lines."""
        # Create test file with various line lengths
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "Short\n"  # 5 chars - should be filtered
            "A bit longer line\n"  # 17 chars - should be filtered with min_length=20
            "This is a sufficiently long sentence.\n"  # 38 chars - should pass
            "x\n"  # 1 char - should be filtered
            "Another long sentence that should pass the filter.\n",  # 51 chars - should pass
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        # Test with min_length=20
        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=True,
            min_length=20
        )

        # Should only include the 2 long sentences
        self.assertEqual(count, 2)

        # Verify content
        with output_file.open('r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]['text'], "This is a sufficiently long sentence.")
            self.assertEqual(lines[1]['text'], "Another long sentence that should pass the filter.")

    def test_configurable_min_length(self):
        """Test different min_length values produce different results."""
        # Create test file
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "Short line\n"  # 10 chars
            "Medium length line here\n"  # 23 chars
            "This is a very long line with lots of text\n",  # 43 chars
            encoding='utf-8'
        )

        texts = [("cleaned_test.txt", "Test Book")]

        # Test with min_length=10
        output1 = self.test_path / "corpus_min10.jsonl"
        count1 = build_corpus(
            self.test_path, output1, texts,
            skip_empty=True, skip_metadata=True, min_length=10
        )
        self.assertEqual(count1, 3)  # All lines pass

        # Test with min_length=20
        output2 = self.test_path / "corpus_min20.jsonl"
        count2 = build_corpus(
            self.test_path, output2, texts,
            skip_empty=True, skip_metadata=True, min_length=20
        )
        self.assertEqual(count2, 2)  # Only 2 longest lines pass

        # Test with min_length=30
        output3 = self.test_path / "corpus_min30.jsonl"
        count3 = build_corpus(
            self.test_path, output3, texts,
            skip_empty=True, skip_metadata=True, min_length=30
        )
        self.assertEqual(count3, 1)  # Only longest line passes

    def test_empty_line_handling(self):
        """Test that empty lines are properly skipped."""
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "First line with content\n"
            "\n"  # Empty line
            "   \n"  # Whitespace only
            "Second line with content\n"
            "\n"
            "Third line with content\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=10
        )

        # Should only count non-empty lines
        self.assertEqual(count, 3)

    def test_metadata_filtering(self):
        """Test that metadata lines are filtered when skip_metadata=True."""
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "CHAPTER ONE\n"  # All caps, short - should be filtered
            "This is actual content that should be kept.\n"
            "*** End of Project Gutenberg ***\n"  # Starts with *** - should be filtered
            "More real content here.\n"
            "Produced by volunteers\n"  # Starts with "Produced by" - should be filtered
            "[Ilustraĵo: picture]\n"  # Starts with [Ilustraĵo: - should be filtered
            "Final real content.\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=True,
            min_length=10
        )

        # Should only include the 3 real content lines
        self.assertEqual(count, 3)

        # Verify content
        with output_file.open('r', encoding='utf-8') as f:
            lines = [json.loads(line)['text'] for line in f]
            self.assertIn("This is actual content that should be kept.", lines)
            self.assertIn("More real content here.", lines)
            self.assertIn("Final real content.", lines)
            self.assertNotIn("CHAPTER ONE", lines)

    def test_no_metadata_filtering(self):
        """Test that metadata is kept when skip_metadata=False."""
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "CHAPTER ONE\n"
            "Content line\n"
            "*** Metadata ***\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,  # Don't skip metadata
            min_length=10
        )

        # Should include all lines
        self.assertEqual(count, 3)

    def test_multiple_texts(self):
        """Test building corpus from multiple text files."""
        # Create multiple test files
        file1 = self.test_path / "cleaned_book1.txt"
        file1.write_text(
            "Book one, sentence one.\n"
            "Book one, sentence two.\n",
            encoding='utf-8'
        )

        file2 = self.test_path / "cleaned_book2.txt"
        file2.write_text(
            "Book two, sentence one.\n"
            "Book two, sentence two.\n"
            "Book two, sentence three.\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [
            ("cleaned_book1.txt", "Book One"),
            ("cleaned_book2.txt", "Book Two")
        ]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=10
        )

        # Should have 5 total sentences
        self.assertEqual(count, 5)

        # Verify sources are tracked
        with output_file.open('r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]

            # First 2 should be from book1
            self.assertEqual(entries[0]['source'], "book1")
            self.assertEqual(entries[0]['source_name'], "Book One")
            self.assertEqual(entries[1]['source'], "book1")

            # Next 3 should be from book2
            self.assertEqual(entries[2]['source'], "book2")
            self.assertEqual(entries[2]['source_name'], "Book Two")
            self.assertEqual(entries[3]['source'], "book2")
            self.assertEqual(entries[4]['source'], "book2")

    def test_missing_file_handling(self):
        """Test that missing files are handled gracefully."""
        output_file = self.test_path / "corpus.jsonl"
        texts = [
            ("cleaned_missing.txt", "Missing Book")
        ]

        # Should not raise exception, just skip missing file
        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=10
        )

        # Should return 0 since no files were processed
        self.assertEqual(count, 0)

    def test_line_numbers_preserved(self):
        """Test that line numbers are correctly preserved in output."""
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "Line 1\n"
            "\n"  # Line 2 - empty, will be skipped
            "Line 3\n"
            "Line 4\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=5
        )

        # Verify line numbers match original file
        with output_file.open('r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[0]['line'], 1)  # Line 1
            self.assertEqual(entries[1]['line'], 3)  # Line 3 (line 2 was empty)
            self.assertEqual(entries[2]['line'], 4)  # Line 4

    def test_unicode_handling(self):
        """Test that Esperanto Unicode characters are handled correctly."""
        test_file = self.test_path / "cleaned_test.txt"
        test_file.write_text(
            "Mi ŝatas la ĉapelon kun ĝi.\n"
            "La knabo manĝas ĵeladon.\n"
            "Ŭa! Tio estas bonega!\n",
            encoding='utf-8'
        )

        output_file = self.test_path / "corpus.jsonl"
        texts = [("cleaned_test.txt", "Test Book")]

        count = build_corpus(
            self.test_path,
            output_file,
            texts,
            skip_empty=True,
            skip_metadata=False,
            min_length=10
        )

        self.assertEqual(count, 3)

        # Verify Unicode is preserved
        with output_file.open('r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]
            self.assertEqual(entries[0]['text'], "Mi ŝatas la ĉapelon kun ĝi.")
            self.assertIn('ŝ', entries[0]['text'])
            self.assertIn('ĉ', entries[0]['text'])
            self.assertIn('ĝ', entries[0]['text'])


if __name__ == '__main__':
    unittest.main()
