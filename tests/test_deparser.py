"""
Tests for the Morpheme-based De-parser.
"""
import unittest
from klareco.deparser import deparse, _reconstruct_word
from klareco.parser import parse

class TestMorphemeDeparser(unittest.TestCase):

    def test_reconstruct_simple_word(self):
        """Tests reconstruction of a simple word (root + ending)."""
        word_ast = {"type": "word", "root": "hund", "endings": ["o"], "suffixes": []}
        self.assertEqual(_reconstruct_word(word_ast), "hundo")

    def test_reconstruct_word_with_accusative(self):
        """Tests reconstruction of a word with accusative ending."""
        word_ast = {"type": "word", "root": "kat", "endings": ["o", "n"], "suffixes": []}
        self.assertEqual(_reconstruct_word(word_ast), "katon")

    def test_reconstruct_word_with_prefix(self):
        """Tests reconstruction of a word with a prefix."""
        word_ast = {"type": "word", "prefix": "mal", "root": "san", "endings": ["a"], "suffixes": []}
        self.assertEqual(_reconstruct_word(word_ast), "malsana")

    def test_reconstruct_word_with_suffix(self):
        """Tests reconstruction of a word with a suffix."""
        word_ast = {"type": "word", "root": "san", "suffixes": ["ul"], "endings": ["o"], "prefix": ''}
        self.assertEqual(_reconstruct_word(word_ast), "sanulo")

    def test_deparse_simple_sentence(self):
        """Tests deparsing a simple sentence parsed by the new parser."""
        original_text = "La hundo amas la katon."
        ast = parse(original_text)
        deparsed_text = deparse(ast)
        self.assertEqual(deparsed_text, "La hundo amas la katon.")

    def test_deparse_sentence_with_pronoun_subject(self):
        """Tests deparsing a sentence with a pronoun subject."""
        original_text = "mi vidas la hundon."
        ast = parse(original_text)
        deparsed_text = deparse(ast)
        self.assertEqual(deparsed_text, "mi vidas la hundon.")

if __name__ == '__main__':
    unittest.main()