"""
Tests for the from-scratch, pure Python Esperanto De-parser.
"""
import unittest
from klareco.parser import parse
from klareco.deparser import deparse, _reconstruct_word

class TestScratchDeparser(unittest.TestCase):

    def test_reconstruct_simple_word(self):
        """Tests reconstructing a simple word from the new AST format."""
        ast = {
            "tipo": "vorto",
            "radiko": "hund",
            "vortspeco": "substantivo",
            "nombro": "singularo",
            "kazo": "nominativo"
        }
        self.assertEqual(_reconstruct_word(ast), "hundo")

    def test_reconstruct_complex_adjective(self):
        """Tests reconstructing a plural, accusative adjective."""
        ast = {
            "tipo": "vorto",
            "radiko": "grand",
            "vortspeco": "adjektivo",
            "nombro": "pluralo",
            "kazo": "akuzativo"
        }
        self.assertEqual(_reconstruct_word(ast), "grandajn")

    def test_reconstruct_word_with_affixes(self):
        """Tests reconstructing a word with both prefix and suffix."""
        ast = {
            "tipo": "vorto",
            "radiko": "san",
            "prefikso": "re",
            "sufiksoj": ["ig"],
            "vortspeco": "verbo",
            "tempo": "futuro"
        }
        self.assertEqual(_reconstruct_word(ast), "resanigos")

    def test_round_trip_simple_sentence(self):
        """
        Tests that parsing and then deparsing a sentence returns the original.
        This is the ultimate integration test for the parser/deparser pair.
        """
        original_sentence = "Mi amas la grandan katon."
        
        # 1. Parse the sentence into our new AST format
        ast = parse(original_sentence)
        
        # 2. Deparse the AST back into a string
        reconstructed_sentence = deparse(ast)
        
        # 3. Compare the result (case-insensitively, ignoring final punctuation)
        self.assertEqual(
            reconstructed_sentence.lower().strip('.'),
            original_sentence.lower().strip('.')
        )

    def test_round_trip_complex_sentence(self):
        """Tests a more complex round-trip operation."""
        original_sentence = "Malgrandaj hundoj vidas la grandan katon."
        ast = parse(original_sentence)
        reconstructed_sentence = deparse(ast)
        self.assertEqual(
            reconstructed_sentence.lower().strip('.'),
            original_sentence.lower().strip('.')
        )


if __name__ == '__main__':
    unittest.main()
