"""
Tests for the from-scratch, pure Python Esperanto Parser.
"""
import unittest
from klareco.parser import parse, parse_word

class TestScratchParser(unittest.TestCase):

    def test_parse_simple_word(self):
        """Tests parsing a simple noun: 'hundo'"""
        ast = parse_word("hundo")
        self.assertEqual(ast['tipo'], 'vorto')
        self.assertEqual(ast['plena_vorto'], 'hundo')
        self.assertEqual(ast['radiko'], 'hund')
        self.assertEqual(ast['vortspeco'], 'substantivo')
        self.assertEqual(ast['nombro'], 'singularo')
        self.assertEqual(ast['kazo'], 'nominativo')

    def test_parse_plural_accusative_adjective(self):
        """Tests a complex adjective: 'grandajn'"""
        ast = parse_word("grandajn")
        self.assertEqual(ast['radiko'], 'grand')
        self.assertEqual(ast['vortspeco'], 'adjektivo')
        self.assertEqual(ast['nombro'], 'pluralo')
        self.assertEqual(ast['kazo'], 'akuzativo')

    def test_parse_word_with_prefix_and_suffix(self):
        """Tests a complex word with multiple morphemes: 'resanigos'"""
        ast = parse_word("resanigos")
        self.assertEqual(ast['radiko'], 'san')
        self.assertEqual(ast['prefikso'], 're')
        self.assertIn('ig', ast['sufiksoj'])
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'futuro')

    def test_unknown_root_fails(self):
        """Tests that a word with an unknown root fails parsing."""
        with self.assertRaises(ValueError):
            parse_word("nekonataradiko") # "nekonataradiko" is not a known root

    def test_parse_simple_sentence(self):
        """Tests parsing a full, simple sentence."""
        text = "mi amas la grandan katon"
        ast = parse(text)

        # Overall structure
        self.assertEqual(ast['tipo'], 'frazo')
        self.assertIsNotNone(ast['subjekto'])
        self.assertIsNotNone(ast['verbo'])
        self.assertIsNotNone(ast['objekto'])

        # Subject: "mi"
        subjekto = ast['subjekto']['kerno']
        self.assertEqual(subjekto['radiko'], 'mi')
        self.assertEqual(subjekto['vortspeco'], 'pronomo')

        # Verb: "amas"
        verbo = ast['verbo']
        self.assertEqual(verbo['radiko'], 'am')
        self.assertEqual(verbo['tempo'], 'prezenco')

        # Object: "la grandan katon"
        objekto_kerno = ast['objekto']['kerno']
        self.assertEqual(objekto_kerno['radiko'], 'kat')
        self.assertEqual(objekto_kerno['kazo'], 'akuzativo')
        
        objekto_priskribo = ast['objekto']['priskriboj'][0]
        self.assertEqual(objekto_priskribo['radiko'], 'grand')
        self.assertEqual(objekto_priskribo['kazo'], 'akuzativo')


if __name__ == '__main__':
    unittest.main()
