"""
Tests for the language identification service.
"""
import unittest
from klareco.lang_id import identify_language

class TestLangId(unittest.TestCase):

    def test_identify_english(self):
        """Tests identification of English text."""
        text = "This is a test sentence in English."
        self.assertEqual(identify_language(text), "en")

    def test_identify_esperanto(self):
        """Tests identification of Esperanto text."""
        text = "Ĉi tio estas testa frazo en Esperanto."
        self.assertEqual(identify_language(text), "eo")

    def test_identify_french(self):
        """Tests identification of French text."""
        text = "Ceci est une phrase de test en français."
        self.assertEqual(identify_language(text), "fr")

    def test_identify_spanish(self):
        """Tests identification of Spanish text."""
        text = "Esta es una oración de prueba en español."
        self.assertEqual(identify_language(text), "es")

    def test_unreliable_short_text(self):
        """Tests that very short text might not be identified reliably."""
        # langdetect is not reliable for very short strings.
        # This test acknowledges that it might return various languages or fail.
        text = "Hi"
        lang = identify_language(text)
        self.assertIsInstance(lang, (str, type(None)))

    def test_empty_string(self):
        """Tests that an empty string returns None."""
        text = ""
        self.assertIsNone(identify_language(text))

    def test_whitespace_string(self):
        """Tests that a string with only whitespace returns None."""
        text = "   \t\n"
        self.assertIsNone(identify_language(text))

    def test_non_string_input(self):
        """Tests that non-string input raises a TypeError."""
        with self.assertRaises(TypeError):
            identify_language(12345)
        with self.assertRaises(TypeError):
            identify_language(None)


if __name__ == '__main__':
    unittest.main()
