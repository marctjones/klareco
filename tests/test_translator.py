"""
Tests for the translation service.
"""
import unittest
from klareco.translator import TranslationService

class TestTranslationService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the translator service once for all tests."""
        cls.translator = TranslationService()

    def test_en_to_eo_translation(self):
        """Tests English to Esperanto translation."""
        text = "My name is John."
        expected = "Mia nomo estas John."
        translation = self.translator.translate(text, 'en', 'eo')
        self.assertEqual(translation, expected)

    def test_eo_to_en_translation(self):
        """Tests Esperanto to English translation."""
        text = "Mi ŝatas programadon."
        expected = "I like programming."
        translation = self.translator.translate(text, 'eo', 'en')
        self.assertEqual(translation, expected)

    def test_fr_to_en_translation(self):
        """Tests French to English translation."""
        text = "Je suis un étudiant."
        possible_translations = ["I am a student.", "I'm a student."]
        translation = self.translator.translate(text, 'fr', 'en')
        self.assertIn(translation, possible_translations)

    def test_empty_string(self):
        """Tests translation of an empty string."""
        text = ""
        translation = self.translator.translate(text, 'en', 'eo')
        self.assertEqual(translation, "")

    def test_whitespace_string(self):
        """Tests translation of a string with only whitespace."""
        text = "   \t\n"
        translation = self.translator.translate(text, 'en', 'eo')
        # The model might return an empty string or whitespace
        self.assertIsInstance(translation, str)

    def test_non_string_input(self):
        """Tests that non-string input raises a TypeError."""
        with self.assertRaises(TypeError):
            self.translator.translate(12345, 'en', 'eo')
        with self.assertRaises(TypeError):
            self.translator.translate(None, 'en', 'eo')

    def test_invalid_language_model(self):
        """Tests that an invalid language pair raises a ValueError."""
        with self.assertRaises(ValueError):
            self.translator.translate("test", "en", "invalidlang")

if __name__ == '__main__':
    unittest.main()
