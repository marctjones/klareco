"""
Tests for the FrontDoor service.
"""
import unittest
from klareco.front_door import FrontDoor

class TestFrontDoor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the FrontDoor service once for all tests."""
        cls.front_door = FrontDoor()

    def test_process_esperanto(self):
        """Tests processing of text that is already in Esperanto."""
        text = "Ĉi tio estas jam en Esperanto."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "eo")
        self.assertEqual(processed_text, text)

    def test_process_english(self):
        """Tests processing of English text."""
        text = "This is a sentence in English."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "en")
        # Translation can vary, so check for key components
        self.assertIn("frazo", processed_text)
        self.assertIn("angla", processed_text)

    def test_process_french(self):
        """Tests processing of French text."""
        text = "Ceci est une phrase en français."
        expected_translation = "Ĉi tiu estas frazo en la franca."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "fr")
        # The translation might vary slightly, so we check for key words.
        self.assertIn("frazo", processed_text)
        self.assertIn("franca", processed_text)

    def test_process_unidentifiable_text(self):
        """Tests that unidentifiable text raises a ValueError."""
        text = "12345"
        with self.assertRaises(ValueError):
            self.front_door.process(text)

    def test_process_non_string_input(self):
        """Tests that non-string input raises a TypeError."""
        with self.assertRaises(TypeError):
            self.front_door.process(123)
        with self.assertRaises(TypeError):
            self.front_door.process(None)

class TestFrontDoorMultipleLanguages(unittest.TestCase):
    """Test suite for multi-language support."""

    @classmethod
    def setUpClass(cls):
        """Set up the FrontDoor service once for all tests."""
        cls.front_door = FrontDoor()

    def test_process_spanish(self):
        """Test processing of Spanish text."""
        text = "Esta es una oración en español."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "es")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_german(self):
        """Test processing of German text."""
        text = "Dies ist ein Satz auf Deutsch."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "de")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_italian(self):
        """Test processing of Italian text."""
        text = "Questa è una frase in italiano."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "it")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_portuguese(self):
        """Test processing of Portuguese text."""
        text = "Esta é uma frase em português."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "pt")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_russian(self):
        """Test processing of Russian text."""
        text = "Это предложение на русском языке."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "ru")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_polish(self):
        """Test processing of Polish text."""
        text = "To jest zdanie po polsku."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "pl")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)

    def test_process_dutch(self):
        """Test processing of Dutch text."""
        text = "Dit is een zin in het Nederlands."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "nl")
        # Should be translated to Esperanto
        self.assertIsNotNone(processed_text)


class TestFrontDoorEdgeCases(unittest.TestCase):
    """Test suite for edge cases."""

    @classmethod
    def setUpClass(cls):
        """Set up the FrontDoor service once for all tests."""
        cls.front_door = FrontDoor()

    def test_process_empty_string(self):
        """Test processing of empty string."""
        with self.assertRaises(ValueError):
            self.front_door.process("")

    def test_process_whitespace_only(self):
        """Test processing of whitespace-only input."""
        with self.assertRaises(ValueError):
            self.front_door.process("   \t\n   ")

    def test_process_very_short_text(self):
        """Test processing of very short text."""
        text = "Hi"
        # Should handle short text gracefully
        try:
            lang, processed_text = self.front_door.process(text)
            self.assertIsNotNone(lang)
        except ValueError:
            # Acceptable if too short to identify
            pass

    def test_process_mixed_case(self):
        """Test processing of mixed case text."""
        text = "ThIs Is MiXeD CaSe TeXt In EnGlIsH."
        lang, processed_text = self.front_door.process(text)
        # Should still identify language correctly
        self.assertEqual(lang, "en")

    def test_process_with_numbers(self):
        """Test processing of text with numbers."""
        text = "I have 3 cats and 2 dogs."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "en")

    def test_process_with_punctuation(self):
        """Test processing of text with extensive punctuation."""
        text = "Hello! How are you? I'm fine, thanks."
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "en")

    def test_process_unicode_characters(self):
        """Test processing of text with Unicode characters."""
        # Esperanto with circumflexes
        text = "Ĉu vi ŝatas ĝin?"
        lang, processed_text = self.front_door.process(text)
        self.assertEqual(lang, "eo")
        self.assertEqual(processed_text, text)

    def test_process_esperanto_without_diacritics(self):
        """Test processing of Esperanto using h-system or x-system."""
        # Using h-system: ch, gh, hh, jh, sh, u
        text = "Chu vi shatas ghin?"
        lang, processed_text = self.front_door.process(text)
        # Might be detected as English or other language
        self.assertIsNotNone(lang)


class TestFrontDoorSpecialCases(unittest.TestCase):
    """Test suite for special cases."""

    @classmethod
    def setUpClass(cls):
        """Set up the FrontDoor service once for all tests."""
        cls.front_door = FrontDoor()

    def test_process_code_like_text(self):
        """Test processing of code-like text."""
        # Should fail or handle gracefully
        try:
            lang, processed_text = self.front_door.process("def hello(): print('world')")
            # If it processes, just verify it returns something
            self.assertIsNotNone(lang)
        except ValueError:
            # Acceptable if unidentifiable
            pass

    def test_process_urls(self):
        """Test processing of URLs."""
        try:
            lang, processed_text = self.front_door.process("https://www.example.com")
            self.assertIsNotNone(lang)
        except ValueError:
            # Acceptable if unidentifiable
            pass

    def test_process_multiple_languages_mixed(self):
        """Test processing of mixed language text."""
        text = "Hello, bonjour, hola, ciao!"
        lang, processed_text = self.front_door.process(text)
        # Should detect one language (likely the dominant one)
        self.assertIsNotNone(lang)


if __name__ == '__main__':
    unittest.main()
