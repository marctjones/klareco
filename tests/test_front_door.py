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

if __name__ == '__main__':
    unittest.main()
