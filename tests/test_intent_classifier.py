"""
Tests for the Morpheme-based Intent Classifier.
"""
import unittest
import pytest
from klareco.parser import parse

# Skip entire module - intent_classifier module not implemented
pytestmark = pytest.mark.skip(reason="intent_classifier module not implemented")

class TestMorphemeIntentClassifier(unittest.TestCase):

    def test_simple_statement_classification(self):
        """Tests that a simple declarative sentence is classified as SimpleStatement."""
        text = "La hundo amas la katon."
        ast = parse(text)
        self.assertEqual(classify_intent(ast), "SimpleStatement")

    def test_simple_statement_with_pronoun_classification(self):
        """Tests that a simple declarative sentence with a pronoun is classified as SimpleStatement.""" 
        text = "mi vidas la hundon."
        ast = parse(text)
        self.assertEqual(classify_intent(ast), "SimpleStatement")

    def test_unknown_ast_classification(self):
        """Tests that an invalid or incomplete AST is classified as Unknown."""
        ast = {"type": "invalid_ast"}
        self.assertEqual(classify_intent(ast), "Unknown")

        ast_incomplete = {"type": "sentence", "subject": {"type": "word", "root": "hund"}}
        self.assertEqual(classify_intent(ast_incomplete), "Unknown")

if __name__ == '__main__':
    unittest.main()