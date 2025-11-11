"""
Tests for the Morpheme-based Responder.
"""
import unittest
from klareco.responder import respond_to_intent, respond_simple_statement
from klareco.parser import parse
from klareco.deparser import deparse # To verify deparser output for expected responses

class TestMorphemeResponder(unittest.TestCase):

    def test_respond_simple_statement(self):
        """Tests that a simple statement intent generates an appropriate response."""
        text = "La hundo amas la katon."
        ast = parse(text)
        expected_response = f"Vi diras, ke {deparse(ast).lower()}."
        # Remove the trailing period from the deparsed sentence if it's already there
        if expected_response.endswith('..'):
            expected_response = expected_response[:-1]
        self.assertEqual(respond_to_intent("SimpleStatement", ast), expected_response)

    def test_respond_simple_statement_with_pronoun(self):
        """Tests that a simple statement with a pronoun generates an appropriate response."""
        text = "mi vidas la hundon."
        ast = parse(text)
        expected_response = f"Vi diras, ke {deparse(ast).lower()}."
        if expected_response.endswith('..'):
            expected_response = expected_response[:-1]
        self.assertEqual(respond_to_intent("SimpleStatement", ast), expected_response)

    def test_respond_unknown_intent(self):
        """Tests that an unknown intent generates the default unknown response."""
        ast = {"type": "invalid_ast"}
        self.assertEqual(respond_to_intent("UnknownIntent", ast), "Mi ne komprenas vian intencon.")

    def test_respond_simple_statement_deparser_error(self):
        """Tests that a deparser error during response generation is handled gracefully."""
        # Create an AST that will cause a deparser error (e.g., missing subject)
        invalid_ast = {"type": "sentence", "verb": {"type": "word", "root": "am", "endings": ["as"]}}
        response = respond_to_intent("SimpleStatement", invalid_ast)
        self.assertTrue("Mi ne povis respondi al via deklaro pro eraro" in response)

if __name__ == '__main__':
    unittest.main()