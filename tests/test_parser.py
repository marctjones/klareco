"""
Tests for the Morpheme-based Lark Parser.
"""
import unittest
from klareco.parser import parse

class TestMorphemeParser(unittest.TestCase):

    def test_simple_sentence_decomposition(self):
        """Tests a simple sentence is decomposed into a morpheme-based AST."""
        text = "hundo amas katon."
        ast = parse(text)

        # Check overall structure
        self.assertEqual(ast['type'], 'sentence')
        
        # Check subject: "hundo"
        subject = ast['subject']
        self.assertEqual(subject['type'], 'word')
        self.assertEqual(subject['root'], 'hund')
        self.assertEqual(subject['endings'], ['o'])
        
        # Check verb: "amas"
        verb = ast['verb']
        self.assertEqual(verb['type'], 'word')
        self.assertEqual(verb['root'], 'am')
        self.assertEqual(verb['endings'], ['as'])

        # Check object: "katon"
        obj = ast['object']
        self.assertEqual(obj['type'], 'word')
        self.assertEqual(obj['root'], 'kat')
        self.assertEqual(obj['endings'], ['o', 'n'])

    def test_sentence_with_prefix_and_suffix(self):
        """Tests a sentence with more complex words."""
        # This will require expanding the grammar's vocabulary
        # For now, this is a placeholder for a future test.
        pass

    def test_invalid_word_fails(self):
        """Tests that a word with an unknown root fails to parse."""
        text = "la homo amas la katon." # "homo" is not in our root vocabulary
        with self.assertRaises(Exception):
            parse(text)

if __name__ == '__main__':
    unittest.main()