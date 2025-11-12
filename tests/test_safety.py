"""
Tests for the SafetyMonitor.
"""
import unittest
from klareco.safety import SafetyMonitor
from klareco.parser import parse

class TestSafetyMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = SafetyMonitor(max_input_length=50, max_ast_nodes=100) # Default for tests

    def test_check_input_length_ok(self):
        """Tests that input within limits passes the length check."""
        self.monitor.check_input_length("short text")
        # No exception means success

    def test_check_input_length_fail(self):
        """Tests that input exceeding limits fails the length check."""
        # Using raw string (r"...") to properly escape regex special characters
        with self.assertRaisesRegex(ValueError, r"Input text length \(.*\) exceeds maximum"):
            self.monitor.check_input_length("a" * 51)

    def test_count_ast_nodes_morpheme_ast(self):
        """Tests node counting for a deep, morpheme-based AST."""
        text = "La hundo amas la katon."
        ast = parse(text)
        # Let's trace _count_ast_nodes(ast) for the example AST:
        # ast (dict) -> count = 1
        #   type (str) -> 0
        #   subject (dict) -> count = 1
        #     type (str) -> 0
        #     suffixes (list) -> count = 1
        #     endings (list) -> count = 1
        #       'o' (str) -> 0
        #     root (str) -> 0
        #   verb (dict) -> count = 1
        #     type (str) -> 0
        #     suffixes (list) -> count = 1
        #     endings (list) -> count = 1
        #       'as' (str) -> 0
        #     root (str) -> 0
        #   object (dict) -> count = 1
        #     type (str) -> 0
        #     suffixes (list) -> count = 1
        #     endings (list) -> count = 1
        #       'o' (str) -> 0
        #       'n' (str) -> 0
        #     root (str) -> 0
        # Total: 1 (sentence) + 3 * (1 (word dict) + 1 (suffixes list) + 1 (endings list)) = 1 + 3 * 3 = 10
        # Updated: Parser now includes "artikolo" fields, adding 2 more nodes = 12 total
        # Further updated: Parser now includes "parse_statistics" dict, adding 2 more nodes = 14 total
        self.assertEqual(self.monitor._count_ast_nodes(ast), 14)

    def test_check_ast_complexity_ok(self):
        """Tests that an AST within limits passes the complexity check."""
        text = "mi vidas la hundon."
        ast = parse(text)
        # Updated limit to accommodate article tracking and parse_statistics in AST
        self.monitor.max_ast_nodes = 14  # Set to accommodate improved AST structure with parse_statistics
        self.monitor.check_ast_complexity(ast)
        # No exception means success

    def test_check_ast_complexity_fail(self):
        """Tests that an AST exceeding limits fails the complexity check."""
        text = "La hundo amas la katon."
        ast = parse(text)
        self.monitor.max_ast_nodes = 11 # Set just below the expected count (12)
        # Using raw string (r"...") to properly escape regex special characters
        with self.assertRaisesRegex(ValueError, r"AST complexity \(.*\) exceeds maximum"):
            self.monitor.check_ast_complexity(ast)

if __name__ == '__main__':
    unittest.main()