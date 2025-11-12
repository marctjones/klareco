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

class TestSafetyMonitorEdgeCases(unittest.TestCase):
    """Test suite for SafetyMonitor edge cases."""

    def setUp(self):
        self.monitor = SafetyMonitor()

    def test_empty_string_passes_length_check(self):
        """Test that empty string passes length check."""
        self.monitor.check_input_length("")
        # Should not raise exception

    def test_exact_max_length_passes(self):
        """Test that input at exactly max length passes."""
        self.monitor.max_input_length = 10
        self.monitor.check_input_length("a" * 10)
        # Should not raise exception

    def test_one_over_max_length_fails(self):
        """Test that input one character over max fails."""
        self.monitor.max_input_length = 10
        with self.assertRaises(ValueError):
            self.monitor.check_input_length("a" * 11)

    def test_very_long_input_fails(self):
        """Test that very long input is rejected."""
        self.monitor.max_input_length = 100
        with self.assertRaises(ValueError):
            self.monitor.check_input_length("a" * 1000)

    def test_unicode_characters_count_correctly(self):
        """Test that Unicode characters are counted correctly."""
        # Esperanto has special characters like ĉ, ĝ, ĥ, ĵ, ŝ, ŭ
        text = "Ĉu vi parolas Esperanton?"
        self.monitor.max_input_length = len(text)
        self.monitor.check_input_length(text)
        # Should not raise exception

    def test_newlines_in_input(self):
        """Test handling of newlines in input."""
        text = "Line 1\nLine 2\nLine 3"
        self.monitor.max_input_length = len(text)
        self.monitor.check_input_length(text)
        # Should not raise exception

    def test_tabs_and_spaces(self):
        """Test handling of tabs and spaces."""
        text = "Word1\t\tWord2    Word3"
        self.monitor.max_input_length = len(text)
        self.monitor.check_input_length(text)
        # Should not raise exception


class TestASTComplexityChecks(unittest.TestCase):
    """Test suite for AST complexity checking."""

    def setUp(self):
        self.monitor = SafetyMonitor()

    def test_minimal_ast_passes(self):
        """Test that minimal AST passes complexity check."""
        minimal_ast = {'tipo': 'vorto'}
        self.monitor.max_ast_nodes = 10
        self.monitor.check_ast_complexity(minimal_ast)
        # Should not raise exception

    def test_empty_ast_passes(self):
        """Test that empty AST passes."""
        self.monitor.check_ast_complexity({})
        # Should not raise exception

    def test_ast_with_lists(self):
        """Test AST complexity calculation with lists."""
        ast = {
            'tipo': 'frazo',
            'aliaj': [
                {'tipo': 'vorto'},
                {'tipo': 'vorto'},
                {'tipo': 'vorto'}
            ]
        }
        node_count = self.monitor._count_ast_nodes(ast)
        # 1 (root) + 1 (list) + 3 (words) = 5
        self.assertGreater(node_count, 0)

    def test_deeply_nested_ast(self):
        """Test deeply nested AST structure."""
        # Create a deeply nested structure
        deep_ast = {'level': 0}
        current = deep_ast
        for i in range(5):
            current['child'] = {'level': i + 1}
            current = current['child']

        node_count = self.monitor._count_ast_nodes(deep_ast)
        self.assertGreater(node_count, 5)

    def test_ast_with_none_values(self):
        """Test AST with None values."""
        ast = {
            'tipo': 'frazo',
            'subjekto': None,
            'verbo': {'radiko': 'vid'},
            'objekto': None
        }
        node_count = self.monitor._count_ast_nodes(ast)
        self.assertGreater(node_count, 0)

    def test_count_ast_nodes_with_mixed_types(self):
        """Test node counting with mixed data types."""
        ast = {
            'string': 'value',
            'number': 42,
            'boolean': True,
            'none': None,
            'dict': {'nested': 'value'},
            'list': [1, 2, 3]
        }
        node_count = self.monitor._count_ast_nodes(ast)
        self.assertGreater(node_count, 0)


class TestSecurityEdgeCases(unittest.TestCase):
    """Test suite for security-related edge cases."""

    def setUp(self):
        self.monitor = SafetyMonitor()

    def test_special_characters_in_input(self):
        """Test handling of special characters."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:',.<>?/"
        self.monitor.check_input_length(special_chars)
        # Should not raise exception

    def test_only_whitespace(self):
        """Test input with only whitespace."""
        self.monitor.check_input_length("   \t\n   ")
        # Should not raise exception

    def test_emoji_in_input(self):
        """Test handling of emoji characters."""
        text_with_emoji = "Hello 👋 world 🌍"
        self.monitor.max_input_length = len(text_with_emoji)
        self.monitor.check_input_length(text_with_emoji)
        # Should not raise exception

    def test_null_byte_in_input(self):
        """Test handling of null bytes."""
        text_with_null = "text\x00with\x00nulls"
        self.monitor.max_input_length = len(text_with_null)
        self.monitor.check_input_length(text_with_null)
        # Should not raise exception


class TestDefaultLimits(unittest.TestCase):
    """Test suite for default safety limits."""

    def test_default_max_input_length(self):
        """Test that default max input length is reasonable."""
        monitor = SafetyMonitor()
        self.assertIsInstance(monitor.max_input_length, int)
        self.assertGreater(monitor.max_input_length, 0)

    def test_default_max_ast_nodes(self):
        """Test that default max AST nodes is reasonable."""
        monitor = SafetyMonitor()
        self.assertIsInstance(monitor.max_ast_nodes, int)
        self.assertGreater(monitor.max_ast_nodes, 0)

    def test_custom_limits_accepted(self):
        """Test that custom limits can be set."""
        monitor = SafetyMonitor(max_input_length=100, max_ast_nodes=50)
        self.assertEqual(monitor.max_input_length, 100)
        self.assertEqual(monitor.max_ast_nodes, 50)


if __name__ == '__main__':
    unittest.main()