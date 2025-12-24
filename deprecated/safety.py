"""
The Safety and Integrity Monitor.

This module provides pragmatic checks to ensure the system operates within
defined computational limits.
"""

class SafetyMonitor:
    """
    Enforces safety policies on inputs and internal states.
    """
    def __init__(self, max_input_length: int = 2048, max_ast_nodes: int = 50):
        self.max_input_length = max_input_length
        self.max_ast_nodes = max_ast_nodes

    def check_input_length(self, text: str):
        """
        Checks if the input text exceeds the maximum allowed length.

        Raises:
            ValueError: If the text is too long.
        """
        if len(text) > self.max_input_length:
            raise ValueError(
                f"Input text length ({len(text)}) exceeds maximum of {self.max_input_length} characters."
            )

    def _count_ast_nodes(self, ast_node) -> int:
        """
        Recursively counts the number of nodes in an AST.
        A 'node' is defined as any dictionary or list in the tree.
        """
        count = 0
        if isinstance(ast_node, dict):
            count += 1  # Count the dictionary itself as a node
            for key, value in ast_node.items():
                count += self._count_ast_nodes(value)
        elif isinstance(ast_node, list):
            count += 1 # Count the list itself as a node
            for item in ast_node:
                count += self._count_ast_nodes(item)
        # Do not count other types (like strings, integers, booleans) as nodes
        return count

    def check_ast_complexity(self, ast: dict):
        """
        Checks if the AST exceeds the maximum allowed complexity (number of nodes).

        Raises:
            ValueError: If the AST is too complex.
        """
        node_count = self._count_ast_nodes(ast)
        if node_count > self.max_ast_nodes:
            raise ValueError(
                f"AST complexity ({node_count} nodes) exceeds maximum of {self.max_ast_nodes}."
            )

if __name__ == '__main__':
    # Example Usage
    monitor = SafetyMonitor(max_input_length=50, max_ast_nodes=10)

    # --- Input Length Check ---
    good_input = "This is a reasonable length input."
    bad_input = "This input is definitely far too long and should fail the safety check."
    
    print("--- Input Length Checks ---")
    try:
        monitor.check_input_length(good_input)
        print(f"'{good_input}' -> OK")
    except ValueError as e:
        print(f"'{good_input}' -> FAILED: {e}")

    try:
        monitor.check_input_length(bad_input)
        print(f"'{bad_input}' -> OK")
    except ValueError as e:
        print(f"'{bad_input}' -> FAILED: {e}")

    # --- AST Complexity Check ---
    simple_ast = {"type": "sentence", "subject": "mi", "verb": "amas", "object": "vin"}
    complex_ast = {
        "type": "sentence", "subject": "a", "verb": "b", "object": "c",
        "clause": {"type": "sub-clause", "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    }

    print("\n--- AST Complexity Checks ---")
    try:
        monitor.check_ast_complexity(simple_ast)
        print(f"Simple AST -> OK (Nodes: {monitor._count_ast_nodes(simple_ast)})")
    except ValueError as e:
        print(f"Simple AST -> FAILED: {e}")

    try:
        monitor.check_ast_complexity(complex_ast)
        print(f"Complex AST -> OK (Nodes: {monitor._count_ast_nodes(complex_ast)})")
    except ValueError as e:
        print(f"Complex AST -> FAILED: {e}")

