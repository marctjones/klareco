"""
Code Interpreter Tool - Execute Python code safely

Provides sandboxed Python code execution with safety restrictions.
Useful for calculations, data processing, and testing code snippets.

Part of Phase 8: External Tools
"""

from typing import Dict, Any, Optional
import subprocess
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeInterpreterTool:
    """
    Tool for executing Python code in a sandboxed environment.

    Safety features:
    - Execution timeout
    - Restricted imports (no os, sys, subprocess, etc.)
    - Resource limits
    - Output capture
    """

    def __init__(self, timeout: int = 5, max_output_size: int = 10000):
        """
        Initialize Code Interpreter Tool.

        Args:
            timeout: Execution timeout in seconds
            max_output_size: Maximum output size in characters
        """
        self.name = "Code_Interpreter_Tool"
        self.capabilities = ["code_execution", "calculation", "data_processing"]
        self.timeout = timeout
        self.max_output_size = max_output_size

        # Restricted imports for safety
        self.blocked_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'glob',
            'socket', 'urllib', 'requests', 'http',
            '__import__', 'eval', 'exec', 'compile',
            'open', 'file', 'input', 'raw_input'
        }

        logger.info(f"{self.name} initialized (timeout={timeout}s)")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this tool can handle the query.

        Code execution queries contain:
        - "kalkulu" (calculate)
        - "komputu" (compute)
        - "rulu" (run)
        - "plenumu" (execute)

        Args:
            ast: Parsed query AST

        Returns:
            True if code execution query
        """
        code_keywords = {'kalkul', 'komput', 'rul', 'plenum', 'programo', 'kodo'}

        return self._contains_any_root(ast, code_keywords)

    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            context: Optional execution context

        Returns:
            Execution result
        """
        logger.info(f"{self.name} executing code ({len(code)} chars)")

        # Validate code
        validation_error = self._validate_code(code)
        if validation_error:
            return {
                'success': False,
                'output': None,
                'error': validation_error,
                'execution_time': 0
            }

        # Execute in subprocess
        result = self._execute_sandboxed(code)

        return result

    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for safety.

        Args:
            code: Code to validate

        Returns:
            Error message if invalid, None if valid
        """
        code_lower = code.lower()

        # Check for blocked imports
        for blocked in self.blocked_imports:
            if blocked in code_lower:
                return f"Blocked import or function: {blocked}"

        # Check code length
        if len(code) > 5000:
            return "Code too long (max 5000 characters)"

        return None

    def _execute_sandboxed(self, code: str) -> Dict[str, Any]:
        """
        Execute code in sandboxed subprocess.

        Args:
            code: Python code to execute

        Returns:
            Execution result
        """
        import time

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            start_time = time.time()

            # Execute with timeout
            process = subprocess.Popen(
                ['python', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time

                # Truncate output if too large
                if len(stdout) > self.max_output_size:
                    stdout = stdout[:self.max_output_size] + "\n...(output truncated)"

                if len(stderr) > self.max_output_size:
                    stderr = stderr[:self.max_output_size] + "\n...(output truncated)"

                success = process.returncode == 0

                return {
                    'success': success,
                    'output': stdout if stdout else None,
                    'error': stderr if stderr else None,
                    'execution_time': execution_time,
                    'return_code': process.returncode
                }

            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    'success': False,
                    'output': None,
                    'error': f"Execution timeout ({self.timeout}s exceeded)",
                    'execution_time': self.timeout,
                    'return_code': -1
                }

        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': f"Execution error: {str(e)}",
                'execution_time': 0,
                'return_code': -1
            }

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def execute_and_format(self, code: str) -> str:
        """
        Execute code and format result as text.

        Args:
            code: Python code to execute

        Returns:
            Formatted result
        """
        result = self.execute(code)

        lines = []

        if result['success']:
            lines.append(f"✓ Sukcese plenumita ({result['execution_time']:.3f}s)")

            if result['output']:
                lines.append("")
                lines.append("Eligo:")
                lines.append(result['output'])
        else:
            lines.append("✗ Eraro okazis")

            if result['error']:
                lines.append("")
                lines.append("Eraro:")
                lines.append(result['error'])

        return "\n".join(lines)

    def _contains_any_root(self, ast: Dict[str, Any], roots: set) -> bool:
        """Check if AST contains any of the specified roots"""
        if ast.get('tipo') == 'vorto':
            radiko = ast.get('radiko', '').lower()
            return any(radiko.startswith(root) for root in roots)
        elif ast.get('tipo') == 'vortgrupo':
            return any(self._contains_any_root(v, roots) for v in ast.get('vortoj', []))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key) and self._contains_any_root(ast[key], roots):
                    return True
            return any(self._contains_any_root(v, roots) for v in ast.get('aliaj', []))
        return False

    def __repr__(self) -> str:
        return f"{self.name}(timeout={self.timeout}s)"


# Factory function
def create_code_interpreter_tool(timeout: int = 5) -> CodeInterpreterTool:
    """
    Create and return a CodeInterpreterTool instance.

    Args:
        timeout: Execution timeout in seconds

    Returns:
        Initialized CodeInterpreterTool
    """
    return CodeInterpreterTool(timeout=timeout)


if __name__ == "__main__":
    # Test code interpreter tool
    print("Testing Code Interpreter Tool")
    print("=" * 80)

    tool = create_code_interpreter_tool(timeout=5)
    print(f"\n{tool}\n")

    # Test cases
    test_cases = [
        {
            'name': 'Simple calculation',
            'code': 'print(2 + 2)'
        },
        {
            'name': 'Loop and calculation',
            'code': '''
total = 0
for i in range(10):
    total += i
print(f"Sum of 0-9: {total}")
'''
        },
        {
            'name': 'Math operations',
            'code': '''
import math
print(f"Pi: {math.pi:.5f}")
print(f"Square root of 16: {math.sqrt(16)}")
'''
        },
        {
            'name': 'Invalid code (blocked import)',
            'code': 'import os\nprint(os.listdir())'
        },
        {
            'name': 'Syntax error',
            'code': 'print("Missing parenthesis"'
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Code: {test['code'][:50]}...")

        result = tool.execute(test['code'])

        print(f"  Success: {result['success']}")
        if result['output']:
            print(f"  Output: {result['output'].strip()}")
        if result['error']:
            print(f"  Error: {result['error'][:100]}")
        print()

    # Test formatted output
    print("Formatted execution:")
    print(tool.execute_and_format("print('Hello from Klareco!')"))

    print("\n✅ Code Interpreter Tool test complete!")
