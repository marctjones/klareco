"""
Math Tool Expert - Handles mathematical computation queries.

This expert can:
- Detect mathematical expressions in queries
- Parse numbers and operators from AST
- Perform symbolic computation
- Return precise numerical results

Examples:
- "Kiom estas du plus tri?" → 5
- "Kio estas la kvadrata radiko de dek ses?" → 4
- "Dividu dek ok per tri" → 6
"""

import re
from typing import Dict, Any, Optional, List
from .base import Expert


class MathExpert(Expert):
    """
    Expert for handling mathematical computations.

    Uses symbolic AST analysis to detect mathematical intent and
    extract operands/operators for precise calculation.
    """

    # Esperanto number words mapping
    NUMBERS = {
        'nul': 0, 'unu': 1, 'du': 2, 'tri': 3, 'kvar': 4,
        'kvin': 5, 'ses': 6, 'sep': 7, 'ok': 8, 'naŭ': 9,
        'dek': 10, 'cent': 100, 'mil': 1000,
        # Also handle numeric literals
    }

    # Mathematical operation keywords in Esperanto
    OPERATIONS = {
        'plus': '+',
        'aldoni': '+',
        'adici': '+',
        'minus': '-',
        'malplus': '-',
        'subtrahi': '-',
        'foje': '*',
        'multiplik': '*',
        'oble': '*',
        'divid': '/',
        'per': '/',  # When used with "divid"
        'potenc': '**',
        'kvadrat': '^2',
        'radik': 'sqrt',
        'kvadrata radiko': 'sqrt',
    }

    # Math-related keywords that indicate mathematical intent
    MATH_KEYWORDS = [
        'kiom',  # how much
        'kalkul',  # calculate
        'komput',  # compute
        'rezult',  # result
        'sumo',  # sum
        'diferenc',  # difference
        'produkt',  # product
        'kvocient',  # quotient
    ]

    def __init__(self):
        """Initialize Math Expert."""
        super().__init__("Math Tool Expert")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this is a mathematical query.

        Looks for:
        - Mathematical operation keywords
        - Number words or numeric literals
        - Question words like "kiom" (how much)

        Args:
            ast: Parsed query AST

        Returns:
            True if this appears to be a math query
        """
        if not ast or ast.get('tipo') != 'frazo':
            return False

        # Extract all words from AST
        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Check for math keywords
        has_math_keyword = any(
            keyword in word
            for word in words_lower
            for keyword in self.MATH_KEYWORDS
        )

        # Check for operation keywords
        has_operation = any(
            op in word
            for word in words_lower
            for op in self.OPERATIONS.keys()
        )

        # Check for numbers
        has_numbers = any(
            num in word or word.isdigit()
            for word in words_lower
            for num in self.NUMBERS.keys()
        )

        # Math query if it has operation + numbers, or math keyword + numbers
        return (has_operation and has_numbers) or (has_math_keyword and has_numbers)

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        High confidence if:
        - Clear mathematical operation detected
        - Numbers are parseable
        - Expression is simple (2-3 operands)

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Count indicators
        num_count = sum(
            1 for word in words_lower
            if any(num in word for num in self.NUMBERS.keys()) or word.isdigit()
        )

        op_count = sum(
            1 for word in words_lower
            if any(op in word for op in self.OPERATIONS.keys())
        )

        # High confidence: clear operation with 2-3 numbers
        if op_count >= 1 and num_count >= 2:
            return 0.95

        # Medium confidence: math keyword with numbers
        if any(kw in ' '.join(words_lower) for kw in self.MATH_KEYWORDS) and num_count >= 2:
            return 0.75

        # Low confidence: ambiguous
        return 0.5

    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mathematical computation.

        Args:
            ast: Parsed query AST

        Returns:
            Response with computed result
        """
        try:
            # Extract mathematical expression
            expression, operation = self._extract_expression(ast)

            if not expression:
                return {
                    'answer': "Mi ne povis trovi matematikan esprimon en via demando.",
                    'confidence': 0.0,
                    'expert': self.name,
                    'error': 'No mathematical expression found'
                }

            # Compute result
            result = self._compute(expression, operation)

            # Format response in Esperanto
            answer = self._format_answer(result, operation)

            return {
                'answer': answer,
                'result': result,
                'expression': expression,
                'operation': operation,
                'confidence': 0.95,
                'expert': self.name,
                'explanation': f"Kalkulis: {expression} = {result}"
            }

        except Exception as e:
            return {
                'answer': f"Eraro dum kalkulo: {str(e)}",
                'confidence': 0.0,
                'expert': self.name,
                'error': str(e)
            }

    def _extract_all_words(self, ast: Dict[str, Any]) -> List[str]:
        """Extract all words from AST recursively."""
        words = []

        if isinstance(ast, dict):
            if ast.get('tipo') == 'vorto':
                word = ast.get('plena_vorto', '') or ast.get('radiko', '')
                if word:
                    words.append(word)

            # Recursively extract from all fields
            for value in ast.values():
                if isinstance(value, (dict, list)):
                    words.extend(self._extract_all_words(value))

        elif isinstance(ast, list):
            for item in ast:
                words.extend(self._extract_all_words(item))

        return words

    def _extract_expression(self, ast: Dict[str, Any]) -> tuple:
        """
        Extract mathematical expression from AST.

        Returns:
            (expression_dict, operation_type)
            expression_dict has: {'operands': [num1, num2, ...], 'operator': '+'}
        """
        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Extract numbers (using set to avoid duplicates from recursive extraction)
        numbers = []
        seen = set()
        for word in words:
            word_lower = word.lower()

            # Try numeric literal
            if word.isdigit():
                num = int(word)
                if num not in seen:
                    numbers.append(num)
                    seen.add(num)
                continue

            # Try Esperanto number word
            for esp_num, value in self.NUMBERS.items():
                if esp_num in word_lower:
                    if value not in seen:
                        numbers.append(value)
                        seen.add(value)
                    break

        # Extract operation
        operator = None
        operation_type = None

        for word in words_lower:
            for esp_op, op_symbol in self.OPERATIONS.items():
                if esp_op in word:
                    operator = op_symbol
                    operation_type = esp_op
                    break
            if operator:
                break

        # Default to addition if numbers found but no operator
        if not operator and len(numbers) >= 2:
            operator = '+'
            operation_type = 'plus'

        if not numbers:
            return None, None

        return {
            'operands': numbers,
            'operator': operator
        }, operation_type

    def _compute(self, expression: Dict[str, Any], operation: str) -> float:
        """
        Perform the computation.

        Args:
            expression: Dict with 'operands' and 'operator'
            operation: Operation name for special handling

        Returns:
            Computed result
        """
        operands = expression['operands']
        operator = expression['operator']

        if not operands:
            raise ValueError("No operands found")

        # Handle special operations
        if operator == 'sqrt':
            return operands[0] ** 0.5

        if operator == '^2':
            return operands[0] ** 2

        # Handle binary operations
        if len(operands) < 2:
            raise ValueError(f"Need at least 2 operands for {operator}")

        result = operands[0]
        for operand in operands[1:]:
            if operator == '+':
                result += operand
            elif operator == '-':
                result -= operand
            elif operator == '*':
                result *= operand
            elif operator == '/':
                if operand == 0:
                    raise ValueError("Divido per nul!")
                result /= operand
            elif operator == '**':
                result **= operand

        return result

    def _format_answer(self, result: float, operation: str) -> str:
        """Format the answer in Esperanto."""
        # Convert to int if it's a whole number
        if result == int(result):
            result = int(result)

        return f"La rezulto estas: {result}"


# Export
__all__ = ['MathExpert']
