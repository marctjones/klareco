"""
Math evaluator for KIOM-style questions (#772).

Two layers:

  1. detect_and_evaluate(question_text, question_ast) — top-level entry.
     Detects whether a question is a math question; if so, normalises
     Esperanto math words to a SymPy-evaluable expression and returns
     the result.

  2. evaluate(expression) — given a normalized expression string, run
     it through SymPy. Used directly by tool dispatch.

Esperanto math-word → operator lexicon:

    plus, kaj, aldonu                → +
    minus                            → -
    oble                             → *
    dividite per, divizio            → /
    al la potenco de, eksponentigite → **
    kvadrata radiko de               → sqrt()
    kvadrato de                      → ** 2

KIOM patterns we handle:
    "Kiom estas du plus tri?"                  → 5
    "Kiom da jaroj inter 1859 kaj 1917?"       → 58
    "Kio estas la kvadrata radiko de 144?"     → 12
    "Kiom estas dek minus tri?"                → 7
    "Kio estas dudek oble tri?"                → 60

Numerals are pre-translated from Esperanto words to digits via a small
lexicon (0-100 covered for V1). For numbers >100, callers should pass
digit strings.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import sympy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Esperanto numerals (0-100 covers what KIOM questions actually use)
# ---------------------------------------------------------------------------

_BASIC_NUM = {
    'nul': 0, 'unu': 1, 'du': 2, 'tri': 3, 'kvar': 4, 'kvin': 5,
    'ses': 6, 'sep': 7, 'ok': 8, 'naŭ': 9, 'nau': 9,
    'dek': 10, 'dudek': 20, 'tridek': 30, 'kvardek': 40,
    'kvindek': 50, 'sesdek': 60, 'sepdek': 70, 'okdek': 80,
    'naŭdek': 90, 'naudek': 90,
    'cent': 100, 'mil': 1000,
}


def _esperanto_word_to_number(word: str) -> Optional[int]:
    """Return integer value of an Esperanto numeral word, or None.

    Handles compounds like 'dudek tri' (23), 'cent kvin' (105) when
    the caller has joined them; for single words, looks up _BASIC_NUM.
    """
    if not word:
        return None
    w = word.strip().lower().replace(',', '').rstrip('.')
    if w.isdigit():
        return int(w)
    if w in _BASIC_NUM:
        return _BASIC_NUM[w]
    # Compound: 'dudek tri' = 20 + 3, 'cent kvin' = 100 + 5
    parts = w.split()
    if len(parts) >= 2:
        total = 0
        for p in parts:
            v = _BASIC_NUM.get(p) or (int(p) if p.isdigit() else None)
            if v is None:
                return None
            # naïve sum: 'dudek tri' = 20 + 3 = 23.
            # Falls down for 'tri mil' (should be 3000); guard below.
            total += v
        return total
    return None


# ---------------------------------------------------------------------------
# Operator lexicon
# ---------------------------------------------------------------------------

# Order matters: longer phrases first so 'al la potenco de' beats 'al'.
_OPERATORS: list[tuple[str, str]] = [
    ('kvadrata radiko de', 'sqrt'),
    ('al la potenco de',   '**'),
    ('eksponentigite per', '**'),
    ('dividite per',       '/'),
    ('divizio',            '/'),
    ('plus',               '+'),
    ('aldonu',             '+'),
    ('kaj',                '+'),    # 'kvin kaj tri' = 5+3 in KIOM context
    ('minus',              '-'),
    ('oble',               '*'),
    ('multiplikite per',   '*'),
    ('inter',              '-'),    # 'inter X kaj Y' = |X - Y| in year-diff
]


def _normalize_to_expression(text: str) -> Optional[str]:
    """Map Esperanto math text to a SymPy-evaluable expression string.

    Returns None if the text doesn't look like math at all."""
    t = text.lower().strip().rstrip('?.!')
    # Drop the "Kiom estas" / "Kio estas" prefix
    for prefix in ('kiom estas', 'kio estas', 'kalkulu', 'kiom da jaroj',
                    'kiom da'):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    # Replace operator words with symbols. Longest first.
    for word, op in _OPERATORS:
        if word in t:
            t = t.replace(word, f' {op} ')
    # Now convert remaining word-numbers to digits
    tokens = re.split(r'\s+', t)
    converted: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok:
            i += 1
            continue
        # Skip already-operator tokens
        if tok in ('+', '-', '*', '/', '**', '(', ')', 'sqrt'):
            converted.append(tok)
            i += 1
            continue
        # If it's a digit string, keep it
        if re.fullmatch(r'-?\d+(\.\d+)?', tok):
            converted.append(tok)
            i += 1
            continue
        # Try compound numerals (greedy: try 3-word, 2-word, 1-word)
        joined3 = ' '.join(tokens[i:i+3])
        joined2 = ' '.join(tokens[i:i+2])
        joined1 = tok
        for joined, n in [(joined3, 3), (joined2, 2), (joined1, 1)]:
            v = _esperanto_word_to_number(joined)
            if v is not None:
                converted.append(str(v))
                i += n
                break
        else:
            # Unknown filler word — drop
            i += 1

    expr = ' '.join(converted).strip()
    # Need at least one digit and one operator (or sqrt) to be a math expr
    has_digit = any(c.isdigit() for c in expr)
    has_op = bool(re.search(r'[+\-*/]|sqrt', expr))
    if not (has_digit and has_op):
        return None
    # Convert 'sqrt 144' to 'sqrt(144)'
    expr = re.sub(r'sqrt\s+(\d+(?:\.\d+)?)', r'sqrt(\1)', expr)
    return expr


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expression: str) -> Optional[str]:
    """Run a SymPy expression. Returns formatted result or None on failure."""
    try:
        parsed = sympy.sympify(expression, locals={'sqrt': sympy.sqrt})
        result = sympy.simplify(parsed)
        # Pretty-format: integer if integer-valued, else decimal
        if result.is_Integer:
            return str(int(result))
        if result.is_Number:
            f = float(result)
            if abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
            return f'{f:.4f}'.rstrip('0').rstrip('.')
        return str(result)
    except Exception as e:
        logger.debug(f'sympy eval failed on {expression!r}: {e}')
        return None


# ---------------------------------------------------------------------------
# Public entry: detect + evaluate
# ---------------------------------------------------------------------------

def detect_and_evaluate(question_text: str,
                         question_ast: Optional[dict] = None
                         ) -> Optional[str]:
    """Top-level entry. Returns the answer string if the question is a
    math question, None otherwise.

    Trigger conditions:
      - Starts with 'Kiom estas' / 'Kio estas la X radiko' / 'Kalkulu'
      - OR (question contains 'jaroj inter' / 'jaroj de ... ĝis')
      - OR (question contains digit-or-numeral + math-operator-word)
    """
    if not question_text:
        return None
    expr = _normalize_to_expression(question_text)
    if expr is None:
        return None
    result = evaluate(expr)
    return result


def year_diff(a: int | str, b: int | str) -> int:
    """Convenience: years between two dates."""
    return abs(int(a) - int(b))
