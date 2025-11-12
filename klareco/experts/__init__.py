"""
Expert system for Klareco.

Experts are specialized components that handle specific types of queries:
- MathExpert: Performs mathematical calculations (symbolic)
- DateExpert: Handles date/time operations (symbolic)
- GrammarExpert: Explains grammatical structure (symbolic)
- Factoid_QA_Expert: Answers factual questions using RAG (neural - future)
- Dictionary_Tool_Expert: Looks up word definitions (symbolic - future)
"""

from .base import Expert
from .math_expert import MathExpert
from .date_expert import DateExpert
from .grammar_expert import GrammarExpert

__all__ = [
    'Expert',
    'MathExpert',
    'DateExpert',
    'GrammarExpert',
]
