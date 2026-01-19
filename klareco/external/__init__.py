"""External knowledge base integration for Klareco.

This module provides integration with external semantic resources
like ConceptNet and Wikidata to expand semantic category coverage.
"""

from .category_mapper import CategoryMapper
from .function_words import (
    FUNCTION_WORDS,
    is_function_word,
    filter_function_words,
    get_function_word_count
)

__all__ = [
    'CategoryMapper',
    'FUNCTION_WORDS',
    'is_function_word',
    'filter_function_words',
    'get_function_word_count'
]
