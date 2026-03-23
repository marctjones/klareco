"""
Morphology module for Esperanto affix semantics.

Provides deterministic semantic transformations for Esperanto affixes.
"""

from .affix_semantics import (
    AFFIX_SEMANTICS,
    get_affix_features,
    compose_word_semantics
)

__all__ = [
    'AFFIX_SEMANTICS',
    'get_affix_features', 
    'compose_word_semantics'
]
