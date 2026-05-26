"""
klareco.generation — multi-sentence output generators (#766 #775).

Public API:
    biography(entity_radiko) → paragraph about a person/organization
    define(entity_radiko)    → 1-3 sentence definition
    compare(a_radiko, b_radiko) → 2-3 sentence comparison
"""
from klareco.generation.discourse import biography, define, compare

__all__ = ['biography', 'define', 'compare']
