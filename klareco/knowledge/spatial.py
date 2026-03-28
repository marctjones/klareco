"""
Spatial Vocabulary and Location Patterns for Esperanto

This module provides location-related vocabulary used for:
- WHERE question answering (location detection)
- Spatial entity validation in answer extraction
- Location preposition recognition

Sources:
- Extracted from klareco/rag/answer_extractor.py
- Expandable with more spatial patterns

Version: v2.1
Created: 2026-03-25
"""

from typing import Set

# Core spatial vocabulary
spatial_vocab: Set[str] = {
    # Places
    'lok', 'teren', 'spac', 'are', 'zon',
    'urb', 'vilaĝ', 'land', 'region', 'kontinént',
    'land', 'ŝtat', 'provinc', 'distrito',

    # Buildings and structures
    'dom', 'konstruaĵ', 'edific', 'palac', 'turm',
    'pont', 'strat', 'plac', 'parko', 'ĝarden',

    # Geographic features
    'mont', 'river', 'lag', 'mar', 'ocean',
    'insul', 'peneninsul', 'valo', 'arbar', 'dezert',

    # Direction/position words (roots)
    'norde', 'sude', 'okcidente', 'oriente',
    'supre', 'malsupre', 'dekstr', 'maldekstr',
    'centr', 'rand', 'angul', 'inter',

    # Movement verbs (spatial)
    'ven', 'ir', 'est', 'situ', 'trov', 'log', 'rest',
}

# Location prepositions
location_prepositions: Set[str] = {
    'en',       # in, inside
    'sur',      # on, on top of
    'sub',      # under, below
    'super',    # above, over
    'apud',     # beside, next to
    'ĉe',       # at, by (location)
    'antaŭ',    # in front of
    'post',     # behind
    'inter',    # between, among
    'ekster',   # outside
    'ĉirkaŭ',   # around
    'trans',    # across, beyond
    'proksime', # near, close to
    'malproksime', # far from
    'kontraŭ',  # opposite, against
    'laŭ',      # along
}

# Direction words
direction_words: Set[str] = {
    'norde', 'sude', 'okcidente', 'oriente',
    'nordo', 'sudo', 'okcidento', 'oriento',
    'nordokcidente', 'nordoriente', 'sudokcidente', 'sudoriente',
}

# Container/location suffixes
location_suffixes: Set[str] = {
    'ej',   # place for (e.g., lernejo = school, vendejo = store)
    'uj',   # container (e.g., mondujo = world, leteruj = mailbox)
}


def looks_like_location(text: str, suffixes: Set[str] = None) -> bool:
    """
    Check if text likely refers to a location.

    Args:
        text: Text to check
        suffixes: Optional set of suffixes from AST analysis

    Returns:
        True if likely a location
    """
    text_lower = text.lower()

    # Check if in spatial vocabulary
    if any(vocab in text_lower for vocab in spatial_vocab):
        return True

    # Check if has location suffix
    if suffixes:
        if any(suf in suffixes for suf in location_suffixes):
            return True

    # Check if it's a direction word
    if text_lower in direction_words:
        return True

    # Check if capitalized (might be place name)
    # Note: This is a weak signal, needs to be combined with other checks
    if text and text[0].isupper():
        # Could be a place name, but need gazetteer confirmation
        # This is handled by gazetteers.is_likely_place()
        pass

    return False


def extract_location_context(text: str) -> dict:
    """
    Extract location context from text (prepositions, direction, container).

    Args:
        text: Text to analyze

    Returns:
        Dict with location context information
    """
    text_lower = text.lower()

    context = {
        'prepositions': [],
        'directions': [],
        'is_container': False,
    }

    # Find location prepositions
    for prep in location_prepositions:
        if prep in text_lower:
            context['prepositions'].append(prep)

    # Find direction words
    for direction in direction_words:
        if direction in text_lower:
            context['directions'].append(direction)

    # Check for container suffixes (requires morphology analysis)
    if 'ej' in text_lower or 'uj' in text_lower:
        context['is_container'] = True

    return context
