"""
Knowledge Module - Unified Entity Knowledge for Klareco

This module provides a single source of truth for entity knowledge used across
the extractive QA system:

- verb_synonyms: Verb synonym relations for answer extraction
- noun_synonyms: Noun synonym relations for query expansion
- place_names: Geographic entity gazetteer
- person_indicators: Patterns for person detection
- temporal_vocab: Time-related vocabulary and patterns
- spatial_vocab: Location-related vocabulary and prepositions
- reflexive: Reflexive ↔ transitive verb normalization

Version: v2.1
Created: 2026-03-25
"""

from .synonyms import verb_synonyms, noun_synonyms, are_synonyms, get_synonyms
from .gazetteers import place_names, person_indicators, is_likely_person, is_likely_place
from .temporal import temporal_vocab, time_prepositions, time_adverbs, month_names, looks_like_time, extract_year
from .spatial import spatial_vocab, location_prepositions, looks_like_location, extract_location_context
from .reflexive import (
    REFLEXIVE_TRANSITIVE_PAIRS,
    normalize_reflexive_root,
    expand_with_morphology,
    is_reflexive_verb,
    get_transitive_base,
    get_reflexive_form,
)

__all__ = [
    # Vocabularies and dictionaries
    'verb_synonyms',
    'noun_synonyms',
    'place_names',
    'person_indicators',
    'temporal_vocab',
    'time_prepositions',
    'time_adverbs',
    'month_names',
    'spatial_vocab',
    'location_prepositions',
    'REFLEXIVE_TRANSITIVE_PAIRS',
    # Helper functions
    'are_synonyms',
    'get_synonyms',
    'is_likely_person',
    'is_likely_place',
    'looks_like_time',
    'extract_year',
    'looks_like_location',
    'extract_location_context',
    'normalize_reflexive_root',
    'expand_with_morphology',
    'is_reflexive_verb',
    'get_transitive_base',
    'get_reflexive_form',
]
