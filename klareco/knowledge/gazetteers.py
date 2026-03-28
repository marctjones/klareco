"""
Gazetteers - Named Entity Lists for Klareco

This module provides named entity gazetteers for:
- Place names (cities, countries, regions)
- Person indicators (names, titles, occupations)

Used for:
- WHERE question answering (location detection)
- WHO question answering (person detection)
- Entity validation in answer extraction

Version: v2.2 (Now uses semantic ontology + fallback)
Created: 2026-03-25
Updated: 2026-03-28 - Integrated with semantic ontology
"""

from typing import Set
from .semantic_bridge import (
    get_merged_places,
    get_merged_persons,
    get_merged_temporal,
    is_person_from_ontology,
    is_place_from_ontology,
    is_time_from_ontology
)

# Fallback Place Names (used if ontology unavailable)
# These are MERGED with ontology data to provide comprehensive coverage
_FALLBACK_PLACE_NAMES: Set[str] = {
    # Major European Cities
    'Barcelono', 'Varsovio', 'Parizo', 'Berlino', 'Londono', 'Romo',
    'Moskvo', 'Pekino', 'Tokio', 'Nov-Jorko', 'Amsterdamo', 'Bruselo',
    'Vieno', 'Prago', 'Budapeŝto', 'Krakovo', 'Sofio', 'Bukareŝto',
    'Aténo', 'Lisbono', 'Dublino', 'Kopenhago', 'Stokholmo', 'Oslo',
    'Helsinko', 'Rejkjaviko',

    # Esperanto-Relevant Cities
    'Bjalistoko',  # Zamenhof's birthplace
    'Suwałki',     # Esperanto history
    'Bulonjo',     # First Esperanto Congress (1905)

    # Countries
    'Pollando', 'Francio', 'Germanio', 'Anglio', 'Italio', 'Rusio',
    'Ĉinio', 'Japanio', 'Usono', 'Hispanio', 'Britio', 'Nederlando',
    'Belgio', 'Aŭstrio', 'Ĉeĥio', 'Slovakio', 'Hungario', 'Rumanio',
    'Bulgario', 'Grekio', 'Portugalio', 'Irlando', 'Danio', 'Svedio',
    'Norvegio', 'Finnlando', 'Islando', 'Svislando', 'Aŭstralio',
    'Brazilo', 'Argentino', 'Meksiko', 'Kanado', 'Hind', 'Koreio',

    # Regions/Continents
    'Eŭropo', 'Azio', 'Afriko', 'Ameriko', 'Okeani', 'Sudo-Ameriko',
    'Norda-Ameriko', 'Orienta-Eŭropo', 'Okcidenta-Eŭropo',
}

# Person Indicators
# Patterns that suggest a word refers to a person
person_indicators: dict = {
    # Suffixes that indicate persons
    'suffixes': {
        'ul',   # Person characterized by X (e.g., saĝulo - wise person)
        'ist',  # Professional/practitioner (e.g., artisto - artist)
        'in',   # Feminine form (e.g., instruistino - female teacher)
        'int',  # Past active participle (e.g., kreinto - creator)
        'ant',  # Present active participle (e.g., helpanto - helper)
        'ont',  # Future active participle (e.g., parolonto - future speaker)
    },

    # Titles and honorifics
    'titles': {
        'Doktoro', 'Profesoro', 'Sinjoro', 'Sinjorino', 'Fraŭlino',
        'Estimata', 'Kara', 'Princo', 'Reĝo', 'Reĝino', 'Prezidento',
    },

    # Known Esperanto personalities (expandable)
    'esperantists': {
        'Zamenhof', 'Zamenhofo', 'Ludoviko', 'Lazaro',
        'Grabowski', 'Waringhien', 'Kalocsay', 'Janton',
        'Lapenna', 'Sikosek', 'Privat',
    },

    # Common occupations (roots)
    'occupations': {
        'kuracist',  # doctor
        'okulist',   # ophthalmologist
        'instruist', # teacher
        'verkist',   # writer
        'aŭtor',     # author
        'redaktor',  # editor
        'tradukist', # translator
        'direktist', # director
        'prezident', # president
    },
}

# Convert fallback place names to lowercase roots
_fallback_place_roots = {p.lower().rstrip('o') if p.endswith('o') else p.lower()
                         for p in _FALLBACK_PLACE_NAMES}

# Convert fallback person data to roots
_fallback_person_roots = set()
for occ in person_indicators['occupations']:
    _fallback_person_roots.add(occ)
for name in person_indicators['esperantists']:
    _fallback_person_roots.add(name.lower())

# PUBLIC API: Merge ontology + fallback data
# This maintains backward compatibility while using semantic ontology
place_names: Set[str] = get_merged_places(_fallback_place_roots)
person_roots: Set[str] = get_merged_persons(_fallback_person_roots)


def is_likely_person(text: str, suffixes: Set[str] = None) -> bool:
    """
    Check if text likely refers to a person based on indicators.

    Args:
        text: Text to check
        suffixes: Optional set of suffixes from AST analysis

    Returns:
        True if likely a person
    """
    # First check semantic ontology
    text_lower = text.lower()
    # Strip -o ending if present for root lookup
    text_root = text_lower.rstrip('o') if text_lower.endswith('o') else text_lower
    if is_person_from_ontology(text_root):
        return True

    # Check if it's a known title
    if text in person_indicators['titles']:
        return True

    # Check if it's a known Esperantist
    if text in person_indicators['esperantists']:
        return True

    # Check if it contains person-indicating suffixes
    if suffixes:
        if any(suf in suffixes for suf in person_indicators['suffixes']):
            return True

    # Check if it's a known occupation
    if any(occ in text_lower for occ in person_indicators['occupations']):
        return True

    # Check if capitalized (proper noun) and not a place
    if text and text[0].isupper() and text not in place_names:
        return True

    return False


def is_likely_place(text: str, suffixes: Set[str] = None) -> bool:
    """
    Check if text likely refers to a place.

    Args:
        text: Text to check
        suffixes: Optional set of suffixes from AST analysis

    Returns:
        True if likely a place
    """
    # First check semantic ontology
    text_lower = text.lower()
    # Strip -o ending if present for root lookup
    text_root = text_lower.rstrip('o') if text_lower.endswith('o') else text_lower
    if is_place_from_ontology(text_root):
        return True

    # Check if in place names gazetteer (merged ontology + fallback)
    if text in place_names:
        return True

    # Check for -ej suffix (place for)
    if suffixes and 'ej' in suffixes:
        return True

    # Check for location-related roots
    location_roots = {'urb', 'vilaĝ', 'land', 'region', 'loko', 'teren'}
    if any(root in text_lower for root in location_roots):
        return True

    return False
