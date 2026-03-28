"""
Gazetteers - Named Entity Lists for Klareco

This module provides named entity gazetteers for:
- Place names (cities, countries, regions)
- Person indicators (names, titles, occupations)

Used for:
- WHERE question answering (location detection)
- WHO question answering (person detection)
- Entity validation in answer extraction

Source: Extracted from klareco/rag/answer_extractor.py
Expandable: Can be loaded from external JSON files

Version: v2.1
Created: 2026-03-25
"""

from typing import Set

# Place Names Gazetteer
# Used in _is_place() to identify locations for WHERE questions
place_names: Set[str] = {
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


def is_likely_person(text: str, suffixes: Set[str] = None) -> bool:
    """
    Check if text likely refers to a person based on indicators.

    Args:
        text: Text to check
        suffixes: Optional set of suffixes from AST analysis

    Returns:
        True if likely a person
    """
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
    text_lower = text.lower()
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
    # Check if in place names gazetteer
    if text in place_names:
        return True

    # Check for -ej suffix (place for)
    if suffixes and 'ej' in suffixes:
        return True

    # Check for location-related roots
    location_roots = {'urb', 'vilaĝ', 'land', 'region', 'loko', 'teren'}
    text_lower = text.lower()
    if any(root in text_lower for root in location_roots):
        return True

    return False
