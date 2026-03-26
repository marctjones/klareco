"""
Temporal Vocabulary and Time Patterns for Esperanto

This module provides time-related vocabulary used for:
- WHEN question answering (time/date detection)
- Temporal entity validation in answer extraction
- Date pattern recognition

Sources:
- Extracted from klareco/rag/answer_extractor.py
- Expandable with more temporal patterns

Version: v2.1
Created: 2026-03-25
"""

from typing import Set

# Core temporal vocabulary
temporal_vocab: Set[str] = {
    # Time units
    'jar', 'jarcent', 'jardek', 'jarmil',
    'monat', 'semajn', 'tag', 'hor', 'minut', 'sekond',

    # Time periods
    'dat', 'temp', 'epok', 'period', 'moment',
    'matén', 'tagmez', 'vesper', 'nokt',
    'printémp', 'somer', 'aŭtun', 'vintr',

    # Time concepts
    'hor', 'dat', 'kalendar', 'horloĝ',
    'pasint', 'nún', 'estoné', 'futur',
    'komenc', 'fin', 'daŭr', 'long',
}

# Time prepositions
time_prepositions: Set[str] = {
    'en',      # in (en 1887, en la jaro)
    'dum',     # during
    'post',    # after
    'antaŭ',   # before
    'ekde',    # since, from
    'ĝis',     # until
    'je',      # at (time point)
    'de',      # from
    'al',      # to (time endpoint)
}

# Time adverbs
time_adverbs: Set[str] = {
    # Relative time
    'hieraŭ', 'hodiaŭ', 'morgaŭ',
    'antaŭhieraŭ', 'postmorgaŭ',

    # General time
    'nun', 'tiam', 'ĉiam', 'neniam',
    'jam', 'ankoraŭ', 'baldaŭ',
    'frue', 'malfrue', 'longe',

    # Frequency
    'ofte', 'malofte', 'ĉiam', 'kelkfoje',
    'unuafoje', 'refoje', 'denove',
}

# Month names (Esperanto)
month_names: Set[str] = {
    'januaro', 'februaro', 'marto', 'aprilo',
    'majo', 'junio', 'julio', 'aŭgusto',
    'septembro', 'oktobro', 'novembro', 'decembro',
}

# Day names (Esperanto)
day_names: Set[str] = {
    'lundo', 'mardo', 'merkredo', 'ĵaŭdo',
    'vendredo', 'sabato', 'dimanĉo',
}

# Numeric patterns for years (common historical ranges)
# Used for pattern matching in dates
YEAR_RANGE = range(1000, 2100)

# Century patterns (Quick Win #2 - WHEN questions)
# Maps century number to year ranges
century_to_years: dict[int, tuple[int, int]] = {
    13: (1200, 1299),  # 13th century
    14: (1300, 1399),  # 14th century
    15: (1400, 1499),  # 15th century
    16: (1500, 1599),  # 16th century
    17: (1600, 1699),  # 17th century
    18: (1700, 1799),  # 18th century
    19: (1800, 1899),  # 19th century
    20: (1900, 1999),  # 20th century
    21: (2000, 2099),  # 21st century
}

# Century name patterns (Esperanto)
century_patterns: dict[str, int] = {
    # Full ordinal forms
    '13-a jarcento': 13,
    '14-a jarcento': 14,
    '15-a jarcento': 15,
    '16-a jarcento': 16,
    '17-a jarcento': 17,
    '18-a jarcento': 18,
    '19-a jarcento': 19,
    '20-a jarcento': 20,
    '21-a jarcento': 21,

    # Variants
    'dektria jarcento': 13,
    'dekkvara jarcento': 14,
    'dekkina jarcento': 15,
    'deksesa jarcento': 16,
    'deksepma jarcento': 17,
    'dekoka jarcento': 18,
    'deknaua jarcento': 19,
    'dudeka jarcento': 20,
    'dudekuna jarcento': 21,
}

# Temporal unit expansions
temporal_units: dict[str, list[str]] = {
    'jarcento': ['century', '100 years', 'centjaro'],
    'jarmilo': ['millennium', '1000 years', 'miljaro'],
    'jardeko': ['decade', '10 years', 'dekjaro'],
    'epoko': ['era', 'epoch', 'age', 'periodo'],
    'erao': ['era', 'epoch', 'epoko'],
    'periodo': ['period', 'time span', 'epoko'],
    'dato': ['date', 'day', 'tago'],
}


def looks_like_time(text: str, suffixes: Set[str] = None) -> bool:
    """
    Check if text likely refers to a time/date expression.

    Args:
        text: Text to check
        suffixes: Optional set of suffixes from AST analysis

    Returns:
        True if likely a time expression
    """
    text_lower = text.lower()

    # Check if in temporal vocabulary
    if any(vocab in text_lower for vocab in temporal_vocab):
        return True

    # Check if it's a month name
    if text_lower in month_names:
        return True

    # Check if it's a day name
    if text_lower in day_names:
        return True

    # Check if it's a year (4 digits)
    if text.isdigit() and len(text) == 4:
        try:
            year = int(text)
            if year in YEAR_RANGE:
                return True
        except ValueError:
            pass

    # Check if it contains a year pattern (e.g., "en 1887", "jaro 1905")
    if any(char.isdigit() for char in text):
        # Extract potential year
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) == 4:
            try:
                year = int(digits)
                if year in YEAR_RANGE:
                    return True
            except ValueError:
                pass

    return False


def extract_year(text: str) -> int:
    """
    Extract a year (4-digit number) from text.

    Args:
        text: Text potentially containing a year

    Returns:
        Year as integer, or None if not found
    """
    # Extract all digit sequences
    import re
    matches = re.findall(r'\b(\d{4})\b', text)

    for match in matches:
        year = int(match)
        if year in YEAR_RANGE:
            return year

    return None


def detect_century(text: str) -> int:
    """
    Detect century reference in text (Quick Win #2).

    Args:
        text: Text potentially containing century reference

    Returns:
        Century number (e.g., 17 for 17th century), or None if not found

    Examples:
        "17-a jarcento" -> 17
        "deksepma jarcento" -> 17
        "en la 18-a jarcento" -> 18
    """
    text_lower = text.lower()

    # Check full patterns
    for pattern, century_num in century_patterns.items():
        if pattern in text_lower:
            return century_num

    # Check for numeric pattern like "17-a"
    import re
    match = re.search(r'(\d{1,2})-a\s+jarcent', text_lower)
    if match:
        return int(match.group(1))

    return None


def expand_century_to_years(century: int) -> list[str]:
    """
    Expand century number to list of years for query expansion (Quick Win #2).

    Args:
        century: Century number (13-21)

    Returns:
        List of year strings to search for

    Examples:
        17 -> ['1600', '1650', '1699', '17th century']
    """
    if century not in century_to_years:
        return []

    start_year, end_year = century_to_years[century]

    # Return representative years from the century
    years = [
        str(start_year),      # Start of century
        str(start_year + 25), # First quarter
        str(start_year + 50), # Mid-century
        str(start_year + 75), # Third quarter
        str(end_year),        # End of century
    ]

    return years


def expand_temporal_query(roots: list[str]) -> list[str]:
    """
    Expand temporal query roots with century patterns and synonyms (Quick Win #2).

    Args:
        roots: List of query roots

    Returns:
        Expanded list with temporal synonyms and century years

    Examples:
        ['17', 'jarcent', 'viv', 'Newton'] -> ['17', 'jarcent', 'viv', 'Newton', '1600', '1650', '1699']
    """
    expanded = list(roots)

    # Check if query contains century reference
    query_text = ' '.join(roots)
    century = detect_century(query_text)

    if century:
        # Add representative years from that century
        century_years = expand_century_to_years(century)
        expanded.extend(century_years)

    # Expand temporal units
    for root in roots:
        if root in temporal_units:
            # Don't add synonyms for now, just mark as temporal
            pass

    return expanded
