#!/usr/bin/env python3
"""
Clean extracted Unua Libro roots to get the definitive 917 original roots.

Removes:
- Proper names with examples (containing capitals or special formatting)
- Example phrases (multi-word entries)
- Grammatical affixes (suffixes/prefixes that aren't lexical roots)
- Entries with complex formatting

Usage:
    python scripts/clean_unua_libro_roots.py
"""
import json
import re
from pathlib import Path

# Known Esperanto grammatical affixes (not lexical roots)
GRAMMATICAL_AFFIXES = {
    # Suffixes
    'ad', 'aĵ', 'an', 'ar', 'ĉj', 'ebl', 'ec', 'eg', 'ej', 'em', 'end',
    'er', 'estr', 'et', 'id', 'ig', 'iĝ', 'il', 'in', 'ind', 'ing',
    'ism', 'ist', 'nj', 'obl', 'on', 'op', 'uj', 'ul', 'um',
    # Verb participle endings
    'ant', 'int', 'ont', 'at', 'it', 'ot', 'ir',
    # Prefixes
    'bo', 'dis', 'ek', 'eks', 'ge', 'mal', 'mis', 'pra', 're',
}

# Known proper name roots (not lexical roots)
PROPER_NAMES = {
    'emili', 'mari', 'miĥael', 'aleksandr', 'paris', 'turk',
}

def is_valid_root(entry: str) -> bool:
    """Check if entry is a valid lexical root (not affix, proper name, or phrase)."""

    # Remove backslash escapes for analysis
    cleaned = entry.replace('\\,', '').replace('\\', '')

    # Skip if contains capital letters (proper names or examples)
    if any(c.isupper() for c in cleaned):
        return False

    # Skip if contains spaces (phrases)
    if ' ' in cleaned:
        return False

    # Skip if contains special punctuation (examples with formatting)
    if any(c in cleaned for c in ['—', ';', '(', ')', '[', ']']):
        return False

    # Skip if contains "da" (measurement phrases like "kilogram\\,o da viand")
    if 'da' in cleaned.split():
        return False

    # Skip if looks like a phrase pattern (multiple word parts)
    # e.g., "mi ir\\,as dom\\,o", "kia hom"
    word_parts = [p for p in cleaned.split() if p]
    if len(word_parts) > 1:
        return False

    # Get the root part (before any comma or space)
    root = cleaned.split(',')[0].split()[0].strip().lower()

    # Skip if empty
    if not root:
        return False

    # Skip if it's a grammatical affix
    if root in GRAMMATICAL_AFFIXES:
        return False

    # Skip if it's a known proper name
    if root in PROPER_NAMES:
        return False

    # Skip if too short (single letter - likely abbreviation)
    if len(root) < 2:
        return False

    # Must be all lowercase letters (with Esperanto diacritics)
    if not re.match(r'^[a-zĉĝĥĵŝŭ]+$', root):
        return False

    return True

def clean_unua_libro_roots():
    """Extract clean lexical roots from Unua Libro extraction."""

    input_file = Path('data/raw/eo/unua_libro/unua_libro_roots_extracted.txt')
    output_file = Path('data/vocabularies/unua_libro_original_roots.json')

    print(f"Reading extracted roots from: {input_file}")

    # Read all entries
    with open(input_file, 'r', encoding='utf-8') as f:
        entries = [line.strip() for line in f if line.strip()]

    print(f"Total extracted entries: {len(entries)}")

    # Filter to valid roots
    valid_roots = {}
    rejected = {
        'proper_names': [],
        'phrases': [],
        'affixes': [],
        'other': []
    }

    for entry in entries:
        # Get cleaned root
        cleaned = entry.replace('\\,', '').replace('\\', '')
        root = cleaned.split(',')[0].split()[0].strip().lower()

        if is_valid_root(entry):
            valid_roots[root] = {
                'root': root,
                'source': 'Unua Libro (1887)',
                'raw_entry': entry
            }
        else:
            # Categorize rejection reason
            if any(c.isupper() for c in cleaned) or root in PROPER_NAMES:
                rejected['proper_names'].append(entry)
            elif ' ' in cleaned or 'da' in cleaned:
                rejected['phrases'].append(entry)
            elif root in GRAMMATICAL_AFFIXES:
                rejected['affixes'].append(entry)
            else:
                rejected['other'].append(entry)

    print(f"\nValid lexical roots: {len(valid_roots)}")
    print(f"Rejected entries:")
    print(f"  Proper names: {len(rejected['proper_names'])}")
    print(f"  Phrases: {len(rejected['phrases'])}")
    print(f"  Affixes: {len(rejected['affixes'])}")
    print(f"  Other: {len(rejected['other'])}")

    # Show some examples of rejected entries
    if rejected['proper_names']:
        print(f"\nExample proper names rejected: {rejected['proper_names'][:3]}")
    if rejected['phrases']:
        print(f"Example phrases rejected: {rejected['phrases'][:3]}")
    if rejected['affixes']:
        print(f"Example affixes rejected: {rejected['affixes'][:3]}")

    # Save clean roots
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_roots, f, ensure_ascii=False, indent=2)

    print(f"\nSaved clean roots to: {output_file}")

    # Show sample of valid roots
    sample_roots = sorted(valid_roots.keys())[:20]
    print(f"\nSample valid roots: {', '.join(sample_roots)}")

    return valid_roots, rejected

if __name__ == '__main__':
    valid_roots, rejected = clean_unua_libro_roots()

    print(f"\n{'='*60}")
    print(f"Expected: ~917 original Unua Libro roots")
    print(f"Extracted: {len(valid_roots)} lexical roots")

    if len(valid_roots) < 900:
        print(f"\n⚠️  WARNING: Found fewer roots than expected!")
        print(f"   Expected ~917, got {len(valid_roots)}")
        print(f"   May need to check if some valid roots were incorrectly rejected.")
    elif len(valid_roots) > 950:
        print(f"\n⚠️  WARNING: Found more roots than expected!")
        print(f"   Expected ~917, got {len(valid_roots)}")
        print(f"   May need to tighten filtering criteria.")
    else:
        print(f"\n✓ Root count looks reasonable!")
