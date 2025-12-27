#!/usr/bin/env python3
"""
Extract semantic relations from ReVo database.

This script extracts curated semantic relations from ReVo:
- Synonyms (sin)
- Antonyms (ant)
- Hypernyms (super) - "is a kind of"
- Hyponyms (sub) - "is a specific type of"
- Part-of (prt) - "is part of"
- Has-part (malprt) - "has as part"

Output: data/revo/revo_semantic_relations.json

These are human-curated by Esperanto lexicographers and provide
strong supervision for semantic similarity training.
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# X-notation to Unicode mapping
X_TO_UNICODE = {
    'cx': 'ĉ', 'Cx': 'Ĉ', 'CX': 'Ĉ',
    'gx': 'ĝ', 'Gx': 'Ĝ', 'GX': 'Ĝ',
    'hx': 'ĥ', 'Hx': 'Ĥ', 'HX': 'Ĥ',
    'jx': 'ĵ', 'Jx': 'Ĵ', 'JX': 'Ĵ',
    'sx': 'ŝ', 'Sx': 'Ŝ', 'SX': 'Ŝ',
    'ux': 'ŭ', 'Ux': 'Ŭ', 'UX': 'Ŭ',
}

# Valid Esperanto letters
VALID_EO_CHARS = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭ')


def normalize_x_notation(text: str) -> str:
    """Convert x-notation to proper Unicode."""
    result = text
    for x_form, unicode_form in X_TO_UNICODE.items():
        result = result.replace(x_form, unicode_form)
    return result


def extract_root(mrk: str) -> str | None:
    """
    Extract root from ReVo marker.

    Examples:
        'hund.0o' -> 'hund'
        'grand.mal0a' -> 'grand'
        'prem.0ajxo.MED' -> 'prem'
        'sxip.0estro' -> 'ŝip'
    """
    if not mrk:
        return None

    # Split on dots, take first part
    parts = mrk.split('.')
    if not parts:
        return None

    root = parts[0]

    # Remove trailing digits
    root = re.sub(r'\d+$', '', root)

    # Normalize x-notation
    root = normalize_x_notation(root)

    # Lowercase
    root = root.lower()

    # Validate - only Esperanto letters
    if not root or not all(c in VALID_EO_CHARS for c in root):
        return None

    # Minimum length
    if len(root) < 2:
        return None

    return root


def main():
    db_path = Path('data/revo/revo.db')
    output_path = Path('data/revo/revo_semantic_relations.json')

    if not db_path.exists():
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Relation types and their semantic meaning
    relation_types = {
        'sin': 'synonym',
        'ant': 'antonym',
        'super': 'hypernym',
        'sub': 'hyponym',
        'prt': 'part_of',
        'malprt': 'has_part'
    }

    relations = {}
    stats = {}

    for rel_code, rel_name in relation_types.items():
        cursor.execute("SELECT mrk, cel FROM referenco WHERE tip=?", (rel_code,))

        pairs = []
        for mrk, cel in cursor.fetchall():
            root1 = extract_root(mrk)
            root2 = extract_root(cel)

            if root1 and root2 and root1 != root2:
                pairs.append([root1, root2])

        # Deduplicate (order matters for hypernym/hyponym, not for synonym/antonym)
        if rel_name in ('synonym', 'antonym'):
            unique = set(tuple(sorted(p)) for p in pairs)
            pairs = [list(p) for p in unique]
        else:
            unique = set(tuple(p) for p in pairs)
            pairs = [list(p) for p in unique]

        relations[rel_name] = pairs
        stats[rel_name] = len(pairs)
        print(f"  {rel_name}: {len(pairs)} pairs")

    conn.close()

    # Save output
    output = {
        'metadata': {
            'source': 'ReVo (Reta Vortaro)',
            'description': 'Human-curated semantic relations from Esperanto lexicographers',
            'statistics': stats,
            'total_pairs': sum(stats.values())
        },
        'relations': relations
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"Total pairs: {sum(stats.values())}")

    # Show samples
    print("\n" + "=" * 50)
    print("Sample relations:")
    print("=" * 50)

    print("\nSynonyms (should be very similar):")
    for r1, r2 in relations['synonym'][:5]:
        print(f"  {r1} ≈ {r2}")

    print("\nAntonyms (should be distant):")
    for r1, r2 in relations['antonym'][:5]:
        print(f"  {r1} ↔ {r2}")

    print("\nHypernyms (X is-a Y):")
    for r1, r2 in relations['hypernym'][:5]:
        print(f"  {r1} → {r2}")


if __name__ == '__main__':
    main()
