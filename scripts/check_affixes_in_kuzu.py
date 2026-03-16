#!/usr/bin/env python3
"""
Check if Esperanto affixes are present in Kuzu database as Radiko nodes.

Affixes are grammatical morphemes (suffixes/prefixes), not lexical roots.
They should be classified separately since they have different linguistic status.

Usage:
    python scripts/check_affixes_in_kuzu.py --kuzu data/indexes/v2.1_kuzu_index_full
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    sys.exit(1)

# Official Esperanto affixes from Fundamento
OFFICIAL_SUFFIXES = {
    # Noun-forming
    'aĵ': 'thing/concrete manifestation',
    'an': 'member/inhabitant',
    'ar': 'collection/group',
    'ej': 'place',
    'er': 'smallest unit',
    'estr': 'chief/leader',
    'id': 'offspring/descendant',
    'ig': 'causative (make/cause to be)',
    'iĝ': 'become',
    'il': 'tool/instrument',
    'in': 'feminine',
    'ind': 'worthy of',
    'ing': 'holder',
    'ism': 'doctrine/system',
    'ist': 'professional/adherent',
    'uj': 'container/country',
    'ul': 'person characterized by',
    'um': 'indefinite meaning',

    # Adjective/adverb-forming
    'ebl': 'possible/able to be',
    'ec': 'quality/abstract property',
    'eg': 'augmentative',
    'em': 'tendency/inclination',
    'end': 'must be done',
    'et': 'diminutive',

    # Verb aspects (participles)
    'ad': 'continuous/repeated action',
    'ant': 'active present participle',
    'int': 'active past participle',
    'ont': 'active future participle',
    'at': 'passive present participle',
    'it': 'passive past participle',
    'ot': 'passive future participle',

    # Special
    'ĉj': 'masculine diminutive (for names)',
    'nj': 'feminine diminutive (for names)',
    'obl': 'multiple',
    'on': 'fraction',
    'op': 'collective number',
}

OFFICIAL_PREFIXES = {
    'bo': 'related by marriage',
    'dis': 'dispersion/separation',
    'ek': 'beginning of action',
    'eks': 'former/ex-',
    'ge': 'both sexes together',
    'mal': 'opposite',
    'mis': 'wrongly/incorrectly',
    'pra': 'remote ancestor/ancient',
    're': 'again/back',
}

def check_affixes_in_database(args):
    """Check which affixes exist as Radiko nodes in database."""

    print(f"Opening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)

    print("\n=== Checking Official Affixes in Database ===\n")

    # Check suffixes
    print("SUFFIXES:")
    suffix_found = []
    suffix_missing = []

    for suffix, meaning in sorted(OFFICIAL_SUFFIXES.items()):
        result = conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{suffix}'}})
            RETURN r.radiko, r.nivelo, r.ofteco
        """)

        if result.has_next():
            radiko, nivelo, ofteco = result.get_next()
            suffix_found.append((suffix, nivelo or 'NULL', ofteco or 0))
            status = f"✓ FOUND - tier: {nivelo or 'NULL'}, usage: {ofteco or 0}"
        else:
            suffix_missing.append(suffix)
            status = "✗ NOT FOUND"

        print(f"  {suffix:6s} ({meaning:40s}) - {status}")

    # Check prefixes
    print("\nPREFIXES:")
    prefix_found = []
    prefix_missing = []

    for prefix, meaning in sorted(OFFICIAL_PREFIXES.items()):
        result = conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{prefix}'}})
            RETURN r.radiko, r.nivelo, r.ofteco
        """)

        if result.has_next():
            radiko, nivelo, ofteco = result.get_next()
            prefix_found.append((prefix, nivelo or 'NULL', ofteco or 0))
            status = f"✓ FOUND - tier: {nivelo or 'NULL'}, usage: {ofteco or 0}"
        else:
            prefix_missing.append(prefix)
            status = "✗ NOT FOUND"

        print(f"  {prefix:6s} ({meaning:40s}) - {status}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Suffixes found: {len(suffix_found)}/{len(OFFICIAL_SUFFIXES)}")
    print(f"  Prefixes found: {len(prefix_found)}/{len(OFFICIAL_PREFIXES)}")

    # Check current tier classifications
    if suffix_found or prefix_found:
        print("\n=== Current Tier Classifications ===")

        tier_counts = {}
        for affix, tier, count in suffix_found + prefix_found:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        for tier, count in sorted(tier_counts.items()):
            print(f"  {tier}: {count} affixes")

    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION:")

    if suffix_found or prefix_found:
        print("""
Affixes ARE present in the database as Radiko nodes.

These should be classified as a separate tier:
  - tier0_afikso: Official Esperanto affixes (grammatical morphemes)

Rationale:
  - Affixes are NOT lexical roots (they don't carry independent meaning)
  - They are productive grammatical morphemes
  - They should be treated differently in training:
    * Excluded from root embedding training (they're compositional)
    * May need special handling in AST processing
    * Useful for morphological analysis but not semantic retrieval

Next step:
  - Create new tier: tier0_afikso
  - Label all official affixes with this tier
  - Update tier assignment script to handle affixes explicitly
""")
    else:
        print("""
Affixes are NOT present in the database as separate Radiko nodes.

This is expected since:
  - Affixes are parsed as morphological components, not root nodes
  - The parser treats them as part of word structure, not independent roots
  - They appear in AST fields (prefikso, sufiksoj) but not as Radiko nodes

No action needed - current architecture handles affixes correctly.
""")

    # Save results for potential tier update
    if suffix_found or prefix_found:
        import json
        output_file = Path('data/vocabularies/official_affixes.json')

        affixes = {
            'suffixes': {s: {'meaning': OFFICIAL_SUFFIXES[s], 'found': True}
                        for s, _, _ in suffix_found},
            'prefixes': {p: {'meaning': OFFICIAL_PREFIXES[p], 'found': True}
                        for p, _, _ in prefix_found},
            'total_found': len(suffix_found) + len(prefix_found),
            'total_official': len(OFFICIAL_SUFFIXES) + len(OFFICIAL_PREFIXES)
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(affixes, f, ensure_ascii=False, indent=2)

        print(f"\nSaved affix inventory to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Check affixes in Kuzu database')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    args = parser.parse_args()

    check_affixes_in_database(args)

if __name__ == '__main__':
    main()
