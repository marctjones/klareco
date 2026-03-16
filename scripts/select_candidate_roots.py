#!/usr/bin/env python3
"""
Select Candidate Roots for Phase 0 Annotation

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Data
DEPENDENCIES: Kuzu database, root frequency data

Description:
    Identifies top 50 candidate roots for manual semantic annotation.
    Prioritizes Fundamento roots, high-frequency roots, and diverse
    semantic categories.

Usage:
    python scripts/select_candidate_roots.py \
        --database data/indexes/v2.1_kuzu_index_full \
        --output data/annotations/candidates.jsonl \
        --count 50

Outputs:
    - candidates.jsonl: List of candidate roots with metadata
    - Selection based on: Fundamento status, frequency, diversity

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #656
See Also: data/annotations/README.md
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)


def get_fundamento_roots(conn) -> List[str]:
    """Query Fundamento roots from database."""
    try:
        result = conn.execute("""
            MATCH (r:Radiko)
            WHERE r.estas_funda = true
            RETURN r.radiko
            ORDER BY r.radiko
        """)

        roots = []
        while result.has_next():
            roots.append(result.get_next()[0])
        return roots
    except:
        # If estas_funda not set, return known Fundamento roots
        return []


def get_high_frequency_roots(conn, limit: int = 100) -> List[tuple]:
    """Get high-frequency roots from corpus."""
    try:
        # Try to query by frequency tier
        result = conn.execute(f"""
            MATCH (r:Radiko)
            WHERE r.ofteca_tavolo IS NOT NULL AND r.ofteca_tavolo <= 1
            RETURN r.radiko, r.ofteca_tavolo
            ORDER BY r.ofteca_tavolo, r.radiko
            LIMIT {limit}
        """)

        roots = []
        while result.has_next():
            row = result.get_next()
            roots.append((row[0], row[1]))
        return roots
    except:
        # If frequency not available, query by root usage count
        try:
            result = conn.execute(f"""
                MATCH (r:Radiko)-[:APERAS_EN]->(s)
                WITH r.radiko as radiko, COUNT(s) as usage_count
                RETURN radiko, usage_count
                ORDER BY usage_count DESC
                LIMIT {limit}
            """)

            roots = []
            while result.has_next():
                row = result.get_next()
                roots.append((row[0], 0))  # Assume tier 0 for high-usage
            return roots
        except:
            return []


def categorize_root(radiko: str) -> Dict[str, Any]:
    """
    Heuristically categorize root based on common patterns.

    This is a starting point - should be manually reviewed.
    """
    # Common verb roots (action/state)
    verb_roots = {
        'est': ('ekzisto-47', 'stato'),
        'hav': ('havado-100', 'stato'),
        'far': ('kreado-26', 'plenumigo'),
        'ir': ('movo-51', 'aktiveco'),
        'ven': ('movo-51', 'atingaĵo'),
        'dir': ('diro-37', 'aktiveco'),
        'sci': ('scio-30', 'stato'),
        'vid': ('vido-30', 'percepto'),
        'aŭd': ('aŭdo-47', 'percepto'),
        'pens': ('pensado-29', 'aktiveco'),
        'don': ('translokigo-11', 'atingaĵo'),
        'pren': ('translokigo-11', 'atingaĵo'),
        'met': ('translokigo-11', 'atingaĵo'),
        'viv': ('ekzisto-47', 'stato'),
        'mort': ('ekzisto-47', 'atingaĵo'),
        'am': ('amo-31', 'stato'),
        'parol': ('diro-37', 'aktiveco'),
        'skrib': ('kreado-26', 'plenumigo'),
        'leg': ('scio-30', 'aktiveco'),
        'aŭskult': ('aŭdo-47', 'aktiveco'),
        'kur': ('movo-51', 'aktiveco'),
        'salt': ('movo-51', 'atingaĵo'),
        'flu': ('movo-51', 'aktiveco'),
        'fal': ('movo-51', 'atingaĵo'),
        'lev': ('movo-51', 'atingaĵo'),
        'sid': ('movo-51', 'stato'),
        'star': ('movo-51', 'stato'),
        'kuŝ': ('movo-51', 'stato'),
        'manĝ': ('kreado-26', 'aktiveco'),
        'trink': ('kreado-26', 'aktiveco'),
        'dorm': ('ekzisto-47', 'stato'),
        'labor': ('kreado-26', 'aktiveco'),
        'lud': ('kreado-26', 'aktiveco'),
        'kant': ('diro-37', 'aktiveco'),
        'danc': ('movo-51', 'aktiveco'),
        'rid': ('amo-31', 'atingaĵo'),
        'plor': ('timo-31', 'atingaĵo'),
        'tim': ('timo-31', 'stato'),
        'esper': ('pensado-29', 'stato'),
        'dezir': ('pensado-29', 'stato'),
        'vol': ('pensado-29', 'stato'),
        'dev': ('pensado-29', 'stato'),
        'pov': ('pensado-29', 'stato'),
        'kre': ('kreado-26', 'plenumigo'),
        'konstru': ('kreado-26', 'plenumigo'),
        'detru': ('detruo-44', 'atingaĵo'),
        'romp': ('detruo-44', 'atingaĵo'),
        'ŝanĝ': ('ŝanĝo-45', 'plenumigo'),
        'kresk': ('kreskado-26', 'aktiveco'),
        'komenc': ('ŝanĝo-45', 'atingaĵo'),
        'fin': ('ŝanĝo-45', 'atingaĵo'),
        'send': ('translokigo-11', 'plenumigo'),
        'port': ('translokigo-11', 'aktiveco'),
        'aĉet': ('translokigo-11', 'atingaĵo'),
        'vend': ('translokigo-11', 'atingaĵo'),
        'don': ('translokigo-11', 'atingaĵo'),
        'ricev': ('translokigo-11', 'atingaĵo'),
        'serv': ('kreado-26', 'aktiveco'),
        'help': ('kreado-26', 'aktiveco'),
        'gvid': ('movo-51', 'aktiveco'),
        'sekv':  ('movo-51', 'aktiveco'),
        'atend': ('pensado-29', 'stato'),
        'serĉ': ('demando-40', 'aktiveco'),
        'trov': ('demando-40', 'atingaĵo'),
        'perd': ('detruo-44', 'atingaĵo'),
        'gajn': ('translokigo-11', 'atingaĵo'),
        'vink': ('kreado-26', 'atingaĵo'),
    }

    # Common noun roots
    noun_roots = {
        'hom': ('persono', 'socia'),
        'vir': ('persono', 'socia'),
        'virin': ('persono', 'socia'),
        'infan': ('persono', 'socia'),
        'patr': ('rolo', 'socia'),
        'patr': ('rolo', 'socia'),
        'matr': ('rolo', 'socia'),
        'frat': ('rolo', 'socia'),
        'fil': ('rolo', 'socia'),
        'amik': ('rolo', 'socia'),
        'hund': ('animalo', 'natura'),
        'kat': ('animalo', 'natura'),
        'ĉeval': ('animalo', 'natura'),
        'bird': ('animalo', 'natura'),
        'fiŝ': ('animalo', 'natura'),
        'arb': ('planto', 'natura'),
        'flor': ('planto', 'natura'),
        'herb': ('planto', 'natura'),
        'dom': ('konstruaĵo', 'socia'),
        'urb': ('loko', 'socia'),
        'land': ('loko', 'socia'),
        'mond': ('loko', 'natura'),
        'ter': ('loko', 'natura'),
        'ĉiel': ('loko', 'natura'),
        'mar': ('natura_loko', 'natura'),
        'river': ('natura_loko', 'natura'),
        'mont': ('natura_loko', 'natura'),
        'sun': ('loko', 'natura'),
        'lun': ('loko', 'natura'),
        'stel': ('loko', 'natura'),
        'akv': ('loko', 'natura'),
        'aer': ('loko', 'natura'),
        'fajr': ('loko', 'natura'),
        'tag': ('koncepto', 'socia'),
        'nokt': ('koncepto', 'socia'),
        'jar': ('koncepto', 'socia'),
        'temp': ('koncepto', 'socia'),
        'lok': ('koncepto', 'socia'),
        'hejm': ('loko', 'socia'),
        'voj': ('konstruaĵo', 'socia'),
        'strat': ('konstruaĵo', 'socia'),
        'urb': ('loko', 'socia'),
        'vilaĝ': ('loko', 'socia'),
        'ŝtat': ('loko', 'socia'),
        'reĝ': ('rolo', 'socia'),
        'prezident': ('rolo', 'socia'),
        'ministr': ('rolo', 'socia'),
        'kuracist': ('profesio', 'socia'),
        'instruist': ('profesio', 'socia'),
        'stud': ('rolo', 'socia'),
        'labor': ('rolo', 'socia'),
        'komercist': ('profesio', 'ekonomia'),
        'mier': ('profesio', 'socia'),
        'nom': ('koncepto', 'socia'),
        'vort': ('koncepto', 'kultura'),
        'lingv': ('koncepto', 'kultura'),
        'libr': ('ilo', 'kultura'),
        'paper': ('ilo', 'kultura'),
        'pen': ('ilo', 'kultura'),
        'krajon': ('ilo', 'kultura'),
        'tabel': ('ilo', 'socia'),
        'seĝ': ('ilo', 'socia'),
        'lit': ('ilo', 'socia'),
        'vest': ('ilo', 'socia'),
        'ŝu': ('ilo', 'socia'),
        'ĉapel': ('ilo', 'socia'),
        'pan': ('manĝaĵo', 'socia'),
        'vian': ('manĝaĵo', 'socia'),
        'legom': ('manĝaĵo', 'natura'),
        'frukt': ('manĝaĵo', 'natura'),
        'aŭt': ('veturilo', 'scienca'),
        'trajn': ('veturilo', 'scienca'),
        'ŝip': ('veturilo', 'scienca'),
        'aviadil': ('veturilo', 'scienca'),
        'bicipl': ('veturilo', 'scienca'),
        'telefon': ('ilo', 'scienca'),
        'komputik': ('ilo', 'scienca'),
        'ide': ('koncepto', 'socia'),
        'pens': ('koncepto', 'socia'),
        'sci': ('koncepto', 'scienca'),
        'art': ('koncepto', 'kultura'),
        'muzik': ('koncepto', 'kultura'),
        'kant': ('koncepto', 'kultura'),
        'poem': ('koncepto', 'kultura'),
        'rakont': ('koncepto', 'kultura'),
        'histori': ('koncepto', 'kultura'),
        'filozofi': ('koncepto', 'kultura'),
        'religi': ('koncepto', 'kultura'),
        'dio': ('koncepto', 'kultura'),
        'eklezi': ('konstruaĵo', 'kultura'),
        'preĝ': ('koncepto', 'kultura'),
        'fest': ('evento', 'kultura'),
        'milit': ('evento', 'socia'),
        'pac': ('koncepto', 'socia'),
        'libert': ('koncepto', 'socia'),
        'rajt': ('koncepto', 'socia'),
        'dev': ('koncepto', 'socia'),
        'leĝ': ('koncepto', 'socia'),
        'justik': ('koncepto', 'socia'),
        'san': ('kvalito', 'socia'),
        'malsam': ('kvalito', 'socia'),
        'bel': ('kvalito', 'socia'),
        'bon': ('kvalito', 'socia'),
        'fort': ('kvalito', 'socia'),
        'saĝ': ('kvalito', 'socia'),
        'riĉ': ('kvalito', 'ekonomia'),
        'pov': ('kvalito', 'ekonomia'),
    }

    category = {}

    if radiko in verb_roots:
        verb_class, aspect = verb_roots[radiko]
        category['verba_klaso'] = verb_class
        category['aspekta_klaso'] = aspect
        category['semantika_kampo'] = 'socia'  # Default, refine manually
    elif radiko in noun_roots:
        noun_class, field = noun_roots[radiko]
        category['substantiva_klaso'] = noun_class
        category['semantika_kampo'] = field
    else:
        # Unknown - mark for manual review
        category['needs_manual_review'] = True

    return category


def select_candidates(db_path: str, target_count: int = 50) -> List[Dict[str, Any]]:
    """Select top candidate roots for annotation."""
    print(f"🔍 Analyzing roots in: {db_path}")

    # Connect to database
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Get already annotated roots
    print("📊 Checking already annotated roots...")
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.verba_klaso IS NOT NULL OR r.substantiva_klaso IS NOT NULL
        RETURN r.radiko
    """)

    already_annotated = set()
    while result.has_next():
        already_annotated.add(result.get_next()[0])

    print(f"✅ Found {len(already_annotated)} already annotated roots")

    # Get Fundamento roots
    print("📚 Querying Fundamento roots...")
    fundamento = get_fundamento_roots(conn)
    print(f"✅ Found {len(fundamento)} Fundamento roots")

    # Get high-frequency roots
    print("📈 Querying high-frequency roots...")
    frequent = get_high_frequency_roots(conn, limit=200)
    print(f"✅ Found {len(frequent)} high-frequency roots")

    # Build candidate list
    candidates = []
    seen = already_annotated.copy()

    # Priority 1: Unannotated Fundamento roots
    for radiko in fundamento:
        if radiko not in seen and len(candidates) < target_count:
            category = categorize_root(radiko)
            candidates.append({
                'radiko': radiko,
                'priority': 1,
                'reason': 'Fundamento root',
                'funda_stato': 'fundamento_kerno',
                'estas_funda': True,
                'ofteca_tavolo': 0,
                **category
            })
            seen.add(radiko)

    # Priority 2: High-frequency roots
    for radiko, tier in frequent:
        if radiko not in seen and len(candidates) < target_count:
            category = categorize_root(radiko)
            candidates.append({
                'radiko': radiko,
                'priority': 2,
                'reason': f'High frequency (tier {tier})',
                'funda_stato': 'vortaro_agnoskita' if radiko not in fundamento else 'fundamento_kerno',
                'estas_funda': radiko in fundamento,
                'ofteca_tavolo': tier,
                **category
            })
            seen.add(radiko)

    print(f"\n✅ Selected {len(candidates)} candidate roots")
    print(f"   - Priority 1 (Fundamento): {sum(1 for c in candidates if c['priority'] == 1)}")
    print(f"   - Priority 2 (High-frequency): {sum(1 for c in candidates if c['priority'] == 2)}")
    print(f"   - Need manual review: {sum(1 for c in candidates if c.get('needs_manual_review'))}")

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Select candidate roots for Phase 0 annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--database',
        type=str,
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/candidates.jsonl',
        help='Output JSONL file for candidates'
    )

    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='Number of candidates to select (default: 50)'
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    # Select candidates
    candidates = select_candidates(str(db_path), args.count)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for candidate in candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + '\n')

    print(f"\n💾 Saved candidates to: {output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Review {output_path}")
    print(f"   2. Manually refine semantic classifications")
    print(f"   3. Add importance scores (graveco_biografia, etc.)")
    print(f"   4. Remove 'priority' and 'reason' fields")
    print(f"   5. Save as data/annotations/phase_0_roots.jsonl")
    print(f"   6. Run: python scripts/load_semantic_annotations.py --annotations phase_0_roots.jsonl")


if __name__ == '__main__':
    main()
