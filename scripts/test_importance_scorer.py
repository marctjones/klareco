#!/usr/bin/env python3
"""
Test Importance Scorer

VERSION: v2.1
STAGE: Evaluation
DEPENDENCIES: klareco.summarization.ImportanceScorer

Description:
    Tests importance scorer on sample facts using annotated roots.

Usage:
    python scripts/test_importance_scorer.py \
        --database data/indexes/v2.1_kuzu_index_full

Last Updated: 2026-03-09
Author: Claude Code
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.summarization import ImportanceScorer


def test_scorer(db_path: str):
    """Test importance scorer on sample facts."""
    print(f"🔍 Testing ImportanceScorer with database: {db_path}\n")

    # Initialize scorer
    scorer = ImportanceScorer(db_path)

    # Sample facts (using our annotated roots)
    test_facts = [
        # Biographical facts
        {
            'schema': 'biographical',
            'facts': [
                {'predicate': 'fond', 'subject': 'Zamenhof', 'object': 'Esperanton', 'subject_root': 'hom', 'object_root': 'nom'},
                {'predicate': 'viv', 'subject': 'Zamenhof', 'object': '', 'subject_root': 'hom', 'object_root': ''},
                {'predicate': 'mort', 'subject': 'Zamenhof', 'object': '1917', 'subject_root': 'hom', 'object_root': 'jar'},
                {'predicate': 'est', 'subject': 'Zamenhof', 'object': 'kuracisto', 'subject_root': 'hom', 'object_root': 'hom'},
                {'predicate': 'kre', 'subject': 'Zamenhof', 'object': 'lingvon', 'subject_root': 'hom', 'object_root': 'nom'},
            ]
        },
        # Definitional facts
        {
            'schema': 'definitional',
            'facts': [
                {'predicate': 'est', 'subject': 'Esperanto', 'object': 'lingvo', 'subject_root': 'nom', 'object_root': 'nom'},
                {'predicate': 'hav', 'subject': 'Esperanto', 'object': 'regulojn', 'subject_root': 'nom', 'object_root': 'nom'},
                {'predicate': 'parol', 'subject': 'homoj', 'object': 'Esperanton', 'subject_root': 'hom', 'object_root': 'nom'},
            ]
        },
        # Event facts
        {
            'schema': 'event',
            'facts': [
                {'predicate': 'okazis', 'subject': 'kongreso', 'object': '1887', 'subject_root': 'okazaĵ', 'object_root': 'jar', 'temporal_marker': True},
                {'predicate': 'venis', 'subject': 'delegitoj', 'object': 'Bulonjo', 'subject_root': 'hom', 'object_root': 'urb', 'spatial_marker': True},
                {'predicate': 'fondis', 'subject': 'Zamenhof', 'object': 'Esperanton', 'subject_root': 'hom', 'object_root': 'nom'},
            ]
        }
    ]

    # Test each schema
    for test_case in test_facts:
        schema = test_case['schema']
        facts = test_case['facts']

        print("=" * 100)
        print(f"SCHEMA: {schema.upper()}")
        print("=" * 100)
        print()

        # Score facts
        scored_facts = scorer.score_facts(facts, schema)

        # Display results
        for i, scored in enumerate(scored_facts, 1):
            print(f"{i}. Score: {scored.score:.3f} | {scored.fact['predicate']:10} | {scored.fact['subject']:15} {scored.fact['object']:15}")
            print(f"   Explanation: {', '.join(scored.explanation)}")
            print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("✅ Scorer initialized successfully")
    print("✅ All sample facts scored")
    print()
    print("Key observations:")
    print("  - High-importance roots (fond, viv, mort, kre) should score ≥0.70")
    print("  - Biografical schema prioritizes life events")
    print("  - Definitional schema prioritizes category/properties")
    print("  - Event schema prioritizes temporal/spatial info")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test importance scorer on sample facts"
    )

    parser.add_argument(
        '--database',
        type=str,
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database'
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    test_scorer(str(db_path))


if __name__ == '__main__':
    main()
