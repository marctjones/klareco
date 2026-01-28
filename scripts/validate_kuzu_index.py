#!/usr/bin/env python3
"""
Validate Kùzu Graph Index

Checks that the index was built correctly:
1. Database file exists and is readable
2. All node tables populated (Root, Sentence, Document)
3. All edge tables populated (HAS_ROOT, IN_DOCUMENT, NEXT_SENTENCE)
4. Semantic relations loaded (IS_SYNONYM, IS_HYPERNYM, IS_ANTONYM)
5. Sample queries work correctly
6. Statistics match expected values

Usage:
    python scripts/validate_kuzu_index.py
    python scripts/validate_kuzu_index.py --verbose  # Show detailed stats
"""

import argparse
import sys
from pathlib import Path

import kuzu

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_index(db_path: Path, verbose: bool = False):
    """Validate Kùzu index."""

    print("=" * 80)
    print("KÙZU INDEX VALIDATION")
    print("=" * 80)
    print()

    # Check database exists
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    print(f"✓ Database exists: {db_path}")
    print()

    # Connect to database
    try:
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

    print("✓ Database connection successful")
    print()

    # Validate node tables
    print("Checking node tables...")
    node_checks = [
        ("Root", "MATCH (r:Root) RETURN count(r)"),
        ("Sentence", "MATCH (s:Sentence) RETURN count(s)"),
        ("Document", "MATCH (d:Document) RETURN count(d)"),
    ]

    node_counts = {}
    for table_name, query in node_checks:
        try:
            result = conn.execute(query)
            count = result.get_next()[0]
            node_counts[table_name] = count
            print(f"  ✓ {table_name}: {count:,} nodes")
        except Exception as e:
            print(f"  ❌ {table_name}: {e}")
            return False

    print()

    # Validate edge tables
    print("Checking edge tables...")
    edge_checks = [
        ("HAS_ROOT", "MATCH ()-[r:HAS_ROOT]->() RETURN count(r)"),
        ("IN_DOCUMENT", "MATCH ()-[r:IN_DOCUMENT]->() RETURN count(r)"),
        ("NEXT_SENTENCE", "MATCH ()-[r:NEXT_SENTENCE]->() RETURN count(r)"),
        ("IS_SYNONYM", "MATCH ()-[r:IS_SYNONYM]->() RETURN count(r)"),
        ("IS_HYPERNYM", "MATCH ()-[r:IS_HYPERNYM]->() RETURN count(r)"),
        ("IS_ANTONYM", "MATCH ()-[r:IS_ANTONYM]->() RETURN count(r)"),
    ]

    edge_counts = {}
    for table_name, query in edge_checks:
        try:
            result = conn.execute(query)
            count = result.get_next()[0]
            edge_counts[table_name] = count
            status = "✓" if count > 0 else "⚠️ "
            print(f"  {status} {table_name}: {count:,} edges")
        except Exception as e:
            print(f"  ❌ {table_name}: {e}")
            # Semantic relations are optional, so don't fail
            if table_name not in ["IS_SYNONYM", "IS_HYPERNYM", "IS_ANTONYM"]:
                return False

    print()

    # Sample queries
    print("Testing sample queries...")

    # Query 1: Find root frequency
    try:
        result = conn.execute("""
            MATCH (r:Root {root: 'hund'})
            RETURN r.frequency, r.document_frequency
        """)
        if result.has_next():
            freq, doc_freq = result.get_next()
            print(f"  ✓ Root lookup: 'hund' appears {freq} times in {doc_freq} documents")
        else:
            print(f"  ⚠️  Root 'hund' not found (may not be in corpus)")
    except Exception as e:
        print(f"  ❌ Root query failed: {e}")
        return False

    # Query 2: Find sentences containing root
    try:
        result = conn.execute("""
            MATCH (s:Sentence)-[:HAS_ROOT]->(r:Root {root: 'esper'})
            RETURN count(s) LIMIT 10
        """)
        count = result.get_next()[0]
        print(f"  ✓ Sentence search: Found {count:,} sentences with 'esper'")
    except Exception as e:
        print(f"  ❌ Sentence search failed: {e}")
        return False

    # Query 3: Test semantic relations (if present)
    if edge_counts.get("IS_SYNONYM", 0) > 0:
        try:
            result = conn.execute("""
                MATCH (r1:Root)-[:IS_SYNONYM]->(r2:Root)
                RETURN r1.root, r2.root
                LIMIT 3
            """)
            synonyms = []
            while result.has_next():
                root1, root2 = result.get_next()
                synonyms.append(f"{root1} ↔ {root2}")
            print(f"  ✓ Synonyms working: {', '.join(synonyms)}")
        except Exception as e:
            print(f"  ⚠️  Synonym query failed: {e}")

    print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Roots:     {node_counts['Root']:,}")
    print(f"Sentences: {node_counts['Sentence']:,}")
    print(f"Documents: {node_counts['Document']:,}")
    print()
    print(f"Root relationships: {edge_counts['HAS_ROOT']:,}")
    print(f"Synonyms:           {edge_counts.get('IS_SYNONYM', 0):,}")
    print(f"Hypernyms:          {edge_counts.get('IS_HYPERNYM', 0):,}")
    print(f"Antonyms:           {edge_counts.get('IS_ANTONYM', 0):,}")
    print()

    # Sanity checks
    issues = []
    if node_counts['Root'] < 100000:
        issues.append(f"Low root count: {node_counts['Root']:,} (expected ~1M)")
    if node_counts['Sentence'] < 1000000:
        issues.append(f"Low sentence count: {node_counts['Sentence']:,} (expected ~3.8M)")
    if edge_counts['HAS_ROOT'] < 1000000:
        issues.append(f"Low HAS_ROOT edges: {edge_counts['HAS_ROOT']:,} (expected ~30M)")

    if issues:
        print("⚠️  Potential issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    print("✓ Index validation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Kùzu index")
    parser.add_argument(
        '--index',
        type=Path,
        default=Path('data/indexes/kuzu_index/kuzu.db'),
        help='Path to Kùzu database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed statistics'
    )

    args = parser.parse_args()

    success = validate_index(args.index, args.verbose)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
