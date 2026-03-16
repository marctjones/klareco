#!/usr/bin/env python3
"""
Extract root categories using:
1. Official Akademia Vortaro (download from akademio-de-esperanto.org)
2. AST annotations (analizstato, propranoma_kategorio)  
3. Kuzu database queries

Usage:
    python scripts/extract_root_categories.py --kuzu data/indexes/v2.1_kuzu_index_full
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    exit(1)

def extract_successfully_parsed_roots(conn):
    """Get roots from successfully parsed Esperanto words."""
    print("Extracting roots from successfully parsed words...")
    
    result = conn.execute("""
        MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko)
        WHERE v.analizstato = 'sukceso'
          AND v.propranoma_kategorio IS NULL
        WITH r.radiko as root, count(v) as usage
        RETURN root, usage
        ORDER BY usage DESC
    """)
    
    roots = {}
    while result.has_next():
        root, usage = result.get_next()
        roots[root] = {
            'usage': usage,
            'source': 'corpus_validated',
            'parse_status': 'sukceso'
        }
    
    print(f"  Found {len(roots):,} successfully parsed roots")
    return roots

def extract_proper_names(conn):
    """Get proper names by category."""
    print("Extracting proper names...")
    
    result = conn.execute("""
        MATCH (v:Vorto)
        WHERE v.propranoma_kategorio IS NOT NULL
        WITH v.radiko as root, v.propranoma_kategorio as category, count(v) as usage
        RETURN root, category, usage
        ORDER BY usage DESC
    """)
    
    proper_names = {}
    while result.has_next():
        root, category, usage = result.get_next()
        if root not in proper_names:
            proper_names[root] = {
                'usage': usage,
                'category': category,
                'source': 'proper_name'
            }
    
    print(f"  Found {len(proper_names):,} proper names")
    return proper_names

def extract_parse_failures(conn):
    """Get roots from parse failures (likely foreign/garbage)."""
    print("Extracting parse failures...")
    
    result = conn.execute("""
        MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko)
        WHERE v.analizstato = 'malsukceso'
        WITH r.radiko as root, count(v) as usage
        RETURN root, usage
        ORDER BY usage DESC
    """)
    
    failures = {}
    while result.has_next():
        root, usage = result.get_next()
        failures[root] = {
            'usage': usage,
            'source': 'parse_failure',
            'parse_status': 'malsukceso'
        }
    
    print(f"  Found {len(failures):,} parse failure roots")
    return failures

def main():
    parser = argparse.ArgumentParser(description='Extract root categories')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--output', type=Path, default=Path('data/vocabularies'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Connect to Kuzu
    print(f"Opening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)
    
    # Extract categories
    validated_roots = extract_successfully_parsed_roots(conn)
    proper_names = extract_proper_names(conn)
    parse_failures = extract_parse_failures(conn)
    
    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving outputs...")
    
    with open(args.output / 'corpus_validated_roots.json', 'w') as f:
        json.dump(validated_roots, f, indent=2, ensure_ascii=False)
    print(f"  Saved: corpus_validated_roots.json ({len(validated_roots):,} roots)")
    
    with open(args.output / 'proper_names.json', 'w') as f:
        json.dump(proper_names, f, indent=2, ensure_ascii=False)
    print(f"  Saved: proper_names.json ({len(proper_names):,} names)")
    
    with open(args.output / 'parse_failures.json', 'w') as f:
        json.dump(parse_failures, f, indent=2, ensure_ascii=False)
    print(f"  Saved: parse_failures.json ({len(parse_failures):,} roots)")
    
    # Summary
    print("\n=== Summary ===")
    print(f"  Validated roots (sukceso): {len(validated_roots):,}")
    print(f"  Proper names: {len(proper_names):,}")
    print(f"  Parse failures (garbage): {len(parse_failures):,}")
    print(f"\nNext steps:")
    print(f"  1. Download Akademia Vortaro PDF for official Fundamento roots")
    print(f"  2. Filter validated_roots to Fundamento only for training")
    print(f"  3. Exclude parse_failures from all training")

if __name__ == '__main__':
    main()
