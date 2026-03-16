#!/usr/bin/env python3
"""
Extract Fundamento roots from the Universala Vortaro section.

The Fundamento text has roots marked with apostrophe ('), e.g.:
    aks' axe | axle | Achse | ocb | oŝ.
    akv' eau | water | Wasser | Boaa | woda.

Usage:
    python scripts/extract_fundamento_roots.py
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def extract_roots_from_fundamento(fundamento_path: Path) -> dict:
    """Extract roots from Fundamento Universala Vortaro."""
    print(f"Reading Fundamento from: {fundamento_path}")

    with open(fundamento_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find start of Universala Vortaro section
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "UNIVERSALA VORTARO":
            start_idx = i
            break

    if start_idx is None:
        print("ERROR: Could not find 'UNIVERSALA VORTARO' section")
        return {}

    print(f"Found Universala Vortaro at line {start_idx + 1}")

    # Extract roots (pattern: word' translation | translation | ...)
    # Roots are marked with apostrophe immediately after the root
    root_pattern = re.compile(r"^\s*([a-z]+(?:aŭ)?)'")

    roots = {}
    root_count = 0

    for i in range(start_idx, len(lines)):
        line = lines[i]
        match = root_pattern.match(line)

        if match:
            root = match.group(1)
            # Remove the aŭ suffix if present (e.g., almenaŭ, anstataŭ)
            clean_root = root.replace('aŭ', '')

            # Extract first translation (usually in Esperanto description)
            # Format: root' translation | translation | ...
            parts = line.split('|')
            if parts:
                # Get text between root' and first |
                desc = parts[0].strip()
                # Remove the root' prefix
                desc = desc[len(root) + 1:].strip()
            else:
                desc = ""

            roots[root] = {
                'fundamento': True,
                'raw_line': line.strip(),
                'description': desc,
                'source': 'Fundamento Universala Vortaro'
            }
            root_count += 1

    print(f"Extracted {root_count} Fundamento roots")
    return roots


def extract_roots_from_revo(revo_db_path: Path) -> dict:
    """Extract roots from ReVo SQLite database."""
    print(f"Reading ReVo database: {revo_db_path}")

    try:
        import sqlite3
    except ImportError:
        print("WARNING: sqlite3 not available, skipping ReVo extraction")
        return {}

    if not revo_db_path.exists():
        print("WARNING: ReVo database not found")
        return {}

    conn = sqlite3.connect(str(revo_db_path))
    cursor = conn.cursor()

    # Get table schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"  ReVo tables: {[t[0] for t in tables]}")

    # Try to extract roots from main table
    # ReVo typically has tables like 'vortoj' or 'entries'
    roots = {}

    for table in ['vortoj', 'entries', 'roots']:
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            columns = [desc[0] for desc in cursor.description]
            print(f"  Table '{table}' columns: {columns}")

            # Try to find root column
            root_col = None
            for col in ['radiko', 'root', 'mrk', 'vorto']:
                if col in columns:
                    root_col = col
                    break

            if root_col:
                cursor.execute(f"SELECT DISTINCT {root_col} FROM {table}")
                revo_roots = cursor.fetchall()
                print(f"  Found {len(revo_roots)} roots in {table}.{root_col}")

                for (root,) in revo_roots[:10]:  # Sample first 10
                    print(f"    Sample: {root}")

        except sqlite3.OperationalError:
            continue

    conn.close()
    return roots


def main():
    parser = argparse.ArgumentParser(description='Extract Fundamento roots')
    parser.add_argument('--fundamento', type=Path,
                       default=Path('data/raw/eo/fundamento/fundamento_de_esperanto.txt'),
                       help='Path to Fundamento text file')
    parser.add_argument('--revo', type=Path,
                       default=Path('data/raw/eo/dictionaries/revo/revo.db'),
                       help='Path to ReVo SQLite database')
    parser.add_argument('--output', type=Path,
                       default=Path('data/vocabularies'),
                       help='Output directory')

    args = parser.parse_args()

    # Extract Fundamento roots
    fundamento_roots = extract_roots_from_fundamento(args.fundamento)

    # Extract ReVo roots (technical vocabulary)
    revo_roots = extract_roots_from_revo(args.revo)

    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)

    print("\nSaving outputs...")

    # Save Fundamento roots
    fundamento_file = args.output / 'fundamento_roots.json'
    with open(fundamento_file, 'w', encoding='utf-8') as f:
        json.dump(fundamento_roots, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {fundamento_file} ({len(fundamento_roots):,} roots)")

    # Statistics
    print("\n=== Summary ===")
    print(f"  Fundamento roots: {len(fundamento_roots):,}")

    # Show sample roots
    print("\nSample Fundamento roots:")
    for root, data in list(fundamento_roots.items())[:20]:
        desc = data.get('description', '')[:50]
        print(f"  {root}: {desc}")

    print("\nNext steps:")
    print("  1. Use these roots as Tier 1 vocabulary for training")
    print("  2. Compare with corpus_validated_roots.json")
    print("  3. Build hierarchical root classification system")


if __name__ == '__main__':
    main()
