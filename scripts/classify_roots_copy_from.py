#!/usr/bin/env python3
"""
FASTEST POSSIBLE: Use COPY FROM + single JOIN UPDATE.

Strategy:
1. Classify all 1.2M nodes in Python (fast dict lookups)
2. Write classifications to CSV file
3. COPY FROM CSV into temporary node table (Kuzu's fastest bulk operation)
4. Single UPDATE query with JOIN (one checkpoint instead of 1.2M!)
5. Drop temp table

Expected: 10-20 seconds total!

Usage:
    python scripts/classify_roots_copy_from.py --kuzu data/indexes/v2.1_kuzu_index_full
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    exit(1)

# Import classification functions
from classify_roots_v2 import (
    load_classification_data,
    determine_classification,
    propagate_to_vorto_nodes,
    verify_classification
)


def classify_radiko_nodes_copy_from(conn, classification_data):
    """FASTEST: Use COPY FROM + single JOIN UPDATE."""

    print("\nClassifying Radiko nodes (COPY FROM method)...")
    print("  Strategy: Classify in Python → CSV → COPY FROM → Single UPDATE with JOIN")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    # Step 1: Fetch all radiko strings
    print("\n  Step 1: Fetching all radiko strings...")
    fetch_start = time.time()

    result = conn.execute("MATCH (r:Radiko) RETURN r.radiko")
    all_radikos = []
    while result.has_next():
        (radiko,) = result.get_next()
        all_radikos.append(radiko)

    fetch_time = time.time() - fetch_start
    print(f"  ✓ Fetched {len(all_radikos):,} radiko strings in {fetch_time:.1f} seconds")

    # Step 2: Classify all in Python
    print("\n  Step 2: Classifying all nodes in Python...")
    classify_start = time.time()

    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    classifications = []
    for radiko in all_radikos:
        nivelo, fonto = determine_classification(
            radiko, tier0_words, unua_libro_roots, tier1b_roots,
            revo_roots, corpus_roots, proper_names, parse_failures
        )
        ofteco = corpus_usage.get(radiko, 0)

        classifications.append((radiko, nivelo, fonto or '', ofteco))

        tier_counts[nivelo] += 1
        if fonto:
            source_counts[fonto] += 1

    classify_time = time.time() - classify_start
    print(f"  ✓ Classified {len(classifications):,} nodes in {classify_time:.1f} seconds")

    # Step 3: Write to CSV
    print("\n  Step 3: Writing classifications to CSV...")
    csv_path = Path('/tmp/radiko_classifications.csv')
    csv_start = time.time()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['radiko', 'nivelo', 'fonto', 'ofteco'])
        # Write data
        writer.writerows(classifications)

    csv_time = time.time() - csv_start
    print(f"  ✓ Wrote {len(classifications):,} rows to {csv_path} in {csv_time:.1f} seconds")

    # Step 4: Create temporary node table
    print("\n  Step 4: Creating temporary classification table...")
    try:
        conn.execute("DROP TABLE RadikoClassificationTemp")
    except:
        pass  # Table doesn't exist yet

    conn.execute("""
        CREATE NODE TABLE RadikoClassificationTemp (
            radiko STRING PRIMARY KEY,
            nivelo STRING,
            fonto STRING,
            ofteco INT64
        )
    """)
    print("  ✓ Created temporary table")

    # Step 5: COPY FROM CSV (fastest bulk operation!)
    print("\n  Step 5: Bulk loading CSV with COPY FROM...")
    copy_start = time.time()

    conn.execute(f"""
        COPY RadikoClassificationTemp FROM '{csv_path}' (HEADER=true)
    """)

    copy_time = time.time() - copy_start
    print(f"  ✓ Loaded {len(classifications):,} rows in {copy_time:.1f} seconds")

    # Step 6: Single UPDATE with JOIN (ONE checkpoint!)
    print("\n  Step 6: Updating Radiko nodes with single JOIN query...")
    update_start = time.time()

    conn.execute("""
        MATCH (r:Radiko), (t:RadikoClassificationTemp)
        WHERE r.radiko = t.radiko
        SET r.nivelo = t.nivelo,
            r.fonto = t.fonto,
            r.ofteco = t.ofteco
    """)

    update_time = time.time() - update_start
    print(f"  ✓ Updated {len(classifications):,} nodes in {update_time:.1f} seconds")

    # Step 7: Cleanup
    print("\n  Step 7: Cleaning up...")
    conn.execute("DROP TABLE RadikoClassificationTemp")
    csv_path.unlink()
    print("  ✓ Dropped temporary table and deleted CSV")

    # Total time
    total_time = fetch_time + classify_time + csv_time + copy_time + update_time
    print(f"\n  ✓ Total classification time: {total_time:.1f} seconds")

    # Print distribution
    print("\n  Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / len(classifications) if len(classifications) > 0 else 0
        print(f"    {tier}: {count:,} ({pct:.1f}%)")

    print("\n  Source distribution:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        pct = 100 * count / len(classifications) if len(classifications) > 0 else 0
        print(f"    {source}: {count:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='FASTEST classification with COPY FROM')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'))

    args = parser.parse_args()

    # Load classification data
    classification_data = load_classification_data(args.vocab_dir)

    # Connect to Kuzu
    print(f"\nOpening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)

    # Ensure schema
    print("\nEnsuring schema has required properties...")
    for table, prop, dtype in [
        ('Radiko', 'nivelo', 'STRING'),
        ('Radiko', 'fonto', 'STRING'),
        ('Radiko', 'ofteco', 'INT64'),
        ('Vorto', 'radiko_nivelo', 'STRING'),
        ('Vorto', 'radiko_fonto', 'STRING'),
        ('Vorto', 'radiko_ofteco', 'INT64'),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD {prop} {dtype}")
            print(f"  Added {table}.{prop}")
        except Exception as e:
            if "already exists" in str(e).lower():
                pass  # Silent
            else:
                print(f"  Error adding {table}.{prop}: {e}")

    # Classify with COPY FROM method
    classify_radiko_nodes_copy_from(conn, classification_data)

    # Propagate to Vorto
    checkpoint = {'radiko_done': True, 'vorto_batch': 0}
    checkpoint_file = Path('data/vocabularies/classification_checkpoint_copy_from.json')
    propagate_to_vorto_nodes(conn, checkpoint, checkpoint_file)

    # Verify
    verify_classification(conn)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print("\n✓ Classification complete!")


if __name__ == '__main__':
    main()
