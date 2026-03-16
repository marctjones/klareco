#!/usr/bin/env python3
"""
Add tier/category properties to Radiko and Vorto nodes in Kuzu database.

This enables querying the graph by root tier for training data generation:
- MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko) WHERE r.nivelo = 'tier1_fundamento'
- MATCH (v:Vorto) WHERE v.radiko_nivelo = 'tier1_fundamento'

Tiers:
- tier1a_fundamento_kerno: Top 900 most frequent Fundamento roots (core)
- tier1b_fundamento_etendita: Remaining 1,273 Fundamento roots (extended)
- tier2_revo: 21,113 ReVo extended vocabulary
- tier3_korpuso: Corpus-validated roots
- tier4_propranomo: Proper names
- tier5_rubaĵo: Parse failures (garbage)
- tier6_nekonata: Unknown/unclassified

Properties added:
- Radiko.nivelo: tier classification
- Radiko.ofteco: usage count in corpus
- Vorto.radiko_nivelo: tier of root (propagated from Radiko)
- Vorto.radiko_ofteco: root usage count (propagated from Radiko)

Usage:
    python scripts/add_root_tiers_to_kuzu.py --kuzu data/indexes/v2.1_kuzu_index_full
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


def load_tier_classification(vocab_dir: Path, core_count: int = 900) -> tuple[dict, dict]:
    """Load root classification from vocabulary files.

    Args:
        vocab_dir: Directory containing vocabulary JSON files
        core_count: Number of top-frequency Fundamento roots to mark as "core"

    Returns:
        (root_to_tier, root_to_frequency) - tier mapping and frequency counts
    """
    print("Loading tier classification...")
    print(f"  Using top {core_count} Fundamento roots as 'core' (frequency-based)")

    # Load classification
    class_file = vocab_dir / 'root_classification.json'
    if not class_file.exists():
        print(f"ERROR: Classification file not found: {class_file}")
        print("Run: python scripts/compare_root_sources.py first")
        exit(1)

    with open(class_file, 'r') as f:
        classification = json.load(f)

    # Load corpus usage data
    corpus_file = vocab_dir / 'corpus_validated_roots_clean.json'
    corpus_usage = {}
    if corpus_file.exists():
        with open(corpus_file, 'r') as f:
            corpus_data = json.load(f)
        corpus_usage = {root: data.get('usage', 0) for root, data in corpus_data.items()}

    # Build root -> tier and root -> frequency mappings
    root_to_tier = {}
    root_to_frequency = {}

    # Tier 1: Fundamento (split into core 900 and extended)
    fundamento_roots = classification['tier1_fundamento']['roots']

    # Get frequency for each Fundamento root
    fundamento_with_freq = []
    for root in fundamento_roots:
        freq = corpus_usage.get(root, 0)
        fundamento_with_freq.append((root, freq))
        root_to_frequency[root] = freq

    # Sort by frequency and split at core_count
    fundamento_with_freq.sort(key=lambda x: x[1], reverse=True)

    tier1a_count = 0
    tier1b_count = 0
    for i, (root, freq) in enumerate(fundamento_with_freq):
        if i < core_count:
            root_to_tier[root] = 'tier1a_fundamento_kerno'
            tier1a_count += 1
        else:
            root_to_tier[root] = 'tier1b_fundamento_etendita'
            tier1b_count += 1

    # Tier 2: ReVo
    for root in classification['tier2_revo']['roots']:
        root_to_tier[root] = 'tier2_revo'
        root_to_frequency[root] = corpus_usage.get(root, 0)

    # Tier 3: Corpus
    for root in classification['tier3_corpus']['roots']:
        root_to_tier[root] = 'tier3_korpuso'
        root_to_frequency[root] = corpus_usage.get(root, 0)

    # Tier 4: Proper names
    for root in classification['tier4_proper_names']['roots']:
        root_to_tier[root] = 'tier4_propranomo'
        root_to_frequency[root] = corpus_usage.get(root, 0)

    print(f"  Tier 1a (Fundamento core): {tier1a_count:,}")
    print(f"  Tier 1b (Fundamento extended): {tier1b_count:,}")
    print(f"  Tier 2 (ReVo): {len(classification['tier2_revo']['roots']):,}")
    print(f"  Tier 3 (Corpus): {len(classification['tier3_corpus']['roots']):,}")
    print(f"  Tier 4 (Proper names): {len(classification['tier4_proper_names']['roots']):,}")

    return root_to_tier, root_to_frequency


def load_parse_failures(vocab_dir: Path) -> set:
    """Load parse failure roots (tier 5 - garbage)."""
    failure_file = vocab_dir / 'parse_failures.json'
    if not failure_file.exists():
        return set()

    with open(failure_file, 'r') as f:
        failures = json.load(f)

    print(f"  Tier 5 (Garbage): {len(failures):,}")
    return set(failures.keys())


def add_tier_to_radiko_nodes(conn, root_to_tier: dict, root_to_frequency: dict, parse_failures: set):
    """Add nivelo (tier) and ofteco (frequency) properties to all Radiko nodes."""
    print("\nAdding tier and frequency properties to Radiko nodes...")

    # Get all Radiko nodes
    result = conn.execute("MATCH (r:Radiko) RETURN count(r)")
    total_radiko = result.get_next()[0]
    print(f"  Total Radiko nodes: {total_radiko:,}")

    # Track statistics
    tier_counts = defaultdict(int)

    # Process in batches to avoid memory issues
    batch_size = 10000
    processed = 0

    result = conn.execute("MATCH (r:Radiko) RETURN r.radiko")

    batch = []
    while result.has_next():
        (radiko,) = result.get_next()

        # Determine tier
        if radiko in parse_failures:
            tier = 'tier5_rubaĵo'
        elif radiko in root_to_tier:
            tier = root_to_tier[radiko]
        else:
            tier = 'tier6_nekonata'

        # Get frequency (default 0 if not found)
        frequency = root_to_frequency.get(radiko, 0)

        batch.append((radiko, tier, frequency))
        tier_counts[tier] += 1

        # Execute batch update
        if len(batch) >= batch_size:
            for root, tier_val, freq in batch:
                # Escape apostrophes
                root_escaped = root.replace("'", "\\'")
                tier_escaped = tier_val.replace("'", "\\'")

                conn.execute(f"""
                    MATCH (r:Radiko {{radiko: '{root_escaped}'}})
                    SET r.nivelo = '{tier_escaped}', r.ofteco = {freq}
                """)

            processed += len(batch)
            print(f"    Progress: {processed:,} / {total_radiko:,} ({100*processed//total_radiko}%)")
            batch = []

    # Process remaining batch
    if batch:
        for root, tier_val, freq in batch:
            root_escaped = root.replace("'", "\\'")
            tier_escaped = tier_val.replace("'", "\\'")

            conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root_escaped}'}})
                SET r.nivelo = '{tier_escaped}', r.ofteco = {freq}
            """)
        processed += len(batch)

    print(f"  Updated {processed:,} Radiko nodes")

    # Print tier distribution
    print("\n  Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / processed
        print(f"    {tier}: {count:,} ({pct:.1f}%)")


def add_tier_to_vorto_nodes(conn):
    """Add radiko_nivelo and radiko_ofteco properties to Vorto nodes (propagate from Radiko)."""
    print("\nAdding radiko_nivelo and radiko_ofteco properties to Vorto nodes...")

    # Get total Vorto count
    result = conn.execute("MATCH (v:Vorto) RETURN count(v)")
    total_vorto = result.get_next()[0]
    print(f"  Total Vorto nodes: {total_vorto:,}")

    # Propagate tier and frequency from Radiko to Vorto via HAVAS_RADIKON relationship
    # Process in batches by Vorto ID range
    batch_size = 1_000_000
    result = conn.execute("MATCH (v:Vorto) RETURN min(v.id), max(v.id)")
    min_id, max_id = result.get_next()

    total_batches = (max_id - min_id + batch_size) // batch_size
    print(f"  Processing {total_batches} batches...")

    for batch_num in range(total_batches):
        start_id = min_id + (batch_num * batch_size)
        end_id = min(start_id + batch_size - 1, max_id)

        # Update Vorto.radiko_nivelo and radiko_ofteco from connected Radiko
        conn.execute(f"""
            MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko)
            WHERE v.id >= {start_id} AND v.id <= {end_id}
              AND r.nivelo IS NOT NULL
            SET v.radiko_nivelo = r.nivelo, v.radiko_ofteco = r.ofteco
        """)

        percent = 100 * (batch_num + 1) // total_batches
        print(f"    Progress: {batch_num + 1}/{total_batches} ({percent}%)")

    print(f"  Updated Vorto.radiko_nivelo and radiko_ofteco properties")


def verify_tier_distribution(conn):
    """Verify tier distribution in database."""
    print("\n=== Verification ===")

    # Count Radiko nodes by tier
    print("\nRadiko tier distribution:")
    result = conn.execute("""
        MATCH (r:Radiko)
        WITH r.nivelo as tier, count(r) as cnt
        RETURN tier, cnt
        ORDER BY tier
    """)

    while result.has_next():
        tier, count = result.get_next()
        print(f"  {tier}: {count:,}")

    # Count Vorto nodes by tier
    print("\nVorto radiko_nivelo distribution:")
    result = conn.execute("""
        MATCH (v:Vorto)
        WHERE v.radiko_nivelo IS NOT NULL
        WITH v.radiko_nivelo as tier, count(v) as cnt
        RETURN tier, cnt
        ORDER BY tier
    """)

    while result.has_next():
        tier, count = result.get_next()
        print(f"  {tier}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description='Add root tier labels to Kuzu database')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'),
                       help='Vocabulary directory')
    parser.add_argument('--core-count', type=int, default=900,
                       help='Number of top-frequency Fundamento roots to mark as "core" (default: 900)')
    parser.add_argument('--skip-vorto', action='store_true',
                       help='Skip updating Vorto nodes (only update Radiko)')

    args = parser.parse_args()

    # Load tier classification
    root_to_tier, root_to_frequency = load_tier_classification(args.vocab_dir, args.core_count)
    parse_failures = load_parse_failures(args.vocab_dir)

    # Connect to Kuzu
    print(f"\nOpening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)

    # Add new properties to schema
    print("\nAdding new properties to schema...")
    try:
        conn.execute("ALTER TABLE Radiko ADD nivelo STRING")
        print("  Added Radiko.nivelo property")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Radiko.nivelo already exists")
        else:
            print(f"  Error adding Radiko.nivelo: {e}")

    try:
        conn.execute("ALTER TABLE Radiko ADD ofteco INT64")
        print("  Added Radiko.ofteco property")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Radiko.ofteco already exists")
        else:
            print(f"  Error adding Radiko.ofteco: {e}")

    try:
        conn.execute("ALTER TABLE Vorto ADD radiko_nivelo STRING")
        print("  Added Vorto.radiko_nivelo property")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Vorto.radiko_nivelo already exists")
        else:
            print(f"  Error adding Vorto.radiko_nivelo: {e}")

    try:
        conn.execute("ALTER TABLE Vorto ADD radiko_ofteco INT64")
        print("  Added Vorto.radiko_ofteco property")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Vorto.radiko_ofteco already exists")
        else:
            print(f"  Error adding Vorto.radiko_ofteco: {e}")

    # Add tier to Radiko nodes
    add_tier_to_radiko_nodes(conn, root_to_tier, root_to_frequency, parse_failures)

    # Add tier to Vorto nodes (propagate from Radiko)
    if not args.skip_vorto:
        add_tier_to_vorto_nodes(conn)

    # Verify
    verify_tier_distribution(conn)

    print("\n✓ Tier labels added successfully!")
    print("\nExample queries:")
    print("  # Get all Tier 1 (Fundamento) roots:")
    print("  MATCH (r:Radiko) WHERE r.nivelo = 'tier1_fundamento' RETURN r.radiko")
    print()
    print("  # Get all words built from Tier 1 roots:")
    print("  MATCH (v:Vorto) WHERE v.radiko_nivelo = 'tier1_fundamento' RETURN v.plena_vorto")
    print()
    print("  # Exclude garbage from queries:")
    print("  MATCH (v:Vorto) WHERE v.radiko_nivelo <> 'tier5_rubaĵo' RETURN v")


if __name__ == '__main__':
    main()
