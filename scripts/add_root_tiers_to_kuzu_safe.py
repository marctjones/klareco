#!/usr/bin/env python3
"""
MEMORY-SAFE version: Add tier/category properties to Radiko and Vorto nodes.

Key improvements:
- Streams updates instead of loading all nodes into memory
- Shows progress output with flush
- Checkpointing for resume capability
- Memory-efficient batching

Usage:
    python scripts/add_root_tiers_to_kuzu_safe.py --kuzu data/indexes/v2.1_kuzu_index_full
    python scripts/add_root_tiers_to_kuzu_safe.py --kuzu data/indexes/v2.1_kuzu_index_full --resume
"""
import argparse
import json
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


def load_tier_classification(vocab_dir: Path, core_count: int = 900) -> tuple[dict, dict, set]:
    """Load root classification from vocabulary files.

    Returns:
        (root_to_tier, root_to_frequency, affixes) - tier mapping, frequency counts, and affix set
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

    # Load official affixes
    affixes = set()
    affix_file = vocab_dir / 'official_affixes.json'
    if affix_file.exists():
        with open(affix_file, 'r') as f:
            affix_data = json.load(f)
        affixes = set(affix_data.get('suffixes', {}).keys()) | set(affix_data.get('prefixes', {}).keys())
    else:
        # Hardcoded official affixes
        affixes = {
            'aĵ', 'an', 'ar', 'ej', 'er', 'estr', 'id', 'ig', 'iĝ', 'il', 'in', 'ind', 'ing',
            'ism', 'ist', 'uj', 'ul', 'um', 'ebl', 'ec', 'eg', 'em', 'end', 'et', 'ad',
            'ant', 'int', 'ont', 'at', 'it', 'ot', 'ĉj', 'nj', 'obl', 'on', 'op',
            'bo', 'dis', 'ek', 'eks', 'ge', 'mal', 'mis', 'pra', 're'
        }

    # Build root -> tier and root -> frequency mappings
    root_to_tier = {}
    root_to_frequency = {}

    # Tier 0: Official affixes (grammatical morphemes, not lexical roots)
    for affix in affixes:
        root_to_tier[affix] = 'tier0_afikso'
        root_to_frequency[affix] = corpus_usage.get(affix, 0)

    # Tier 1: Fundamento (split into core 900 and extended)
    fundamento_roots = classification['tier1_fundamento']['roots']

    # Get frequency for each Fundamento root
    fundamento_with_freq = []
    for root in fundamento_roots:
        if root in affixes:  # Skip if already classified as affix
            continue
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
        if root not in affixes and root not in root_to_tier:
            root_to_tier[root] = 'tier2_revo'
            root_to_frequency[root] = corpus_usage.get(root, 0)

    # Tier 3: Corpus
    for root in classification['tier3_corpus']['roots']:
        if root not in affixes and root not in root_to_tier:
            root_to_tier[root] = 'tier3_korpuso'
            root_to_frequency[root] = corpus_usage.get(root, 0)

    # Tier 4: Proper names
    for root in classification['tier4_proper_names']['roots']:
        if root not in affixes and root not in root_to_tier:
            root_to_tier[root] = 'tier4_propranomo'
            root_to_frequency[root] = corpus_usage.get(root, 0)

    print(f"  Tier 0 (Affixes): {len(affixes):,}")
    print(f"  Tier 1a (Fundamento core): {tier1a_count:,}")
    print(f"  Tier 1b (Fundamento extended): {tier1b_count:,}")
    print(f"  Tier 2 (ReVo): {len(classification['tier2_revo']['roots']):,}")
    print(f"  Tier 3 (Corpus): {len(classification['tier3_corpus']['roots']):,}")
    print(f"  Tier 4 (Proper names): {len(classification['tier4_proper_names']['roots']):,}")

    return root_to_tier, root_to_frequency, affixes


def load_parse_failures(vocab_dir: Path) -> set:
    """Load parse failure roots (tier 5 - garbage)."""
    failure_file = vocab_dir / 'parse_failures.json'
    if not failure_file.exists():
        return set()

    with open(failure_file, 'r') as f:
        failures = json.load(f)

    print(f"  Tier 5 (Garbage): {len(failures):,}")
    return set(failures.keys())


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'radiko_done': False, 'vorto_batch': 0, 'radiko_processed': 0, 'vorto_processed': 0}


def save_checkpoint(checkpoint_file: Path, state: dict):
    """Save checkpoint atomically."""
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f)
    temp_file.rename(checkpoint_file)


def add_tier_to_radiko_nodes(conn, root_to_tier: dict, root_to_frequency: dict,
                              parse_failures: set, checkpoint: dict, checkpoint_file: Path):
    """Add nivelo and ofteco properties to Radiko nodes (MEMORY-SAFE: streaming updates)."""

    if checkpoint['radiko_done']:
        print("\nRadiko nodes already processed (from checkpoint)")
        return

    print("\nAdding tier and frequency properties to Radiko nodes...")

    # Get total count
    result = conn.execute("MATCH (r:Radiko) RETURN count(r)")
    total_radiko = result.get_next()[0]
    print(f"  Total Radiko nodes: {total_radiko:,}")

    # Track statistics
    tier_counts = defaultdict(int)
    processed = checkpoint.get('radiko_processed', 0)

    # MEMORY-SAFE: Iterate through our classification dictionaries instead of loading all nodes
    # Process in small batches to avoid building up queries
    batch_size = 100
    batch_updates = []

    print(f"  Resuming from: {processed:,} nodes")

    # Process classified roots first
    all_classified = set(root_to_tier.keys()) | parse_failures
    classified_list = sorted(all_classified)  # Deterministic order for checkpointing

    start_idx = processed
    start_time = time.time()

    for idx in range(start_idx, len(classified_list)):
        radiko = classified_list[idx]

        # Determine tier
        if radiko in parse_failures:
            tier = 'tier5_rubaĵo'
        elif radiko in root_to_tier:
            tier = root_to_tier[radiko]
        else:
            tier = 'tier6_nekonata'

        frequency = root_to_frequency.get(radiko, 0)

        batch_updates.append((radiko, tier, frequency))
        tier_counts[tier] += 1

        # Execute batch
        if len(batch_updates) >= batch_size:
            for root, tier_val, freq in batch_updates:
                root_escaped = root.replace("'", "\\'")
                tier_escaped = tier_val.replace("'", "\\'")

                conn.execute(f"""
                    MATCH (r:Radiko {{radiko: '{root_escaped}'}})
                    SET r.nivelo = '{tier_escaped}', r.ofteco = {freq}
                """)

            processed = idx + 1
            elapsed = time.time() - start_time
            rate = len(batch_updates) / elapsed if elapsed > 0 else 0
            pct = 100 * processed / len(classified_list)

            print(f"    Progress: {processed:,} / {len(classified_list):,} ({pct:.1f}%) - {rate:.0f} nodes/sec", flush=True)

            # Save checkpoint
            checkpoint['radiko_processed'] = processed
            save_checkpoint(checkpoint_file, checkpoint)

            batch_updates = []
            start_time = time.time()

    # Process remaining batch
    if batch_updates:
        for root, tier_val, freq in batch_updates:
            root_escaped = root.replace("'", "\\'")
            tier_escaped = tier_val.replace("'", "\\'")

            conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root_escaped}'}})
                SET r.nivelo = '{tier_escaped}', r.ofteco = {freq}
            """)
        processed = len(classified_list)

    # Mark any remaining unclassified roots as tier6_nekonata
    # Use SKIP/LIMIT pagination to avoid loading all into memory
    print(f"\n  Marking unclassified roots as tier6_nekonata...")
    batch_size = 10000
    offset = 0
    unclassified_count = 0

    while True:
        result = conn.execute(f"""
            MATCH (r:Radiko)
            WHERE r.nivelo IS NULL
            RETURN r.radiko
            SKIP {offset}
            LIMIT {batch_size}
        """)

        batch = []
        while result.has_next():
            (radiko,) = result.get_next()
            batch.append(radiko)

        if not batch:
            break

        for radiko in batch:
            radiko_escaped = radiko.replace("'", "\\'")
            conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{radiko_escaped}'}})
                SET r.nivelo = 'tier6_nekonata', r.ofteco = 0
            """)
            tier_counts['tier6_nekonata'] += 1
            unclassified_count += 1

        print(f"    Unclassified: {unclassified_count:,}", flush=True)
        offset += batch_size

    checkpoint['radiko_done'] = True
    checkpoint['radiko_processed'] = processed
    save_checkpoint(checkpoint_file, checkpoint)

    print(f"\n  Updated {processed:,} classified + {unclassified_count:,} unclassified = {processed + unclassified_count:,} total")

    # Print tier distribution
    print("\n  Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / (processed + unclassified_count) if (processed + unclassified_count) > 0 else 0
        print(f"    {tier}: {count:,} ({pct:.1f}%)")


def add_tier_to_vorto_nodes(conn, checkpoint: dict, checkpoint_file: Path):
    """Add radiko_nivelo and radiko_ofteco to Vorto nodes (MEMORY-SAFE: ID range batching)."""
    print("\nAdding radiko_nivelo and radiko_ofteco properties to Vorto nodes...")

    # Get total Vorto count and ID range
    result = conn.execute("MATCH (v:Vorto) RETURN count(v)")
    total_vorto = result.get_next()[0]
    print(f"  Total Vorto nodes: {total_vorto:,}")

    result = conn.execute("MATCH (v:Vorto) RETURN min(v.id), max(v.id)")
    min_id, max_id = result.get_next()

    batch_size = 1_000_000
    total_batches = (max_id - min_id + batch_size) // batch_size

    start_batch = checkpoint.get('vorto_batch', 0)
    print(f"  Processing {total_batches} batches (resuming from batch {start_batch})...")

    start_time = time.time()

    for batch_num in range(start_batch, total_batches):
        start_id = min_id + (batch_num * batch_size)
        end_id = min(start_id + batch_size - 1, max_id)

        batch_start = time.time()

        # Update Vorto.radiko_nivelo and radiko_ofteco from connected Radiko
        conn.execute(f"""
            MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko)
            WHERE v.id >= {start_id} AND v.id <= {end_id}
              AND r.nivelo IS NOT NULL
            SET v.radiko_nivelo = r.nivelo, v.radiko_ofteco = r.ofteco
        """)

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        percent = 100 * (batch_num + 1) / total_batches

        # Estimate remaining time
        batches_done = batch_num - start_batch + 1
        avg_time_per_batch = elapsed / batches_done if batches_done > 0 else 0
        batches_remaining = total_batches - batch_num - 1
        eta_seconds = avg_time_per_batch * batches_remaining
        eta_minutes = eta_seconds / 60

        print(f"    Batch {batch_num + 1}/{total_batches} ({percent:.1f}%) - {batch_time:.1f}s - ETA: {eta_minutes:.1f}m", flush=True)

        # Save checkpoint
        checkpoint['vorto_batch'] = batch_num + 1
        checkpoint['vorto_processed'] = (batch_num + 1 - start_batch) * batch_size
        save_checkpoint(checkpoint_file, checkpoint)

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
    parser = argparse.ArgumentParser(description='Add root tier labels to Kuzu database (MEMORY-SAFE)')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'),
                       help='Vocabulary directory')
    parser.add_argument('--core-count', type=int, default=900,
                       help='Number of top-frequency Fundamento roots to mark as "core" (default: 900)')
    parser.add_argument('--skip-vorto', action='store_true',
                       help='Skip updating Vorto nodes (only update Radiko)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    args = parser.parse_args()

    checkpoint_file = Path('data/vocabularies/tier_labeling_checkpoint.json')

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        checkpoint = {'radiko_done': False, 'vorto_batch': 0, 'radiko_processed': 0, 'vorto_processed': 0}

    # Load tier classification
    root_to_tier, root_to_frequency, affixes = load_tier_classification(args.vocab_dir, args.core_count)
    parse_failures = load_parse_failures(args.vocab_dir)

    # Connect to Kuzu
    print(f"\nOpening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)

    # Add new properties to schema
    print("\nAdding new properties to schema...")
    for table, prop, dtype in [
        ('Radiko', 'nivelo', 'STRING'),
        ('Radiko', 'ofteco', 'INT64'),
        ('Vorto', 'radiko_nivelo', 'STRING'),
        ('Vorto', 'radiko_ofteco', 'INT64'),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD {prop} {dtype}")
            print(f"  Added {table}.{prop} property")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  {table}.{prop} already exists")
            else:
                print(f"  Error adding {table}.{prop}: {e}")

    # Add tier to Radiko nodes
    add_tier_to_radiko_nodes(conn, root_to_tier, root_to_frequency, parse_failures, checkpoint, checkpoint_file)

    # Add tier to Vorto nodes (propagate from Radiko)
    if not args.skip_vorto:
        add_tier_to_vorto_nodes(conn, checkpoint, checkpoint_file)

    # Verify
    verify_tier_distribution(conn)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"\nCheckpoint file removed: {checkpoint_file}")

    print("\n✓ Tier labels added successfully!")
    print("\nExample queries:")
    print("  # Get all Tier 0 (Affixes):")
    print("  MATCH (r:Radiko) WHERE r.nivelo = 'tier0_afikso' RETURN r.radiko")
    print()
    print("  # Get all Tier 1a (Core 900 Fundamento) roots:")
    print("  MATCH (r:Radiko) WHERE r.nivelo = 'tier1a_fundamento_kerno' RETURN r.radiko")
    print()
    print("  # Get all words built from core roots:")
    print("  MATCH (v:Vorto) WHERE v.radiko_nivelo = 'tier1a_fundamento_kerno' RETURN v.plena_vorto LIMIT 100")
    print()
    print("  # Exclude garbage and affixes from training data queries:")
    print("  MATCH (v:Vorto) WHERE v.radiko_nivelo NOT IN ['tier0_afikso', 'tier5_rubaĵo', 'tier6_nekonata'] RETURN v")


if __name__ == '__main__':
    main()
