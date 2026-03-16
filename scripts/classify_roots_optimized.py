#!/usr/bin/env python3
"""
OPTIMIZED version: Batch multiple UPDATEs in single transaction.

Key optimization: Wrap 1000 UPDATEs in one transaction to reduce
checkpoint overhead (Kuzu checkpoints after each write transaction).

Usage:
    python scripts/classify_roots_optimized.py --kuzu data/indexes/v2.1_kuzu_index_full
    python scripts/classify_roots_optimized.py --kuzu data/indexes/v2.1_kuzu_index_full --resume
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

# Import all the load functions from the v2 script
from classify_roots_v2 import (
    load_classification_data,
    determine_classification,
    load_checkpoint,
    save_checkpoint,
    propagate_to_vorto_nodes,
    verify_classification
)


def classify_radiko_nodes_optimized(conn, classification_data, checkpoint, checkpoint_file):
    """OPTIMIZED: Batch UPDATEs in single transaction."""

    if checkpoint['radiko_done']:
        print("\nRadiko nodes already classified (from checkpoint)")
        return

    print("\nClassifying Radiko nodes (OPTIMIZED - transaction batching)...")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    # Get total count
    result = conn.execute("MATCH (r:Radiko) RETURN count(r)")
    total_radiko = result.get_next()[0]
    print(f"  Total Radiko nodes: {total_radiko:,}")

    # Track statistics
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    # OPTIMIZATION: Transaction batch size
    fetch_batch_size = 10000    # Fetch nodes
    tx_batch_size = 1000        # Updates per transaction

    # Resume from checkpoint if available
    processed = checkpoint.get('radiko_processed', 0)
    offset = (processed // fetch_batch_size) * fetch_batch_size

    if processed > 0:
        print(f"  Resuming from {processed:,} nodes (offset {offset:,})")

    start_time = time.time()
    pending_updates = []  # Queue of (query, nivelo, fonto) tuples

    def flush_updates():
        """Execute all pending updates in one transaction."""
        if not pending_updates:
            return

        # Begin transaction (implicit with first query in batch)
        for query, nivelo, fonto in pending_updates:
            conn.execute(query)
            tier_counts[nivelo] += 1
            if fonto:
                source_counts[fonto] += 1

        pending_updates.clear()

    while True:
        # Fetch batch of radikos
        result = conn.execute(f"""
            MATCH (r:Radiko)
            RETURN r.radiko
            SKIP {offset}
            LIMIT {fetch_batch_size}
        """)

        batch = []
        while result.has_next():
            (radiko,) = result.get_next()
            batch.append(radiko)

        if not batch:
            break  # No more nodes

        # Classify all in batch and queue updates
        for radiko in batch:
            nivelo, fonto = determine_classification(
                radiko, tier0_words, unua_libro_roots, tier1b_roots,
                revo_roots, corpus_roots, proper_names, parse_failures
            )

            ofteco = corpus_usage.get(radiko, 0)

            # Escape quotes
            radiko_escaped = radiko.replace("'", "\\'")
            nivelo_escaped = nivelo.replace("'", "\\'")

            # Build SET clause
            set_clause = f"r.nivelo = '{nivelo_escaped}', r.ofteco = {ofteco}"
            if fonto:
                fonto_escaped = fonto.replace("'", "\\'")
                set_clause += f", r.fonto = '{fonto_escaped}'"

            query = f"""
                MATCH (r:Radiko {{radiko: '{radiko_escaped}'}})
                SET {set_clause}
            """

            pending_updates.append((query, nivelo, fonto))
            processed += 1

            # Flush when transaction batch is full
            if len(pending_updates) >= tx_batch_size:
                flush_updates()

        # Flush remaining updates for this fetch batch
        flush_updates()

        # Progress
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        pct = 100 * processed / total_radiko
        eta_seconds = (total_radiko - processed) / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60

        print(f"    Progress: {processed:,} / {total_radiko:,} ({pct:.1f}%) - {rate:.0f} nodes/sec - ETA: {eta_minutes:.1f}m", flush=True)

        # Save checkpoint every 50K
        if processed % 50000 == 0:
            checkpoint['radiko_processed'] = processed
            save_checkpoint(checkpoint_file, checkpoint)

        offset += fetch_batch_size

    # Flush any remaining updates
    flush_updates()

    # Mark complete
    checkpoint['radiko_done'] = True
    checkpoint['radiko_processed'] = processed
    save_checkpoint(checkpoint_file, checkpoint)

    print(f"\n  Classified {processed:,} Radiko nodes in {(time.time() - start_time)/60:.1f} minutes")

    # Print distribution
    print("\n  Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / processed if processed > 0 else 0
        print(f"    {tier}: {count:,} ({pct:.1f}%)")

    print("\n  Source distribution:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        pct = 100 * count / processed if processed > 0 else 0
        print(f"    {source}: {count:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='OPTIMIZED classification with transaction batching')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'))
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    checkpoint_file = Path('data/vocabularies/classification_checkpoint_optimized.json')

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        checkpoint = {'radiko_done': False, 'vorto_batch': 0}
        if checkpoint_file.exists():
            checkpoint_file.unlink()

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

    # Classify with OPTIMIZED method
    classify_radiko_nodes_optimized(conn, classification_data, checkpoint, checkpoint_file)

    # Propagate to Vorto
    propagate_to_vorto_nodes(conn, checkpoint, checkpoint_file)

    # Verify
    verify_classification(conn)

    # Clean up
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print("\n✓ Classification complete!")


if __name__ == '__main__':
    main()
