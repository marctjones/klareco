#!/usr/bin/env python3
"""
FAST version: Batch UPDATE queries for 5-10x speedup.

Changes from v2:
- Build batched UPDATE queries (100 nodes per query)
- Use UNWIND for bulk updates
- Fewer database round-trips

Estimated time: ~15 minutes (down from ~100 minutes)

Usage:
    python scripts/classify_roots_fast.py --kuzu data/indexes/v2.1_kuzu_index_full
    python scripts/classify_roots_fast.py --kuzu data/indexes/v2.1_kuzu_index_full --resume
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
    verify_classification
)


def classify_radiko_nodes_fast(conn, classification_data, checkpoint, checkpoint_file):
    """FAST: Classify Radiko nodes using batched UPDATEs."""

    if checkpoint['radiko_done']:
        print("\nRadiko nodes already classified (from checkpoint)")
        return

    print("\nClassifying Radiko nodes (FAST MODE - batched updates)...")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    # Get total count
    result = conn.execute("MATCH (r:Radiko) RETURN count(r)")
    total_radiko = result.get_next()[0]
    print(f"  Total Radiko nodes: {total_radiko:,}")

    # Track statistics
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    # OPTIMIZATION: Larger pagination, smaller update batches
    fetch_size = 50000  # Fetch more roots at once
    update_batch_size = 100  # Update in batches of 100

    # Resume from checkpoint if available
    processed = checkpoint.get('radiko_processed', 0)
    offset = (processed // fetch_size) * fetch_size  # Round down to batch boundary

    if processed > 0:
        print(f"  Resuming from {processed:,} nodes (offset {offset:,})")

    start_time = time.time()

    while True:
        # Fetch batch of radikos
        result = conn.execute(f"""
            MATCH (r:Radiko)
            RETURN r.radiko
            SKIP {offset}
            LIMIT {fetch_size}
        """)

        batch = []
        while result.has_next():
            (radiko,) = result.get_next()
            batch.append(radiko)

        if not batch:
            break  # No more nodes

        # Classify all in batch
        classifications = []
        for radiko in batch:
            nivelo, fonto = determine_classification(
                radiko, tier0_words, unua_libro_roots, tier1b_roots,
                revo_roots, corpus_roots, proper_names, parse_failures
            )
            ofteco = corpus_usage.get(radiko, 0)
            classifications.append((radiko, nivelo, fonto, ofteco))

            tier_counts[nivelo] += 1
            if fonto:
                source_counts[fonto] += 1

        # OPTIMIZATION: Update in sub-batches using UNWIND
        for i in range(0, len(classifications), update_batch_size):
            sub_batch = classifications[i:i + update_batch_size]

            # Build UNWIND query for batch update
            # Format: UNWIND [{radiko: 'x', nivelo: 'y', fonto: 'z', ofteco: n}, ...] AS row
            rows = []
            for radiko, nivelo, fonto, ofteco in sub_batch:
                # Escape for JSON
                radiko_escaped = radiko.replace('\\', '\\\\').replace("'", "\\'")
                nivelo_escaped = nivelo.replace('\\', '\\\\').replace("'", "\\'")
                fonto_escaped = fonto.replace('\\', '\\\\').replace("'", "\\'") if fonto else None

                row = f"{{radiko: '{radiko_escaped}', nivelo: '{nivelo_escaped}', "
                if fonto_escaped:
                    row += f"fonto: '{fonto_escaped}', "
                else:
                    row += "fonto: NULL, "
                row += f"ofteco: {ofteco}}}"
                rows.append(row)

            rows_str = ', '.join(rows)

            # Execute batched update
            query = f"""
                UNWIND [{rows_str}] AS row
                MATCH (r:Radiko {{radiko: row.radiko}})
                SET r.nivelo = row.nivelo,
                    r.fonto = row.fonto,
                    r.ofteco = row.ofteco
            """

            try:
                conn.execute(query)
            except Exception as e:
                # Fallback to individual updates if batch fails
                print(f"    Warning: Batch update failed, falling back to individual updates: {e}")
                for radiko, nivelo, fonto, ofteco in sub_batch:
                    radiko_escaped = radiko.replace("'", "\\'")
                    nivelo_escaped = nivelo.replace("'", "\\'")

                    set_clause = f"r.nivelo = '{nivelo_escaped}', r.ofteco = {ofteco}"
                    if fonto:
                        fonto_escaped = fonto.replace("'", "\\'")
                        set_clause += f", r.fonto = '{fonto_escaped}'"

                    conn.execute(f"""
                        MATCH (r:Radiko {{radiko: '{radiko_escaped}'}})
                        SET {set_clause}
                    """)

        processed += len(batch)

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

        offset += fetch_size

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


def propagate_to_vorto_nodes(conn, checkpoint, checkpoint_file):
    """Propagate classification to Vorto nodes (same as v2)."""
    print("\nPropagating classification to Vorto nodes...")

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

        conn.execute(f"""
            MATCH (v:Vorto)-[:HAVAS_RADIKON]->(r:Radiko)
            WHERE v.id >= {start_id} AND v.id <= {end_id}
              AND r.nivelo IS NOT NULL
            SET v.radiko_nivelo = r.nivelo,
                v.radiko_fonto = r.fonto,
                v.radiko_ofteco = r.ofteco
        """)

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        percent = 100 * (batch_num + 1) / total_batches

        batches_done = batch_num - start_batch + 1
        avg_time_per_batch = elapsed / batches_done if batches_done > 0 else 0
        batches_remaining = total_batches - batch_num - 1
        eta_seconds = avg_time_per_batch * batches_remaining
        eta_minutes = eta_seconds / 60

        print(f"    Batch {batch_num + 1}/{total_batches} ({percent:.1f}%) - {batch_time:.1f}s - ETA: {eta_minutes:.1f}m", flush=True)

        checkpoint['vorto_batch'] = batch_num + 1
        save_checkpoint(checkpoint_file, checkpoint)

    print(f"  Propagated classification to Vorto nodes in {(time.time() - start_time)/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='FAST classification with batched updates')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'))
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    checkpoint_file = Path('data/vocabularies/classification_checkpoint_fast.json')

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

    # Classify with FAST method
    classify_radiko_nodes_fast(conn, classification_data, checkpoint, checkpoint_file)

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
