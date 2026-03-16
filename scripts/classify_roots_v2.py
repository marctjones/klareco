#!/usr/bin/env python3
"""
Root Classification Script v2 - Memory-safe, restartable, complete taxonomy

Implements the full tier system:
- Tier 0: 169 grammatical words (10 subcategories)
- Tier 1: Core vocabulary (787 Unua Libro + extended Fundamento)
- Tier 2: ReVo technical terms
- Tier 3: Corpus-validated
- Tier 4: Proper entities
- Tier 5: Garbage
- Tier 6: Unknown

Properties added:
- nivelo: Tier classification (grammatical/semantic role)
- fonto: Historical source (unua_libro, fundamento, revo, korpuso)
- ofteco: Usage frequency
- jaro_unua_vido: First year seen (for neologism detection)

Memory-safe features:
- Streams updates (doesn't load all nodes)
- Checkpointing for resume
- Progress indicators
- Batch processing

Usage:
    python scripts/classify_roots_v2.py --kuzu data/indexes/v2.1_kuzu_index_full
    python scripts/classify_roots_v2.py --kuzu data/indexes/v2.1_kuzu_index_full --resume
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


def load_tier0_grammatical_words(vocab_dir: Path) -> dict:
    """Load all tier0 grammatical words (10 subcategories, 169 total)."""
    tier0_file = vocab_dir / 'tier0_grammatical_words.json'

    if not tier0_file.exists():
        print(f"ERROR: {tier0_file} not found")
        exit(1)

    with open(tier0_file, 'r') as f:
        tier0_data = json.load(f)

    # Build mapping: word -> tier0 subcategory
    word_to_tier0 = {}

    for category_key in ['tier0_afikso', 'tier0_korelativo', 'tier0_pronomo',
                          'tier0_prepozicio', 'tier0_konjunkcio', 'tier0_partiklo',
                          'tier0_nombro', 'tier0_finaco', 'tier0_artikolo', 'tier0_komparativo']:
        if category_key in tier0_data:
            category_data = tier0_data[category_key]
            words = category_data.get('words', [])

            # Handle both dict (affixes) and list (others)
            if isinstance(words, dict):
                for word in words.keys():
                    word_to_tier0[word] = category_key
            elif isinstance(words, list):
                for word in words:
                    word_to_tier0[word] = category_key

            # Also check 'suffixes' and 'prefixes' keys for affixes
            for key in ['suffixes', 'prefixes']:
                if key in category_data:
                    for word in category_data[key].keys():
                        word_to_tier0[word] = category_key

    print(f"Loaded {len(word_to_tier0)} tier0 grammatical words")
    return word_to_tier0


def load_unua_libro_roots(vocab_dir: Path) -> set:
    """Load the 787 Unua Libro lexical roots."""
    ub_file = vocab_dir / 'unua_libro_original_roots.json'

    if not ub_file.exists():
        print(f"ERROR: {ub_file} not found")
        exit(1)

    with open(ub_file, 'r') as f:
        ub_data = json.load(f)

    roots = set(ub_data.keys())
    print(f"Loaded {len(roots)} Unua Libro lexical roots")
    return roots


def load_classification_data(vocab_dir: Path) -> tuple:
    """Load all classification sources."""
    print("Loading classification data...")

    # Tier 0: Grammatical words (all from Unua Libro)
    tier0_words = load_tier0_grammatical_words(vocab_dir)

    # Tier 1a: Unua Libro lexical roots
    unua_libro_roots = load_unua_libro_roots(vocab_dir)

    # Load Fundamento, ReVo, corpus data
    class_file = vocab_dir / 'root_classification.json'
    if not class_file.exists():
        print(f"ERROR: {class_file} not found - run compare_root_sources.py")
        exit(1)

    with open(class_file, 'r') as f:
        classification = json.load(f)

    # Tier 1: Fundamento (split into Unua Libro vs extended)
    fundamento_all = set(classification['tier1_fundamento']['roots'])
    tier1b_roots = fundamento_all - unua_libro_roots - set(tier0_words.keys())

    # Tier 2: ReVo
    revo_roots = set(classification['tier2_revo']['roots']) - fundamento_all - set(tier0_words.keys())

    # Tier 3: Corpus
    corpus_roots = set(classification['tier3_corpus']['roots']) - fundamento_all - revo_roots - set(tier0_words.keys())

    # Tier 4: Proper names
    proper_names = set(classification['tier4_proper_names']['roots'])

    # Tier 5: Parse failures
    failure_file = vocab_dir / 'parse_failures.json'
    parse_failures = set()
    if failure_file.exists():
        with open(failure_file, 'r') as f:
            failures = json.load(f)
        parse_failures = set(failures.keys())

    # Load corpus usage for frequency data
    corpus_file = vocab_dir / 'corpus_validated_roots_clean.json'
    corpus_usage = {}
    if corpus_file.exists():
        with open(corpus_file, 'r') as f:
            corpus_data = json.load(f)
        corpus_usage = {root: data.get('usage', 0) for root, data in corpus_data.items()}

    print(f"  Tier 0 (grammatical): {len(tier0_words):,}")
    print(f"  Tier 1a (Unua Libro): {len(unua_libro_roots):,}")
    print(f"  Tier 1b (Fundamento extended): {len(tier1b_roots):,}")
    print(f"  Tier 2 (ReVo): {len(revo_roots):,}")
    print(f"  Tier 3 (Corpus): {len(corpus_roots):,}")
    print(f"  Tier 4 (Proper names): {len(proper_names):,}")
    print(f"  Tier 5 (Parse failures): {len(parse_failures):,}")

    return (tier0_words, unua_libro_roots, tier1b_roots, revo_roots,
            corpus_roots, proper_names, parse_failures, corpus_usage)


def determine_classification(root: str, tier0_words, unua_libro_roots, tier1b_roots,
                             revo_roots, corpus_roots, proper_names, parse_failures) -> tuple:
    """Determine tier (nivelo) and source (fonto) for a root.

    Returns: (nivelo, fonto)
    """
    # Tier 0: Grammatical words (all from Unua Libro)
    if root in tier0_words:
        return (tier0_words[root], 'unua_libro')

    # Tier 1a: Unua Libro lexical roots
    if root in unua_libro_roots:
        return ('tier1a_unua_libro', 'unua_libro')

    # Tier 1b: Fundamento extended
    if root in tier1b_roots:
        return ('tier1b_fundamento', 'fundamento')

    # Tier 2: ReVo
    if root in revo_roots:
        return ('tier2_revo', 'revo')

    # Tier 3: Corpus
    if root in corpus_roots:
        return ('tier3_korpuso', 'korpuso')

    # Tier 4: Proper names
    if root in proper_names:
        return ('tier4_propranomo', 'propranomo')

    # Tier 5: Parse failures
    if root in parse_failures:
        return ('tier5_rubaĵo', None)

    # Tier 6: Unknown
    return ('tier6_nekonata', None)


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'radiko_done': False, 'vorto_batch': 0}


def save_checkpoint(checkpoint_file: Path, state: dict):
    """Save checkpoint atomically."""
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f)
    temp_file.rename(checkpoint_file)


def classify_radiko_nodes(conn, classification_data, checkpoint, checkpoint_file):
    """Classify all Radiko nodes (memory-safe: pagination)."""

    if checkpoint['radiko_done']:
        print("\nRadiko nodes already classified (from checkpoint)")
        return

    print("\nClassifying Radiko nodes...")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    # Get total count
    result = conn.execute("MATCH (r:Radiko) RETURN count(r)")
    total_radiko = result.get_next()[0]
    print(f"  Total Radiko nodes: {total_radiko:,}")

    # Track statistics
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    # MEMORY-SAFE: Use SKIP/LIMIT pagination
    batch_size = 10000

    # Resume from checkpoint if available
    processed = checkpoint.get('radiko_processed', 0)
    offset = (processed // batch_size) * batch_size  # Round down to batch boundary

    if processed > 0:
        print(f"  Resuming from {processed:,} nodes (offset {offset:,})")

    start_time = time.time()

    while True:
        # Fetch batch of radikos
        result = conn.execute(f"""
            MATCH (r:Radiko)
            RETURN r.radiko
            SKIP {offset}
            LIMIT {batch_size}
        """)

        batch = []
        while result.has_next():
            (radiko,) = result.get_next()
            batch.append(radiko)

        if not batch:
            break  # No more nodes

        # Classify and update each root in batch
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

            conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{radiko_escaped}'}})
                SET {set_clause}
            """)

            tier_counts[nivelo] += 1
            if fonto:
                source_counts[fonto] += 1
            processed += 1

        # Progress
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        pct = 100 * processed / total_radiko
        print(f"    Progress: {processed:,} / {total_radiko:,} ({pct:.1f}%) - {rate:.0f} nodes/sec", flush=True)

        # Save checkpoint
        checkpoint['radiko_processed'] = processed
        save_checkpoint(checkpoint_file, checkpoint)

        offset += batch_size

    # Mark Radiko classification complete
    checkpoint['radiko_done'] = True
    checkpoint['radiko_processed'] = processed
    save_checkpoint(checkpoint_file, checkpoint)

    print(f"\n  Classified {processed:,} Radiko nodes")

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
    """Propagate nivelo, fonto, ofteco from Radiko to Vorto nodes."""
    print("\nPropagating classification to Vorto nodes...")

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

        # Propagate all properties from Radiko to Vorto
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

        # Estimate remaining time
        batches_done = batch_num - start_batch + 1
        avg_time_per_batch = elapsed / batches_done if batches_done > 0 else 0
        batches_remaining = total_batches - batch_num - 1
        eta_seconds = avg_time_per_batch * batches_remaining
        eta_minutes = eta_seconds / 60

        print(f"    Batch {batch_num + 1}/{total_batches} ({percent:.1f}%) - {batch_time:.1f}s - ETA: {eta_minutes:.1f}m", flush=True)

        # Save checkpoint
        checkpoint['vorto_batch'] = batch_num + 1
        save_checkpoint(checkpoint_file, checkpoint)

    print(f"  Propagated classification to Vorto nodes")


def verify_classification(conn):
    """Verify classification distribution."""
    print("\n=== Verification ===")

    # Radiko tier distribution
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

    # Radiko source distribution
    print("\nRadiko source distribution:")
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.fonto IS NOT NULL
        WITH r.fonto as source, count(r) as cnt
        RETURN source, cnt
        ORDER BY source
    """)

    while result.has_next():
        source, count = result.get_next()
        print(f"  {source}: {count:,}")

    # Vorto tier distribution
    print("\nVorto radiko_nivelo distribution:")
    result = conn.execute("""
        MATCH (v:Vorto)
        WHERE v.radiko_nivelo IS NOT NULL
        WITH v.radiko_nivelo as tier, count(v) as cnt
        RETURN tier, cnt
        ORDER BY tier
        LIMIT 15
    """)

    while result.has_next():
        tier, count = result.get_next()
        print(f"  {tier}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description='Classify roots with full tier taxonomy v2')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'),
                       help='Vocabulary directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    args = parser.parse_args()

    checkpoint_file = Path('data/vocabularies/classification_checkpoint_v2.json')

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        checkpoint = {'radiko_done': False, 'vorto_batch': 0}
        # Remove old checkpoint if starting fresh
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # Load all classification data
    classification_data = load_classification_data(args.vocab_dir)

    # Connect to Kuzu
    print(f"\nOpening Kuzu database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)

    # Add new properties to schema (if not already added)
    print("\nEnsuring schema has required properties...")
    for table, prop, dtype in [
        ('Radiko', 'nivelo', 'STRING'),
        ('Radiko', 'fonto', 'STRING'),
        ('Radiko', 'ofteco', 'INT64'),
        ('Radiko', 'jaro_unua_vido', 'INT64'),
        ('Vorto', 'radiko_nivelo', 'STRING'),
        ('Vorto', 'radiko_fonto', 'STRING'),
        ('Vorto', 'radiko_ofteco', 'INT64'),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD {prop} {dtype}")
            print(f"  Added {table}.{prop}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  {table}.{prop} already exists")
            else:
                print(f"  Error adding {table}.{prop}: {e}")

    # Classify Radiko nodes
    classify_radiko_nodes(conn, classification_data, checkpoint, checkpoint_file)

    # Propagate to Vorto nodes
    propagate_to_vorto_nodes(conn, checkpoint, checkpoint_file)

    # Verify
    verify_classification(conn)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"\nCheckpoint file removed: {checkpoint_file}")

    print("\n✓ Classification complete!")
    print("\nExample queries:")
    print("  # All Unua Libro words (lexical + grammatical):")
    print("  MATCH (r:Radiko) WHERE r.fonto = 'unua_libro' RETURN r.radiko, r.nivelo")
    print()
    print("  # Just Unua Libro lexical roots:")
    print("  MATCH (r:Radiko) WHERE r.nivelo = 'tier1a_unua_libro' RETURN r.radiko")
    print()
    print("  # Grammatical words from Unua Libro:")
    print("  MATCH (r:Radiko) WHERE r.fonto = 'unua_libro' AND r.nivelo STARTS WITH 'tier0_' RETURN r")
    print()
    print("  # Most frequent foundational words:")
    print("  MATCH (r:Radiko) WHERE r.fonto = 'unua_libro' RETURN r ORDER BY r.ofteco DESC LIMIT 100")


if __name__ == '__main__':
    main()
