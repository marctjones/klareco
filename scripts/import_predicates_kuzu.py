#!/usr/bin/env python3
"""
Import Predicate Triples into Kuzu Graph Database.

Part of Issue #254: Add Predicate table to Kuzu schema.

This script imports the predicates.jsonl file (from extract_predicates.py)
into the existing Kuzu database, adding:
- Predicate nodes (verb, subj, obj combinations)
- HAS_PREDICATE edges (Sentence → Predicate)

IMPORTANT: Must be run AFTER extract_predicates.py and build_kuzu_index.py

Usage:
    python scripts/import_predicates_kuzu.py
    python scripts/import_predicates_kuzu.py --limit 10000  # Test with subset

Input:
    data/indexes/kuzu_index/predicates.jsonl

Output:
    Updates data/indexes/kuzu_index/kuzu.db with:
    - Predicate nodes
    - HAS_PREDICATE edges

Note: This script is transactional - it clears existing predicates before import.
      No checkpointing needed as Kuzu COPY operations are atomic.
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

# Script version for tracking
VERSION = "1.0.0"

try:
    import kuzu
except ImportError:
    print("Error: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)


def setup_logging(log_dir: Path) -> Path:
    """Set up logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"import_predicates_kuzu_{timestamp}.log"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    return log_file


logger = logging.getLogger(__name__)


def add_predicate_schema(conn: kuzu.Connection):
    """Add Predicate node and HAS_PREDICATE edge tables to existing schema."""
    logger.info("Adding Predicate schema to Kuzu database...")

    # Check if Predicate table already exists
    try:
        result = conn.execute("MATCH (p:Predicate) RETURN count(p) LIMIT 1")
        count = result.get_next()[0]
        if count > 0:
            logger.info(f"  Predicate table already exists with {count:,}+ entries")
            return False  # Already exists
    except Exception:
        pass  # Table doesn't exist, continue

    # Create Predicate node table
    # Uses composite key: (verb, subj, obj) where subj/obj can be null
    # We use a generated string key: "verb:subj:obj"
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Predicate (
            pred_key STRING,
            verb STRING,
            subj STRING,
            obj STRING,
            is_copula BOOLEAN DEFAULT FALSE,
            is_passive BOOLEAN DEFAULT FALSE,
            doc_count INT64 DEFAULT 0,
            PRIMARY KEY (pred_key)
        )
    """)
    logger.info("  ✓ Created Predicate node table")

    # Create HAS_PREDICATE edge table (Sentence → Predicate)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS HAS_PREDICATE (
            FROM Sentence TO Predicate,
            clause_depth INT64
        )
    """)
    logger.info("  ✓ Created HAS_PREDICATE edge table")

    return True


def generate_pred_key(verb: str, subj: Optional[str], obj: Optional[str]) -> str:
    """Generate a unique key for a predicate triple."""
    subj_str = subj if subj else "_"
    obj_str = obj if obj else "_"
    return f"{verb}:{subj_str}:{obj_str}"


def import_predicates(
    kuzu_path: Path,
    predicates_path: Path,
    limit: Optional[int] = None,
):
    """Import predicates into Kuzu database."""
    logger.info("=" * 60)
    logger.info(f"Importing Predicates into Kuzu (v{VERSION})")
    logger.info("=" * 60)
    logger.info(f"Kuzu database: {kuzu_path}")
    logger.info(f"Predicates:    {predicates_path}")
    if limit:
        logger.info(f"Limit:         {limit:,}")
    logger.info("")

    if not predicates_path.exists():
        logger.error(f"Predicates file not found: {predicates_path}")
        logger.error("Run extract_predicates.py first!")
        sys.exit(1)

    start_time = datetime.now()

    # Open database
    db = kuzu.Database(str(kuzu_path))
    conn = kuzu.Connection(db)

    # Add schema
    schema_added = add_predicate_schema(conn)
    if not schema_added:
        logger.info("Predicate schema already exists, clearing existing data...")
        # Clear existing data
        try:
            conn.execute("MATCH ()-[e:HAS_PREDICATE]->() DELETE e")
            conn.execute("MATCH (p:Predicate) DELETE p")
            logger.info("  ✓ Cleared existing predicates")
        except Exception as e:
            logger.warning(f"  Could not clear: {e}")

    # Count predicates
    logger.info("")
    logger.info("Counting predicates...")
    with open(predicates_path) as f:
        total_preds = sum(1 for _ in f)
    logger.info(f"  Total: {total_preds:,}")

    if limit:
        total_preds = min(total_preds, limit)

    # Create temp directory for CSVs
    temp_dir = kuzu_path.parent / "temp_pred_csv"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Stream predicates to CSV files (memory-efficient two-pass approach)
    # Pass 1: Build unique predicates dict (small - ~50MB for 237K predicates)
    # Pass 2: Stream edges directly to CSV (no memory overhead)
    logger.info("")
    logger.info("Pass 1: Collecting unique predicates...")

    pred_csv = temp_dir / "predicates.csv"
    edge_csv = temp_dir / "has_predicate.csv"

    predicates_seen: Dict[str, dict] = {}  # pred_key -> {verb, subj, obj, is_copula, is_passive}

    processed = 0
    skipped_verb_only = 0
    last_log_time = datetime.now()

    with open(predicates_path) as f:
        for line in f:
            if limit and processed >= limit:
                break

            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            verb = pred['verb']
            subj = pred.get('subj')
            obj = pred.get('obj')
            is_copula = pred.get('is_copula', False)
            is_passive = pred.get('is_passive', False)

            # Skip verb-only predicates (no subject AND no object = noise)
            if not subj and not obj:
                skipped_verb_only += 1
                continue

            # Generate predicate key
            pred_key = generate_pred_key(verb, subj, obj)

            # Track unique predicates
            if pred_key not in predicates_seen:
                predicates_seen[pred_key] = {
                    'verb': verb,
                    'subj': subj,
                    'obj': obj,
                    'is_copula': is_copula,
                    'is_passive': is_passive,
                    'doc_count': 0,
                }
            predicates_seen[pred_key]['doc_count'] += 1

            processed += 1

            # Log progress every 10 seconds
            now = datetime.now()
            if (now - last_log_time).total_seconds() >= 10:
                elapsed = (now - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                pct = processed / total_preds * 100
                eta_seconds = (total_preds - processed) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.info(
                    f"  {processed:,}/{total_preds:,} ({pct:.1f}%) - "
                    f"{len(predicates_seen):,} unique - {rate:.0f}/sec - ETA: {eta_minutes:.1f}m"
                )
                last_log_time = now

    logger.info(f"  Total processed: {processed:,}")
    logger.info(f"  Skipped (verb-only): {skipped_verb_only:,}")
    logger.info(f"  Unique predicates: {len(predicates_seen):,}")

    # Write predicate CSV
    logger.info("")
    logger.info("Writing predicate CSV...")
    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pred_key', 'verb', 'subj', 'obj', 'is_copula', 'is_passive', 'doc_count'])

        for pred_key, info in predicates_seen.items():
            subj = info['subj'] if info['subj'] else ""
            obj = info['obj'] if info['obj'] else ""
            writer.writerow([
                pred_key,
                info['verb'],
                subj,
                obj,
                info['is_copula'],
                info['is_passive'],
                info['doc_count'],
            ])

    logger.info(f"  ✓ Wrote {len(predicates_seen):,} predicates")

    # Pass 2: Stream edges directly to CSV (memory-efficient)
    logger.info("")
    logger.info("Pass 2: Streaming edges to CSV...")
    edge_count = 0
    last_log_time = datetime.now()

    with open(edge_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['sent_id', 'pred_key', 'clause_depth'])

        with open(predicates_path) as f_in:
            for line in f_in:
                if limit and edge_count >= limit:
                    break

                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                subj = pred.get('subj')
                obj = pred.get('obj')

                # Skip verb-only (same filter as pass 1)
                if not subj and not obj:
                    continue

                doc_id = pred['doc_id']
                verb = pred['verb']
                clause_depth = pred.get('clause_depth', 0)
                pred_key = generate_pred_key(verb, subj, obj)

                writer.writerow([doc_id, pred_key, clause_depth])
                edge_count += 1

                # Log progress every 10 seconds
                now = datetime.now()
                if (now - last_log_time).total_seconds() >= 10:
                    logger.info(f"  {edge_count:,} edges written...")
                    last_log_time = now

    logger.info(f"  ✓ Wrote {edge_count:,} edges")

    # Bulk load into Kuzu
    logger.info("")
    logger.info("Bulk loading into Kuzu...")

    t0 = datetime.now()
    try:
        conn.execute(f"COPY Predicate FROM '{pred_csv}' (header=true)")
        logger.info(f"  ✓ Loaded Predicate nodes in {(datetime.now() - t0).total_seconds():.1f}s")
    except Exception as e:
        logger.error(f"  Error loading Predicate: {e}")
        raise

    t0 = datetime.now()
    try:
        conn.execute(f"COPY HAS_PREDICATE FROM '{edge_csv}' (header=true)")
        logger.info(f"  ✓ Loaded HAS_PREDICATE edges in {(datetime.now() - t0).total_seconds():.1f}s")
    except Exception as e:
        logger.error(f"  Error loading HAS_PREDICATE: {e}")
        raise

    # Verify counts
    logger.info("")
    logger.info("Verifying counts...")

    result = conn.execute("MATCH (p:Predicate) RETURN count(p)")
    pred_count = result.get_next()[0]
    logger.info(f"  Predicate nodes: {pred_count:,}")

    result = conn.execute("MATCH ()-[e:HAS_PREDICATE]->() RETURN count(e)")
    edge_count = result.get_next()[0]
    logger.info(f"  HAS_PREDICATE edges: {edge_count:,}")

    # Show some predicate stats
    logger.info("")
    logger.info("Predicate statistics:")

    # Most common predicates
    result = conn.execute("""
        MATCH (p:Predicate)
        RETURN p.verb, p.subj, p.obj, p.doc_count
        ORDER BY p.doc_count DESC
        LIMIT 10
    """)
    logger.info("  Top 10 predicates by frequency:")
    while result.has_next():
        row = result.get_next()
        verb, subj, obj, count = row
        subj_str = subj if subj else "_"
        obj_str = obj if obj else "_"
        logger.info(f"    ({verb}, {subj_str}, {obj_str}): {count:,}")

    # Copula vs non-copula
    result = conn.execute("MATCH (p:Predicate) WHERE p.is_copula = true RETURN count(p)")
    copula_count = result.get_next()[0]
    logger.info(f"  Copula predicates: {copula_count:,}")

    # Passive
    result = conn.execute("MATCH (p:Predicate) WHERE p.is_passive = true RETURN count(p)")
    passive_count = result.get_next()[0]
    logger.info(f"  Passive predicates: {passive_count:,}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info("Import Complete")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Predicate nodes: {pred_count:,}")
    logger.info(f"HAS_PREDICATE edges: {edge_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Import predicate triples into Kuzu database"
    )
    parser.add_argument(
        "--kuzu",
        type=Path,
        default=Path("data/indexes/kuzu_index/kuzu.db"),
        help="Path to Kuzu database",
    )
    parser.add_argument(
        "--predicates",
        type=Path,
        default=Path("data/indexes/kuzu_index/predicates.jsonl"),
        help="Path to predicates.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of predicates (for testing)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"import_predicates_kuzu.py v{VERSION}",
    )

    args = parser.parse_args()

    # Set up logging (console + file)
    log_file = setup_logging(Path("logs"))

    logger.info(f"import_predicates_kuzu.py v{VERSION}")
    logger.info(f"Log file: {log_file}")

    if not args.kuzu.exists():
        logger.error(f"Kuzu database not found: {args.kuzu}")
        logger.error("Run build_kuzu_index.py first!")
        sys.exit(1)

    import_predicates(args.kuzu, args.predicates, args.limit)


if __name__ == "__main__":
    main()
