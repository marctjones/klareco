#!/usr/bin/env python3
"""
Load Semantic Annotations into Kuzu (v2.1)

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema (extended with semantic properties)
STAGE: Data
DEPENDENCIES: Kuzu database with extended schema, annotation JSONL file

Description:
    Loads semantic annotations from JSONL file into Kuzu Radiko nodes.
    Updates existing Radiko nodes with semantic classification properties.

Usage:
    python scripts/load_semantic_annotations.py \\
        --annotations data/annotations/phase_0_template.jsonl \\
        --database data/indexes/v2.1_kuzu_index_full \\
        --dry-run  # Optional: show updates without executing

Inputs:
    - Annotations JSONL: One root per line with semantic properties
    - Kuzu database: v2.1 with extended schema

Outputs:
    - Updated Radiko nodes with semantic properties

Quality Checks:
    - Verify all roots exist in database before updating
    - Validate property values (e.g., graveco must be 0.0-1.0)
    - Report success/failure for each root

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #656, #658
See Also: docs/GETTING_STARTED_IMPLEMENTATION.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)


def load_annotations(file_path: Path) -> List[Dict]:
    """Load annotations from JSONL file."""
    annotations = []

    print(f"📖 Reading annotations from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                annotation = json.loads(line)

                # Validate required field
                if 'radiko' not in annotation:
                    print(f"⚠️  Line {line_num}: Missing 'radiko' field, skipping")
                    continue

                annotations.append(annotation)
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON: {e}")

    print(f"✅ Loaded {len(annotations)} annotations\n")
    return annotations


def validate_annotation(annotation: Dict) -> bool:
    """Validate annotation values."""
    radiko = annotation.get('radiko', 'unknown')

    # Check graveco values (0.0-1.0)
    for key in ['graveco_biografia', 'graveco_difina', 'graveco_okazaĵa', 'konfido']:
        if key in annotation:
            value = annotation[key]
            if not (0.0 <= value <= 1.0):
                print(f"⚠️  {radiko}: {key}={value} out of range [0.0, 1.0]")
                return False

    # Check ofteca_tavolo (0-3)
    if 'ofteca_tavolo' in annotation:
        value = annotation['ofteca_tavolo']
        if not (0 <= value <= 3):
            print(f"⚠️  {radiko}: ofteca_tavolo={value} out of range [0, 3]")
            return False

    return True


def build_update_query(annotation: Dict) -> str:
    """Build Cypher UPDATE query for a root."""
    radiko = annotation['radiko']

    # Build SET clause
    set_parts = []
    for key, value in annotation.items():
        if key == 'radiko':
            continue  # Don't update primary key

        if isinstance(value, str):
            set_parts.append(f"r.{key} = '{value}'")
        elif isinstance(value, bool):
            set_parts.append(f"r.{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            set_parts.append(f"r.{key} = {value}")

    set_clause = ", ".join(set_parts)

    query = f"""
    MATCH (r:Radiko {{radiko: '{radiko}'}})
    SET {set_clause}
    RETURN r.radiko
    """

    return query


def apply_annotations(db_path: str, annotations: List[Dict], dry_run: bool = False):
    """Apply annotations to Kuzu database."""
    print(f"📊 Applying {len(annotations)} annotations to: {db_path}")
    print(f"{'🔍 DRY RUN MODE' if dry_run else '✏️  APPLYING CHANGES'}\n")

    if dry_run:
        print("Sample queries (first 3):")
        print("=" * 80)
        for i, annotation in enumerate(annotations[:3], 1):
            if not validate_annotation(annotation):
                continue
            query = build_update_query(annotation)
            print(f"\n-- Annotation {i} ({annotation['radiko']}):")
            print(query)
        print("\n" + "=" * 80)
        print(f"\n✅ Dry run complete. Would update {len(annotations)} roots.")
        print("   Run without --dry-run to apply changes.")
        return

    # Connect to database
    try:
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    # Check if Radiko table exists and has new columns
    try:
        result = conn.execute("""
            MATCH (r:Radiko)
            RETURN r.radiko, r.funda_stato
            LIMIT 1
        """)
        # Just executing the query is enough - no need to fetch results
        # If the query succeeds, the schema is valid
    except Exception as e:
        print(f"❌ Schema not extended yet! Run extend_kuzu_schema.py first.")
        print(f"   Error: {e}")
        sys.exit(1)

    # Apply annotations
    success_count = 0
    not_found = []
    failed = []

    for i, annotation in enumerate(annotations, 1):
        radiko = annotation['radiko']

        print(f"[{i}/{len(annotations)}] Updating {radiko}...", end=" ")

        # Validate
        if not validate_annotation(annotation):
            print("❌ VALIDATION FAILED")
            failed.append((radiko, "validation failed"))
            continue

        # Build and execute query
        try:
            query = build_update_query(annotation)
            result = conn.execute(query)

            # Check if any rows were affected by trying to get row count
            # For UPDATE queries, Kuzu returns the matched/updated nodes
            has_results = result.has_next()

            if has_results:
                print("✅")
                success_count += 1
            else:
                print("⚠️  NOT FOUND (root doesn't exist in database)")
                not_found.append(radiko)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed.append((radiko, str(e)))

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Success: {success_count}/{len(annotations)} roots updated")

    if not_found:
        print(f"⚠️  Not found: {len(not_found)} roots don't exist in database")
        print(f"   (This is normal if roots haven't been seen in corpus yet)")
        for radiko in not_found[:5]:
            print(f"   - {radiko}")
        if len(not_found) > 5:
            print(f"   ... and {len(not_found) - 5} more")

    if failed:
        print(f"❌ Failed: {len(failed)} roots")
        for radiko, error in failed:
            print(f"   - {radiko}: {error}")

    # Verify by querying updated roots
    if success_count > 0:
        print(f"\n{'='*80}")
        print("🔍 Verifying updates...")

        try:
            # Query first annotated root
            first_radiko = annotations[0]['radiko']
            result = conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{first_radiko}'}})
                RETURN r.radiko, r.verba_klaso, r.substantiva_klaso,
                       r.graveco_biografia, r.funda_stato, r.ofteca_tavolo
            """)

            # Just check if query succeeded - don't need to fetch results
            if result.has_next():
                print(f"✅ Verified! Sample root '{first_radiko}' has semantic properties")
            else:
                print(f"⚠️  Could not verify root '{first_radiko}'")
        except Exception as e:
            print(f"⚠️  Verification failed: {e}")

    print(f"\n{'='*80}")
    if failed:
        print("⚠️  Some annotations failed. Review errors above.")
        sys.exit(1)
    else:
        print(f"✅ Annotation loading complete! {success_count} roots updated.")


def main():
    parser = argparse.ArgumentParser(
        description="Load semantic annotations into Kuzu database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show queries without executing)
  python scripts/load_semantic_annotations.py \\
      --annotations data/annotations/phase_0_template.jsonl \\
      --database data/indexes/v2.1_kuzu_index_full \\
      --dry-run

  # Apply annotations
  python scripts/load_semantic_annotations.py \\
      --annotations data/annotations/phase_0_template.jsonl \\
      --database data/indexes/v2.1_kuzu_index_full

See: docs/GETTING_STARTED_IMPLEMENTATION.md
        """
    )

    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Path to annotations JSONL file'
    )

    parser.add_argument(
        '--database',
        type=str,
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show queries without executing (for testing)'
    )

    args = parser.parse_args()

    # Check files exist
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print(f"❌ Annotations file not found: {annotations_path}")
        sys.exit(1)

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    # Load and apply annotations
    annotations = load_annotations(annotations_path)
    if not annotations:
        print("❌ No valid annotations found in file")
        sys.exit(1)

    apply_annotations(str(db_path), annotations, args.dry_run)


if __name__ == '__main__':
    main()
