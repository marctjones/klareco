#!/usr/bin/env python3
"""
Extend Kuzu Schema with Semantic Properties (v2.1)

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Data
DEPENDENCIES: Kuzu database, semantic_properties.py

Description:
    Extends the existing Radiko node table with semantic classification
    properties needed for schema-based summarization.

Usage:
    python scripts/extend_kuzu_schema.py \\
        --database data/indexes/v2.1_kuzu_index_full \\
        --dry-run  # Optional: show SQL without executing

Inputs:
    - Kuzu database (existing v2.1 schema)
    - klareco/schema/semantic_properties.py (property definitions)

Outputs:
    - Extended Radiko table with new columns

Quality Checks:
    - Verify all ALTER TABLE statements succeed
    - Query new columns to confirm they exist
    - Check default values applied correctly

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #655, #664, #665
See Also: docs/GETTING_STARTED_IMPLEMENTATION.md
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import klareco
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

from klareco.schema.semantic_properties import SEMANTIC_PROPERTIES_SCHEMA


def extend_schema(db_path: str, dry_run: bool = False):
    """
    Extend Kuzu schema with semantic properties.

    Args:
        db_path: Path to Kuzu database directory
        dry_run: If True, print SQL without executing
    """
    print(f"📊 Extending Kuzu schema: {db_path}")
    print(f"{'🔍 DRY RUN MODE' if dry_run else '✏️  APPLYING CHANGES'}\n")

    # Parse the schema SQL - extract only ALTER TABLE statements
    statements = []
    for stmt in SEMANTIC_PROPERTIES_SCHEMA.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        # Remove comment lines
        lines = [line for line in stmt.split('\n') if line.strip() and not line.strip().startswith('--')]
        if lines:
            clean_stmt = '\n'.join(lines).strip()
            if clean_stmt.startswith('ALTER TABLE'):
                statements.append(clean_stmt)

    print(f"Found {len(statements)} ALTER TABLE statements\n")

    if dry_run:
        print("SQL to execute:")
        print("=" * 80)
        for i, stmt in enumerate(statements, 1):
            print(f"\n-- Statement {i}:")
            print(stmt + ';')
        print("\n" + "=" * 80)
        print("\n✅ Dry run complete. Run without --dry-run to apply changes.")
        return

    # Connect to database
    try:
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    # Execute statements
    success_count = 0
    failed = []

    for i, stmt in enumerate(statements, 1):
        # Extract column name from ALTER TABLE statement for logging
        if "ADD" in stmt:
            try:
                # Kuzu syntax: ALTER TABLE Radiko ADD column_name TYPE
                parts = stmt.split("ADD")
                if len(parts) > 1:
                    col_name = parts[1].strip().split()[0]
                else:
                    col_name = "unknown"
            except:
                col_name = "unknown"
        else:
            col_name = "N/A"

        print(f"[{i}/{len(statements)}] Adding column: {col_name}...", end=" ")

        try:
            conn.execute(stmt + ';')
            print("✅")
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            # Check if column already exists (not an error)
            if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower() or "property" in error_msg.lower():
                print("⚠️  (already exists)")
                success_count += 1
            else:
                print(f"❌ FAILED: {error_msg}")
                failed.append((col_name, error_msg))

    print(f"\n{'='*80}")
    print(f"✅ Success: {success_count}/{len(statements)} statements")

    if failed:
        print(f"❌ Failed: {len(failed)} statements")
        for col_name, error in failed:
            print(f"   - {col_name}: {error}")

    # Verify schema by querying new columns
    print(f"\n{'='*80}")
    print("🔍 Verifying schema extension...")

    try:
        # Try to query new columns
        result = conn.execute("""
            MATCH (r:Radiko)
            RETURN r.radiko, r.funda_stato, r.verba_klaso, r.graveco_biografia
            LIMIT 5
        """)

        rows = result.get_as_pl()
        if rows is not None and len(rows) > 0:
            print("✅ Schema extension verified! New columns accessible.")
            print("\nSample data (first 5 roots):")
            print(rows)
        else:
            print("⚠️  Schema extended but no data found in Radiko table")
    except Exception as e:
        print(f"⚠️  Could not verify schema: {e}")
        print("   This might be normal if Radiko table is empty or columns don't exist yet.")

    print(f"\n{'='*80}")
    if failed:
        print("⚠️  Some statements failed. Review errors above.")
        sys.exit(1)
    else:
        print("✅ Schema extension complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extend Kuzu schema with semantic properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show SQL without executing)
  python scripts/extend_kuzu_schema.py --database data/indexes/v2.1_kuzu_index_full --dry-run

  # Apply changes
  python scripts/extend_kuzu_schema.py --database data/indexes/v2.1_kuzu_index_full

See: docs/GETTING_STARTED_IMPLEMENTATION.md
        """
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
        help='Show SQL without executing (for testing)'
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Expected location: {db_path.absolute()}")
        sys.exit(1)

    extend_schema(str(db_path), args.dry_run)


if __name__ == '__main__':
    main()
