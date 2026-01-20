#!/usr/bin/env python3
"""
Clean up old corpus files to free disk space.

This script identifies and removes outdated corpus files that have been
superseded by the new unified corpus with GOLD/SILVER/BRONZE quality system.

Usage:
    python scripts/cleanup_old_corpus.py --dry-run  # See what would be deleted
    python scripts/cleanup_old_corpus.py            # Actually delete files
"""

import argparse
from pathlib import Path
import sys


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def find_old_corpus_files(corpus_dir: Path) -> list:
    """Find old corpus files that can be safely deleted."""

    # New corpus file (keep this!)
    new_corpus = corpus_dir / 'corpus_with_metadata.jsonl'

    # Old corpus files to remove
    old_files = [
        'books_corpus.jsonl',           # Pre-merge books only
        'wikipedia_corpus.jsonl',       # Pre-merge Wikipedia only
        'corpus_full_with_tier0.jsonl', # Old tier numbering system
    ]

    files_to_delete = []

    for filename in old_files:
        file_path = corpus_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            files_to_delete.append({
                'path': file_path,
                'name': filename,
                'size': size,
                'size_human': format_size(size)
            })

    # Also find backup files
    for backup_file in corpus_dir.glob('*.backup*'):
        if backup_file != new_corpus:
            size = backup_file.stat().st_size
            files_to_delete.append({
                'path': backup_file,
                'name': backup_file.name,
                'size': size,
                'size_human': format_size(size)
            })

    return files_to_delete


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old corpus files"
    )
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        default=Path('data/enhanced_corpus'),
        help='Corpus directory to clean'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    if not args.corpus_dir.exists():
        print(f"Error: Directory not found: {args.corpus_dir}")
        return 1

    # Find old files
    files_to_delete = find_old_corpus_files(args.corpus_dir)

    if not files_to_delete:
        print("✓ No old corpus files found - directory is clean!")
        return 0

    # Show what will be deleted
    print("Old corpus files found:")
    print()

    total_size = 0
    for file_info in files_to_delete:
        print(f"  {file_info['name']}")
        print(f"    Size: {file_info['size_human']}")
        total_size += file_info['size']

    print()
    print(f"Total size to free: {format_size(total_size)}")
    print()

    # Dry run or actual deletion
    if args.dry_run:
        print("DRY RUN - No files deleted")
        print()
        print("To actually delete these files, run without --dry-run:")
        print("  python scripts/cleanup_old_corpus.py")
        return 0

    # Confirm deletion
    print("WARNING: This will permanently delete the files listed above!")
    print()
    response = input("Continue? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Aborted - no files deleted")
        return 0

    # Delete files
    print()
    print("Deleting files...")

    deleted_count = 0
    deleted_size = 0

    for file_info in files_to_delete:
        try:
            file_info['path'].unlink()
            print(f"  ✓ Deleted: {file_info['name']}")
            deleted_count += 1
            deleted_size += file_info['size']
        except Exception as e:
            print(f"  ✗ Failed to delete {file_info['name']}: {e}")

    print()
    print("=" * 80)
    print(f"✓ Deleted {deleted_count} files")
    print(f"✓ Freed {format_size(deleted_size)} of disk space")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
