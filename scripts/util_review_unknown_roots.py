#!/usr/bin/env python3
"""
Review Unknown Roots for Vocabulary Expansion.

Displays unknown roots that have been tracked during inference,
allowing you to decide which ones to add to the vocabulary.

Usage:
    python scripts/review_unknown_roots.py
    python scripts/review_unknown_roots.py --min-count 20
    python scripts/review_unknown_roots.py --export candidates.txt
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.unknown_tracker import UnknownRootTracker


def print_table(candidates: list, show_contexts: bool = True) -> None:
    """Print candidates in a nice table format."""
    if not candidates:
        print("No candidates found.")
        return

    # Header
    print()
    print(f"{'#':<4} {'Root':<20} {'Count':>8} {'Words':<30} {'First Seen':<12}")
    print("-" * 80)

    for i, c in enumerate(candidates, 1):
        words = ", ".join(c['words'][:3])
        if len(c['words']) > 3:
            words += "..."
        first_seen = c['first_seen'][:10] if c['first_seen'] else "?"

        print(f"{i:<4} {c['root']:<20} {c['count']:>8} {words:<30} {first_seen:<12}")

        if show_contexts and c['contexts']:
            for ctx in c['contexts'][:2]:
                # Truncate long contexts
                ctx_display = ctx[:70] + "..." if len(ctx) > 70 else ctx
                print(f"     └─ \"{ctx_display}\"")

    print("-" * 80)
    print(f"Total: {len(candidates)} candidates")
    print()


def interactive_selection(candidates: list) -> list:
    """Interactive mode to select which roots to add."""
    print("\nInteractive Selection Mode")
    print("Enter numbers to select (e.g., '1,3,5-10'), 'all' for all, or 'q' to quit:")

    selection = input("> ").strip().lower()

    if selection == 'q':
        return []

    if selection == 'all':
        return [c['root'] for c in candidates]

    # Parse selection like "1,3,5-10"
    selected_indices = set()
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            selected_indices.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            selected_indices.add(int(part))

    # Convert to roots (1-indexed)
    selected_roots = []
    for idx in sorted(selected_indices):
        if 1 <= idx <= len(candidates):
            selected_roots.append(candidates[idx - 1]['root'])

    return selected_roots


def main():
    parser = argparse.ArgumentParser(
        description="Review unknown roots for vocabulary expansion"
    )
    parser.add_argument(
        "--tracker-file",
        type=Path,
        default=Path("data/unknown_roots.json"),
        help="Path to unknown roots tracker file"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum occurrence count to show (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum candidates to show (default: 50)"
    )
    parser.add_argument(
        "--sort",
        choices=["count", "recent"],
        default="count",
        help="Sort by count or most recent (default: count)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't show example contexts"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export selected roots to file"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode to select roots"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, not individual roots"
    )

    args = parser.parse_args()

    # Load tracker
    if not args.tracker_file.exists():
        print(f"Tracker file not found: {args.tracker_file}")
        print("\nNo unknown roots have been tracked yet.")
        print("Unknown roots are tracked when you use the compositional embeddings")
        print("with tracking enabled:")
        print()
        print("    embedding.enable_unknown_tracking()")
        print()
        return 1

    tracker = UnknownRootTracker(args.tracker_file)

    # Show statistics
    stats = tracker.get_stats()
    print("\n" + "=" * 60)
    print("Unknown Roots Tracker Statistics")
    print("=" * 60)
    print(f"Total unknown roots seen:     {stats['total_unknown_roots']:,}")
    print(f"Total occurrences:            {stats['total_occurrences']:,}")
    print(f"Roots seen 10+ times:         {stats['roots_seen_10plus']:,}")
    print(f"Roots seen 100+ times:        {stats['roots_seen_100plus']:,}")

    if stats.get('top_5'):
        print("\nTop 5 unknown roots:")
        for root, count in stats['top_5']:
            print(f"  {root}: {count:,}")

    if args.stats_only:
        return 0

    # Get candidates
    candidates = tracker.get_candidates(
        min_count=args.min_count,
        limit=args.limit,
        sort_by=args.sort,
    )

    if not candidates:
        print(f"\nNo roots found with count >= {args.min_count}")
        return 0

    print(f"\nCandidates (count >= {args.min_count}):")
    print_table(candidates, show_contexts=not args.no_context)

    # Interactive selection
    selected_roots = []
    if args.interactive:
        selected_roots = interactive_selection(candidates)

        if selected_roots:
            print(f"\nSelected {len(selected_roots)} roots:")
            for root in selected_roots:
                print(f"  - {root}")

    # Export if requested
    if args.export:
        roots_to_export = selected_roots if selected_roots else [c['root'] for c in candidates]

        with open(args.export, 'w', encoding='utf-8') as f:
            for root in roots_to_export:
                f.write(root + '\n')

        print(f"\nExported {len(roots_to_export)} roots to {args.export}")
        print(f"\nTo expand vocabulary, run:")
        print(f"  python scripts/expand_vocabulary.py --roots-file {args.export}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
