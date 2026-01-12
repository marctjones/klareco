#!/usr/bin/env python3
"""
Extract proper nouns from corpus to build a dictionary.

Features:
- Resumable with checkpoints
- Line-buffered logging for real-time progress
- Output file protection (won't overwrite without --clean)
- Memory-efficient streaming
"""

import json
import sys
import time
import gc
from pathlib import Path
from typing import Optional
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKPOINT_FILENAME = "build_proper_noun_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, line_number: int, proper_nouns: dict):
    """Save checkpoint with current progress."""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'line_number': line_number,
            'proper_nouns': proper_nouns,
        }, f)


def strip_esperanto_endings(word: str) -> str:
    """
    Remove Esperanto noun endings to get the base form.

    Handles: -ojn, -oj, -on, -o (in order of length)
    """
    # Strip common noun endings (order matters - longest first)
    for ending in ('ojn', 'oj', 'on', 'o'):
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[:-len(ending)]
    return word


def is_proper_noun_candidate(word: str) -> bool:
    """
    Check if a word looks like a proper noun.

    Criteria:
    - Starts with uppercase letter
    - At least 2 characters
    - Not all uppercase (avoid acronyms like "UNESCO")
    """
    if len(word) < 2:
        return False
    if not word[0].isupper():
        return False
    if word.isupper():
        return False  # Skip acronyms
    return True


def extract_proper_nouns_from_ast(ast: dict) -> list:
    """
    Recursively extract proper noun candidates from AST.

    Returns list of (word, base_form) tuples.
    """
    proper_nouns = []

    if not isinstance(ast, dict):
        return proper_nouns

    # Check if this node is a proper noun
    vortspeco = ast.get('vortspeco', '')
    plena_vorto = ast.get('plena_vorto', '')

    if vortspeco == 'propra_nomo' and plena_vorto:
        base = strip_esperanto_endings(plena_vorto)
        proper_nouns.append((plena_vorto, base))

    # Also check for capitalized unknown words that might be proper nouns
    if ast.get('parse_status') in ('partial', 'failed'):
        word = ast.get('plena_vorto', '') or ast.get('vorto', '')
        if word and is_proper_noun_candidate(word):
            base = strip_esperanto_endings(word)
            proper_nouns.append((word, base))

    # Recurse into all dict values and lists
    for key, value in ast.items():
        if isinstance(value, dict):
            proper_nouns.extend(extract_proper_nouns_from_ast(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    proper_nouns.extend(extract_proper_nouns_from_ast(item))

    return proper_nouns


def infer_category(base: str, contexts: list) -> str:
    """
    Try to infer proper noun category from context.

    Categories: person, place, organization, other
    """
    # Simple heuristics based on common patterns
    base_lower = base.lower()

    # Place indicators (Esperanto country/city suffixes)
    if base_lower.endswith(('io', 'ujo', 'lando')):
        return 'place'

    # Check contexts for clues
    context_text = ' '.join(contexts).lower()

    if any(word in context_text for word in ['urbo', 'lando', 'regiono', 'rivero', 'monto']):
        return 'place'
    if any(word in context_text for word in ['sinjoro', 'sinjorino', 'reĝo', 'reĝino', 'princo']):
        return 'person'

    # Default based on ending patterns
    if base_lower.endswith(('a', 'o')) and len(base) < 10:
        return 'person'  # Short names ending in vowels often personal

    return 'other'


def build_proper_noun_dict(
    corpus_path: Path,
    output_path: Path,
    min_frequency: int = 3,
    checkpoint_path: Optional[Path] = None,
    batch_size: int = 1000,
    clean_start: bool = False,
) -> dict:
    """
    Build proper noun dictionary from corpus.

    Args:
        corpus_path: Path to corpus JSONL file
        output_path: Output JSON file for dictionary
        min_frequency: Minimum occurrences to include (default: 3)
        checkpoint_path: Path for checkpoint file
        batch_size: Lines between checkpoints/progress updates
        clean_start: If True, delete existing output and checkpoint

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_lines': 0,
        'lines_with_ast': 0,
        'proper_nouns_found': 0,
        'unique_bases': 0,
        'included_in_dict': 0,
    }

    # Handle clean start
    if clean_start:
        if output_path.exists():
            print(f"Removing existing output: {output_path}")
            output_path.unlink()
        if checkpoint_path and checkpoint_path.exists():
            print(f"Removing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()

    # Load checkpoint or start fresh
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None

    if checkpoint:
        start_line = checkpoint['line_number']
        proper_noun_counts = Counter(checkpoint['proper_nouns'])
        print(f"Resuming from line {start_line:,} ({len(proper_noun_counts):,} unique bases so far)")
    else:
        # Check for existing output without checkpoint
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"ERROR: Output file exists but no checkpoint found!")
            print(f"   Output: {output_path}")
            print(f"   Use --clean to start fresh")
            raise RuntimeError("Refusing to overwrite existing output without --clean")

        start_line = 0
        proper_noun_counts = Counter()

    # Track contexts for category inference
    proper_noun_contexts = {}  # base -> list of context sentences

    # Count total lines for progress
    print("Counting lines...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")

    # Process corpus
    start_time = time.time()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        # Skip to checkpoint position
        if start_line > 0:
            print(f"Skipping to line {start_line:,}...")
            for _ in range(start_line):
                next(f)

        for i, line in enumerate(f, start=start_line):
            stats['total_lines'] = i + 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract proper nouns from AST
            ast = entry.get('ast')
            if not ast:
                continue

            stats['lines_with_ast'] += 1

            proper_nouns = extract_proper_nouns_from_ast(ast)

            for word, base in proper_nouns:
                stats['proper_nouns_found'] += 1
                proper_noun_counts[base] += 1

                # Store context for category inference (limit to 5 per base)
                if base not in proper_noun_contexts:
                    proper_noun_contexts[base] = []
                if len(proper_noun_contexts[base]) < 5:
                    text = entry.get('text', '')
                    if text:
                        proper_noun_contexts[base].append(text)

            # Progress and checkpoint
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1 - start_line) / elapsed if elapsed > 0 else 0
                pct = 100.0 * (i + 1) / total_lines

                print(f"   {i + 1:,}/{total_lines:,} ({pct:.1f}%) | "
                      f"{rate:.0f} lines/sec | "
                      f"{len(proper_noun_counts):,} unique bases")

                # Save checkpoint
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, i + 1, dict(proper_noun_counts))

                # Garbage collection
                gc.collect()

    # Build final dictionary
    print(f"\nFiltering by frequency >= {min_frequency}...")

    proper_noun_dict = {}
    for base, count in proper_noun_counts.items():
        if count >= min_frequency:
            contexts = proper_noun_contexts.get(base, [])
            category = infer_category(base, contexts)

            proper_noun_dict[base] = {
                'frequency': count,
                'category': category,
                'source': 'corpus',
            }
            stats['included_in_dict'] += 1

    stats['unique_bases'] = len(proper_noun_counts)

    # Save dictionary
    print(f"Saving dictionary to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(proper_noun_dict, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint on success
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print("Checkpoint removed (build complete)")
        except OSError as e:
            print(f"Warning: Could not remove checkpoint: {e}")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build proper noun dictionary from corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus_with_sources_v2.jsonl"),
        help="Corpus JSONL file (default: data/corpus_with_sources_v2.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/proper_nouns_dynamic.json"),
        help="Output dictionary file (default: data/proper_nouns_dynamic.json)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/build_proper_noun_checkpoint.json"),
        help="Checkpoint file (default: data/build_proper_noun_checkpoint.json)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum frequency to include (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Lines between progress updates (default: 10000)"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output and checkpoint before starting"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Proper Noun Dictionary")
    print("=" * 60)
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Checkpointing: {not args.no_checkpoint}")
    print()

    # Check corpus exists
    if not args.corpus.exists():
        print(f"ERROR: Corpus file not found: {args.corpus}")
        sys.exit(1)

    # Build dictionary
    stats = build_proper_noun_dict(
        corpus_path=args.corpus,
        output_path=args.output,
        min_frequency=args.min_frequency,
        checkpoint_path=args.checkpoint if not args.no_checkpoint else None,
        batch_size=args.batch_size,
        clean_start=args.clean,
    )

    # Print summary
    print()
    print("=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"Total lines processed: {stats['total_lines']:,}")
    print(f"Lines with AST: {stats['lines_with_ast']:,}")
    print(f"Proper nouns found: {stats['proper_nouns_found']:,}")
    print(f"Unique base forms: {stats['unique_bases']:,}")
    print(f"Included in dictionary (freq >= {args.min_frequency}): {stats['included_in_dict']:,}")
    print(f"Output file: {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
