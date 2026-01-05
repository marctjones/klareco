#!/usr/bin/env python3
"""
Extract root vocabulary from corpus for compositional embeddings.

Extracts all unique roots (radiko) from parsed ASTs along with frequency counts.
Also creates affix vocabularies for prefixes and suffixes.

Features:
- Resumable with checkpoints
- Line-buffered logging
- Memory-efficient streaming
"""

import json
import sys
import time
import gc
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKPOINT_FILENAME = "extract_root_vocab_checkpoint.json"

# Standard Esperanto affixes
STANDARD_PREFIXES = [
    'mal', 're', 'ge', 'eks', 'ek', 'pra', 'for', 'mis', 'bo', 'dis',
    'fi', 'vic', 'ĉef', 'duon', 'pseudo', 'kvazaŭ'
]

STANDARD_SUFFIXES = [
    'ul', 'ej', 'in', 'et', 'ad', 'ig', 'iĝ', 'ism', 'ist', 'ar',
    'aĉ', 'aĵ', 'ebl', 'end', 'ec', 'eg', 'em', 'er', 'estr', 'id',
    'il', 'ind', 'ing', 'uj', 'um', 'ant', 'int', 'ont', 'at', 'it', 'ot',
    'an', 'op', 'obl', 'on', 'ĉj', 'nj'
]


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, line_number: int, root_counts: dict,
                   prefix_counts: dict, suffix_counts: dict):
    """Save checkpoint with current progress."""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'line_number': line_number,
            'root_counts': root_counts,
            'prefix_counts': prefix_counts,
            'suffix_counts': suffix_counts,
        }, f)


def extract_words_from_ast(ast: dict) -> List[dict]:
    """
    Recursively extract all word ASTs from a sentence AST.

    Returns list of word dictionaries with radiko, prefikso, sufiksoj, etc.
    """
    words = []

    if not isinstance(ast, dict):
        return words

    tipo = ast.get('tipo')

    if tipo == 'vorto':
        # This is a word node
        if ast.get('radiko'):
            words.append(ast)

    # Recurse into all values
    for key, value in ast.items():
        if isinstance(value, dict):
            words.extend(extract_words_from_ast(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    words.extend(extract_words_from_ast(item))

    return words


def extract_root_vocabulary(
    corpus_path: Path,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
    batch_size: int = 50000,
    min_frequency: int = 2,
    clean_start: bool = False,
) -> dict:
    """
    Extract root vocabulary from corpus.

    Args:
        corpus_path: Path to corpus JSONL file (with ASTs)
        output_dir: Output directory for vocabulary files
        checkpoint_path: Path for checkpoint file
        batch_size: Lines between checkpoints/progress updates
        min_frequency: Minimum frequency to include in vocabulary
        clean_start: If True, delete existing output and checkpoint

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_lines': 0,
        'total_words': 0,
        'unique_roots': 0,
        'unique_prefixes': 0,
        'unique_suffixes': 0,
        'roots_in_vocab': 0,
    }

    # Handle clean start
    if clean_start:
        if output_dir.exists():
            import shutil
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        if checkpoint_path and checkpoint_path.exists():
            print(f"Removing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint or start fresh
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None

    if checkpoint:
        start_line = checkpoint['line_number']
        root_counts = Counter(checkpoint['root_counts'])
        prefix_counts = Counter(checkpoint['prefix_counts'])
        suffix_counts = Counter(checkpoint['suffix_counts'])
        print(f"Resuming from line {start_line:,}")
    else:
        start_line = 0
        root_counts = Counter()
        prefix_counts = Counter()
        suffix_counts = Counter()

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

            # Get AST
            ast = entry.get('ast')
            if not ast:
                continue

            # Extract all words
            words = extract_words_from_ast(ast)

            for word in words:
                stats['total_words'] += 1

                # Extract root
                radiko = word.get('radiko', '')
                if radiko:
                    root_counts[radiko] += 1

                # Extract prefix
                prefikso = word.get('prefikso')
                if prefikso:
                    prefix_counts[prefikso] += 1

                # Extract suffixes
                sufiksoj = word.get('sufiksoj', [])
                for suf in sufiksoj:
                    suffix_counts[suf] += 1

            # Progress and checkpoint
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1 - start_line) / elapsed if elapsed > 0 else 0
                pct = 100.0 * (i + 1) / total_lines

                print(f"   {i + 1:,}/{total_lines:,} ({pct:.1f}%) | "
                      f"{rate:.0f} lines/sec | "
                      f"{len(root_counts):,} roots | "
                      f"{stats['total_words']:,} words")

                # Save checkpoint
                if checkpoint_path:
                    save_checkpoint(
                        checkpoint_path, i + 1,
                        dict(root_counts), dict(prefix_counts), dict(suffix_counts)
                    )

                # Garbage collection
                gc.collect()

    # Build vocabularies
    print(f"\nBuilding vocabularies (min_frequency={min_frequency})...")

    # Root vocabulary - filter by frequency and sort
    filtered_roots = {r: c for r, c in root_counts.items() if c >= min_frequency}
    sorted_roots = sorted(filtered_roots.items(), key=lambda x: x[1], reverse=True)
    root_vocab = {root: idx for idx, (root, _) in enumerate(sorted_roots)}

    # Add special tokens
    root_vocab['<PAD>'] = len(root_vocab)
    root_vocab['<UNK>'] = len(root_vocab)

    stats['unique_roots'] = len(root_counts)
    stats['roots_in_vocab'] = len(root_vocab)

    # Prefix vocabulary - use standard + any found in corpus
    all_prefixes = set(STANDARD_PREFIXES) | set(prefix_counts.keys())
    prefix_vocab = {p: idx for idx, p in enumerate(sorted(all_prefixes))}
    prefix_vocab['<NONE>'] = len(prefix_vocab)
    stats['unique_prefixes'] = len(prefix_vocab)

    # Suffix vocabulary - use standard + any found in corpus
    all_suffixes = set(STANDARD_SUFFIXES) | set(suffix_counts.keys())
    suffix_vocab = {s: idx for idx, s in enumerate(sorted(all_suffixes))}
    suffix_vocab['<NONE>'] = len(suffix_vocab)
    stats['unique_suffixes'] = len(suffix_vocab)

    # Save vocabularies
    print("Saving vocabularies...")

    with open(output_dir / "root_vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump(root_vocab, f, ensure_ascii=False, indent=2)

    with open(output_dir / "root_statistics.json", 'w', encoding='utf-8') as f:
        json.dump({
            'total_unique': len(root_counts),
            'in_vocabulary': len(root_vocab),
            'min_frequency': min_frequency,
            'top_100': sorted_roots[:100],
        }, f, ensure_ascii=False, indent=2)

    with open(output_dir / "affix_vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'prefixes': prefix_vocab,
            'suffixes': suffix_vocab,
        }, f, ensure_ascii=False, indent=2)

    with open(output_dir / "affix_statistics.json", 'w', encoding='utf-8') as f:
        json.dump({
            'prefix_counts': dict(prefix_counts.most_common(50)),
            'suffix_counts': dict(suffix_counts.most_common(50)),
        }, f, ensure_ascii=False, indent=2)

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
        description="Extract root vocabulary from corpus for compositional embeddings"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus_with_sources_v2.jsonl"),
        help="Corpus JSONL file with ASTs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vocabularies"),
        help="Output directory for vocabulary files"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/extract_root_vocab_checkpoint.json"),
        help="Checkpoint file"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency to include in vocabulary (default: 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Lines between progress updates (default: 50000)"
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
    print("Extracting Root Vocabulary for Compositional Embeddings")
    print("=" * 60)
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Min frequency: {args.min_frequency}")
    print()

    # Check corpus exists
    if not args.corpus.exists():
        print(f"ERROR: Corpus file not found: {args.corpus}")
        sys.exit(1)

    # Extract vocabulary
    stats = extract_root_vocabulary(
        corpus_path=args.corpus,
        output_dir=args.output,
        checkpoint_path=args.checkpoint if not args.no_checkpoint else None,
        batch_size=args.batch_size,
        min_frequency=args.min_frequency,
        clean_start=args.clean,
    )

    # Print summary
    print()
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Total lines processed: {stats['total_lines']:,}")
    print(f"Total words extracted: {stats['total_words']:,}")
    print(f"Unique roots found: {stats['unique_roots']:,}")
    print(f"Roots in vocabulary (freq >= {args.min_frequency}): {stats['roots_in_vocab']:,}")
    print(f"Unique prefixes: {stats['unique_prefixes']:,}")
    print(f"Unique suffixes: {stats['unique_suffixes']:,}")
    print(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()
