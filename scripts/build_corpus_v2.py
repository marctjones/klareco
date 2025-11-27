#!/usr/bin/env python3
"""
Build corpus v2 with proper sentence extraction.

This version:
- Uses extract_sentences.py to get complete sentences (not line fragments)
- Stores AST metadata for faster indexing
- Filters by parse quality
- Resumable with checkpoints
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_sentences import extract_sentences_from_file

# Texts to include
TEXTS_TO_INDEX = [
    ("cleaned_la_mastro_de_l_ringoj.txt", "La Mastro de l' Ringoj (Lord of the Rings)"),
    ("cleaned_la_hobito.txt", "La Hobito (The Hobbit)"),
    ("cleaned_kadavrejo_strato.txt", "Kadavrejo Strato (Poe)"),
    ("cleaned_la_korvo.txt", "La Korvo (The Raven)"),
    ("cleaned_puto_kaj_pendolo.txt", "Puto kaj Pendolo (Pit and Pendulum)"),
    ("cleaned_ses_noveloj.txt", "Ses Noveloj (Six Stories)"),
    ("cleaned_usxero_domo.txt", "Usxero Domo (Fall of House of Usher)"),
]

CHECKPOINT_FILENAME = "build_corpus_v2_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, file: str, total: int):
    """Save checkpoint."""
    with open(checkpoint_path, 'w') as f:
        json.dump({'file': file, 'total': total}, f)


def build_corpus_v2(
    cleaned_dir: Path,
    output_file: Path,
    texts: List[Tuple[str, str]],
    min_words: int = 3,
    max_words: int = 100,
    min_parse_rate: float = 0.0,  # 0.0 = no filtering, 0.7 = strict
    with_ast: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """
    Build corpus v2 with proper sentence extraction.

    Args:
        cleaned_dir: Directory with cleaned text files
        output_file: Output JSONL file
        texts: List of (filename, display_name) tuples
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence
        min_parse_rate: Minimum parse success rate (0.0-1.0)
        with_ast: Store AST in corpus
        checkpoint_path: Path for checkpoint file

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_sentences': 0,
        'total_files': 0,
        'skipped_files': 0,
        'filtered_by_parse': 0,
    }

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
    start_file = checkpoint.get('file') if checkpoint else None
    if checkpoint:
        stats['total_sentences'] = checkpoint.get('total', 0)
        print(f"📁 Resuming from: {start_file} (already processed: {stats['total_sentences']} sentences)")

    # Open output file (append if resuming, write if new)
    mode = 'a' if checkpoint else 'w'

    with output_file.open(mode, encoding='utf-8', buffering=1) as out:
        skip_until_found = bool(start_file)

        for filename, display_name in texts:
            # Skip files until we reach checkpoint
            if skip_until_found:
                if filename == start_file:
                    skip_until_found = False
                    continue  # Already processed this file
                else:
                    print(f"⏭️  Skipping (already done): {display_name}")
                    continue

            file_path = cleaned_dir / filename

            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                stats['skipped_files'] += 1
                continue

            print(f"📖 Processing: {display_name}")

            # Extract sentences
            try:
                sentences = extract_sentences_from_file(
                    file_path,
                    min_words=min_words,
                    max_words=max_words,
                    with_ast=with_ast
                )
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                stats['skipped_files'] += 1
                continue

            # Filter by parse quality if requested
            if min_parse_rate > 0 and with_ast:
                original_count = len(sentences)
                sentences = [s for s in sentences if s.get('parse_rate', 0) >= min_parse_rate]
                filtered = original_count - len(sentences)
                stats['filtered_by_parse'] += filtered
                if filtered > 0:
                    print(f"   🔍 Filtered {filtered} low-quality sentences (parse_rate < {min_parse_rate})")

            # Write to corpus
            source_id = filename.replace("cleaned_", "").replace(".txt", "")
            count = 0

            for sent_data in sentences:
                entry = {
                    'text': sent_data['text'],
                    'source': source_id,
                    'source_name': display_name,
                    'paragraph': sent_data['paragraph'],
                    'word_count': sent_data['word_count'],
                }

                # Include AST if available
                if with_ast and 'ast' in sent_data:
                    entry['ast'] = sent_data['ast']
                    entry['parse_rate'] = sent_data.get('parse_rate', 0.0)

                out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1
                stats['total_sentences'] += 1

            print(f"   ✅ Added {count:,} sentences")
            stats['total_files'] += 1

            # Save checkpoint
            if checkpoint_path:
                save_checkpoint(checkpoint_path, filename, stats['total_sentences'])

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build corpus v2 with sentence extraction")
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=Path("data/cleaned"),
        help="Directory with cleaned texts (default: data/cleaned)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus_with_sources_v2.jsonl"),
        help="Output JSONL file (default: data/corpus_with_sources_v2.jsonl)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/build_corpus_v2_checkpoint.json"),
        help="Checkpoint file (default: data/build_corpus_v2_checkpoint.json)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum words per sentence (default: 3)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=100,
        help="Maximum words per sentence (default: 100)"
    )
    parser.add_argument(
        "--min-parse-rate",
        type=float,
        default=0.5,
        help="Minimum parse success rate 0.0-1.0 (default: 0.5, use 0.0 for no filtering)"
    )
    parser.add_argument(
        "--no-ast",
        action="store_true",
        help="Don't generate AST (faster but no quality filtering)"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Corpus v2 with Sentence Extraction")
    print("=" * 60)
    print(f"Cleaned dir: {args.cleaned_dir}")
    print(f"Output: {args.output}")
    print(f"Min words: {args.min_words}, Max words: {args.max_words}")
    print(f"Min parse rate: {args.min_parse_rate}")
    print(f"Generate AST: {not args.no_ast}")
    print(f"Checkpointing: {not args.no_checkpoint}")
    print()

    # Build corpus
    stats = build_corpus_v2(
        cleaned_dir=args.cleaned_dir,
        output_file=args.output,
        texts=TEXTS_TO_INDEX,
        min_words=args.min_words,
        max_words=args.max_words,
        min_parse_rate=args.min_parse_rate,
        with_ast=not args.no_ast,
        checkpoint_path=args.checkpoint if not args.no_checkpoint else None,
    )

    # Print summary
    print()
    print("=" * 60)
    print("✅ Corpus v2 Build Complete!")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total sentences: {stats['total_sentences']:,}")
    print(f"Filtered by parse quality: {stats['filtered_by_parse']:,}")
    print(f"Output file: {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Clean up checkpoint on success
    if not args.no_checkpoint and args.checkpoint.exists():
        args.checkpoint.unlink()
        print("🧹 Checkpoint file removed (build complete)")


if __name__ == '__main__':
    main()
