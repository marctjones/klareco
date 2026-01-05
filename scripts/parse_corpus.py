#!/usr/bin/env python3
"""
Build corpus v2 with proper sentence extraction.

This version:
- Uses extract_sentences.py to get complete sentences (not line fragments)
- Stores AST metadata for faster indexing
- Filters by parse quality
- Resumable with checkpoints
- Memory-efficient batch processing
- CPU throttling to prevent system freeze
- Fine-grained progress indicators
"""

import json
import sys
import time
import gc
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_sentences import extract_sentences_streaming

# Texts to include
TEXTS_TO_INDEX = [
    ("cleaned_la_mastro_de_l_ringoj.txt", "La Mastro de l' Ringoj (Lord of the Rings)"),
    ("cleaned_la_hobito.txt", "La Hobito (The Hobbit)"),
    ("cleaned_kadavrejo_strato.txt", "Kadavrejo Strato (Poe)"),
    ("cleaned_la_korvo.txt", "La Korvo (The Raven)"),
    ("cleaned_puto_kaj_pendolo.txt", "Puto kaj Pendolo (Pit and Pendulum)"),
    ("cleaned_ses_noveloj.txt", "Ses Noveloj (Six Stories)"),
    ("cleaned_usxero_domo.txt", "Usxero Domo (Fall of House of Usher)"),
    ("cleaned_wikipedia.txt", "Vikipedio Esperanto (Wikipedia)"),
]

CHECKPOINT_FILENAME = "build_corpus_v2_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, file: str, total: int, sentence_offset: int = 0,
                    byte_position: int = 0, file_size: int = 0):
    """Save checkpoint with sentence offset and byte position for intra-file resumption."""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'file': file,
            'total': total,
            'sentence_offset': sentence_offset,
            'byte_position': byte_position,
            'file_size': file_size,
        }, f)


def get_next_file(texts: List[Tuple[str, str]], current_file: str) -> Optional[str]:
    """Get the next file after current_file in the texts list."""
    for i, (filename, _) in enumerate(texts):
        if filename == current_file:
            if i + 1 < len(texts):
                return texts[i + 1][0]
            return None  # current_file is the last one
    return None


def build_corpus_v2(
    cleaned_dir: Path,
    output_file: Path,
    texts: List[Tuple[str, str]],
    min_words: int = 3,
    max_words: int = 100,
    min_parse_rate: float = 0.0,  # 0.0 = no filtering, 0.7 = strict
    checkpoint_path: Optional[Path] = None,
    batch_size: int = 100,
    throttle_delay: float = 0.0,
    parse_timeout: int = 30,
    clean_start: bool = False,
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
        checkpoint_path: Path for checkpoint file
        batch_size: Number of sentences to process before checkpointing (default: 100)
        throttle_delay: Delay in seconds between batches to reduce CPU load (default: 0.0)
        parse_timeout: Max seconds to wait for parsing a sentence (default: 30)
        clean_start: If True, delete existing output and checkpoint files

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_sentences': 0,
        'total_files': 0,
        'skipped_files': 0,
        'filtered_by_parse': 0,
    }

    # Handle clean start - delete existing files
    if clean_start:
        if output_file.exists():
            print(f"🗑️  Removing existing output file: {output_file}")
            output_file.unlink()
        if checkpoint_path and checkpoint_path.exists():
            print(f"🗑️  Removing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
    start_file = checkpoint.get('file') if checkpoint else None
    byte_offset = checkpoint.get('byte_position', 0) if checkpoint else 0

    # CRITICAL: Check if output file exists but checkpoint is missing
    # This prevents accidental data loss
    output_exists = output_file.exists() and output_file.stat().st_size > 0

    if checkpoint:
        # Checkpoint exists - we're resuming
        stats['total_sentences'] = checkpoint.get('total', 0)
        print(f"📁 Resuming from: {start_file} (already processed: {stats['total_sentences']:,} sentences)")
        if byte_offset > 0:
            file_size = checkpoint.get('file_size', 0)
            if file_size > 0:
                pct = 100.0 * byte_offset / file_size
                print(f"   Byte position: {byte_offset:,} / {file_size:,} ({pct:.1f}%)")
            else:
                print(f"   Byte position: {byte_offset:,}")
        mode = 'a'
    elif output_exists:
        # No checkpoint but output file has data - REFUSE to overwrite
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"❌ ERROR: Output file exists with data but no checkpoint found!")
        print(f"   Output file: {output_file}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print()
        print(f"   This would overwrite existing data. To proceed, either:")
        print(f"   1. Use --clean to start fresh (deletes existing data)")
        print(f"   2. Restore the checkpoint file: {checkpoint_path}")
        print()
        raise RuntimeError("Refusing to overwrite existing corpus without --clean flag")
    else:
        # No checkpoint, no output file - fresh start
        mode = 'w'

    with output_file.open(mode, encoding='utf-8', buffering=1) as out:
        skip_until_found = bool(start_file)
        resuming_file = False

        for filename, display_name in texts:
            # Skip files until we reach checkpoint
            if skip_until_found:
                if filename == start_file:
                    skip_until_found = False
                    resuming_file = True  # We're resuming within this file
                else:
                    print(f"⏭️  Skipping (already done): {display_name}")
                    continue

            file_path = cleaned_dir / filename

            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                stats['skipped_files'] += 1
                continue

            print(f"📖 Processing: {display_name}")
            file_start_time = time.time()

            # Write to corpus using streaming to avoid loading all sentences in memory
            source_id = filename.replace("cleaned_", "").replace(".txt", "")
            count = 0
            filtered_count = 0

            # If resuming this file, use byte position to skip efficiently
            start_byte = byte_offset if resuming_file else 0
            file_size_on_disk = file_path.stat().st_size
            if resuming_file:
                # Check if we already completed this file (byte_position >= file_size)
                checkpoint_file_size = checkpoint.get('file_size', 0) if checkpoint else 0
                if start_byte > 0 and start_byte >= checkpoint_file_size and checkpoint_file_size > 0:
                    # File was already fully processed, skip to next
                    print(f"   ⏩ File already completed (byte {start_byte:,} >= {checkpoint_file_size:,}), skipping")
                    resuming_file = False
                    byte_offset = 0
                    continue  # Skip to next file
                elif start_byte > 0:
                    print(f"   ⏩ Resuming from byte position {start_byte:,}")
                resuming_file = False
                byte_offset = 0

            # Stream sentences one at a time (memory efficient!)
            # Use start_byte for efficient resumption (no re-parsing!)
            try:
                sentence_iterator = extract_sentences_streaming(
                    file_path,
                    min_words=min_words,
                    max_words=max_words,
                    with_ast=True,  # Always generate AST
                    batch_size=batch_size,  # GC frequency in extractor
                    parse_timeout=parse_timeout,
                    start_byte=start_byte,  # Resume from byte position
                )

                for sent_data in sentence_iterator:
                    # Filter by parse quality if requested
                    # Note: We want to KEEP sentences that parse well (high parse_rate)
                    # A low parse_rate might indicate non-Esperanto text or malformed sentences
                    if min_parse_rate > 0:
                        parse_rate = sent_data.get('parse_rate', 0)
                        # Keep sentences with parse_rate >= min_parse_rate
                        if parse_rate < min_parse_rate:
                            filtered_count += 1
                            continue

                    entry = {
                        'text': sent_data['text'],
                        'source': source_id,
                        'source_name': display_name,
                        'paragraph': sent_data['paragraph'],
                        'word_count': sent_data['word_count'],
                    }

                    # Include AST if available
                    if 'ast' in sent_data:
                        entry['ast'] = sent_data['ast']
                        entry['parse_rate'] = sent_data.get('parse_rate', 0.0)

                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    count += 1
                    stats['total_sentences'] += 1

                    # Track byte position for progress and resumption (from chunked reader)
                    byte_pos = sent_data.get('_byte_position', 0)
                    file_size = sent_data.get('_file_size', 0)

                    # Progress indicator every batch_size sentences
                    if count % batch_size == 0:
                        elapsed = time.time() - file_start_time
                        rate = count / elapsed if elapsed > 0 else 0

                        # Calculate file progress percentage
                        if file_size > 0:
                            pct = 100.0 * byte_pos / file_size
                            progress_str = f"{pct:.1f}% of file"
                        else:
                            progress_str = "N/A"

                        # Show progress with file percentage
                        try:
                            import psutil
                            process = psutil.Process()
                            mem_mb = process.memory_info().rss / 1024 / 1024
                            print(f"   ⏳ {count:,} sent ({rate:.1f}/sec) | {progress_str} | {filtered_count:,} filtered | {mem_mb:.0f}MB")
                        except ImportError:
                            print(f"   ⏳ {count:,} sent ({rate:.1f}/sec) | {progress_str} | {filtered_count:,} filtered")

                        # Save checkpoint within file (use byte_pos for efficient resumption)
                        if checkpoint_path:
                            save_checkpoint(checkpoint_path, filename, stats['total_sentences'], 0,
                                          byte_pos, file_size)

                        # Throttle CPU if requested
                        if throttle_delay > 0:
                            time.sleep(throttle_delay)

                        # Force garbage collection to free memory
                        gc.collect()

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                stats['skipped_files'] += 1
                continue

            # Update stats
            stats['filtered_by_parse'] += filtered_count
            if filtered_count > 0:
                print(f"   🔍 Filtered {filtered_count:,} low-quality sentences (parse_rate < {min_parse_rate})")

            elapsed = time.time() - file_start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"   ✅ Added {count:,} sentences in {elapsed:.1f}s ({rate:.1f} sent/sec)")
            stats['total_files'] += 1

            # Save checkpoint pointing to NEXT file (not current with offset=0)
            # This prevents duplicate processing if we crash right after this
            if checkpoint_path:
                next_file = get_next_file(texts, filename)
                if next_file:
                    # More files to process - point to next one
                    save_checkpoint(checkpoint_path, next_file, stats['total_sentences'], 0, 0, 0)
                # If no next file, we'll delete checkpoint at the end anyway

            # Force final garbage collection
            gc.collect()

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
        default=0.0,
        help="Minimum parse success rate 0.0-1.0 (default: 0.0 = no filtering, use 0.3-0.5 to filter out low-quality/non-Esperanto sentences)"
    )
    parser.add_argument(
        "--parse-timeout",
        type=int,
        default=30,
        help="Max seconds to wait for parsing a sentence (default: 30, problem sentences logged to data/problem_sentences.jsonl)"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sentences to process before checkpointing and showing progress (default: 20, lower = more frequent checkpoints/progress)"
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.1,
        help="Delay in seconds between batches to reduce CPU load (default: 0.1s, use 0.0 for max speed or increase to 0.2-0.5 to prevent freezing)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output and checkpoint files before starting (required if output exists without checkpoint)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Corpus v2 with Sentence Extraction")
    print("=" * 60)
    print(f"Cleaned dir: {args.cleaned_dir}")
    print(f"Output: {args.output}")
    print(f"Min words: {args.min_words}, Max words: {args.max_words}")
    print(f"Min parse rate: {args.min_parse_rate}")
    print(f"Parse timeout: {args.parse_timeout}s (problem sentences logged to data/problem_sentences.jsonl)")
    print(f"Checkpointing: {not args.no_checkpoint}")
    print(f"Batch size: {args.batch_size} sentences")
    if args.throttle > 0:
        print(f"CPU throttle: {args.throttle}s delay between batches")
    print()

    # Build corpus
    stats = build_corpus_v2(
        cleaned_dir=args.cleaned_dir,
        output_file=args.output,
        texts=TEXTS_TO_INDEX,
        min_words=args.min_words,
        max_words=args.max_words,
        min_parse_rate=args.min_parse_rate,
        checkpoint_path=args.checkpoint if not args.no_checkpoint else None,
        batch_size=args.batch_size,
        throttle_delay=args.throttle,
        parse_timeout=args.parse_timeout,
        clean_start=args.clean,
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
        try:
            args.checkpoint.unlink()
            print("🧹 Checkpoint file removed (build complete)")
        except OSError as e:
            print(f"⚠️  Could not remove checkpoint file: {e}")


if __name__ == '__main__':
    main()
