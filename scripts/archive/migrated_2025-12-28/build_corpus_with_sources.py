"""
Build unified corpus from cleaned texts with source metadata.

Design goals:
- Interruptible/resumable via checkpoints (no lost work on interruption).
- Line-buffered output for real-time tailing.
- Source attribution for downstream RAG.

Output format (JSONL):
{"text": "sentence", "source": "la_mastro_de_l_ringoj", "line": 1234}
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

# Texts to include (Wikipedia optional via CLI flags)
TEXTS_TO_INDEX = [
    ("cleaned_la_mastro_de_l_ringoj.txt", "La Mastro de l' Ringoj (Lord of the Rings)"),
    ("cleaned_la_hobito.txt", "La Hobito (The Hobbit)"),
    ("cleaned_kadavrejo_strato.txt", "Kadavrejo Strato (Poe)"),
    ("cleaned_la_korvo.txt", "La Korvo (The Raven)"),
    ("cleaned_puto_kaj_pendolo.txt", "Puto kaj Pendolo (Pit and Pendulum)"),
    ("cleaned_ses_noveloj.txt", "Ses Noveloj (Six Stories)"),
    ("cleaned_usxero_domo.txt", "Usxero Domo (Fall of House of Usher)"),
]

CHECKPOINT_FILENAME = "build_corpus_checkpoint.json"


def build_corpus(
    cleaned_dir: Path,
    output_file: Path,
    texts: List[Tuple[str, str]],
    skip_empty: bool = True,
    skip_metadata: bool = True,
    min_length: int = 20,
    checkpoint_path: Optional[Path] = None,
) -> int:
    """
    Build corpus from cleaned texts with source metadata.

    Args:
        cleaned_dir: Directory containing cleaned text files
        output_file: Output JSONL file path
        texts: List of (filename, display_name) tuples
        skip_empty: Skip empty lines
        skip_metadata: Skip likely metadata lines (very short, all caps, etc.)
        min_length: Minimum sentence length in characters (default: 20)

    Returns:
        Total number of sentences written
    """
    total = 0
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else None

    # Append mode if resuming, otherwise truncate
    mode = "a" if (checkpoint_path and checkpoint) else "w"
    with output_file.open(mode, encoding='utf-8') as out:
        for filename, display_name in texts:
            file_path = cleaned_dir / filename

            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue

            print(f"📖 Processing: {display_name}")
            start_line = 0
            if checkpoint and checkpoint.get("file") == filename:
                start_line = checkpoint.get("line", 0)
                total = checkpoint.get("total_written", total)

            count = 0
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num <= start_line:
                        continue
                    line = line.strip()

                    # Skip empty lines
                    if skip_empty and not line:
                        continue

                    # Skip likely metadata
                    if skip_metadata:
                        # Skip very short lines (< min_length chars)
                        if len(line) < min_length:
                            continue

                        # Skip lines that are all caps (likely headers)
                        if line.isupper() and len(line) < 50:
                            continue

                        # Skip lines starting with "Produced by", "Distributed", etc.
                        if any(line.startswith(prefix) for prefix in [
                            "Produced by", "Distributed", "[Ilustraĵo:",
                            "***", "---", "==", "Project Gutenberg"
                        ]):
                            continue

                    # Write to corpus with metadata
                    entry = {
                        "text": line,
                        "source": filename.replace("cleaned_", "").replace(".txt", ""),
                        "source_name": display_name,
                        "line": line_num
                    }
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    out.flush()
                    count += 1
                    total += 1

                    if checkpoint_path and (line_num % 1000 == 0):
                        _save_checkpoint(checkpoint_path, filename, line_num, total)

            print(f"   ✅ Added {count:,} lines")
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, filename, line_num, total)

    return total


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build corpus with source metadata")
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=Path("data/cleaned"),
        help="Directory with cleaned texts (default: data/cleaned)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus_with_sources.jsonl"),
        help="Output JSONL file (default: data/corpus_with_sources.jsonl)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file for resumable runs (default: data/build_corpus_checkpoint.json)"
    )
    parser.add_argument(
        "--include-wikipedia",
        action="store_true",
        help="Include Wikipedia (warning: 18M+ lines, very slow)"
    )
    parser.add_argument(
        "--wikipedia-limit",
        type=int,
        default=None,
        help="Limit Wikipedia to N lines (e.g., 100000 for first 100K)"
    )
    parser.add_argument(
        "--no-skip-metadata",
        action="store_true",
        help="Include all lines (don't skip metadata)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum sentence length in characters (default: 20)"
    )

    args = parser.parse_args()

    texts = TEXTS_TO_INDEX.copy()

    # Add Wikipedia if requested
    if args.include_wikipedia:
        texts.append((
            "cleaned_wikipedia.txt",
            "Vikipedio (Wikipedia)"
        ))
        if args.wikipedia_limit:
            print(f"⚠️  Wikipedia will be limited to first {args.wikipedia_limit:,} lines")

    print(f"🔧 Building corpus from {len(texts)} texts")
    print(f"📁 Output: {args.output}")
    print()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.output.parent / CHECKPOINT_FILENAME

    total = build_corpus(
        args.cleaned_dir,
        args.output,
        texts,
        skip_empty=True,
        skip_metadata=not args.no_skip_metadata,
        min_length=args.min_length,
        checkpoint_path=checkpoint_path,
    )

    print()
    print(f"✅ Done! Total sentences: {total:,}")
    print(f"📄 Corpus saved to: {args.output}")
    print()
    print("Next step: Index the corpus with:")
    print(f"  python scripts/index_corpus.py --corpus {args.output} --output data/corpus_index_new")


if __name__ == "__main__":
    main()


def _load_checkpoint(path: Optional[Path]) -> Optional[dict]:
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(path: Path, filename: str, line: int, total: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"file": filename, "line": line, "total_written": total}
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)
