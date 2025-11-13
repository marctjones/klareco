"""
Build unified corpus from all cleaned texts with source metadata.

This script combines all Esperanto texts into one corpus file with
source attribution (which book, which section) for RAG retrieval.

Output format (JSONL):
{"text": "sentence", "source": "la_mastro_de_l_ringoj", "line": 1234}
"""

import json
from pathlib import Path
from typing import List, Tuple

# Texts to include (excluding Wikipedia for now - too large)
TEXTS_TO_INDEX = [
    ("cleaned_la_mastro_de_l_ringoj.txt", "La Mastro de l' Ringoj (Lord of the Rings)"),
    ("cleaned_la_hobito.txt", "La Hobito (The Hobbit)"),
    ("cleaned_kadavrejo_strato.txt", "Kadavrejo Strato (Poe)"),
    ("cleaned_la_korvo.txt", "La Korvo (The Raven)"),
    ("cleaned_puto_kaj_pendolo.txt", "Puto kaj Pendolo (Pit and Pendulum)"),
    ("cleaned_ses_noveloj.txt", "Ses Noveloj (Six Stories)"),
    ("cleaned_usxero_domo.txt", "Usxero Domo (Fall of House of Usher)"),
]


def build_corpus(
    cleaned_dir: Path,
    output_file: Path,
    texts: List[Tuple[str, str]],
    skip_empty: bool = True,
    skip_metadata: bool = True
) -> int:
    """
    Build corpus from cleaned texts with source metadata.

    Args:
        cleaned_dir: Directory containing cleaned text files
        output_file: Output JSONL file path
        texts: List of (filename, display_name) tuples
        skip_empty: Skip empty lines
        skip_metadata: Skip likely metadata lines (very short, all caps, etc.)

    Returns:
        Total number of sentences written
    """
    total = 0

    with output_file.open('w', encoding='utf-8') as out:
        for filename, display_name in texts:
            file_path = cleaned_dir / filename

            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue

            print(f"📖 Processing: {display_name}")
            count = 0

            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines
                    if skip_empty and not line:
                        continue

                    # Skip likely metadata
                    if skip_metadata:
                        # Skip very short lines (< 10 chars)
                        if len(line) < 10:
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
                    count += 1

            print(f"   ✅ Added {count:,} lines")
            total += count

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

    total = build_corpus(
        args.cleaned_dir,
        args.output,
        texts,
        skip_empty=True,
        skip_metadata=not args.no_skip_metadata
    )

    print()
    print(f"✅ Done! Total sentences: {total:,}")
    print(f"📄 Corpus saved to: {args.output}")
    print()
    print("Next step: Index the corpus with:")
    print(f"  python scripts/index_corpus.py --corpus {args.output} --output data/corpus_index_new")


if __name__ == "__main__":
    main()
