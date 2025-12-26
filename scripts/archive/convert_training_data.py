#!/usr/bin/env python3
"""
Convert tab-separated training pairs to JSONL format for Tree-LSTM training.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

def convert_file(txt_file: Path, jsonl_file: Path, label: int):
    """Convert tab-separated file to JSONL with ASTs."""
    print(f"Converting {txt_file} → {jsonl_file}")

    count = 0
    with open(txt_file, 'r', encoding='utf-8') as f_in, \
         open(jsonl_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                continue

            sent1, sent2 = parts

            try:
                # Parse both sentences to ASTs
                ast1 = parse(sent1)
                ast2 = parse(sent2)

                # Create training pair entry
                entry = {
                    "ast1": {"sentence": sent1, "ast": ast1},
                    "ast2": {"sentence": sent2, "ast": ast2},
                    "label": label
                }

                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1

                if count % 1000 == 0:
                    print(f"  Converted {count:,} pairs...")

            except Exception as e:
                # Skip pairs that fail to parse
                continue

    print(f"  ✓ Converted {count:,} pairs total")
    return count

def main():
    input_dir = Path("data/training_pairs_v2")

    print("Converting training data to JSONL format...")
    print()

    # Convert positive pairs
    pos_count = convert_file(
        input_dir / "positive_pairs.txt",
        input_dir / "positive_pairs.jsonl",
        label=1
    )

    print()

    # Convert negative pairs
    neg_count = convert_file(
        input_dir / "negative_pairs.txt",
        input_dir / "negative_pairs.jsonl",
        label=0
    )

    print()
    print("=" * 70)
    print(f"✓ Conversion complete!")
    print(f"  Positive pairs: {pos_count:,}")
    print(f"  Negative pairs: {neg_count:,}")
    print(f"  Total pairs:    {pos_count + neg_count:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
