#!/usr/bin/env python3
"""
Validate model vocabulary against authoritative Esperanto sources.

This script checks what percentage of the trained vocabulary consists of
valid Esperanto roots vs. junk (OCR artifacts, foreign words, etc.)

Run: python scripts/validate_vocabulary.py
"""

import json
import torch
from pathlib import Path
from collections import Counter


def load_authoritative_roots():
    """Load roots from Fundamento and ReVo."""
    roots = set()

    # Fundamento roots
    fundamento_path = Path('data/vocabularies/fundamento_roots.json')
    if fundamento_path.exists():
        with open(fundamento_path) as f:
            data = json.load(f)
            if 'roots' in data:
                roots.update(data['roots'].keys())
                print(f"Fundamento roots: {len(data['roots'])}")

    # ReVo headwords
    revo_path = Path('data/revo/revo_definitions_with_roots.json')
    if revo_path.exists():
        with open(revo_path) as f:
            data = json.load(f)
            roots.update(data.keys())
            print(f"ReVo headwords: {len(data)}")

    return roots


def categorize_junk(roots: set) -> dict:
    """Categorize junk roots by type."""
    categories = {
        'starts_with_digit': [],
        'contains_digit': [],
        'x_notation': [],  # cx, gx, etc
        'foreign_letters': [],  # w, x, y, q
        'too_short': [],  # 1-2 chars
        'cyrillic_like': [],
        'other': []
    }

    for root in roots:
        if not root:
            continue
        if root[0].isdigit():
            categories['starts_with_digit'].append(root)
        elif any(c.isdigit() for c in root):
            categories['contains_digit'].append(root)
        elif any(x in root for x in ['cx', 'gx', 'hx', 'jx', 'sx', 'ux']):
            categories['x_notation'].append(root)
        elif any(c in root for c in 'wxyq'):
            categories['foreign_letters'].append(root)
        elif len(root) <= 2:
            categories['too_short'].append(root)
        else:
            # Check for patterns that look like OCR errors
            has_weird_combo = any(x in root for x in ['tb', 'bii', 'tbi', 'iib', 'btb'])
            if has_weird_combo:
                categories['cyrillic_like'].append(root)
            else:
                categories['other'].append(root)

    return categories


def main():
    print("=" * 60)
    print("VOCABULARY VALIDATION REPORT")
    print("=" * 60)

    # Load model vocabulary
    model_path = Path('models/root_embeddings/best_model.pt')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    model_roots = set(checkpoint['root_to_idx'].keys())
    print(f"\nModel vocabulary size: {len(model_roots)}")

    # Load authoritative roots
    print()
    authoritative = load_authoritative_roots()
    print(f"Combined authoritative: {len(authoritative)}")

    # Compute overlap
    valid = model_roots & authoritative
    junk = model_roots - authoritative
    missing = authoritative - model_roots

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid roots (in authoritative): {len(valid):,} ({100*len(valid)/len(model_roots):.1f}%)")
    print(f"Junk roots (not authoritative): {len(junk):,} ({100*len(junk)/len(model_roots):.1f}%)")
    print(f"Missing from model: {len(missing):,} ({100*len(missing)/len(authoritative):.1f}% of authoritative)")

    # Categorize junk
    print()
    print("=" * 60)
    print("JUNK CATEGORIZATION")
    print("=" * 60)
    categories = categorize_junk(junk)
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(items) / len(junk) if junk else 0
        print(f"{cat:20}: {len(items):,} ({pct:.1f}%)")
        if items:
            print(f"  Sample: {sorted(items)[:5]}")

    # Summary
    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"""
The model vocabulary contains {len(junk):,} junk roots ({100*len(junk)/len(model_roots):.1f}%).

To fix this, the training script should:
1. ONLY include roots from Fundamento + ReVo headwords
2. Filter out roots not in the authoritative list
3. This would reduce vocabulary from {len(model_roots):,} to ~{len(authoritative):,} roots

The current model's correlation of 0.79 is impressive given the noise.
Retraining with a clean vocabulary should improve results.
""")


if __name__ == '__main__':
    main()
