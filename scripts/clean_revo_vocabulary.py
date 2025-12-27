#!/usr/bin/env python3
"""
Clean ReVo vocabulary for training.

This script:
1. Normalizes x-notation to proper Unicode (abajxur -> abaĵur)
2. Strips homograph markers (a1 -> a)
3. Filters out non-Esperanto entries
4. Validates against known-good sources

Output: data/vocabularies/clean_roots.json
"""

import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict


# Valid Esperanto letters (lowercase)
VALID_EO_CHARS = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭ')

# X-notation to Unicode mapping
X_TO_UNICODE = {
    'cx': 'ĉ', 'Cx': 'Ĉ', 'CX': 'Ĉ',
    'gx': 'ĝ', 'Gx': 'Ĝ', 'GX': 'Ĝ',
    'hx': 'ĥ', 'Hx': 'Ĥ', 'HX': 'Ĥ',
    'jx': 'ĵ', 'Jx': 'Ĵ', 'JX': 'Ĵ',
    'sx': 'ŝ', 'Sx': 'Ŝ', 'SX': 'Ŝ',
    'ux': 'ŭ', 'Ux': 'Ŭ', 'UX': 'Ŭ',
}


def normalize_x_notation(text: str) -> str:
    """Convert x-notation to proper Unicode."""
    result = text
    for x_form, unicode_form in X_TO_UNICODE.items():
        result = result.replace(x_form, unicode_form)
    return result


def strip_homograph_marker(headword: str) -> str:
    """Remove trailing digit from homograph markers like 'a1' -> 'a'."""
    return re.sub(r'\d+$', '', headword)


def is_valid_esperanto_root(root: str) -> bool:
    """Check if root contains only valid Esperanto characters."""
    if not root or len(root) < 2:
        return False
    
    # Must use only valid Esperanto letters
    root_lower = root.lower()
    if not all(c in VALID_EO_CHARS for c in root_lower):
        return False
    
    # No digits allowed
    if any(c.isdigit() for c in root):
        return False
    
    return True


def load_revo_headwords(db_path: Path) -> dict:
    """Load headwords from ReVo SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all article headwords (mrk column)
    cursor.execute("SELECT mrk FROM artikolo")
    rows = cursor.fetchall()
    
    headwords = {}
    for (mrk,) in rows:
        if mrk:
            # Normalize and clean
            normalized = normalize_x_notation(mrk)
            base = strip_homograph_marker(normalized)
            
            if is_valid_esperanto_root(base):
                headwords[base] = {
                    'original': mrk,
                    'normalized': normalized,
                    'source': 'revo'
                }
    
    conn.close()
    return headwords


def load_fundamento_roots(json_path: Path) -> dict:
    """Load Fundamento roots and fix known OCR errors."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Known OCR corrections
    OCR_FIXES = {
        'akuŝlst': 'akuŝist',  # midwife
        'oficlst': 'oficist',  # official
        'yiŝ': 'viŝ',          # wipe/towel
    }
    
    roots = {}
    for root, info in data.get('roots', {}).items():
        # Apply OCR fixes
        fixed_root = OCR_FIXES.get(root, root)
        
        if is_valid_esperanto_root(fixed_root):
            roots[fixed_root] = {
                'original': root,
                'translations': info.get('translations', {}),
                'source': 'fundamento'
            }
        else:
            print(f"  Skipping invalid Fundamento root: {root}")
    
    return roots


def load_vortlisto_roots(json_path: Path) -> set:
    """Load Vortlisto roots for validation."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Clean roots (remove affix markers like -ad-)
    roots = set()
    for root in data.get('roots', []):
        clean = root.strip('-')
        if is_valid_esperanto_root(clean):
            roots.add(clean)
    
    return roots


def main():
    print("=" * 60)
    print("CLEANING REVO VOCABULARY")
    print("=" * 60)
    
    # Paths
    revo_db = Path('data/revo/revo.db')
    fundamento_json = Path('data/vocabularies/fundamento_roots.json')
    vortlisto_json = Path('data/vocabularies/reference/vortlisto_words.json')
    output_path = Path('data/vocabularies/clean_roots.json')
    
    # Load sources
    print("\n1. Loading ReVo headwords...")
    revo_roots = load_revo_headwords(revo_db)
    print(f"   Valid ReVo roots: {len(revo_roots)}")
    
    print("\n2. Loading Fundamento roots...")
    fundamento_roots = load_fundamento_roots(fundamento_json)
    print(f"   Valid Fundamento roots: {len(fundamento_roots)}")
    
    print("\n3. Loading Vortlisto for validation...")
    vortlisto_roots = load_vortlisto_roots(vortlisto_json)
    print(f"   Vortlisto roots: {len(vortlisto_roots)}")
    
    # Merge sources (Fundamento takes priority)
    print("\n4. Merging sources...")
    all_roots = {}
    
    # Add ReVo first
    for root, info in revo_roots.items():
        all_roots[root] = info
    
    # Fundamento overwrites (higher authority)
    for root, info in fundamento_roots.items():
        if root in all_roots:
            all_roots[root]['source'] = 'fundamento+revo'
            all_roots[root]['translations'] = info.get('translations', {})
        else:
            all_roots[root] = info
    
    # Statistics
    fundamento_only = set(fundamento_roots.keys()) - set(revo_roots.keys())
    revo_only = set(revo_roots.keys()) - set(fundamento_roots.keys())
    both = set(fundamento_roots.keys()) & set(revo_roots.keys())
    
    print(f"\n5. Statistics:")
    print(f"   Total clean roots: {len(all_roots)}")
    print(f"   In Fundamento only: {len(fundamento_only)}")
    print(f"   In ReVo only: {len(revo_only)}")
    print(f"   In both: {len(both)}")
    
    # Validate against Vortlisto
    vortlisto_coverage = len(vortlisto_roots & set(all_roots.keys()))
    print(f"\n6. Vortlisto coverage:")
    print(f"   Vortlisto roots in our vocabulary: {vortlisto_coverage}/{len(vortlisto_roots)} ({100*vortlisto_coverage/len(vortlisto_roots):.1f}%)")
    
    missing_from_vortlisto = vortlisto_roots - set(all_roots.keys())
    if missing_from_vortlisto:
        print(f"   Missing from our vocab: {sorted(list(missing_from_vortlisto))[:20]}...")
    
    # Tier assignment
    print("\n7. Assigning tiers...")
    tier_counts = defaultdict(int)
    for root in all_roots:
        if root in fundamento_roots:
            all_roots[root]['tier'] = 1  # Fundamento
            tier_counts[1] += 1
        elif root in vortlisto_roots:
            all_roots[root]['tier'] = 2  # Core vocabulary
            tier_counts[2] += 1
        else:
            all_roots[root]['tier'] = 3  # Extended (ReVo only)
            tier_counts[3] += 1
    
    print(f"   Tier 1 (Fundamento): {tier_counts[1]}")
    print(f"   Tier 2 (Core/Vortlisto): {tier_counts[2]}")
    print(f"   Tier 3 (Extended/ReVo): {tier_counts[3]}")
    
    # Save output
    output = {
        'metadata': {
            'total_roots': len(all_roots),
            'sources': {
                'revo': len(revo_roots),
                'fundamento': len(fundamento_roots),
                'vortlisto_reference': len(vortlisto_roots)
            },
            'tiers': dict(tier_counts),
            'description': 'Clean Esperanto roots for training'
        },
        'roots': all_roots
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n8. Saved to {output_path}")
    print("=" * 60)
    
    # Sample output
    print("\nSample roots:")
    for i, (root, info) in enumerate(sorted(all_roots.items())):
        if i >= 10:
            break
        print(f"   {root}: tier={info.get('tier')}, source={info.get('source')}")


if __name__ == '__main__':
    main()
