#!/usr/bin/env python3
"""
Compare roots from different sources and create hierarchical classification.

Sources:
1. Fundamento (Tier 1): 2,173 official roots from Universala Vortaro
2. ReVo (Tier 2): Extended vocabulary including technical terms
3. Corpus-validated (Tier 3): Successfully parsed from corpus
4. Proper names: Tagged by parser
5. Parse failures: Garbage/unknown words

Usage:
    python scripts/compare_root_sources.py
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_json_roots(path: Path) -> set:
    """Load roots from JSON file."""
    if not path.exists():
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return set(data.keys())


def filter_affixes(roots: set) -> set:
    """Remove affixes (prefixes/suffixes) from root set."""
    # Common Esperanto affixes
    affixes = {
        # Suffixes
        '-a', '-e', '-i', '-o', '-u',
        '-aĉ', '-ad', '-aĵ', '-an', '-ar',
        '-ebl', '-ec', '-ed', '-eg', '-ej', '-em', '-end', '-er', '-esk', '-estr', '-et',
        '-id', '-ig', '-iĝ', '-il', '-in', '-ind', '-ing', '-ism', '-ist',
        '-obl', '-on', '-op', '-oz',
        '-uj', '-ul', '-um',
        # Prefixes
        'bo-', 'dis-', 'ek-', 'eks-', 'fi-', 'ge-', 'mal-', 'mis-', 'pra-', 're-', 'vic-',
        # Correlatives
        'kio', 'tio', 'ĉio', 'io', 'nenio',
        'kiu', 'tiu', 'ĉiu', 'iu', 'neniu',
        'kia', 'tia', 'ĉia', 'ia', 'nenia',
        'kie', 'tie', 'ĉie', 'ie', 'nenie',
        'kiel', 'tiel', 'ĉiel', 'iel', 'neniel',
    }

    # Remove affixes and very short roots (< 2 chars)
    filtered = set()
    for root in roots:
        if root in affixes:
            continue
        if len(root) < 2:
            continue
        if root.startswith('-') or root.endswith('-'):
            continue
        filtered.add(root)

    return filtered


def main():
    parser = argparse.ArgumentParser(description='Compare root sources')
    parser.add_argument('--vocab-dir', type=Path,
                       default=Path('data/vocabularies'),
                       help='Vocabulary directory')

    args = parser.parse_args()

    # Load all root sources
    print("Loading root sources...")

    fundamento = load_json_roots(args.vocab_dir / 'fundamento_roots.json')
    print(f"  Fundamento: {len(fundamento):,} roots")

    revo_raw = load_json_roots(args.vocab_dir / 'revo_roots.json')
    revo = filter_affixes(revo_raw)
    print(f"  ReVo: {len(revo):,} roots (filtered from {len(revo_raw):,})")

    # Try cleaned corpus first, fall back to original
    corpus_clean_file = args.vocab_dir / 'corpus_validated_roots_clean.json'
    corpus_orig_file = args.vocab_dir / 'corpus_validated_roots.json'

    if corpus_clean_file.exists():
        corpus_validated = load_json_roots(corpus_clean_file)
        print(f"  Corpus-validated (clean): {len(corpus_validated):,} roots")
    else:
        corpus_validated = load_json_roots(corpus_orig_file)
        print(f"  Corpus-validated (original): {len(corpus_validated):,} roots")

    proper_names = load_json_roots(args.vocab_dir / 'proper_names.json')
    print(f"  Proper names: {len(proper_names):,} names")

    parse_failures = load_json_roots(args.vocab_dir / 'parse_failures.json')
    print(f"  Parse failures: {len(parse_failures):,} roots")

    # Analyze overlaps
    print("\n=== Overlap Analysis ===")

    # Fundamento coverage in corpus
    fundamento_in_corpus = fundamento & corpus_validated
    fundamento_coverage = 100 * len(fundamento_in_corpus) / len(fundamento) if fundamento else 0
    print(f"Fundamento roots found in corpus: {len(fundamento_in_corpus):,} / {len(fundamento):,} ({fundamento_coverage:.1f}%)")

    # ReVo coverage in corpus
    revo_in_corpus = revo & corpus_validated
    revo_coverage = 100 * len(revo_in_corpus) / len(revo) if revo else 0
    print(f"ReVo roots found in corpus: {len(revo_in_corpus):,} / {len(revo):,} ({revo_coverage:.1f}%)")

    # Corpus roots not in Fundamento or ReVo
    corpus_only = corpus_validated - fundamento - revo
    print(f"Corpus roots not in Fundamento or ReVo: {len(corpus_only):,}")

    # Show sample of corpus-only roots (likely loanwords/modern terms)
    print("\nSample corpus-only roots (first 30):")
    for root in sorted(corpus_only)[:30]:
        print(f"  {root}")

    # Create hierarchical classification
    print("\n=== Creating Hierarchical Classification ===")

    tier1 = fundamento  # Official Fundamento
    tier2 = revo - fundamento  # ReVo extended vocabulary
    tier3 = corpus_validated - fundamento - revo  # Corpus-validated only

    print(f"Tier 1 (Fundamento): {len(tier1):,} roots")
    print(f"Tier 2 (ReVo extended): {len(tier2):,} roots")
    print(f"Tier 3 (Corpus-only): {len(tier3):,} roots")
    print(f"Tier 4 (Proper names): {len(proper_names):,} names")
    print(f"Tier 5 (Parse failures): {len(parse_failures):,} entries")

    # Save hierarchical classification
    print("\nSaving hierarchical classification...")

    classification = {
        'tier1_fundamento': {
            'count': len(tier1),
            'description': 'Official Fundamento roots (Universala Vortaro)',
            'roots': sorted(tier1)
        },
        'tier2_revo': {
            'count': len(tier2),
            'description': 'ReVo extended vocabulary (technical terms, neologisms)',
            'roots': sorted(tier2)
        },
        'tier3_corpus': {
            'count': len(tier3),
            'description': 'Corpus-validated roots not in Fundamento or ReVo',
            'roots': sorted(tier3)
        },
        'tier4_proper_names': {
            'count': len(proper_names),
            'description': 'Proper names (places, people, organizations)',
            'roots': sorted(proper_names)
        },
        'tier5_parse_failures': {
            'count': len(parse_failures),
            'description': 'Parse failures (garbage, typos, foreign words)',
            'note': 'EXCLUDE from training'
        }
    }

    output_file = args.vocab_dir / 'root_classification.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classification, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_file}")

    # Create training-ready vocabulary (Tier 1 + Tier 2 + Tier 3)
    training_roots = tier1 | tier2 | tier3
    training_vocab = {
        'count': len(training_roots),
        'description': 'Combined Fundamento + ReVo + Corpus roots for training',
        'tiers': {
            'tier1_fundamento': len(tier1),
            'tier2_revo': len(tier2),
            'tier3_corpus': len(tier3)
        },
        'roots': {root: {'tier': 1 if root in tier1 else 2 if root in tier2 else 3}
                  for root in training_roots}
    }

    training_file = args.vocab_dir / 'training_vocabulary.json'
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_vocab, f, indent=2, ensure_ascii=False)

    print(f"Saved: {training_file}")

    print("\n=== Summary ===")
    print(f"Total training vocabulary: {len(training_roots):,} roots")
    print(f"  - Tier 1 (Fundamento): {len(tier1):,}")
    print(f"  - Tier 2 (ReVo): {len(tier2):,}")
    print(f"  - Tier 3 (Corpus): {len(tier3):,}")
    print(f"\nExcluded from training:")
    print(f"  - Proper names: {len(proper_names):,}")
    print(f"  - Parse failures: {len(parse_failures):,}")


if __name__ == '__main__':
    main()
