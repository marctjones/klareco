#!/usr/bin/env python3
"""
Extract ROOT semantic training data from corpus.

Key insight: Label ROOTS (radiko), not full words.
Deterministic composition handles affixes.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_word_from_ast(ast, target_index: int, current_index: int = 0):
    """Extract a specific word from AST by index."""
    if not ast:
        return None, current_index

    if isinstance(ast, dict):
        if ast.get('tipo') == 'vorto':
            if current_index == target_index:
                return ast, current_index + 1
            return None, current_index + 1

        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key):
                    result, current_index = extract_word_from_ast(ast[key], target_index, current_index)
                    if result:
                        return result, current_index

            for alia in ast.get('aliaj', []):
                result, current_index = extract_word_from_ast(alia, target_index, current_index)
                if result:
                    return result, current_index

        elif ast.get('tipo') == 'vortgrupo':
            if ast.get('kerno'):
                result, current_index = extract_word_from_ast(ast['kerno'], target_index, current_index)
                if result:
                    return result, current_index

            for priskribo in ast.get('priskriboj', []):
                result, current_index = extract_word_from_ast(priskribo, target_index, current_index)
                if result:
                    return result, current_index

    return None, current_index


def count_words_in_ast(ast) -> int:
    """Count total words in AST."""
    if not ast:
        return 0

    if isinstance(ast, dict):
        if ast.get('tipo') == 'vorto':
            return 1

        elif ast.get('tipo') == 'frazo':
            count = 0
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key):
                    count += count_words_in_ast(ast[key])
            for alia in ast.get('aliaj', []):
                count += count_words_in_ast(alia)
            return count

        elif ast.get('tipo') == 'vortgrupo':
            count = 0
            if ast.get('kerno'):
                count += count_words_in_ast(ast['kerno'])
            for priskribo in ast.get('priskriboj', []):
                count += count_words_in_ast(priskribo)
            return count

    return 0


def is_function_word(radiko: str) -> bool:
    """
    Check if root is a function word (grammatical, not semantic).

    Function words are handled deterministically, not learned.
    """
    function_words = {
        # Articles
        'la',
        # Pronouns
        'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si',
        # Correlatives (40 total)
        'kiu', 'tiu', 'iu', 'ĉiu', 'neniu',
        'kio', 'tio', 'io', 'ĉio', 'nenio',
        'kia', 'tia', 'ia', 'ĉia', 'nenia',
        'kie', 'tie', 'ie', 'ĉie', 'nenie',
        'kiam', 'tiam', 'iam', 'ĉiam', 'neniam',
        'kiel', 'tiel', 'iel', 'ĉiel', 'neniel',
        'kial', 'tial', 'ial', 'ĉial', 'nenial',
        'kiom', 'tiom', 'iom', 'ĉiom', 'neniom',
        'kies', 'ties', 'ies', 'ĉies', 'nenies',
        # Prepositions
        'al', 'antaŭ', 'anstataŭ', 'apud', 'ĉe', 'ĉirkaŭ', 'da', 'de', 'dum',
        'ekster', 'el', 'en', 'ĝis', 'inter', 'je', 'kontraŭ', 'krom', 'kun',
        'laŭ', 'malgraŭ', 'per', 'po', 'por', 'post', 'preter', 'pri', 'pro',
        'sen', 'sub', 'super', 'sur', 'tra', 'trans',
        # Conjunctions
        'kaj', 'aŭ', 'nek', 'sed', 'se', 'ĉar', 'ke', 'kvankam', 'ol',
        # Other grammatical
        'ne', 'jes', 'ja'
    }
    return radiko in function_words


def should_label_root(word_ast: dict) -> bool:
    """
    Determine if this root needs semantic labeling.

    Skip:
    - Function words (grammatical)
    - Proper names (already specific)
    - Affixes that fully determine meaning
    """
    radiko = word_ast.get('radiko', '')
    vortspeco = word_ast.get('vortspeco', '')

    # Skip function words
    if is_function_word(radiko):
        return False

    # Skip proper names (already maximally specific)
    if word_ast.get('parse_status') == 'proper_name':
        return False

    # Skip if it's just an affix with no root
    if not radiko or radiko == '':
        return False

    return True


def extract_root_contexts_from_corpus(
    corpus_path: Path,
    output_path: Path,
    max_sentences: int = 100000,
    contexts_per_root: int = 50
):
    """
    Extract training data: roots with their contexts.

    For each unique root:
    - Collect up to N sentences where it appears
    - Include full sentence AST for context
    - Include derived forms (with affixes)
    - Group for efficient annotation
    """
    print("="*70)
    print("EXTRACT ROOT SEMANTIC TRAINING DATA")
    print("="*70)
    print()

    # Group contexts by root
    root_contexts = defaultdict(list)
    root_forms = defaultdict(Counter)  # Track different derived forms

    sentences_processed = 0

    print(f"Processing corpus: {corpus_path}")
    print(f"Target: {contexts_per_root} contexts per root")
    print()

    with open(corpus_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            sentences_processed += 1
            if sentences_processed > max_sentences:
                break

            try:
                sentence_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            ast = sentence_data.get('ast')
            if not ast:
                continue

            text = sentence_data.get('text', '')
            num_words = count_words_in_ast(ast)

            # Extract all roots from this sentence
            for word_index in range(num_words):
                word_ast, _ = extract_word_from_ast(ast, word_index)

                if not word_ast:
                    continue

                # Only label content roots (not function words)
                if not should_label_root(word_ast):
                    continue

                radiko = word_ast.get('radiko', '')

                # Skip if we already have enough contexts for this root
                if len(root_contexts[radiko]) >= contexts_per_root:
                    continue

                # Extract context
                context = {
                    'sentence': text,
                    'sentence_ast': ast,
                    'word_index': word_index,
                    'word_ast': word_ast,
                    'derived_form': word_ast.get('plena_vorto', ''),
                    'affixes': {
                        'prefixes': word_ast.get('prefiksoj', []),
                        'suffixes': word_ast.get('sufiksoj', [])
                    },
                    'vortspeco': word_ast.get('vortspeco', ''),
                    'sentence_role': None,  # TODO: Extract from position in AST
                    'source': sentence_data.get('source', {})
                }

                root_contexts[radiko].append(context)
                root_forms[radiko][word_ast.get('plena_vorto', '')] += 1

            if sentences_processed % 1000 == 0:
                print(f"  Processed: {sentences_processed:,} sentences")
                print(f"  Unique roots found: {len(root_contexts):,}")

    print()
    print(f"✓ Processed {sentences_processed:,} sentences")
    print(f"✓ Found {len(root_contexts):,} unique content roots")
    print()

    # Show root diversity
    print("Root diversity statistics:")
    contexts_counts = [len(contexts) for contexts in root_contexts.values()]
    print(f"  Roots with {contexts_per_root}+ contexts: {sum(1 for c in contexts_counts if c >= contexts_per_root):,}")
    print(f"  Roots with 10-{contexts_per_root-1} contexts: {sum(1 for c in contexts_counts if 10 <= c < contexts_per_root):,}")
    print(f"  Roots with 1-9 contexts: {sum(1 for c in contexts_counts if c < 10):,}")
    print()

    # Save training data grouped by root
    training_data = []
    for radiko, contexts in root_contexts.items():
        training_data.append({
            'root': radiko,
            'num_contexts': len(contexts),
            'derived_forms': dict(root_forms[radiko].most_common(10)),
            'contexts': contexts[:contexts_per_root],  # Limit to N contexts
            'label': {
                'domain': None,  # To be annotated
                'subdomain': None,
                'specific': None,
                'needs_annotation': True
            }
        })

    # Sort by number of contexts (most common first - easier to annotate)
    training_data.sort(key=lambda x: x['num_contexts'], reverse=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for root_data in training_data:
            f.write(json.dumps(root_data, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(training_data):,} roots to: {output_path}")
    print()

    # Show examples
    print("Top 10 most frequent roots (by context count):")
    for i, root_data in enumerate(training_data[:10], 1):
        root = root_data['root']
        count = root_data['num_contexts']
        forms = ', '.join(list(root_data['derived_forms'].keys())[:3])
        print(f"  {i:2}. {root:15s} ({count:3} contexts) - forms: {forms}")
    print()

    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("These roots need SEMANTIC DOMAIN ANNOTATION:")
    print("  - domain: e.g., 'action', 'entity', 'property', 'relation'")
    print("  - subdomain: e.g., 'cognitive', 'physical', 'emotion'")
    print("  - specific: e.g., 'teaching', 'learning', 'understanding'")
    print()
    print("Options:")
    print("  1. Manual annotation (highest quality)")
    print("  2. Semi-automatic with Claude (faster)")
    print("  3. Clustering + manual refinement (scalable)")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract root semantic training data')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Path to unified corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/root_semantics/roots_to_annotate.jsonl'),
        help='Output path for root training data'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=100000,
        help='Max sentences to process'
    )
    parser.add_argument(
        '--contexts-per-root',
        type=int,
        default=50,
        help='Number of example contexts to collect per root'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: Corpus not found: {args.corpus}")
        sys.exit(1)

    extract_root_contexts_from_corpus(
        args.corpus,
        args.output,
        args.max_sentences,
        args.contexts_per_root
    )
