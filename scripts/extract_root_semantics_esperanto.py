#!/usr/bin/env python3
"""
Extract ROOT semantic training data with Esperanto category labels.

Focus: Label ROOTS with semantic categories that grammar doesn't capture.
Labels: All in Esperanto (Pure Esperanto philosophy)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.semantic_enrichment.radiko_semantiko import (
    ĈIUJ_KATEGORIOJ,
    kategorio_al_teksto
)


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
        'la',
        'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si',
        'kiu', 'tiu', 'iu', 'ĉiu', 'neniu',
        'kio', 'tio', 'io', 'ĉio', 'nenio',
        'kia', 'tia', 'ia', 'ĉia', 'nenia',
        'kie', 'tie', 'ie', 'ĉie', 'nenie',
        'kiam', 'tiam', 'iam', 'ĉiam', 'neniam',
        'kiel', 'tiel', 'iel', 'ĉiel', 'neniel',
        'kial', 'tial', 'ial', 'ĉial', 'nenial',
        'kiom', 'tiom', 'iom', 'ĉiom', 'neniom',
        'kies', 'ties', 'ies', 'ĉies', 'nenies',
        'al', 'antaŭ', 'anstataŭ', 'apud', 'ĉe', 'ĉirkaŭ', 'da', 'de', 'dum',
        'ekster', 'el', 'en', 'ĝis', 'inter', 'je', 'kontraŭ', 'krom', 'kun',
        'laŭ', 'malgraŭ', 'per', 'po', 'por', 'post', 'preter', 'pri', 'pro',
        'sen', 'sub', 'super', 'sur', 'tra', 'trans',
        'kaj', 'aŭ', 'nek', 'sed', 'se', 'ĉar', 'ke', 'kvankam', 'ol',
        'ne', 'jes', 'ja'
    }
    return radiko in function_words


def should_label_root(word_ast: dict) -> bool:
    """Determine if this root needs semantic labeling."""
    radiko = word_ast.get('radiko', '')

    # Skip function words
    if is_function_word(radiko):
        return False

    # Skip proper names
    if word_ast.get('parse_status') == 'proper_name':
        return False

    # Skip empty roots
    if not radiko or radiko == '':
        return False

    return True


def extract_root_contexts(
    corpus_path: Path,
    output_path: Path,
    max_sentences: int = 100000,
    contexts_per_root: int = 20
):
    """Extract training data: roots with contexts for annotation."""
    print("="*70)
    print("EKSTRAKTI RADIKAN SEMANTIKAN TREJNAN DATUMARON")
    print("(Extract Root Semantic Training Data)")
    print("="*70)
    print()
    print(f"Korpuso: {corpus_path}")
    print(f"Celo: {contexts_per_root} kuntekstoj por radiko")
    print(f"Disponebla kategorioj: {len(ĈIUJ_KATEGORIOJ)}")
    print()

    # Group contexts by root
    root_contexts = defaultdict(list)
    root_forms = defaultdict(Counter)

    sentences_processed = 0

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

            # Filter out grammar-focused sources
            source_info = sentence_data.get('source', {})
            source_type = source_info.get('source_type', '')
            if source_type in ['grammar_reference', 'grammar_qa', 'grammar_manual']:
                continue  # Skip meta-linguistic texts

            ast = sentence_data.get('ast')
            if not ast:
                continue

            text = sentence_data.get('text', '')
            num_words = count_words_in_ast(ast)

            for word_index in range(num_words):
                word_ast, _ = extract_word_from_ast(ast, word_index)

                if not word_ast or not should_label_root(word_ast):
                    continue

                radiko = word_ast.get('radiko', '')

                if len(root_contexts[radiko]) >= contexts_per_root:
                    continue

                context = {
                    'frazo': text,
                    'frazo_ast': ast,
                    'vorto_indekso': word_index,
                    'vorto_ast': word_ast,
                    'derivita_formo': word_ast.get('plena_vorto', ''),
                    'afiksoj': {
                        'prefiksoj': word_ast.get('prefiksoj', []),
                        'sufiksoj': word_ast.get('sufiksoj', [])
                    },
                    'vortspeco': word_ast.get('vortspeco', ''),
                    'fonto': sentence_data.get('source', {})
                }

                root_contexts[radiko].append(context)
                root_forms[radiko][word_ast.get('plena_vorto', '')] += 1

            if sentences_processed % 5000 == 0:
                print(f"  Traktita: {sentences_processed:,} frazoj")
                print(f"  Trovita unikaj radikoj: {len(root_contexts):,}")

    print()
    print(f"✓ Traktita {sentences_processed:,} frazoj")
    print(f"✓ Trovita {len(root_contexts):,} unikaj enhavaj radikoj")
    print()

    # Statistics
    contexts_counts = [len(contexts) for contexts in root_contexts.values()]
    print("Radika diverseco:")
    print(f"  Radikoj kun {contexts_per_root}+ kuntekstoj: {sum(1 for c in contexts_counts if c >= contexts_per_root):,}")
    print(f"  Radikoj kun 5-{contexts_per_root-1} kuntekstoj: {sum(1 for c in contexts_counts if 5 <= c < contexts_per_root):,}")
    print(f"  Radikoj kun 1-4 kuntekstoj: {sum(1 for c in contexts_counts if c < 5):,}")
    print()

    # Prepare training data
    training_data = []
    for radiko, contexts in root_contexts.items():
        training_data.append({
            'radiko': radiko,
            'nombro_kuntekstoj': len(contexts),
            'derivitaj_formoj': dict(root_forms[radiko].most_common(10)),
            'kuntekstoj': contexts[:contexts_per_root],
            'etikedo': {
                'kategorio': None,           # To be annotated (e.g., "besto:mamulo")
                'rilataj_radikoj': [],       # To be annotated (e.g., ["kat", "best"])
                'bezonas_anoton': True
            }
        })

    # Sort by frequency
    training_data.sort(key=lambda x: x['nombro_kuntekstoj'], reverse=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for root_data in training_data:
            f.write(json.dumps(root_data, ensure_ascii=False) + '\n')

    print(f"✓ Konservita {len(training_data):,} radikoj al: {output_path}")
    print()

    # Show examples
    print("Plej oftaj radikoj (por ekzemplo):")
    for i, root_data in enumerate(training_data[:15], 1):
        root = root_data['radiko']
        count = root_data['nombro_kuntekstoj']
        forms = ', '.join(list(root_data['derivitaj_formoj'].keys())[:3])
        print(f"  {i:2}. {root:15s} ({count:2} kuntekstoj) - formoj: {forms}")
    print()

    # Show available categories
    print("="*70)
    print("DISPONEBLAJ KATEGORIOJ (286 total)")
    print("="*70)
    print()

    category_examples = {
        "Bestoj": ["besto:mamulo", "besto:birdo", "besto:fiŝo", "besto:insekto"],
        "Plantoj": ["planto:arbo", "planto:floro", "planto:herbo", "planto:legomo"],
        "Agoj": ["ago:vidi", "ago:aŭdi", "ago:pensi", "ago:instrui", "ago:ami"],
        "Sentoj": ["sento:ĝojo", "sento:amo", "sento:timo", "sento:kolero"],
        "Koloroj": ["eco:ruĝa", "eco:blua", "eco:verda", "eco:flava"],
        "Konstruaĵoj": ["konstruaĵo:loĝejo", "konstruaĵo:religia", "konstruaĵo:komerca"],
    }

    for group, examples in category_examples.items():
        print(f"{group}:")
        for ex in examples:
            print(f"  - {ex}")
    print()
    print("(Vidu radiko_semantiko.py por kompleta listo de 286 kategorioj)")
    print()

    print("="*70)
    print("SEKVAJ PAŜOJ")
    print("="*70)
    print()
    print("Tiuj radikoj bezonas SEMANTIKAN KATEGORION:")
    print("  - kategorio: ekz., 'besto:mamulo', 'ago:instrui', 'sento:amo'")
    print("  - rilataj_radikoj: ekz., ['kat', 'ĉeval', 'best']")
    print()
    print("Elektoj:")
    print("  1. Mana anotado (plej alta kvalito)")
    print("  2. Duon-aŭtomata kun Claude (pli rapida)")
    print("  3. Grupigo + mana rafinado (skalebla)")
    print()

    return len(training_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ekstrakti radikan semantikan trejnan datumaron')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Vojo al unuigita korpuso'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/root_semantics/radikoj_por_anoti.jsonl'),
        help='Elira vojo por radika trejndata'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=100000,
        help='Maksimuma nombro de frazoj por trakti'
    )
    parser.add_argument(
        '--contexts-per-root',
        type=int,
        default=20,
        help='Nombro de ekzemplaj kuntekstoj por kolekti por radiko'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERARO: Korpuso ne trovita: {args.corpus}")
        sys.exit(1)

    num_roots = extract_root_contexts(
        args.corpus,
        args.output,
        args.max_sentences,
        args.contexts_per_root
    )

    print(f"✓ Sukcese ekstraktita {num_roots:,} radikoj por semantika anotado")
