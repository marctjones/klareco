#!/usr/bin/env python3
"""
Extract Highest Quality Training Data from Corpus.

Strategy:
1. Use REAL SENTENCES from unified_corpus.jsonl (not isolated words)
2. Focus on SEMANTIC GAP (where deterministic is uncertain)
3. Include full sentence context for each word
4. Prioritize cases where context matters for classification

This generates the highest quality training data because:
- Real usage patterns (not synthetic)
- Full sentence context (not isolated words)
- Actual semantic gap cases (not what deterministic already knows)
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.semantic_enrichment.deterministic import DeterministicFeatureExtractor


def extract_word_from_ast(ast, target_index: int, current_index: int = 0):
    """
    Extract a specific word from AST by index.
    Returns (word_ast, new_index) or (None, new_index) if not found.
    """
    if not ast:
        return None, current_index

    if isinstance(ast, dict):
        if ast.get('tipo') == 'vorto':
            if current_index == target_index:
                return ast, current_index + 1
            return None, current_index + 1

        elif ast.get('tipo') == 'frazo':
            # Search in sentence components
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


def extract_training_examples_from_sentence(
    sentence_data: dict,
    classifier: DeterministicFeatureExtractor,
    confidence_threshold_low: float = 0.0,
    confidence_threshold_high: float = 0.7
) -> List[dict]:
    """
    Extract training examples from a sentence.

    Focus on words where deterministic is UNCERTAIN (confidence < 0.7).
    These are the semantic gap cases we need to learn.

    Returns list of training examples with full sentence context.
    """
    ast = sentence_data.get('ast')
    if not ast:
        return []

    text = sentence_data.get('text', '')
    source = sentence_data.get('source', {})

    num_words = count_words_in_ast(ast)
    examples = []

    for word_index in range(num_words):
        word_ast, _ = extract_word_from_ast(ast, word_index)

        if not word_ast:
            continue

        # Skip if not substantivo (entity classification is for nouns)
        if word_ast.get('vortspeco') != 'substantivo':
            continue

        # Get deterministic classification
        det_result = classifier.extract(word_ast)

        tier1 = det_result.get('tier1_category')
        tier2 = det_result.get('tier2_type')
        tier3 = det_result.get('tier3_type')
        confidence = det_result.get('confidence', 0.0)

        # Convert enum types to strings for JSON serialization
        if hasattr(tier1, 'value'):
            tier1 = tier1.value
        if hasattr(tier2, 'value'):
            tier2 = tier2.value
        if hasattr(tier3, 'value'):
            tier3 = tier3.value

        # Filter by confidence: Keep UNCERTAIN examples (semantic gap!)
        if confidence >= confidence_threshold_high:
            continue  # Skip high-confidence (deterministic already works)

        if tier3 is None and confidence < confidence_threshold_low:
            continue  # Skip completely unknown (no signal at all)

        # This is a semantic gap example!
        example = {
            'word_ast': word_ast,
            'sentence_ast': ast,  # ← FULL SENTENCE CONTEXT!
            'sentence_text': text,
            'word_index': word_index,
            'deterministic_priors': {
                'tier1_category': tier1,
                'tier2_type': tier2,
                'tier3_type': tier3,
                'confidence': confidence,
                'evidence': det_result.get('evidence', {})
            },
            'label': {
                'tier3_type': tier3,  # Use deterministic prediction as weak label
                'confidence': confidence,
                'source': 'corpus_sentence',
                'needs_manual_review': True  # Flag for manual annotation
            },
            'metadata': {
                'source_name': source.get('source_name'),
                'author': source.get('author'),
                'quality': source.get('quality'),
                'sentence_type': source.get('sentence_type')
            }
        }

        examples.append(example)

    return examples


def extract_highest_quality_data(
    corpus_path: Path,
    output_path: Path,
    max_sentences: int = 100000,
    target_examples: int = 10000,
    confidence_low: float = 0.0,
    confidence_high: float = 0.7
):
    """
    Extract highest quality training data from corpus.

    Strategy:
    1. Process sentences from unified_corpus.jsonl
    2. Extract words where deterministic is UNCERTAIN (confidence < 0.7)
    3. Include FULL sentence context for each word
    4. Prioritize high-quality sources (GOLD > authored > web)

    Args:
        corpus_path: Path to unified_corpus.jsonl
        output_path: Output path for training examples
        max_sentences: Max sentences to process (for speed)
        target_examples: Target number of examples to extract
        confidence_low: Minimum confidence to keep (skip completely unknown)
        confidence_high: Maximum confidence to keep (skip high-confidence)
    """
    print("="*70)
    print("EXTRACT HIGHEST QUALITY TRAINING DATA")
    print("="*70)
    print()
    print(f"Corpus: {corpus_path}")
    print(f"Target: {target_examples:,} examples")
    print(f"Confidence range: {confidence_low:.2f} - {confidence_high:.2f}")
    print(f"  (Focus on SEMANTIC GAP where deterministic is uncertain)")
    print()

    classifier = DeterministicFeatureExtractor()

    examples_by_quality = {
        'GOLD': [],
        'authored': [],
        'web': [],
        'unknown': []
    }

    total_sentences = 0
    total_words = 0
    total_extracted = 0

    print("Processing corpus...")
    print()

    with open(corpus_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            total_sentences += 1

            if total_sentences > max_sentences:
                print(f"Reached max sentences limit: {max_sentences:,}")
                break

            try:
                sentence_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            examples = extract_training_examples_from_sentence(
                sentence_data,
                classifier,
                confidence_low,
                confidence_high
            )

            total_extracted += len(examples)

            # Categorize by quality
            for example in examples:
                quality = example.get('metadata', {}).get('quality', 'unknown')
                if quality not in examples_by_quality:
                    quality = 'unknown'
                examples_by_quality[quality].append(example)

            if total_sentences % 1000 == 0:
                print(f"  Sentences: {total_sentences:,} | Extracted: {total_extracted:,}")

            # Stop if we have enough
            if total_extracted >= target_examples * 2:
                print(f"Extracted enough examples: {total_extracted:,}")
                break

    print()
    print(f"✓ Processed {total_sentences:,} sentences")
    print(f"✓ Extracted {total_extracted:,} semantic gap examples")
    print()

    # Show quality distribution
    print("Quality distribution:")
    for quality in ['GOLD', 'authored', 'web', 'unknown']:
        count = len(examples_by_quality[quality])
        pct = count / total_extracted * 100 if total_extracted > 0 else 0
        print(f"  {quality:10s}: {count:6,} ({pct:5.1f}%)")
    print()

    # Sample to target size, prioritizing quality
    print(f"Sampling to target size: {target_examples:,}")
    print("  Priority: GOLD > authored > web > unknown")
    print()

    final_examples = []

    # Take all GOLD examples first
    gold_count = min(len(examples_by_quality['GOLD']), target_examples)
    final_examples.extend(random.sample(examples_by_quality['GOLD'], gold_count))
    remaining = target_examples - len(final_examples)

    # Then authored
    if remaining > 0:
        authored_count = min(len(examples_by_quality['authored']), remaining)
        final_examples.extend(random.sample(examples_by_quality['authored'], authored_count))
        remaining = target_examples - len(final_examples)

    # Then web
    if remaining > 0:
        web_count = min(len(examples_by_quality['web']), remaining)
        final_examples.extend(random.sample(examples_by_quality['web'], web_count))
        remaining = target_examples - len(final_examples)

    # Finally unknown
    if remaining > 0:
        unknown_count = min(len(examples_by_quality['unknown']), remaining)
        final_examples.extend(random.sample(examples_by_quality['unknown'], unknown_count))

    # Shuffle
    random.shuffle(final_examples)

    print(f"✓ Selected {len(final_examples):,} highest quality examples")
    print()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for example in final_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved to: {output_path}")
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("These examples need MANUAL ANNOTATION because:")
    print("  - Deterministic labels are uncertain (confidence < 0.7)")
    print("  - These are the semantic gap cases model must learn")
    print()
    print("Options:")
    print("  1. Use deterministic labels as weak supervision (quick)")
    print("  2. Manually annotate subset for high-quality labels")
    print("  3. Use active learning: train model, find uncertain cases, annotate")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract highest quality training data')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Path to unified corpus with full sentences'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/entity_classifier_highest_quality/semantic_gap.jsonl'),
        help='Output path'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=100000,
        help='Max sentences to process'
    )
    parser.add_argument(
        '--target-examples',
        type=int,
        default=10000,
        help='Target number of examples'
    )
    parser.add_argument(
        '--conf-low',
        type=float,
        default=0.0,
        help='Minimum confidence threshold'
    )
    parser.add_argument(
        '--conf-high',
        type=float,
        default=0.7,
        help='Maximum confidence threshold (focus on uncertain)'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: Corpus not found: {args.corpus}")
        sys.exit(1)

    extract_highest_quality_data(
        args.corpus,
        args.output,
        args.max_sentences,
        args.target_examples,
        args.conf_low,
        args.conf_high
    )
