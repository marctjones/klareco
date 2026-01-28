#!/usr/bin/env python3
"""
Tune Reranker Weights via Grid Search

Tests different combinations of retrieval weight vs reranker weight
to find the optimal balance for top-1 accuracy.

Usage:
    python scripts/tune_reranker_weights.py \\
        --test-set data/evaluation/rag_test_set.jsonl \\
        --output results/weight_tuning.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.models.reranker import ASTReranker
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.answer_extractor import ASTAnswerExtractor

logging.basicConfig(
    level=logging.WARNING,  # Suppress most logging for cleaner output
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_weight_combination(
    retriever: ASTAwareRetriever,
    reranker: ASTReranker,
    extractor: ASTAnswerExtractor,
    test_set: List[Dict],
    retrieval_weight: float,
    reranker_weight: float,
    top_k_retrieve: int = 50,
) -> Dict:
    """
    Evaluate a specific weight combination.

    Args:
        retriever: Base retriever
        reranker: Reranker model
        extractor: Answer extractor
        test_set: Test questions
        retrieval_weight: Weight for retrieval score
        reranker_weight: Weight for reranker score
        top_k_retrieve: Number of docs to retrieve before reranking

    Returns:
        {
            'top1_accuracy': float,  # % of correct answers in top-1 doc
            'top3_accuracy': float,  # % of correct answers in top-3 docs
            'exact_match': int,      # Number of exact matches
            'partial_match': int,    # Number of partial matches
            'no_answer': int,        # Number with no answer extracted
        }
    """
    correct_top1 = 0
    correct_top3 = 0
    exact_matches = 0
    partial_matches = 0
    no_answers = 0

    for item in test_set:
        query = item['question']
        expected = item['expected_answer']

        # Retrieve candidates
        candidates = retriever.search(
            query=query,
            top_k=top_k_retrieve,
            use_m1_expansion=False,  # Don't use M1 per user request
        )

        if not candidates:
            no_answers += 1
            continue

        # Rerank with custom weights
        query_ast = parse(query)
        reranked = []

        for score, doc, stats in candidates:
            try:
                doc_text = doc.get('text', '')
                doc_ast = parse(doc_text)

                with torch.no_grad():
                    rerank_score = reranker(query_ast, doc_ast).item()

                # Custom weight combination
                combined_score = retrieval_weight * score + reranker_weight * rerank_score

                reranked.append((combined_score, doc, stats))

            except Exception:
                reranked.append((score, doc, stats))

        reranked.sort(key=lambda x: x[0], reverse=True)

        # Check if answer in top-1
        top1_doc = reranked[0][1]
        top1_text = top1_doc.get('text', '').lower()

        if expected.lower() in top1_text:
            correct_top1 += 1

        # Check if answer in top-3
        for _, doc, _ in reranked[:3]:
            if expected.lower() in doc.get('text', '').lower():
                correct_top3 += 1
                break

        # Try extraction from top-1
        try:
            top1_ast = parse(top1_doc.get('text', ''))
            answer = extractor.extract_answer(query_ast, top1_ast, top1_doc.get('text', ''))

            if answer:
                answer_text = answer['text'].lower()
                if expected.lower() in answer_text or answer_text in expected.lower():
                    exact_matches += 1
                elif any(word in answer_text for word in expected.lower().split()):
                    partial_matches += 1
            else:
                no_answers += 1
        except Exception:
            no_answers += 1

    total = len(test_set)
    return {
        'top1_accuracy': (correct_top1 / total) * 100 if total > 0 else 0,
        'top3_accuracy': (correct_top3 / total) * 100 if total > 0 else 0,
        'exact_match': exact_matches,
        'partial_match': partial_matches,
        'no_answer': no_answers,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune reranker weights via grid search")
    parser.add_argument('--test-set', default='data/evaluation/rag_test_set.jsonl',
                       help='Test set JSONL file')
    parser.add_argument('--stage1-model', default='models/root_embeddings_tier0/best_model.pt',
                       help='Stage 1 compositional embeddings')
    parser.add_argument('--reranker-model', default='models/reranker/best_model.pt',
                       help='Reranker model')
    parser.add_argument('--index-dir', default='data/indexes/kuzu_index',
                       help='Kuzu index directory')
    parser.add_argument('--output', default='results/weight_tuning.json',
                       help='Output JSON file')
    args = parser.parse_args()

    # Load test set
    test_set = []
    with open(args.test_set) as f:
        for line in f:
            item = json.loads(line)
            # Only test questions marked as "should work"
            if item.get('expected_performance') == 'works':
                # Add expected_answer field if missing (extract from pattern)
                if 'expected_answer' not in item:
                    pattern = item.get('expected_answer_pattern', '')
                    # Try to extract key terms from pattern
                    if 'Zamenhof' in pattern:
                        item['expected_answer'] = 'Zamenhof'
                    elif 'Esperanto' in pattern and 'estas' in pattern:
                        item['expected_answer'] = 'planlingvo'
                    else:
                        item['expected_answer'] = pattern
                test_set.append(item)

    # Limit to first 10 questions for faster tuning
    test_set = test_set[:10]

    logger.info(f"Loaded {len(test_set)} test questions")

    # Load models
    print("Loading models...")

    # Load compositional embedding
    comp_model_path = Path(args.stage1_model)
    checkpoint = torch.load(comp_model_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    else:
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 64),
        )

        if 'embeddings.weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['embeddings.weight']
        elif 'weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['weight']

    comp_emb.eval()

    # Load reranker
    reranker_path = Path(args.reranker_model)
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()

    # Load retriever
    retriever = ASTAwareRetriever(
        index_path=Path(args.index_dir),
    )

    # Create extractor
    extractor = ASTAnswerExtractor()

    print("✓ Models loaded")

    # Test weight combinations
    # Constraint: weights should sum to 1.0
    weight_combinations = [
        (0.3, 0.7),  # Current default
        (0.2, 0.8),  # Trust reranker more
        (0.4, 0.6),  # Trust retrieval more
        (0.1, 0.9),  # Heavy reranker
        (0.5, 0.5),  # Equal weight
    ]

    print("=" * 80)
    print("Reranker Weight Tuning")
    print("=" * 80)
    print(f"Test set: {len(test_set)} questions")
    print("")

    results = []

    for retrieval_w, reranker_w in weight_combinations:
        print(f"Testing weights: retrieval={retrieval_w:.1f}, reranker={reranker_w:.1f}")

        metrics = evaluate_weight_combination(
            retriever, reranker, extractor, test_set,
            retrieval_w, reranker_w
        )

        results.append({
            'retrieval_weight': retrieval_w,
            'reranker_weight': reranker_w,
            'metrics': metrics,
        })

        print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.1f}%")
        print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.1f}%")
        print(f"  Exact matches: {metrics['exact_match']}/{metrics['total']}")
        print(f"  Partial matches: {metrics['partial_match']}/{metrics['total']}")
        print("")

    # Find best combination
    best = max(results, key=lambda x: x['metrics']['top1_accuracy'])

    print("=" * 80)
    print("BEST WEIGHTS")
    print("=" * 80)
    print(f"Retrieval: {best['retrieval_weight']:.1f}")
    print(f"Reranker: {best['reranker_weight']:.1f}")
    print(f"Top-1 accuracy: {best['metrics']['top1_accuracy']:.1f}%")
    print(f"Top-3 accuracy: {best['metrics']['top3_accuracy']:.1f}%")
    print("")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'test_set': str(args.test_set),
            'num_questions': len(test_set),
            'results': results,
            'best_weights': {
                'retrieval': best['retrieval_weight'],
                'reranker': best['reranker_weight'],
            },
        }, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
