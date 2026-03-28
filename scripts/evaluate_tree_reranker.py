#!/usr/bin/env python3
"""
Evaluate TreeMatchReranker on Test Set

VERSION: v2.1
COMPATIBLE WITH: v2.1 database, TreeMatchReranker
DEPENDENCIES: TreeMatchReranker, Whoosh retriever
STAGE: Evaluation

Description:
    Evaluates TreeMatchReranker on 30-question test set.
    Compares with baselines: no reranker (73.3%), old MLP reranker (56.7%).
    Target: 75%+ accuracy.

Usage:
    python scripts/evaluate_tree_reranker.py \\
        --test-set data/test_sets/qa_test_diverse_30.jsonl \\
        --model models/tree_reranker/best_model.pt \\
        --output results/tree_reranker_eval.json

Inputs:
    - Test set: 30-question JSONL
    - Trained model: TreeMatchReranker checkpoint

Outputs:
    - Evaluation results: JSON with accuracy, breakdowns
    - Score breakdowns: Interpretable score components

Metrics:
    - Overall accuracy
    - Per-question-type accuracy (WHO, WHAT, WHERE, WHEN)
    - Score component analysis (syntax vs comp vs semantic)

Last Updated: 2026-03-26
Author: Claude + Marc
Related Issues: #704
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.tree_match_reranker import TreeMatchReranker
from klareco.embeddings import CompositionalEmbedding
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.importance_scorer import classify_question_type
from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TreeRerankerEvaluator:
    """Evaluator for TreeMatchReranker."""

    def __init__(
        self,
        model: TreeMatchReranker,
        retriever: WhooshRetriever
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained TreeMatchReranker
            retriever: Whoosh retriever for candidate retrieval
        """
        self.model = model
        self.retriever = retriever
        self.model.eval()

        logger.info("TreeRerankerEvaluator initialized")

    def evaluate_question(
        self,
        question: str,
        expected_keywords: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate single question.

        Args:
            question: Question text
            expected_keywords: Expected keywords in answer
            top_k: Number of results to rerank

        Returns:
            Result dict with correct, scores, breakdowns
        """
        # Parse question
        try:
            query_ast = parse(question)
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return {'correct': False, 'error': 'parse_failed'}

        # Extract roots
        roots = self._extract_roots(query_ast)
        if not roots:
            return {'correct': False, 'error': 'no_roots'}

        # Classify question type
        question_type = classify_question_type(question)

        # Retrieve candidates
        try:
            candidates = self.retriever.retrieve(
                query_roots=roots,
                question_type=question_type.value,
                query_entity=None,
                top_k=50  # Get top 50 for reranking
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {'correct': False, 'error': 'retrieval_failed'}

        if not candidates:
            return {'correct': False, 'error': 'no_results'}

        # Rerank candidates
        reranked = self._rerank_candidates(query_ast, candidates, top_k)

        # Check if answer is correct
        correct = self._check_answer(reranked, expected_keywords)

        return {
            'correct': correct,
            'num_candidates': len(candidates),
            'top_scores': [r['rerank_score'] for r in reranked],
            'top_breakdowns': [r['breakdown'] for r in reranked],
            'top_texts': [r['text'][:100] for r in reranked]
        }

    def _extract_roots(self, ast: Dict) -> List[str]:
        """Extract roots from AST for retrieval."""
        roots = []

        def traverse(node):
            if node is None:
                return

            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    root = node.get('radiko', '').lower()
                    if root and root not in ['ki', 'kiu', 'kio', 'kie', 'kiam']:
                        roots.append(root)

                for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                    traverse(node.get(key))
                for item in node.get('priskriboj', []) + node.get('aliaj', []):
                    traverse(item)

        traverse(ast)
        return list(set(roots))

    def _rerank_candidates(
        self,
        query_ast: Dict,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank candidates with TreeMatchReranker."""
        scored_candidates = []

        for doc in candidates:
            # Parse doc if not already parsed
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc.get('text', ''))
                except:
                    continue

            # Score with TreeMatchReranker
            with torch.no_grad():
                score, breakdown = self.model(query_ast, doc_ast)

            scored_candidates.append({
                'text': doc.get('text', ''),
                'bm25_score': doc.get('score', 0.0),
                'rerank_score': score.item(),
                'breakdown': breakdown,
                'id': doc.get('id', ''),
                'ast': doc_ast
            })

        # Sort by rerank score
        scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        return scored_candidates[:top_k]

    def _check_answer(
        self,
        reranked: List[Dict],
        expected_keywords: List[str]
    ) -> bool:
        """Check if any top result contains expected keywords."""
        if not reranked or not expected_keywords:
            return False

        # Check top 3 results
        for doc in reranked[:3]:
            text_lower = doc['text'].lower()

            # Must match ALL expected keywords
            if all(kw.lower() in text_lower for kw in expected_keywords):
                return True

        return False

    def evaluate_test_set(self, test_questions: List[Dict]) -> Dict:
        """
        Evaluate full test set.

        Args:
            test_questions: List of test question dicts

        Returns:
            Evaluation results with accuracy, breakdowns
        """
        results = []
        correct_count = 0
        by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        logger.info(f"\nEvaluating {len(test_questions)} questions...")

        for i, q in enumerate(test_questions):
            question = q.get('question', '')
            expected = q.get('expected_keywords', [])

            logger.info(f"\n[{i+1}/{len(test_questions)}] {question}")

            # Evaluate
            result = self.evaluate_question(question, expected)
            result['question'] = question
            result['expected_keywords'] = expected

            # Update counts
            question_type = classify_question_type(question)
            type_name = question_type.value

            if result.get('correct'):
                correct_count += 1
                by_type[type_name]['correct'] += 1
                logger.info("✓ CORRECT")
            else:
                error = result.get('error', 'wrong_answer')
                logger.info(f"✗ INCORRECT ({error})")

            by_type[type_name]['total'] += 1

            # Log score breakdown if available
            if 'top_breakdowns' in result and result['top_breakdowns']:
                breakdown = result['top_breakdowns'][0]
                logger.info(f"  Syntax: {breakdown.get('syntax_score', 0):.3f}, "
                           f"Comp: {breakdown.get('compositional_score', 0):.3f}, "
                           f"Semantic: {breakdown.get('semantic_score', 0):.3f}, "
                           f"Final: {breakdown.get('final_score', 0):.3f}")

            results.append(result)

        # Compute statistics
        overall_acc = correct_count / len(test_questions) if test_questions else 0.0

        type_accs = {}
        for type_name, stats in by_type.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            type_accs[type_name] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }

        # Aggregate score breakdowns
        all_breakdowns = [r.get('top_breakdowns', [None])[0] for r in results if r.get('top_breakdowns')]
        all_breakdowns = [b for b in all_breakdowns if b is not None]

        avg_breakdown = {}
        if all_breakdowns:
            avg_breakdown = {
                'avg_syntax_score': sum(b.get('syntax_score', 0) for b in all_breakdowns) / len(all_breakdowns),
                'avg_comp_score': sum(b.get('compositional_score', 0) for b in all_breakdowns) / len(all_breakdowns),
                'avg_semantic_score': sum(b.get('semantic_score', 0) for b in all_breakdowns) / len(all_breakdowns),
                'avg_syntax_weight': sum(b.get('syntax_weight', 0) for b in all_breakdowns) / len(all_breakdowns),
                'avg_comp_weight': sum(b.get('compositional_weight', 0) for b in all_breakdowns) / len(all_breakdowns),
                'avg_semantic_weight': sum(b.get('semantic_weight', 0) for b in all_breakdowns) / len(all_breakdowns)
            }

        return {
            'overall_accuracy': overall_acc,
            'correct': correct_count,
            'total': len(test_questions),
            'by_type': type_accs,
            'avg_breakdown': avg_breakdown,
            'results': results
        }


def load_compositional_embeddings(checkpoint_path):
    """Load compositional embeddings from checkpoint."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        # Full CompositionalEmbedding checkpoint
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Simple root embeddings with model_state_dict
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 128)
        )
        # Load state dict (contains root_embeddings.weight)
        comp_emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")

    return comp_emb


def main():
    parser = argparse.ArgumentParser(description="Evaluate TreeMatchReranker")
    parser.add_argument('--test-set', type=Path, required=True,
                       help='Test set JSONL file')
    parser.add_argument('--model', type=Path, required=True,
                       help='Trained TreeMatchReranker model')
    parser.add_argument('--comp-emb', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Compositional embedding model')
    parser.add_argument('--whoosh-index', type=Path,
                       default=Path('data/indexes/whoosh_fts'),
                       help='Whoosh index directory')
    parser.add_argument('--kuzu-db', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Kuzu database path')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file for results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top results to consider (default: 5)')
    args = parser.parse_args()

    # Load compositional embeddings
    logger.info("Loading compositional embeddings...")
    comp_emb = load_compositional_embeddings(args.comp_emb)

    # Load TreeMatchReranker
    logger.info(f"Loading TreeMatchReranker from {args.model}...")
    model = TreeMatchReranker.load(args.model, comp_emb)

    # Load retriever
    logger.info("Loading Whoosh retriever...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.kuzu_db
    )

    # Load test set
    logger.info(f"Loading test set from {args.test_set}...")
    test_questions = []
    with open(args.test_set, 'r', encoding='utf-8') as f:
        for line in f:
            test_questions.append(json.loads(line))

    logger.info(f"Loaded {len(test_questions)} test questions")

    # Create evaluator
    evaluator = TreeRerankerEvaluator(model, retriever)

    # Evaluate
    results = evaluator.evaluate_test_set(test_questions)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.1%} ({results['correct']}/{results['total']})")
    logger.info(f"\nBy Question Type:")
    for type_name, stats in sorted(results['by_type'].items()):
        logger.info(f"  {type_name.upper()}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    if results['avg_breakdown']:
        logger.info(f"\nAverage Score Components:")
        logger.info(f"  Syntax: {results['avg_breakdown']['avg_syntax_score']:.3f}")
        logger.info(f"  Compositional: {results['avg_breakdown']['avg_comp_score']:.3f}")
        logger.info(f"  Semantic: {results['avg_breakdown']['avg_semantic_score']:.3f}")
        logger.info(f"\nAverage Mixing Weights:")
        logger.info(f"  Syntax: {results['avg_breakdown']['avg_syntax_weight']:.3f}")
        logger.info(f"  Compositional: {results['avg_breakdown']['avg_comp_weight']:.3f}")
        logger.info(f"  Semantic: {results['avg_breakdown']['avg_semantic_weight']:.3f}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {args.output}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
