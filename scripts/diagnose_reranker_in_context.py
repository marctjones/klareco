#!/usr/bin/env python3
"""
Diagnose TreeMatchReranker Performance In Context

This script answers:
1. Does BM25 retrieve the source sentence (ground truth)?
2. If yes, does the reranker promote it to top-1?
3. Where exactly is the pipeline failing?

This reveals whether the problem is:
- Retrieval (BM25 doesn't find the answer)
- Reranking (reranker doesn't recognize the answer)
- Both

Usage:
    python scripts/diagnose_reranker_in_context.py \
        --model models/tree_reranker_v2/best_model.pt \
        --test-set data/test_sets/test_questions_50.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.importance_scorer import classify_question_type
from klareco.models.tree_match_reranker import TreeMatchReranker
from klareco.embeddings import CompositionalEmbedding

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class RerankerDiagnostics:
    """Diagnose reranker performance in full pipeline context."""

    def __init__(
        self,
        model_path: Path,
        comp_emb_path: Path,
        whoosh_index: Path,
        kuzu_db: Path
    ):
        """Initialize diagnostics."""
        # Load models
        logger.info("Loading models...")
        comp_emb = self._load_comp_emb(comp_emb_path)
        self.reranker = TreeMatchReranker.load(model_path, comp_emb)
        self.reranker.eval()

        # Load retriever
        logger.info("Loading retriever...")
        self.retriever = WhooshRetriever(
            whoosh_index_dir=whoosh_index,
            kuzu_db_path=kuzu_db
        )

    def _load_comp_emb(self, path: Path) -> CompositionalEmbedding:
        """Load compositional embeddings."""
        import torch
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

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

        comp_emb.eval()
        return comp_emb

    def diagnose_question(self, question_data: Dict) -> Dict:
        """
        Diagnose a single question through full pipeline.

        Returns:
            {
                'question': str,
                'source_in_retrieval': bool,
                'source_bm25_rank': int or None,
                'source_reranked_rank': int or None,
                'bm25_top1_is_source': bool,
                'reranked_top1_is_source': bool,
                'retrieval_size': int,
                'diagnosis': str  # 'retrieval_miss', 'reranker_miss', 'success'
            }
        """
        question = question_data['question']
        source_sentence = question_data['source_sentence']
        expected_keywords = question_data.get('expected_keywords', [])

        logger.info(f"\n{'='*60}")
        logger.info(f"Question: {question}")
        logger.info(f"Source (answer): {source_sentence[:80]}...")

        # Parse query
        try:
            query_ast = parse(question)
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return {'diagnosis': 'parse_error'}

        # Extract roots
        roots = self._extract_roots(query_ast)
        logger.info(f"Query roots: {roots}")

        # Retrieve with BM25
        question_type = classify_question_type(question)
        try:
            candidates = self.retriever.retrieve(
                query_roots=roots,
                question_type=question_type.value,
                query_entity=None,
                top_k=200  # Get more to check if source is present
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {'diagnosis': 'retrieval_error'}

        if not candidates:
            logger.info("❌ No retrieval results")
            return {
                'question': question,
                'diagnosis': 'retrieval_empty',
                'retrieval_size': 0
            }

        logger.info(f"Retrieved: {len(candidates)} candidates")

        # Check: Is source in retrieval results?
        source_bm25_rank = None
        for i, doc in enumerate(candidates):
            if doc['text'] == source_sentence:
                source_bm25_rank = i + 1  # 1-indexed
                break

        if source_bm25_rank is None:
            logger.info(f"❌ RETRIEVAL MISS: Source not in BM25 top-{len(candidates)}")
            return {
                'question': question,
                'source_in_retrieval': False,
                'source_bm25_rank': None,
                'retrieval_size': len(candidates),
                'diagnosis': 'retrieval_miss'
            }

        logger.info(f"✓ Source found at BM25 rank: {source_bm25_rank}")

        # Rerank top-50
        reranked = self._rerank_candidates(query_ast, candidates[:50])

        # Check: Is source at top-1 after reranking?
        source_reranked_rank = None
        for i, doc in enumerate(reranked):
            if doc['text'] == source_sentence:
                source_reranked_rank = i + 1  # 1-indexed
                break

        if source_reranked_rank is None:
            # Source was in top-200 but not top-50, so reranker didn't see it
            logger.info(f"⚠ Source was at rank {source_bm25_rank} > 50, reranker didn't see it")
            return {
                'question': question,
                'source_in_retrieval': True,
                'source_bm25_rank': source_bm25_rank,
                'source_reranked_rank': None,
                'retrieval_size': len(candidates),
                'diagnosis': 'source_beyond_top50'
            }

        logger.info(f"Reranked: {source_bm25_rank} → {source_reranked_rank}")

        # Check success
        if source_reranked_rank == 1:
            logger.info("✅ SUCCESS: Source promoted to rank 1")
            diagnosis = 'success'
        elif source_bm25_rank <= 50:
            logger.info(f"❌ RERANKER MISS: Source at rank {source_reranked_rank} (was {source_bm25_rank})")
            diagnosis = 'reranker_miss'
        else:
            diagnosis = 'source_beyond_top50'

        return {
            'question': question,
            'source_in_retrieval': True,
            'source_bm25_rank': source_bm25_rank,
            'source_reranked_rank': source_reranked_rank,
            'bm25_top1_is_source': (source_bm25_rank == 1),
            'reranked_top1_is_source': (source_reranked_rank == 1),
            'retrieval_size': len(candidates),
            'diagnosis': diagnosis
        }

    def _extract_roots(self, ast: Dict) -> List[str]:
        """Extract roots from AST."""
        roots = []

        def traverse(node):
            if node is None or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '').lower()
                if root and root not in ['ki', 'kiu', 'kio', 'kie', 'kiam']:
                    roots.append(root)
            for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                traverse(node.get(key))
            for item in node.get('priskriboj', []) + node.get('aliaj', []):
                traverse(item)

        traverse(ast)
        return roots

    def _rerank_candidates(self, query_ast: Dict, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates using TreeMatchReranker."""
        import torch

        reranked = []
        for doc in candidates:
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue

            with torch.no_grad():
                score, breakdown = self.reranker(query_ast, doc_ast)

            reranked.append({
                'text': doc['text'],
                'rerank_score': score.item() if hasattr(score, 'item') else float(score),
                'breakdown': breakdown
            })

        # Sort by score descending
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    def run_diagnostics(self, test_questions: List[Dict]) -> Dict:
        """Run diagnostics on all test questions."""
        results = []

        for i, q in enumerate(test_questions):
            logger.info(f"\n[{i+1}/{len(test_questions)}]")
            result = self.diagnose_question(q)
            results.append(result)

        # Compute statistics
        stats = self._compute_stats(results)
        return {
            'results': results,
            'stats': stats
        }

    def _compute_stats(self, results: List[Dict]) -> Dict:
        """Compute diagnostic statistics."""
        total = len(results)

        # Filter valid results
        valid = [r for r in results if 'diagnosis' in r]

        # Count diagnoses
        retrieval_miss = sum(1 for r in valid if r['diagnosis'] == 'retrieval_miss')
        reranker_miss = sum(1 for r in valid if r['diagnosis'] == 'reranker_miss')
        success = sum(1 for r in valid if r['diagnosis'] == 'success')
        beyond_top50 = sum(1 for r in valid if r['diagnosis'] == 'source_beyond_top50')

        # Coverage: % where source is in BM25 top-50
        sources_in_top50 = sum(
            1 for r in valid
            if r.get('source_bm25_rank') and r['source_bm25_rank'] <= 50
        )
        coverage_top50 = sources_in_top50 / total if total > 0 else 0

        # Reranker effectiveness: given source in top-50, does it reach top-1?
        effectiveness = success / sources_in_top50 if sources_in_top50 > 0 else 0

        # Overall accuracy
        overall_accuracy = success / total if total > 0 else 0

        return {
            'total_questions': total,
            'retrieval_miss': retrieval_miss,
            'reranker_miss': reranker_miss,
            'success': success,
            'beyond_top50': beyond_top50,
            'coverage_top50': coverage_top50,
            'reranker_effectiveness': effectiveness,
            'overall_accuracy': overall_accuracy
        }


def main():
    parser = argparse.ArgumentParser(description='Diagnose TreeMatchReranker in context')
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to trained TreeMatchReranker')
    parser.add_argument('--comp-emb', type=Path,
                       default=Path('models/root_embeddings_fundamento_enhanced/root_embeddings_best.pt'),
                       help='Path to compositional embeddings')
    parser.add_argument('--test-set', type=Path, required=True,
                       help='Path to test questions JSONL')
    parser.add_argument('--whoosh-index', type=Path,
                       default=Path('data/indexes/whoosh_fts'),
                       help='Path to Whoosh index')
    parser.add_argument('--kuzu-db', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kuzu database')
    parser.add_argument('--output', type=Path,
                       help='Path to save diagnostic results')

    args = parser.parse_args()

    # Load test questions
    logger.info(f"Loading test questions from {args.test_set}...")
    with open(args.test_set) as f:
        test_questions = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(test_questions)} questions")

    # Initialize diagnostics
    diagnostics = RerankerDiagnostics(
        model_path=args.model,
        comp_emb_path=args.comp_emb,
        whoosh_index=args.whoosh_index,
        kuzu_db=args.kuzu_db
    )

    # Run diagnostics
    results = diagnostics.run_diagnostics(test_questions)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)

    stats = results['stats']
    logger.info(f"\nTotal questions: {stats['total_questions']}")
    logger.info(f"\nBREAKDOWN:")
    logger.info(f"  ✅ Success (source → rank 1): {stats['success']} ({stats['success']/stats['total_questions']*100:.1f}%)")
    logger.info(f"  ❌ Retrieval miss (source not in BM25): {stats['retrieval_miss']} ({stats['retrieval_miss']/stats['total_questions']*100:.1f}%)")
    logger.info(f"  ❌ Reranker miss (source in top-50 but not → rank 1): {stats['reranker_miss']} ({stats['reranker_miss']/stats['total_questions']*100:.1f}%)")
    logger.info(f"  ⚠  Source beyond top-50: {stats['beyond_top50']} ({stats['beyond_top50']/stats['total_questions']*100:.1f}%)")

    logger.info(f"\nPERFORMANCE METRICS:")
    logger.info(f"  Retrieval Coverage (source in BM25 top-50): {stats['coverage_top50']*100:.1f}%")
    logger.info(f"  Reranker Effectiveness (source in top-50 → rank 1): {stats['reranker_effectiveness']*100:.1f}%")
    logger.info(f"  Overall Accuracy: {stats['overall_accuracy']*100:.1f}%")

    logger.info(f"\nFORMULA:")
    logger.info(f"  Overall Accuracy = Coverage × Effectiveness")
    logger.info(f"  {stats['overall_accuracy']*100:.1f}% = {stats['coverage_top50']*100:.1f}% × {stats['reranker_effectiveness']*100:.1f}%")

    logger.info(f"\nBOTTLENECK ANALYSIS:")
    if stats['coverage_top50'] < 0.7:
        logger.info("  🔴 PRIMARY BOTTLENECK: Retrieval")
        logger.info("     → BM25 isn't finding answer documents")
        logger.info("     → Need better retrieval or query expansion")
    elif stats['reranker_effectiveness'] < 0.5:
        logger.info("  🔴 PRIMARY BOTTLENECK: Reranker")
        logger.info("     → Reranker isn't recognizing answer documents")
        logger.info("     → Need better training data or model")
    else:
        logger.info("  🟢 Both components working reasonably well")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
