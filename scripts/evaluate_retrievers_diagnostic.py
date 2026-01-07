#!/usr/bin/env python3
"""
Enhanced Retriever Evaluation with Diagnostic Logging.

Evaluates all 4 active retrievers with detailed diagnostic information:
1. ASTAwareRetriever - Question classification + entity recognition + pattern matching
2. HNSWSlotRetriever - HNSW pre-filter + mmap slot reranking
3. FAISSSlotRetriever - FAISS pre-filter + slot reranking
4. HybridFAISSMmapRetriever - FAISS + mmap hybrid (best accuracy expected)

Provides detailed logs for understanding WHY retrievers succeed or fail:
- Query parsing details (AST, slots extracted, features)
- Pre-filter stage results (candidates returned, scores)
- Reranking stage results (slot similarities, feature bonuses)
- Final ranking with explanations

Usage:
    python scripts/evaluate_retrievers_diagnostic.py
    python scripts/evaluate_retrievers_diagnostic.py --fresh
    python scripts/evaluate_retrievers_diagnostic.py --retriever ASTAware
    python scripts/evaluate_retrievers_diagnostic.py --diagnostic  # Run diagnostic questions only
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Global hybrid embeddings analyzer (loaded lazily)
_hybrid_embedder = None


def get_hybrid_embedder():
    """Lazily load hybrid embeddings for analysis."""
    global _hybrid_embedder
    if _hybrid_embedder is None:
        try:
            from klareco.embeddings.hybrid_embeddings import HybridEmbeddings
            from pathlib import Path

            root_model = Path("models/root_embeddings/best_model.pt")
            topical_model = Path("models/topical_embeddings/best_model.pt")

            if root_model.exists() and topical_model.exists():
                _hybrid_embedder = HybridEmbeddings.from_checkpoints(
                    linguistic_checkpoint=root_model,
                    topical_checkpoint=topical_model,
                    pad_missing=True,
                    default_mode='hybrid'
                )
                logger.info(f"Loaded hybrid embedder for analysis")
        except Exception as e:
            logger.warning(f"Could not load hybrid embedder: {e}")
    return _hybrid_embedder


def analyze_root_embeddings(roots: List[str]) -> Dict[str, Any]:
    """
    Analyze which embedding spaces cover each root.

    Returns:
        Dict with:
        - per_root: {root: {linguistic: bool, topical: bool, type: str}}
        - summary: {both: N, ling_only: N, top_only: N, neither: N}
    """
    embedder = get_hybrid_embedder()

    analysis = {
        'per_root': {},
        'summary': {
            'both': 0,
            'linguistic_only': 0,
            'topical_only': 0,
            'neither': 0,
        },
        'roots_by_type': {
            'both': [],
            'linguistic_only': [],
            'topical_only': [],
            'neither': [],
        }
    }

    if embedder is None:
        return analysis

    for root in roots:
        if not root or len(root) < 2:
            continue

        info = embedder.analyze_root(root)

        analysis['per_root'][root] = {
            'linguistic': info['has_linguistic'],
            'topical': info['has_topical'],
            'type': info['type'],
        }

        if info['has_linguistic'] and info['has_topical']:
            analysis['summary']['both'] += 1
            analysis['roots_by_type']['both'].append(root)
        elif info['has_linguistic']:
            analysis['summary']['linguistic_only'] += 1
            analysis['roots_by_type']['linguistic_only'].append(root)
        elif info['has_topical']:
            analysis['summary']['topical_only'] += 1
            analysis['roots_by_type']['topical_only'].append(root)
        else:
            analysis['summary']['neither'] += 1
            analysis['roots_by_type']['neither'].append(root)

    return analysis


# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
INDEX_DIR = Path("data/indexes/slot_hybrid")
BENCHMARK_FILE = Path("data/benchmarks/datasets/qa_benchmark_v1.jsonl")
DIAGNOSTIC_FILE = Path("data/benchmarks/datasets/diagnostic_retriever_questions.jsonl")
RESULTS_DIR = Path("data/benchmarks/results")
CHECKPOINT_FILE = Path("data/benchmarks/diagnostic_eval_checkpoint.json")


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_questions(benchmark_only: bool = False, diagnostic_only: bool = False) -> List[Dict]:
    """Load questions for evaluation.

    Args:
        benchmark_only: Only load benchmark questions (requires_retrieval=True)
        diagnostic_only: Only load diagnostic questions

    Returns:
        List of question dicts
    """
    questions = []

    if not diagnostic_only:
        # Load benchmark questions that require retrieval
        with open(BENCHMARK_FILE) as f:
            for line in f:
                q = json.loads(line)
                if q.get('requires_retrieval', False):
                    q['question_set'] = 'benchmark'
                    questions.append(q)
        logger.info(f"Loaded {len(questions)} benchmark questions (retrieval-requiring)")

    if not benchmark_only and DIAGNOSTIC_FILE.exists():
        # Load diagnostic questions
        diag_count = 0
        with open(DIAGNOSTIC_FILE) as f:
            for line in f:
                q = json.loads(line)
                q['question_set'] = 'diagnostic'
                questions.append(q)
                diag_count += 1
        logger.info(f"Loaded {diag_count} diagnostic questions")

    return questions


def contains_answer(text: str, acceptable_answers: List[str]) -> bool:
    """Check if text contains any acceptable answer."""
    text_lower = text.lower()
    for answer in acceptable_answers:
        if answer.lower() in text_lower:
            return True
    return False


def extract_query_diagnostics(query: str) -> Dict[str, Any]:
    """Extract detailed diagnostic information from query parsing."""
    diagnostics = {
        'original_query': query,
        'parse_success': False,
        'ast': None,
        'extracted_roots': [],
        'question_word': None,
        'sentence_type': None,
        'slots': {'SUBJ': None, 'VERB': None, 'OBJ': None},
        'parse_error': None,
        'embedding_analysis': None,  # NEW: linguistic vs topical coverage
    }

    try:
        ast = parse(query)
        diagnostics['parse_success'] = True
        diagnostics['sentence_type'] = ast.get('fraztipo', 'unknown')

        # Extract roots from AST
        roots = []
        def extract_roots(node):
            if not node or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    roots.append(root.lower())
                # Check for question words
                vortspeco = node.get('vortspeco', '')
                if vortspeco == 'korelativo' or root.lower() in ['kiu', 'kio', 'kie', 'kiam', 'kiel', 'kiom', 'kial']:
                    diagnostics['question_word'] = root.lower()
            elif node.get('tipo') == 'vortgrupo':
                extract_roots(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract_roots(p)
            elif node.get('tipo') == 'frazo':
                extract_roots(node.get('subjekto'))
                extract_roots(node.get('verbo'))
                extract_roots(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract_roots(a)

        extract_roots(ast)
        diagnostics['extracted_roots'] = roots

        # Extract slot roots
        def get_slot_root(slot_node):
            if not slot_node:
                return None
            if slot_node.get('tipo') == 'vorto':
                return slot_node.get('radiko', '').lower()
            elif slot_node.get('tipo') == 'vortgrupo':
                kerno = slot_node.get('kerno')
                if kerno:
                    return kerno.get('radiko', '').lower()
            return None

        diagnostics['slots']['SUBJ'] = get_slot_root(ast.get('subjekto'))
        diagnostics['slots']['VERB'] = get_slot_root(ast.get('verbo'))
        diagnostics['slots']['OBJ'] = get_slot_root(ast.get('objekto'))

        # Store minimal AST for logging
        diagnostics['ast'] = {
            'fraztipo': ast.get('fraztipo'),
            'has_subjekto': ast.get('subjekto') is not None,
            'has_verbo': ast.get('verbo') is not None,
            'has_objekto': ast.get('objekto') is not None,
            'aliaj_count': len(ast.get('aliaj', [])),
        }

        # Analyze embedding coverage for query roots
        diagnostics['embedding_analysis'] = analyze_root_embeddings(roots)

    except Exception as e:
        diagnostics['parse_error'] = str(e)

    return diagnostics


def evaluate_single_query(
    retriever,
    retriever_name: str,
    question: Dict,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate a single query with detailed diagnostics."""

    query = question['question']
    acceptable_answers = question['acceptable_answers']

    result = {
        'id': question['id'],
        'question': query,
        'category': question.get('category', 'unknown'),
        'question_set': question.get('question_set', 'benchmark'),
        'retriever': retriever_name,
        'acceptable_answers': acceptable_answers,
        'found_at_rank': None,
        'found_answer': None,
        'latency_ms': 0,
        'top_results': [],
        'query_diagnostics': None,
        'retrieval_diagnostics': {},
        'error': None,
    }

    # Step 1: Parse query and extract diagnostics
    result['query_diagnostics'] = extract_query_diagnostics(query)

    # Step 2: Run retrieval
    start = time.time()
    try:
        search_results = retriever.search(query, top_k=top_k)
        result['latency_ms'] = (time.time() - start) * 1000

        # Record top results with details
        for rank, (score, doc) in enumerate(search_results, 1):
            doc_text = doc.get('text', '')
            doc_info = {
                'rank': rank,
                'score': float(score),
                'text_preview': doc_text[:200] + '...' if len(doc_text) > 200 else doc_text,
                'contains_answer': contains_answer(doc_text, acceptable_answers),
                'source': doc.get('source', {}),
            }

            # Add slot info if available
            if 'features' in doc:
                doc_info['features'] = doc['features']

            result['top_results'].append(doc_info)

            # Check if this is the first answer match
            if result['found_at_rank'] is None and doc_info['contains_answer']:
                result['found_at_rank'] = rank
                # Find which answer matched
                for ans in acceptable_answers:
                    if ans.lower() in doc_text.lower():
                        result['found_answer'] = ans
                        break

        # Retrieval diagnostics
        result['retrieval_diagnostics'] = {
            'num_results': len(search_results),
            'top_score': float(search_results[0][0]) if search_results else 0,
            'score_range': (
                float(search_results[-1][0]) if search_results else 0,
                float(search_results[0][0]) if search_results else 0,
            ),
            'any_answer_found': result['found_at_rank'] is not None,
        }

    except Exception as e:
        result['error'] = str(e)
        result['latency_ms'] = (time.time() - start) * 1000
        import traceback
        logger.error(f"Query failed: {query}\n{traceback.format_exc()}")

    return result


def load_checkpoint() -> Dict:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed_retrievers': [], 'results': {}}


def save_checkpoint(data: Dict):
    """Save checkpoint atomically."""
    temp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp.rename(CHECKPOINT_FILE)


def initialize_retriever(name: str) -> Optional[object]:
    """Initialize a retriever by name."""
    logger.info(f"Initializing {name}...")

    try:
        if name == "ASTAware":
            from klareco.rag.ast_aware_retriever import ASTAwareRetriever
            return ASTAwareRetriever(
                index_path=INDEX_DIR,
                use_prefilter=True,
                use_keyword_prefilter=True,
            )

        elif name == "HNSW":
            from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            mmap_dir = INDEX_DIR / "mmap"
            if not mmap_dir.exists():
                logger.warning(f"  mmap/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return HNSWSlotRetriever(INDEX_DIR, indexer)

        elif name == "FAISS":
            from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            faiss_dir = INDEX_DIR / "faiss"
            if not faiss_dir.exists():
                logger.warning(f"  faiss/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return FAISSSlotRetriever(INDEX_DIR, indexer)

        elif name == "HybridFAISS":
            from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            mmap_dir = INDEX_DIR / "mmap"
            faiss_dir = INDEX_DIR / "faiss"
            if not mmap_dir.exists() or not faiss_dir.exists():
                logger.warning(f"  mmap/ or faiss/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return HybridFAISSMmapRetriever(INDEX_DIR, indexer)

        else:
            logger.error(f"Unknown retriever: {name}")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize {name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def evaluate_retriever(
    retriever,
    retriever_name: str,
    questions: List[Dict],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate a retriever on all questions with full diagnostics."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {retriever_name} on {len(questions)} questions")
    logger.info(f"{'='*60}")

    baseline_mem = get_memory_mb()
    peak_mem = baseline_mem

    # Separate questions by type
    benchmark_qs = [q for q in questions if q.get('question_set') == 'benchmark']
    diagnostic_qs = [q for q in questions if q.get('question_set') == 'diagnostic']

    # Filter diagnostic questions for this retriever
    retriever_diagnostic_qs = [
        q for q in diagnostic_qs
        if q.get('retriever', retriever_name) == retriever_name
    ]

    all_questions = benchmark_qs + retriever_diagnostic_qs

    logger.info(f"  Benchmark questions: {len(benchmark_qs)}")
    logger.info(f"  Diagnostic questions for {retriever_name}: {len(retriever_diagnostic_qs)}")

    results = {
        'name': retriever_name,
        'total_questions': len(all_questions),
        'benchmark_questions': len(benchmark_qs),
        'diagnostic_questions': len(retriever_diagnostic_qs),
        'recall_at_1': 0,
        'recall_at_5': 0,
        'recall_at_10': 0,
        'mrr': 0.0,
        'avg_latency_ms': 0,
        'peak_memory_mb': 0,
        'questions': [],
        'summary_by_category': {},
        'diagnostic_results': [],
    }

    total_time = 0
    mrr_sum = 0.0

    for i, qa in enumerate(all_questions):
        # Evaluate single query
        q_result = evaluate_single_query(retriever, retriever_name, qa, top_k=top_k)

        total_time += q_result['latency_ms']
        peak_mem = max(peak_mem, get_memory_mb())

        # Update metrics
        found_at = q_result['found_at_rank']
        if found_at:
            if found_at == 1:
                results['recall_at_1'] += 1
            if found_at <= 5:
                results['recall_at_5'] += 1
            if found_at <= 10:
                results['recall_at_10'] += 1
            mrr_sum += 1.0 / found_at

        # Store result
        if q_result.get('question_set') == 'diagnostic':
            results['diagnostic_results'].append(q_result)
        else:
            results['questions'].append(q_result)

        # Update category summary
        category = qa.get('category', 'unknown')
        if category not in results['summary_by_category']:
            results['summary_by_category'][category] = {
                'total': 0, 'found': 0, 'recall_at_1': 0, 'recall_at_5': 0
            }
        results['summary_by_category'][category]['total'] += 1
        if found_at:
            results['summary_by_category'][category]['found'] += 1
            if found_at == 1:
                results['summary_by_category'][category]['recall_at_1'] += 1
            if found_at <= 5:
                results['summary_by_category'][category]['recall_at_5'] += 1

        # Log progress with details
        status = f"rank {found_at}" if found_at else "NOT FOUND"
        q_diag = q_result['query_diagnostics']
        roots = q_diag.get('extracted_roots', [])[:4]
        slots = q_diag.get('slots', {})

        logger.info(
            f"  [{i+1}/{len(all_questions)}] {qa['id']}: {status} "
            f"({q_result['latency_ms']:.0f}ms) "
            f"roots={roots} slots=[S:{slots.get('SUBJ')}, V:{slots.get('VERB')}, O:{slots.get('OBJ')}]"
        )

        # Log diagnostic details for failed queries
        if not found_at and q_result.get('question_set') == 'benchmark':
            logger.info(f"      Query: {qa['question']}")
            logger.info(f"      Expected: {qa['acceptable_answers'][:3]}")
            if q_result['top_results']:
                top = q_result['top_results'][0]
                logger.info(f"      Top result (score={top['score']:.3f}): {top['text_preview'][:100]}...")
            # Log embedding analysis for failed queries
            emb_analysis = q_diag.get('embedding_analysis')
            if emb_analysis and emb_analysis.get('per_root'):
                summary = emb_analysis.get('summary', {})
                logger.info(f"      Embedding coverage: both={summary.get('both', 0)}, ling_only={summary.get('linguistic_only', 0)}, top_only={summary.get('topical_only', 0)}, neither={summary.get('neither', 0)}")
                # Show roots missing from vocabularies
                neither_roots = emb_analysis.get('roots_by_type', {}).get('neither', [])
                if neither_roots:
                    logger.info(f"      Missing from both vocabs: {neither_roots}")
                top_only_roots = emb_analysis.get('roots_by_type', {}).get('topical_only', [])
                if top_only_roots:
                    logger.info(f"      Topical only (no linguistic): {top_only_roots}")

    # Final stats
    results['avg_latency_ms'] = total_time / len(all_questions) if all_questions else 0
    results['peak_memory_mb'] = peak_mem
    results['memory_delta_mb'] = peak_mem - baseline_mem
    results['mrr'] = mrr_sum / len(all_questions) if all_questions else 0

    # Print summary
    logger.info(f"\n  Summary for {retriever_name}:")
    logger.info(f"    Recall@1:  {results['recall_at_1']}/{len(all_questions)} ({100*results['recall_at_1']/len(all_questions):.1f}%)")
    logger.info(f"    Recall@5:  {results['recall_at_5']}/{len(all_questions)} ({100*results['recall_at_5']/len(all_questions):.1f}%)")
    logger.info(f"    Recall@10: {results['recall_at_10']}/{len(all_questions)} ({100*results['recall_at_10']/len(all_questions):.1f}%)")
    logger.info(f"    MRR:       {results['mrr']:.3f}")
    logger.info(f"    Latency:   {results['avg_latency_ms']:.1f}ms avg")
    logger.info(f"    Memory:    {results['peak_memory_mb']:.0f}MB peak (+{results['memory_delta_mb']:.0f}MB)")

    # Category breakdown
    logger.info(f"\n  By category:")
    for cat, stats in sorted(results['summary_by_category'].items()):
        pct = 100 * stats['found'] / stats['total'] if stats['total'] > 0 else 0
        logger.info(f"    {cat}: {stats['found']}/{stats['total']} ({pct:.0f}%)")

    # Aggregate embedding analysis
    all_emb_results = results['questions'] + results.get('diagnostic_results', [])
    total_roots = 0
    coverage = {'both': 0, 'linguistic_only': 0, 'topical_only': 0, 'neither': 0}

    for q_result in all_emb_results:
        q_diag = q_result.get('query_diagnostics', {})
        emb_analysis = q_diag.get('embedding_analysis', {})
        summary = emb_analysis.get('summary', {})
        for key in coverage:
            coverage[key] += summary.get(key, 0)
        total_roots += sum(summary.get(k, 0) for k in coverage)

    if total_roots > 0:
        results['embedding_coverage'] = {
            'total_roots': total_roots,
            'both_pct': 100 * coverage['both'] / total_roots,
            'linguistic_only_pct': 100 * coverage['linguistic_only'] / total_roots,
            'topical_only_pct': 100 * coverage['topical_only'] / total_roots,
            'neither_pct': 100 * coverage['neither'] / total_roots,
        }
        logger.info(f"\n  Embedding coverage across all query roots ({total_roots} roots):")
        logger.info(f"    Both (hybrid 128d):      {coverage['both']:>4} ({100*coverage['both']/total_roots:>5.1f}%)")
        logger.info(f"    Linguistic only (64d):   {coverage['linguistic_only']:>4} ({100*coverage['linguistic_only']/total_roots:>5.1f}%)")
        logger.info(f"    Topical only (64d):      {coverage['topical_only']:>4} ({100*coverage['topical_only']/total_roots:>5.1f}%)")
        logger.info(f"    Neither (zeros):         {coverage['neither']:>4} ({100*coverage['neither']/total_roots:>5.1f}%)")

    return results


def print_comparison(all_results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("RETRIEVER COMPARISON - DIAGNOSTIC EVALUATION")
    print("=" * 110)
    print()

    header = f"{'Retriever':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Latency':>12} {'Memory':>10} {'Diag':>8}"
    print(header)
    print("-" * 110)

    sorted_results = sorted(all_results, key=lambda x: x['recall_at_10'], reverse=True)

    for r in sorted_results:
        total = r['total_questions']
        r1 = 100 * r['recall_at_1'] / total if total > 0 else 0
        r5 = 100 * r['recall_at_5'] / total if total > 0 else 0
        r10 = 100 * r['recall_at_10'] / total if total > 0 else 0

        # Diagnostic success rate
        diag_results = r.get('diagnostic_results', [])
        diag_found = sum(1 for d in diag_results if d.get('found_at_rank'))
        diag_pct = 100 * diag_found / len(diag_results) if diag_results else 0

        print(f"{r['name']:<20} "
              f"{r1:>7.1f}% "
              f"{r5:>7.1f}% "
              f"{r10:>7.1f}% "
              f"{r['mrr']:>7.3f} "
              f"{r['avg_latency_ms']:>10.1f}ms "
              f"{r['peak_memory_mb']:>8.0f}MB "
              f"{diag_pct:>6.0f}%")

    print()
    print("Legend:")
    print("  R@k = Recall at k (% of questions where answer found in top k)")
    print("  MRR = Mean Reciprocal Rank (higher is better)")
    print("  Diag = Diagnostic question success rate (retriever-specific test questions)")
    print()

    # Category breakdown across all retrievers
    print("Category Breakdown (Recall@10):")
    print("-" * 80)

    categories = set()
    for r in all_results:
        categories.update(r.get('summary_by_category', {}).keys())

    cat_header = f"{'Category':<15}"
    for r in sorted_results:
        cat_header += f" {r['name'][:12]:>12}"
    print(cat_header)

    for cat in sorted(categories):
        row = f"{cat:<15}"
        for r in sorted_results:
            stats = r.get('summary_by_category', {}).get(cat, {'found': 0, 'total': 0})
            pct = 100 * stats['found'] / stats['total'] if stats['total'] > 0 else 0
            row += f" {pct:>11.0f}%"
        print(row)

    print()

    # Embedding coverage analysis (same for all retrievers since it's query-based)
    if sorted_results and sorted_results[0].get('embedding_coverage'):
        emb_cov = sorted_results[0]['embedding_coverage']
        print("Embedding Coverage Analysis (Query Roots):")
        print("-" * 80)
        print(f"  Total roots across queries: {emb_cov['total_roots']}")
        print(f"  Both (hybrid 128d):         {emb_cov['both_pct']:>5.1f}%  - Full semantic + contextual representation")
        print(f"  Linguistic only (64d):      {emb_cov['linguistic_only_pct']:>5.1f}%  - Dictionary semantics only")
        print(f"  Topical only (64d):         {emb_cov['topical_only_pct']:>5.1f}%  - Corpus context only (proper nouns, rare terms)")
        print(f"  Neither (zeros):            {emb_cov['neither_pct']:>5.1f}%  - Unknown roots, no embedding")
        print()
        print("Interpretation:")
        print("  - 'Both' roots have full 128d embeddings capturing semantic AND contextual meaning")
        print("  - 'Topical only' roots (like proper nouns) lack linguistic similarity but have corpus context")
        print("  - 'Neither' roots are out-of-vocabulary and use zero vectors (hurts retrieval)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrievers with diagnostics")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--retriever", type=str, help="Evaluate single retriever")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic questions only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark questions only")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    args = parser.parse_args()

    # Load questions
    questions = load_questions(
        benchmark_only=args.benchmark,
        diagnostic_only=args.diagnostic,
    )

    if not questions:
        logger.error("No questions to evaluate!")
        sys.exit(1)

    # Determine which retrievers to test
    if args.retriever:
        retriever_names = [args.retriever]
    else:
        retriever_names = ["ASTAware", "HNSW", "FAISS", "HybridFAISS"]

    # Load checkpoint
    checkpoint = load_checkpoint() if not args.fresh else {'completed_retrievers': [], 'results': {}}

    all_results = list(checkpoint.get('results', {}).values())

    # Evaluate each retriever
    for name in retriever_names:
        if name in checkpoint.get('completed_retrievers', []):
            logger.info(f"Skipping {name} (already evaluated, use --fresh to re-run)")
            continue

        retriever = initialize_retriever(name)
        if retriever is None:
            logger.warning(f"Skipping {name} (initialization failed)")
            continue

        results = evaluate_retriever(retriever, name, questions, top_k=args.top_k)
        all_results.append(results)

        # Save checkpoint
        checkpoint['completed_retrievers'].append(name)
        checkpoint['results'][name] = results
        save_checkpoint(checkpoint)

        # Clean up to free memory
        del retriever

    # Print comparison
    print_comparison(all_results)

    # Save final results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"diagnostic_retriever_comparison_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'index_dir': str(INDEX_DIR),
            'benchmark_file': str(BENCHMARK_FILE),
            'diagnostic_file': str(DIAGNOSTIC_FILE),
            'total_questions': len(questions),
            'retrievers': all_results,
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")

    # Clean up checkpoint on success
    if len(all_results) == len(retriever_names):
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("Evaluation complete, checkpoint removed")


if __name__ == "__main__":
    main()
