#!/usr/bin/env python3
"""
Evaluate RAG System Against Test Set

Runs the RAG pipeline on each test question and compares output to expected answers.

Usage:
    # Evaluate current RAG system
    python scripts/evaluate_rag_test_set.py \\
        --test-set data/evaluation/rag_test_set.jsonl \\
        --output data/evaluation/rag_results.jsonl

    # Test with different pipeline configurations
    python scripts/evaluate_rag_test_set.py --no-m1        # Without M1
    python scripts/evaluate_rag_test_set.py --no-rerank   # Without reranker

    # Filter by category
    python scripts/evaluate_rag_test_set.py \\
        --category factual_simple \\
        --category grammar

    # Filter by expected performance
    python scripts/evaluate_rag_test_set.py \\
        --expected works  # Only test questions that should work
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.models.reranker import ASTReranker
from klareco.models.m1_inference import M1Inference
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.answer_extractor import ASTAnswerExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_answer_in_documents(passages: List[Dict], expected_answer: str, expected_keywords: List[str]) -> Dict:
    """
    Check if answer appears in retrieved documents and at what rank.

    Returns:
        {
            'answer_in_retrieved': bool,
            'answer_in_rank': int or None  # Rank where answer first appears (1-indexed), None if not found
            'keyword_matches_per_rank': List[int]  # Number of keywords in each rank
        }
    """
    expected_lower = expected_answer.lower()
    keywords_lower = [kw.lower() for kw in expected_keywords]

    answer_in_rank = None
    keyword_matches_per_rank = []

    for i, passage in enumerate(passages):
        text_lower = passage['text'].lower()

        # Count keyword matches
        matches = sum(1 for kw in keywords_lower if kw in text_lower)
        keyword_matches_per_rank.append(matches)

        # Check if answer appears
        if answer_in_rank is None:
            if expected_lower and expected_lower in text_lower:
                answer_in_rank = i + 1  # 1-indexed
            elif matches >= len(keywords_lower) * 0.5:  # At least 50% of keywords
                answer_in_rank = i + 1

    return {
        'answer_in_retrieved': answer_in_rank is not None,
        'answer_in_rank': answer_in_rank,
        'keyword_matches_per_rank': keyword_matches_per_rank,
    }


def format_status_line(diagnostics: Dict, evaluation: Dict) -> str:
    """
    Format compact pipeline status line.

    Example: "RETRIEVAL(3 match,0.85) BOOST(+0.12) RERANK(#3→#1) EXTRACT(✓ exact) EVAL(✓)"
    """
    parts = []

    # Retrieval
    ret = diagnostics.get('retrieval', {})
    ret_metrics = ret.get('metrics', {})
    if ret:
        match_count = ret_metrics.get('answer_keywords_in_top_3_count', 0)
        top_score = ret_metrics.get('top_document_score', 0.0)
        parts.append(f"RETRIEVAL({match_count} match,{top_score:.2f})")

    # Entity boost (if applied)
    boost = diagnostics.get('entity_boost', {})
    if boost and boost.get('status') == 'applied':
        boost_amt = boost.get('metrics', {}).get('boost_amount', 0.0)
        parts.append(f"BOOST(+{boost_amt:.2f})")

    # Quality filter (if documents removed)
    quality = diagnostics.get('quality_filter', {})
    if quality and quality.get('status') == 'removed':
        removed = quality.get('metrics', {}).get('documents_removed', 0)
        parts.append(f"FILTER({removed} removed)")
    elif quality and quality.get('status') == 'pass':
        parts.append(f"FILTER(pass)")

    # Reranking
    rerank = diagnostics.get('reranking', {})
    if rerank and rerank.get('status') == 'success':
        rank_change = rerank.get('metrics', {}).get('rank_change', 0)
        if rank_change != 0:
            parts.append(f"RERANK(moved {rank_change:+d})")
        else:
            parts.append(f"RERANK(#1→#1)")

    # Extraction
    extraction = diagnostics.get('extraction', {})
    if extraction:
        status = extraction.get('status')
        if status == 'success':
            method = extraction.get('metrics', {}).get('match_type', 'exact')
            parts.append(f"EXTRACT(✓ {method})")
        elif status == 'no_match':
            parts.append(f"EXTRACT(✗ failed)")
        elif status == 'fallback':
            parts.append(f"EXTRACT([full doc])")

    # Evaluation
    if evaluation.get('correct'):
        parts.append("EVAL(✓)")
    elif evaluation.get('partial'):
        parts.append("EVAL(~)")
    else:
        parts.append("EVAL(✗)")

    return " ".join(parts)


def load_test_set(
    test_set_path: Path,
    categories: Optional[List[str]] = None,
    expected_performance: Optional[str] = None
) -> List[Dict]:
    """Load test set with optional filtering."""
    questions = []

    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)

            # Filter by category
            if categories and q['category'] not in categories:
                continue

            # Filter by expected performance
            if expected_performance and q['expected_performance'] != expected_performance:
                continue

            questions.append(q)

    return questions


class RAGEvaluator:
    """RAG pipeline evaluator."""

    def __init__(
        self,
        retriever: ASTAwareRetriever,
        reranker: Optional[ASTReranker] = None,
        m1: Optional[M1Inference] = None,
        use_m1: bool = True,
        use_reranking: bool = True,
        use_extraction: bool = True,
        m1_threshold: float = 0.5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.m1 = m1
        self.use_m1 = use_m1 and m1 is not None
        self.use_reranking = use_reranking and reranker is not None
        self.use_extraction = use_extraction
        self.m1_threshold = m1_threshold

        # Initialize answer extractor
        self.extractor = ASTAnswerExtractor() if use_extraction else None

    def extract_svo_triple(self, ast: Dict):
        """Extract subject-verb-object triple from AST."""
        def get_root(node):
            if node is None:
                return None
            if isinstance(node, dict):
                if node.get('tipo') == 'vortgrupo':
                    kerno = node.get('kerno', {})
                    return kerno.get('radiko')
                elif node.get('tipo') == 'vorto':
                    return node.get('radiko')
            return None

        subj = get_root(ast.get('subjekto'))
        verb = get_root(ast.get('verbo'))
        obj = get_root(ast.get('objekto'))

        return (subj, verb, obj)

    def run_query(self, question: str, top_k: int = 10, rerank_top_k: int = 20) -> Dict:
        """
        Run RAG pipeline on a single question.

        Args:
            question: Query string
            top_k: Number of final results to return
            rerank_top_k: Number of candidates to rerank (default: 20, optimization)

        Returns:
            {
                'answer': str,  # Top result text
                'retrieved_passages': List[Dict],
                'confidence': float,
                'pipeline_used': str
            }
        """
        try:
            # Stage 1: Retrieval (retrieve more than rerank_top_k to allow for M1 filtering)
            retrieve_k = max(50, rerank_top_k * 2)
            candidates = self.retriever.search(question, top_k=retrieve_k)

            if not candidates:
                return {
                    'answer': None,
                    'retrieved_passages': [],
                    'confidence': 0.0,
                    'pipeline_used': 'retrieval_only',
                    'error': 'No results found'
                }

            # Stage 2: M1 filtering (if enabled)
            if self.use_m1:
                filtered = []
                for score, doc, stats in candidates:
                    try:
                        doc_text = doc.get('text', '')
                        doc_ast = parse(doc_text)
                        subj, verb, obj = self.extract_svo_triple(doc_ast)

                        if subj and verb and obj:
                            m1_score = self.m1.score_triple(subj, verb, obj)
                            if m1_score >= self.m1_threshold:
                                filtered.append((score, doc, stats, m1_score))
                        else:
                            filtered.append((score, doc, stats, 0.5))
                    except Exception as e:
                        logger.debug(f"M1 scoring failed: {e}")
                        filtered.append((score, doc, stats, 0.5))

                candidates = filtered
            else:
                candidates = [(score, doc, stats, 0.5) for score, doc, stats in candidates]

            if not candidates:
                return {
                    'answer': None,
                    'retrieved_passages': [],
                    'confidence': 0.0,
                    'pipeline_used': 'retrieval+m1',
                    'error': 'All results filtered by M1'
                }

            # Stage 3: Reranking (if enabled)
            if self.use_reranking:
                query_ast = parse(question)

                # Limit candidates to rerank (optimization: only rerank top-K)
                candidates_to_rerank = candidates[:rerank_top_k]

                # Parse all documents once (batch optimization)
                doc_asts = []
                doc_indices = []
                for i, (score, doc, stats, m1_score) in enumerate(candidates_to_rerank):
                    try:
                        doc_text = doc.get('text', '')
                        doc_ast = parse(doc_text)
                        doc_asts.append(doc_ast)
                        doc_indices.append(i)
                    except Exception as e:
                        logger.debug(f"Parsing failed for doc {i}: {e}")

                # Batch reranking (single forward pass for all docs)
                if doc_asts:
                    try:
                        with torch.no_grad():
                            rerank_scores = self.reranker.score_batch(query_ast, doc_asts)
                            rerank_scores = rerank_scores.cpu().numpy()
                    except Exception as e:
                        logger.debug(f"Batch reranking failed: {e}")
                        rerank_scores = [0.0] * len(doc_asts)
                else:
                    rerank_scores = []

                # Combine scores
                reranked = []
                rerank_idx = 0
                for i, (score, doc, stats, m1_score) in enumerate(candidates):
                    if i in doc_indices:
                        rerank_score = float(rerank_scores[rerank_idx])
                        rerank_idx += 1
                    else:
                        rerank_score = 0.0

                    combined_score = 0.2 * score + 0.3 * m1_score + 0.5 * rerank_score
                    reranked.append((combined_score, doc, stats, m1_score, rerank_score))

                reranked.sort(key=lambda x: x[0], reverse=True)
                candidates = reranked
            else:
                candidates = [(score, doc, stats, m1_score, 0.0) for score, doc, stats, m1_score in candidates]

            # Extract top results
            passages = []
            for i, result in enumerate(candidates[:top_k]):
                score = result[0]
                doc = result[1]
                m1_score = result[3] if len(result) > 3 else 0.5
                rerank_score = result[4] if len(result) > 4 else 0.0

                passages.append({
                    'rank': i + 1,
                    'text': doc.get('text', ''),
                    'source': doc.get('source', {}).get('name', 'unknown'),
                    'score': score,
                    'm1_score': m1_score,
                    'rerank_score': rerank_score,
                })

            # Determine pipeline used
            pipeline_parts = ['retrieval']
            if self.use_m1:
                pipeline_parts.append('m1')
            if self.use_reranking:
                pipeline_parts.append('reranker')

            # Stage 4: Answer extraction (if enabled)
            extracted_answer = None
            extraction_confidence = 0.0
            extraction_method = None

            if self.use_extraction and passages:
                try:
                    # Parse query and top document
                    query_ast = parse(question)
                    top_doc_text = passages[0]['text']
                    top_doc_ast = parse(top_doc_text)

                    # Extract answer
                    extraction_result = self.extractor.extract_answer(
                        query_ast, top_doc_ast, top_doc_text
                    )

                    if extraction_result:
                        extracted_answer = extraction_result['text']
                        extraction_confidence = extraction_result['confidence']
                        extraction_method = extraction_result['method']
                        pipeline_parts.append('extraction')
                        logger.info(f"  ✓ Extracted: '{extracted_answer}' ({extraction_method}, conf={extraction_confidence:.2f})")
                    else:
                        logger.info("  ⚠ No answer extracted, returning full document")
                except Exception as e:
                    logger.info(f"  ⚠ Extraction failed: {e}")

            pipeline_used = '+'.join(pipeline_parts)

            # Use extracted answer if available, otherwise fall back to full document
            final_answer = extracted_answer if extracted_answer else (passages[0]['text'] if passages else None)

            # Build pipeline diagnostics (simplified version)
            pipeline_diagnostics = {
                'retrieval': {
                    'status': 'success',
                    'metrics': {
                        'candidates_retrieved': len(passages),
                        'top_document_score': passages[0]['score'] if passages else 0.0,
                    }
                },
                'reranking': {
                    'status': 'success' if self.use_reranking else 'not_applicable',
                    'metrics': {
                        'rerank_score': passages[0].get('rerank_score', 0.0) if passages else 0.0,
                    }
                },
                'extraction': {
                    'status': 'success' if extracted_answer else ('fallback' if passages else 'no_match'),
                    'metrics': {
                        'extraction_confidence': extraction_confidence,
                        'extraction_method': extraction_method,
                        'match_type': 'exact' if extracted_answer else 'document',
                    }
                }
            }

            return {
                'answer': final_answer,
                'extracted_answer': extracted_answer,
                'extraction_confidence': extraction_confidence,
                'extraction_method': extraction_method,
                'retrieved_passages': passages,
                'confidence': passages[0]['score'] if passages else 0.0,
                'pipeline_used': pipeline_used,
                'pipeline_diagnostics': pipeline_diagnostics,
            }

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                'answer': None,
                'retrieved_passages': [],
                'confidence': 0.0,
                'pipeline_used': 'error',
                'error': str(e)
            }


def evaluate_answer(result: Dict, expected: Dict) -> Dict:
    """
    Evaluate RAG answer against expected answer.

    For extracted answers: Uses direct matching and fuzzy keyword matching.
    For full documents: Falls back to keyword-based evaluation.

    Returns:
        {
            'correct': bool,
            'partial': bool,
            'confidence': float,
            'notes': str
        }
    """
    if result.get('error'):
        return {
            'correct': False,
            'partial': False,
            'confidence': 0.0,
            'notes': f"Error: {result['error']}"
        }

    answer = result.get('answer', '')
    if not answer:
        return {
            'correct': False,
            'partial': False,
            'confidence': 0.0,
            'notes': "No answer found"
        }

    # Get expected answer
    expected_answer = expected.get('answer', '').lower()
    expected_keywords = [kw.lower() for kw in expected.get('expected_keywords', [])]

    answer_lower = answer.lower().strip()

    # Check if extraction was used
    extracted_answer = result.get('extracted_answer')
    is_extracted = extracted_answer is not None

    if is_extracted:
        # Evaluation for extracted answers (short spans)

        # 1. Exact match (case-insensitive)
        if answer_lower == expected_answer:
            return {
                'correct': True,
                'partial': False,
                'confidence': result.get('extraction_confidence', result.get('confidence', 0.0)),
                'notes': f"Exact match: '{answer}' == '{expected.get('answer', '')}'"
            }

        # 2. Fuzzy match: answer contains expected or vice versa
        if expected_answer in answer_lower or answer_lower in expected_answer:
            return {
                'correct': True,
                'partial': False,
                'confidence': result.get('extraction_confidence', result.get('confidence', 0.0)),
                'notes': f"Fuzzy match: '{answer}' ~= '{expected.get('answer', '')}'"
            }

        # 3. Keyword match: answer matches any expected keyword
        for keyword in expected_keywords:
            if keyword in answer_lower or answer_lower in keyword:
                return {
                    'correct': False,
                    'partial': True,
                    'confidence': result.get('extraction_confidence', result.get('confidence', 0.0)),
                    'notes': f"Keyword match: '{answer}' contains '{keyword}'"
                }

        # 4. No match
        return {
            'correct': False,
            'partial': False,
            'confidence': result.get('extraction_confidence', result.get('confidence', 0.0)),
            'notes': f"No match: '{answer}' != '{expected.get('answer', '')}' (keywords: {expected_keywords})"
        }

    else:
        # Evaluation for full documents (fallback when extraction fails)

        # Check for expected answer in full document
        if expected_answer and expected_answer in answer_lower:
            return {
                'correct': False,
                'partial': True,
                'confidence': result['confidence'],
                'notes': f"Document contains expected answer '{expected_answer}'"
            }

        # Check for keywords
        keyword_matches = sum(1 for kw in expected_keywords if kw in answer_lower)
        keyword_ratio = keyword_matches / len(expected_keywords) if expected_keywords else 0

        if keyword_ratio >= 0.5:
            return {
                'correct': False,
                'partial': True,
                'confidence': result['confidence'],
                'notes': f"Document contains {keyword_ratio:.0%} keywords ({keyword_matches}/{len(expected_keywords)})"
            }

        return {
            'correct': False,
            'partial': False,
            'confidence': result['confidence'],
            'notes': f"Weak match: {keyword_ratio:.0%} keywords in full document"
        }


def run_evaluation(
    rag_evaluator: RAGEvaluator,
    test_set: List[Dict],
    output_path: Optional[Path] = None
) -> Dict:
    """
    Run evaluation on test set.

    Returns summary statistics.
    """
    results = []
    stats = {
        'total': len(test_set),
        'correct': 0,
        'partial': 0,
        'incorrect': 0,
        'errors': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0}),
        'by_performance': defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0}),
    }

    logger.info(f"Running evaluation on {len(test_set)} questions...")
    logger.info("")

    for i, test_q in enumerate(test_set, 1):
        question_id = test_q['id']
        question = test_q['question']
        category = test_q['category']
        expected_perf = test_q['expected_performance']

        logger.info(f"[{i}/{len(test_set)}] {question_id}: {question}")

        # Run RAG
        try:
            rag_result = rag_evaluator.run_query(question)
            evaluation = evaluate_answer(rag_result, test_q)

            # Check where answer appears in retrieved documents
            passages = rag_result.get('retrieved_passages', [])
            expected_answer = test_q.get('answer', test_q.get('expected_answer_pattern', ''))
            expected_keywords = test_q.get('expected_keywords', [])

            quality_metrics = {
                'answer_present_in_retrieved': False,
                'answer_in_top_1': False,
                'answer_in_top_3': False,
                'answer_in_top_5': False,
                'answer_in_rank': None,
            }

            if passages and expected_answer:
                answer_check = check_answer_in_documents(passages, expected_answer, expected_keywords)
                quality_metrics.update(answer_check)
                quality_metrics['answer_in_top_1'] = answer_check['answer_in_rank'] == 1 if answer_check['answer_in_rank'] else False
                quality_metrics['answer_in_top_3'] = answer_check['answer_in_rank'] and answer_check['answer_in_rank'] <= 3
                quality_metrics['answer_in_top_5'] = answer_check['answer_in_rank'] and answer_check['answer_in_rank'] <= 5

            result = {
                'question_id': question_id,
                'question': question,
                'category': category,
                'expected_performance': expected_perf,
                'rag_answer': rag_result.get('answer'),
                'extracted_answer': rag_result.get('extracted_answer'),
                'extraction_confidence': rag_result.get('extraction_confidence'),
                'extraction_method': rag_result.get('extraction_method'),
                'pipeline_used': rag_result.get('pipeline_used'),
                'pipeline_diagnostics': rag_result.get('pipeline_diagnostics', {}),
                'quality_metrics': quality_metrics,
                'expected_answer_pattern': test_q['expected_answer_pattern'],
                'evaluation': evaluation,
                'retrieved_passages': rag_result.get('retrieved_passages', []),
            }

            # Log compact status line
            symbol = "✓" if evaluation['correct'] else ("⚠" if evaluation['partial'] else "✗")
            status_line = format_status_line(rag_result.get('pipeline_diagnostics', {}), evaluation)
            logger.info(f"{symbol} Q{question_id} | {status_line}")

            # Log top document (show what was selected after all processing)
            if passages:
                top_doc_snippet = passages[0]['text'][:120] + "..." if len(passages[0]['text']) > 120 else passages[0]['text']
                logger.info(f"  └─ Top doc: {top_doc_snippet}")

            # Log expected vs actual
            extracted = rag_result.get('extracted_answer')
            expected = test_q['expected_answer_pattern']
            logger.info(f"  └─ Expected: {expected} | Got: {extracted if extracted else '[full document]'}")

            # Log failure reason if incorrect
            if not evaluation['correct'] and not evaluation['partial']:
                if not quality_metrics['answer_in_retrieved']:
                    logger.info(f"  └─ [FAIL: answer not in retrieved documents]")
                elif not quality_metrics['answer_in_top_1']:
                    rank = quality_metrics.get('answer_in_rank', '?')
                    logger.info(f"  └─ [FAIL: answer in rank #{rank}, not extracted]")
                elif not extracted:
                    logger.info(f"  └─ [FAIL: extraction failed]")
                else:
                    logger.info(f"  └─ [FAIL: extracted wrong answer]")

            # Update stats
            if evaluation['correct']:
                stats['correct'] += 1
                stats['by_category'][category]['correct'] += 1
                stats['by_performance'][expected_perf]['correct'] += 1
            elif evaluation['partial']:
                stats['partial'] += 1
                stats['by_category'][category]['partial'] += 1
                stats['by_performance'][expected_perf]['partial'] += 1
            else:
                stats['incorrect'] += 1

            stats['by_category'][category]['total'] += 1
            stats['by_performance'][expected_perf]['total'] += 1

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            result = {
                'question_id': question_id,
                'question': question,
                'category': category,
                'error': str(e)
            }
            stats['errors'] += 1

        results.append(result)
        logger.info("")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        logger.info(f"✓ Results saved to: {output_path}")

    return stats, results


def load_reranker() -> ASTReranker:
    """Load reranker model."""
    logger.info("Loading reranker...")
    comp_model_path = Path('models/root_embeddings/best_model.pt')
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

    reranker_path = Path('models/reranker/best_model.pt')
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()

    logger.info("  ✓ Reranker loaded")
    return reranker


def print_statistics(stats: Dict, results: List[Dict]):
    """Print comprehensive evaluation statistics with bottleneck analysis."""
    print()
    print("=" * 80)
    print("                           EVALUATION SUMMARY")
    print("=" * 80)
    print()

    total = stats['total']
    correct = stats['correct']
    partial = stats['partial']
    incorrect = stats['incorrect']
    errors = stats['errors']

    accuracy = (correct / total * 100) if total > 0 else 0
    partial_accuracy = ((correct + partial) / total * 100) if total > 0 else 0

    # Overall performance
    print("OVERALL PERFORMANCE:")
    print(f"  Total questions:     {total}")
    print(f"  Correct:             {correct} ({accuracy:.1f}%)   ✓")
    print(f"  Partial:             {partial} ({partial/total*100 if total else 0:.1f}%)   ⚠")
    print(f"  Incorrect:           {incorrect} ({incorrect/total*100 if total else 0:.1f}%)   ✗")
    if errors > 0:
        print(f"  Errors:              {errors}")
    print()
    print(f"  Partial accuracy:    {partial_accuracy:.1f}% (correct + partial)")
    print()
    print()

    # Bottleneck analysis
    print("PIPELINE COMPONENT BOTTLENECKS:")
    print()

    # 1. Retrieval analysis
    retrieval_stats = {
        'answer_in_top_1': 0,
        'answer_in_top_3': 0,
        'answer_in_top_5': 0,
        'answer_in_top_10': 0,
        'answer_not_found': 0,
    }

    for r in results:
        if 'quality_metrics' in r:
            qm = r['quality_metrics']
            if qm.get('answer_in_top_1'):
                retrieval_stats['answer_in_top_1'] += 1
            if qm.get('answer_in_top_3'):
                retrieval_stats['answer_in_top_3'] += 1
            if qm.get('answer_in_top_5'):
                retrieval_stats['answer_in_top_5'] += 1
            if qm.get('answer_in_rank') and qm['answer_in_rank'] <= 10:
                retrieval_stats['answer_in_top_10'] += 1
            if not qm.get('answer_in_retrieved'):
                retrieval_stats['answer_not_found'] += 1

    top_10_pct = (retrieval_stats['answer_in_top_10'] / total * 100) if total > 0 else 0
    print(f"  1. RETRIEVAL: {top_10_pct:.0f}% healthy [{retrieval_stats['answer_in_top_10']}/{total} have answer in top-10]")
    print(f"     ├─ Answer in top-1:  {retrieval_stats['answer_in_top_1']}/{total} ({retrieval_stats['answer_in_top_1']/total*100 if total else 0:.0f}%)")
    print(f"     ├─ Answer in top-3:  {retrieval_stats['answer_in_top_3']}/{total} ({retrieval_stats['answer_in_top_3']/total*100 if total else 0:.0f}%)")
    print(f"     ├─ Answer in top-5:  {retrieval_stats['answer_in_top_5']}/{total} ({retrieval_stats['answer_in_top_5']/total*100 if total else 0:.0f}%)")
    print(f"     ├─ Answer in top-10: {retrieval_stats['answer_in_top_10']}/{total} ({top_10_pct:.0f}%)")
    print(f"     └─ Failures: {retrieval_stats['answer_not_found']} questions ({retrieval_stats['answer_not_found']/total*100 if total else 0:.0f}%) - answer not retrievable")
    print()

    # 2. Extraction analysis
    extraction_stats = {
        'exact_match': 0,
        'fuzzy_match': 0,
        'fallback_to_fulltext': 0,
        'no_extraction': 0,
    }

    for r in results:
        if r.get('extracted_answer'):
            if r.get('evaluation', {}).get('correct'):
                extraction_stats['exact_match'] += 1
            else:
                extraction_stats['fuzzy_match'] += 1
        elif r.get('rag_answer'):
            extraction_stats['fallback_to_fulltext'] += 1
        else:
            extraction_stats['no_extraction'] += 1

    successful_extractions = extraction_stats['exact_match'] + extraction_stats['fuzzy_match']
    extraction_pct = (successful_extractions / total * 100) if total > 0 else 0

    print(f"  2. EXTRACTION: {extraction_pct:.0f}% healthy [{successful_extractions}/{total} successful extractions]")
    print(f"     ├─ Exact match:          {extraction_stats['exact_match']}/{total} ({extraction_stats['exact_match']/total*100 if total else 0:.0f}%)")
    print(f"     ├─ Fuzzy match:          {extraction_stats['fuzzy_match']}/{total} ({extraction_stats['fuzzy_match']/total*100 if total else 0:.0f}%)")
    print(f"     ├─ Fallback to fulltext: {extraction_stats['fallback_to_fulltext']}/{total} ({extraction_stats['fallback_to_fulltext']/total*100 if total else 0:.0f}%)")
    print(f"     └─ No extraction:        {extraction_stats['no_extraction']}/{total} ({extraction_stats['no_extraction']/total*100 if total else 0:.0f}%)  - BOTTLENECK")
    print()

    # Failure analysis
    print()
    print(f"FAILURE ANALYSIS ({incorrect} Incorrect Questions):")
    print()

    if incorrect > 0:
        failure_causes = {
            'retrieval_failure': [],
            'extraction_failed': [],
            'wrong_extraction': [],
            'ranking_issue': [],
        }

        for r in results:
            if r.get('evaluation', {}).get('correct') or r.get('evaluation', {}).get('partial'):
                continue  # Skip correct/partial

            qm = r.get('quality_metrics', {})
            qid = r.get('question_id', '?')

            if not qm.get('answer_in_retrieved'):
                failure_causes['retrieval_failure'].append(qid)
            elif qm.get('answer_in_top_1') and not r.get('extracted_answer'):
                failure_causes['extraction_failed'].append(qid)
            elif r.get('extracted_answer'):
                failure_causes['wrong_extraction'].append(qid)
            elif qm.get('answer_in_retrieved'):
                failure_causes['ranking_issue'].append(qid)

        print("  Root Cause:")
        for cause, qids in failure_causes.items():
            if qids:
                pct = len(qids) / incorrect * 100 if incorrect > 0 else 0
                cause_name = cause.replace('_', ' ').title()
                print(f"    {cause_name}: {len(qids)} ({pct:.1f}%)")
                if len(qids) <= 5:
                    print(f"      Questions: {', '.join(map(str, qids))}")
        print()

    # By category breakdown
    print()
    print("BY CATEGORY BREAKDOWN:")
    print()
    for category, cat_stats in sorted(stats['by_category'].items()):
        cat_total = cat_stats['total']
        cat_correct = cat_stats['correct']
        cat_partial = cat_stats['partial']
        cat_incorrect = cat_total - cat_correct - cat_partial
        cat_acc = (cat_correct / cat_total * 100) if cat_total > 0 else 0

        symbol = "✓" if cat_acc >= 70 else ("⚠" if cat_acc >= 40 else "✗")
        print(f"  {category} ({cat_total} questions):")
        print(f"    {symbol} {cat_correct}/{cat_total} correct ({cat_acc:.0f}%)")
        if cat_partial > 0:
            print(f"    ⚠ {cat_partial}/{cat_total} partial")
        if cat_incorrect > 0:
            print(f"    ✗ {cat_incorrect}/{cat_total} incorrect")
    print()

    # By expected performance
    print("BY EXPECTED PERFORMANCE:")
    print()
    for perf, perf_stats in sorted(stats['by_performance'].items()):
        perf_total = perf_stats['total']
        perf_correct = perf_stats['correct']
        perf_partial = perf_stats['partial']
        perf_acc = (perf_correct / perf_total * 100) if perf_total > 0 else 0
        symbol = {"works": "✅", "partial": "⚠️", "fails": "❌"}.get(perf, "?")

        print(f"  Expected: '{perf}' ({perf_total} questions)")
        print(f"    {symbol} Achieved {perf_correct}/{perf_total} correct ({perf_acc:.0f}%)")
        if perf_partial > 0:
            print(f"    ⚠ {perf_partial}/{perf_total} partial")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system against test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        default=Path('data/evaluation/rag_test_set.jsonl'),
        help='Path to test set'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/evaluation/rag_results.jsonl'),
        help='Output path for results'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/indexes/kuzu_index',
        help='Path to Kuzu index'
    )
    parser.add_argument(
        '--m1-model',
        type=str,
        default='models/m1_semantic_tier_priority/best_model.pt',
        help='Path to M1 model'
    )
    parser.add_argument(
        '--stage1-model',
        type=str,
        default='models/root_embeddings_tier0/best_model.pt',
        help='Path to Stage 1 embeddings'
    )
    parser.add_argument(
        '--no-m1',
        action='store_true',
        help='Disable M1 plausibility filtering'
    )
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable reranking'
    )
    parser.add_argument(
        '--no-extraction',
        action='store_true',
        help='Disable answer extraction (return full document instead)'
    )
    parser.add_argument(
        '--category',
        action='append',
        help='Filter by category (can specify multiple times)'
    )
    parser.add_argument(
        '--expected',
        choices=['works', 'partial', 'fails'],
        help='Filter by expected performance'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed pipeline logs (entity boost, quality filter details)'
    )

    args = parser.parse_args()

    # Configure logging verbosity
    if not args.verbose:
        # Suppress verbose retriever logs
        logging.getLogger('klareco.rag.ast_aware_retriever').setLevel(logging.WARNING)
        logging.getLogger('klareco.rag.kuzu_inverted_index').setLevel(logging.WARNING)

    logger.info("=" * 80)
    logger.info("RAG System Evaluation")
    logger.info("=" * 80)

    # Load test set
    logger.info("Loading test set...")
    test_set = load_test_set(
        args.test_set,
        categories=args.category,
        expected_performance=args.expected
    )

    if not test_set:
        logger.error("No questions match filters")
        return

    logger.info(f"Loaded {len(test_set)} questions")

    # Load RAG components
    logger.info("\nLoading RAG components...")

    # Load retriever
    logger.info("  Loading retriever...")
    index_path = Path(args.index_dir)
    retriever = ASTAwareRetriever(index_path=index_path)
    logger.info("    ✓ Retriever loaded")

    # Load M1 (optional)
    m1 = None
    if not args.no_m1:
        m1_path = Path(args.m1_model)
        stage1_path = Path(args.stage1_model)

        if m1_path.exists() and stage1_path.exists():
            logger.info("  Loading M1 model...")
            try:
                m1 = M1Inference(
                    model_path=m1_path,
                    stage1_path=stage1_path,
                    device='cpu'
                )
                logger.info("    ✓ M1 loaded")
            except Exception as e:
                logger.warning(f"    ⚠ M1 loading failed: {e}")
                logger.warning("    Continuing without M1")
        else:
            logger.warning(f"  ⚠ M1 model not found, continuing without M1")

    # Load reranker (optional)
    reranker = None
    if not args.no_rerank:
        reranker_path = Path('models/reranker/best_model.pt')
        if reranker_path.exists():
            try:
                reranker = load_reranker()
            except Exception as e:
                logger.warning(f"  ⚠ Reranker loading failed: {e}")
                logger.warning("  Continuing without reranker")
        else:
            logger.warning(f"  ⚠ Reranker not found, continuing without reranker")

    # Create RAG evaluator
    pipeline_parts = ['retrieval']
    if m1 and not args.no_m1:
        pipeline_parts.append('M1')
    if reranker and not args.no_rerank:
        pipeline_parts.append('reranker')
    if not args.no_extraction:
        pipeline_parts.append('extraction')

    logger.info(f"\nPipeline configuration: {' → '.join(pipeline_parts)}")

    rag_evaluator = RAGEvaluator(
        retriever=retriever,
        reranker=reranker,
        m1=m1,
        use_m1=not args.no_m1,
        use_reranking=not args.no_rerank,
        use_extraction=not args.no_extraction,
    )

    # Run evaluation
    stats, results = run_evaluation(rag_evaluator, test_set, args.output)

    # Print statistics
    print_statistics(stats, results)

    retriever.close()


if __name__ == '__main__':
    main()
