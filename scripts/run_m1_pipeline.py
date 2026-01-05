#!/usr/bin/env python3
"""
M1 Full Pipeline (CP7): End-to-end Q&A with full explainability.

This script implements the complete M1 pipeline:
    Question → Parser → AST → Retriever → Reranker → Extractor → Answer

Key features:
- Full AST trace for explainability
- Deterministic grammar via parser/deparser
- Zero learned parameters in extraction pipeline
- Comparison against OLMo baseline

Usage:
    python scripts/run_m1_pipeline.py                      # Run full benchmark
    python scripts/run_m1_pipeline.py -i                   # Interactive mode
    python scripts/run_m1_pipeline.py --question "Kiu..."  # Single question
    python scripts/run_m1_pipeline.py --compare-olmo       # Compare to OLMo
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PipelineTrace:
    """Full trace of pipeline execution for explainability."""
    question: str
    question_ast: Optional[Dict] = None
    question_type: str = ""
    key_terms: List[str] = field(default_factory=list)

    # Retrieval stage
    retrieved_count: int = 0
    top_retrieved: List[str] = field(default_factory=list)

    # Reranking stage
    rerank_scores: List[Dict] = field(default_factory=list)

    # Extraction stage
    extraction_method: str = ""
    extraction_confidence: float = 0.0
    source_sentence: str = ""

    # Final answer
    answer: str = ""

    # Timing
    parse_time_ms: float = 0.0
    retrieve_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    extract_time_ms: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class M1Result:
    """Result of M1 pipeline on a single question."""
    question_id: str
    question: str
    category: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    partial_match: bool
    f1_score: float
    trace: PipelineTrace
    latency_ms: float


@dataclass
class M1EvaluationResults:
    """Aggregated M1 evaluation results."""
    total_questions: int = 0
    exact_match: float = 0.0
    partial_match: float = 0.0
    f1_score: float = 0.0
    avg_latency_ms: float = 0.0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[M1Result] = field(default_factory=list)


class M1Pipeline:
    """
    Full M1 Q&A Pipeline with explainability.

    Components:
    - Parser: Text → AST (0 params, deterministic)
    - Retriever: Semantic search (733K params in embeddings)
    - Reranker: AST-based scoring (0 params, deterministic)
    - Extractor: Pattern-based extraction (0 params, deterministic)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._parser = None
        self._retriever = None
        self._reranker = None
        self._extractor = None

    def _load_components(self):
        """Lazy load pipeline components."""
        if self._parser is None:
            if self.verbose:
                print("Loading pipeline components...")

            from klareco.parser import parse as parser_parse
            from klareco import Retriever
            from klareco.qa import DeterministicReranker, AnswerExtractor

            self._parser = parser_parse
            self._retriever = Retriever.load()
            self._reranker = DeterministicReranker()
            self._extractor = AnswerExtractor()

            if self.verbose:
                print("  - Parser: loaded")
                print("  - Retriever: loaded")
                print("  - Reranker: loaded (0 params)")
                print("  - Extractor: loaded (0 params)")

    def answer(
        self,
        question: str,
        top_k: int = 10,
        trace: bool = True
    ) -> tuple:
        """
        Answer a question using the full pipeline.

        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            trace: Whether to generate full trace

        Returns:
            tuple: (answer, PipelineTrace or None)
        """
        self._load_components()

        pipeline_trace = PipelineTrace(question=question) if trace else None
        total_start = time.time()

        # Stage 1: Parse question
        parse_start = time.time()
        try:
            question_ast = self._parser(question)
            if pipeline_trace:
                pipeline_trace.question_ast = question_ast
        except Exception as e:
            if self.verbose:
                print(f"  Parse error: {e}")
            question_ast = None
        parse_time = (time.time() - parse_start) * 1000
        if pipeline_trace:
            pipeline_trace.parse_time_ms = parse_time

        # Stage 2: Analyze question
        question_type, key_terms = self._extractor.analyze_question(question)
        if pipeline_trace:
            pipeline_trace.question_type = question_type
            pipeline_trace.key_terms = key_terms

        # Stage 3: Retrieve documents
        retrieve_start = time.time()
        results = self._retriever.search(question, top_k=top_k * 2)
        retrieved_docs = [r.text for r in results]
        original_scores = [getattr(r, 'score', 1.0 - i/len(results))
                          for i, r in enumerate(results)]
        retrieve_time = (time.time() - retrieve_start) * 1000

        if pipeline_trace:
            pipeline_trace.retrieve_time_ms = retrieve_time
            pipeline_trace.retrieved_count = len(retrieved_docs)
            pipeline_trace.top_retrieved = retrieved_docs[:3]

        # Stage 4: Rerank documents
        rerank_start = time.time()
        if retrieved_docs:
            reranked = self._reranker.rerank(
                question,
                retrieved_docs,
                original_scores=original_scores,
                top_k=top_k
            )
            reranked_docs = [r.text for r in reranked]
            if pipeline_trace:
                pipeline_trace.rerank_scores = [
                    {
                        'text': r.text[:100] + '...' if len(r.text) > 100 else r.text,
                        'original': r.original_score,
                        'rerank': r.rerank_score,
                        'combined': r.combined_score,
                        'features': r.features,
                    }
                    for r in reranked[:5]
                ]
        else:
            reranked_docs = []
        rerank_time = (time.time() - rerank_start) * 1000
        if pipeline_trace:
            pipeline_trace.rerank_time_ms = rerank_time

        # Stage 5: Extract answer
        extract_start = time.time()
        if reranked_docs:
            extraction = self._extractor.extract(question, reranked_docs)
            answer = extraction.answer
            if pipeline_trace:
                pipeline_trace.extraction_method = extraction.method
                pipeline_trace.extraction_confidence = extraction.confidence
                pipeline_trace.source_sentence = extraction.source_sentence
        else:
            answer = "Mi ne scias."
            if pipeline_trace:
                pipeline_trace.extraction_method = 'no_docs'
                pipeline_trace.extraction_confidence = 0.0
        extract_time = (time.time() - extract_start) * 1000
        if pipeline_trace:
            pipeline_trace.extract_time_ms = extract_time

        # Finalize
        total_time = (time.time() - total_start) * 1000
        if pipeline_trace:
            pipeline_trace.answer = answer
            pipeline_trace.total_time_ms = total_time

        return answer, pipeline_trace

    def explain(self, trace: PipelineTrace) -> str:
        """Generate human-readable explanation from trace."""
        lines = []
        lines.append("=" * 60)
        lines.append("M1 PIPELINE EXPLANATION")
        lines.append("=" * 60)

        lines.append(f"\n1. QUESTION ANALYSIS")
        lines.append(f"   Question: {trace.question}")
        lines.append(f"   Type: {trace.question_type}")
        lines.append(f"   Key terms: {', '.join(trace.key_terms)}")
        lines.append(f"   Parse time: {trace.parse_time_ms:.1f}ms")

        lines.append(f"\n2. RETRIEVAL")
        lines.append(f"   Documents retrieved: {trace.retrieved_count}")
        lines.append(f"   Retrieval time: {trace.retrieve_time_ms:.1f}ms")
        if trace.top_retrieved:
            lines.append(f"   Top match preview:")
            preview = trace.top_retrieved[0][:100]
            lines.append(f"     \"{preview}...\"")

        lines.append(f"\n3. RERANKING")
        lines.append(f"   Rerank time: {trace.rerank_time_ms:.1f}ms")
        if trace.rerank_scores:
            lines.append(f"   Top reranked document:")
            top = trace.rerank_scores[0]
            lines.append(f"     Original score: {top['original']:.3f}")
            lines.append(f"     Rerank score: {top['rerank']:.3f}")
            lines.append(f"     Combined: {top['combined']:.3f}")
            lines.append(f"     Features: {top['features']}")

        lines.append(f"\n4. EXTRACTION")
        lines.append(f"   Method: {trace.extraction_method}")
        lines.append(f"   Confidence: {trace.extraction_confidence:.2f}")
        lines.append(f"   Extract time: {trace.extract_time_ms:.1f}ms")
        if trace.source_sentence:
            source_preview = trace.source_sentence[:150]
            lines.append(f"   Source: \"{source_preview}...\"")

        lines.append(f"\n5. ANSWER")
        lines.append(f"   {trace.answer}")

        lines.append(f"\n6. TIMING")
        lines.append(f"   Total: {trace.total_time_ms:.1f}ms")
        lines.append(f"   Parse: {trace.parse_time_ms:.1f}ms ({100*trace.parse_time_ms/trace.total_time_ms:.0f}%)")
        lines.append(f"   Retrieve: {trace.retrieve_time_ms:.1f}ms ({100*trace.retrieve_time_ms/trace.total_time_ms:.0f}%)")
        lines.append(f"   Rerank: {trace.rerank_time_ms:.1f}ms ({100*trace.rerank_time_ms/trace.total_time_ms:.0f}%)")
        lines.append(f"   Extract: {trace.extract_time_ms:.1f}ms ({100*trace.extract_time_ms/trace.total_time_ms:.0f}%)")

        return '\n'.join(lines)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\sĉĝĥĵŝŭ]', '', text)
    text = ' '.join(text.split())
    return text


def compute_f1(prediction: str, reference: str) -> float:
    """Compute F1 score between prediction and reference."""
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def check_exact_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check exact match."""
    pred_norm = normalize_text(prediction)
    if pred_norm == normalize_text(gold):
        return True
    for acc in acceptable:
        if pred_norm == normalize_text(acc):
            return True
    return False


def check_partial_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check partial match."""
    pred_norm = normalize_text(prediction)
    if normalize_text(gold) in pred_norm:
        return True
    for acc in acceptable:
        if normalize_text(acc) in pred_norm:
            return True
    return False


def load_benchmark(benchmark_path: Path) -> List[Dict]:
    """Load benchmark questions."""
    questions = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def evaluate_benchmark(
    pipeline: M1Pipeline,
    questions: List[Dict],
    verbose: bool = False
) -> M1EvaluationResults:
    """Run full benchmark evaluation."""
    from collections import defaultdict

    results = M1EvaluationResults()
    results.total_questions = len(questions)

    category_counts = defaultdict(int)
    category_exact = defaultdict(int)
    category_partial = defaultdict(int)
    category_f1 = defaultdict(float)

    total_exact = 0
    total_partial = 0
    total_f1 = 0.0
    total_latency = 0.0

    for i, q in enumerate(questions):
        if verbose:
            print(f"\r[{i+1}/{len(questions)}] {q['id']}: {q['question'][:50]}...", end='', flush=True)

        answer, trace = pipeline.answer(q['question'])

        gold = q['gold_answer']
        acceptable = q.get('acceptable_answers', [])

        exact = check_exact_match(answer, gold, acceptable)
        partial = check_partial_match(answer, gold, acceptable)
        f1 = compute_f1(answer, gold)

        result = M1Result(
            question_id=q['id'],
            question=q['question'],
            category=q['category'],
            gold_answer=gold,
            predicted_answer=answer,
            exact_match=exact,
            partial_match=partial,
            f1_score=f1,
            trace=trace,
            latency_ms=trace.total_time_ms if trace else 0.0
        )
        results.results.append(result)

        # Aggregate
        cat = q['category']
        category_counts[cat] += 1
        if exact:
            total_exact += 1
            category_exact[cat] += 1
        if partial:
            total_partial += 1
            category_partial[cat] += 1
        total_f1 += f1
        category_f1[cat] += f1
        total_latency += result.latency_ms

    if verbose:
        print()

    # Compute averages
    n = results.total_questions
    results.exact_match = total_exact / n if n > 0 else 0
    results.partial_match = total_partial / n if n > 0 else 0
    results.f1_score = total_f1 / n if n > 0 else 0
    results.avg_latency_ms = total_latency / n if n > 0 else 0

    for cat in category_counts:
        cat_n = category_counts[cat]
        results.by_category[cat] = {
            'count': cat_n,
            'exact_match': category_exact[cat] / cat_n if cat_n > 0 else 0,
            'partial_match': category_partial[cat] / cat_n if cat_n > 0 else 0,
            'f1_score': category_f1[cat] / cat_n if cat_n > 0 else 0,
        }

    return results


def print_results(results: M1EvaluationResults, compare_olmo: bool = False):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("M1 PIPELINE EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal Questions: {results.total_questions}")
    print(f"\nOverall Metrics:")
    print(f"  Exact Match:   {results.exact_match:.1%}")
    print(f"  Partial Match: {results.partial_match:.1%}")
    print(f"  F1 Score:      {results.f1_score:.3f}")
    print(f"  Avg Latency:   {results.avg_latency_ms:.0f}ms")

    print(f"\nBy Category:")
    print(f"  {'Category':<15} {'Count':>6} {'Exact':>8} {'Partial':>8} {'F1':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for cat, metrics in sorted(results.by_category.items()):
        print(f"  {cat:<15} {metrics['count']:>6} "
              f"{metrics['exact_match']:>7.1%} "
              f"{metrics['partial_match']:>7.1%} "
              f"{metrics['f1_score']:>7.3f}")

    if compare_olmo:
        print_olmo_comparison(results)


def print_olmo_comparison(results: M1EvaluationResults):
    """Print comparison with OLMo results if available."""
    olmo_path = PROJECT_ROOT / 'data' / 'benchmarks' / 'results_olmo.json'
    if not olmo_path.exists():
        print("\n[OLMo results not found - run OLMo baseline first]")
        return

    with open(olmo_path) as f:
        olmo = json.load(f)

    print("\n" + "=" * 60)
    print("M1 vs OLMo 1B COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'M1 Pipeline':>15} {'OLMo 1B':>15} {'Winner':>12}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*12}")

    # Exact match
    m1_exact = results.exact_match
    olmo_exact = olmo.get('exact_match', 0)
    winner = "M1" if m1_exact > olmo_exact else ("OLMo" if olmo_exact > m1_exact else "Tie")
    print(f"{'Exact Match':<20} {m1_exact:>14.1%} {olmo_exact:>14.1%} {winner:>12}")

    # Partial match
    m1_partial = results.partial_match
    olmo_partial = olmo.get('partial_match', 0)
    winner = "M1" if m1_partial > olmo_partial else ("OLMo" if olmo_partial > m1_partial else "Tie")
    print(f"{'Partial Match':<20} {m1_partial:>14.1%} {olmo_partial:>14.1%} {winner:>12}")

    # F1
    m1_f1 = results.f1_score
    olmo_f1 = olmo.get('f1_score', 0)
    winner = "M1" if m1_f1 > olmo_f1 else ("OLMo" if olmo_f1 > m1_f1 else "Tie")
    print(f"{'F1 Score':<20} {m1_f1:>14.3f} {olmo_f1:>14.3f} {winner:>12}")

    # Latency
    m1_lat = results.avg_latency_ms
    olmo_lat = olmo.get('avg_latency_ms', 0)
    speedup = olmo_lat / m1_lat if m1_lat > 0 else 0
    print(f"{'Latency (ms)':<20} {m1_lat:>14.0f} {olmo_lat:>14.0f} {'M1 ('+f'{speedup:.0f}x)':>12}")

    # Parameters (M1 uses ~733K, OLMo uses 1.18B)
    m1_params = 733_000
    olmo_params = 1_180_000_000
    ratio = olmo_params / m1_params
    print(f"{'Parameters':<20} {m1_params:>14,} {olmo_params:>14,} {'M1 ('+f'{ratio:.0f}x)':>12}")

    print("\n" + "-" * 60)
    print("Key Insights:")
    print(f"  - M1 is {speedup:.0f}x faster than OLMo")
    print(f"  - M1 uses {ratio:.0f}x fewer parameters")
    print(f"  - M1 provides full explainability (AST trace)")
    print(f"  - M1 grammar is 100% deterministic (parser-based)")


def save_results(results: M1EvaluationResults, output_path: Path):
    """Save results to JSON."""
    output = {
        'total_questions': results.total_questions,
        'exact_match': results.exact_match,
        'partial_match': results.partial_match,
        'f1_score': results.f1_score,
        'avg_latency_ms': results.avg_latency_ms,
        'by_category': results.by_category,
        'results': [
            {
                'question_id': r.question_id,
                'question': r.question,
                'category': r.category,
                'gold_answer': r.gold_answer,
                'predicted_answer': r.predicted_answer,
                'exact_match': r.exact_match,
                'partial_match': r.partial_match,
                'f1_score': r.f1_score,
                'latency_ms': r.latency_ms,
                'trace': {
                    'question_type': r.trace.question_type,
                    'key_terms': r.trace.key_terms,
                    'extraction_method': r.trace.extraction_method,
                    'extraction_confidence': r.trace.extraction_confidence,
                } if r.trace else None
            }
            for r in results.results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def interactive_mode(pipeline: M1Pipeline):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("M1 INTERACTIVE Q&A")
    print("=" * 60)
    print("Enter questions in Esperanto. Type 'quit' to exit.")
    print("Add '!' after question for full explanation.")
    print()

    while True:
        try:
            question = input("Demando: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĜis revido!")
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print("Ĝis revido!")
            break

        # Check for explanation request
        explain = question.endswith('!')
        if explain:
            question = question[:-1].strip()

        answer, trace = pipeline.answer(question)

        print(f"\nRespondo: {answer}")
        print(f"[{trace.extraction_method}, confidence={trace.extraction_confidence:.2f}, {trace.total_time_ms:.0f}ms]")

        if explain:
            print(pipeline.explain(trace))

        print()


def main():
    parser = argparse.ArgumentParser(description='M1 Full Pipeline (CP7)')
    parser.add_argument('--benchmark', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'qa_benchmark_v1.jsonl',
                        help='Path to benchmark JSONL file')
    parser.add_argument('--question', '-q', type=str, default=None,
                        help='Single question to answer')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive Q&A mode')
    parser.add_argument('--compare-olmo', action='store_true',
                        help='Compare results with OLMo baseline')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--explain', '-e', action='store_true',
                        help='Show full explanation for single question')
    parser.add_argument('--category', type=str, default=None,
                        help='Only evaluate specific category')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = M1Pipeline(verbose=args.verbose)

    # Single question mode
    if args.question:
        print(f"\nQuestion: {args.question}")
        answer, trace = pipeline.answer(args.question)
        print(f"\nAnswer: {answer}")
        print(f"[{trace.extraction_method}, confidence={trace.extraction_confidence:.2f}, {trace.total_time_ms:.0f}ms]")

        if args.explain:
            print(pipeline.explain(trace))
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(pipeline)
        return

    # Benchmark mode
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    questions = load_benchmark(args.benchmark)
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    # Filter by category if specified
    if args.category:
        questions = [q for q in questions if q['category'] == args.category]
        print(f"Filtered to {len(questions)} questions in category '{args.category}'")

    # Run evaluation
    print("\nRunning M1 Pipeline evaluation...")
    results = evaluate_benchmark(pipeline, questions, verbose=args.verbose)

    # Print results
    print_results(results, compare_olmo=args.compare_olmo)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        default_output = PROJECT_ROOT / 'data' / 'benchmarks' / 'results_m1_pipeline.json'
        save_results(results, default_output)


if __name__ == '__main__':
    main()
