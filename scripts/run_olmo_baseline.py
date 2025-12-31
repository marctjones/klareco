#!/usr/bin/env python3
"""
OLMo 1B Baseline Evaluation Script (CP3)

Evaluates OLMo 1B Instruct on the M1 Q&A benchmark for comparison with Klareco.
Uses the same evaluation metrics as evaluate_qa.py.

Usage:
    python scripts/run_olmo_baseline.py                    # Run full evaluation
    python scripts/run_olmo_baseline.py --dry-run          # Show questions only
    python scripts/run_olmo_baseline.py --limit 5          # Test with 5 questions
    python scripts/run_olmo_baseline.py --device cpu       # Force CPU mode
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class OLMoResult:
    """Result of OLMo answering a single question."""
    question_id: str
    question: str
    category: str
    gold_answer: str
    acceptable_answers: List[str]
    predicted_answer: str
    exact_match: bool = False
    partial_match: bool = False
    f1_score: float = 0.0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    error: Optional[str] = None


@dataclass
class OLMoEvaluationResults:
    """Aggregated OLMo evaluation results."""
    model_name: str = ""
    total_questions: int = 0
    exact_match: float = 0.0
    partial_match: float = 0.0
    f1_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    memory_gb: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    question_results: List[OLMoResult] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove punctuation except Esperanto special chars
    text = re.sub(r'[^\w\sĉĝĥĵŝŭ]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> set:
    """Tokenize text into words."""
    return set(normalize_text(text).split())


def compute_f1(prediction: str, reference: str) -> float:
    """Compute F1 score between prediction and reference."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def check_exact_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check if prediction exactly matches gold or any acceptable answer."""
    pred_norm = normalize_text(prediction)

    if pred_norm == normalize_text(gold):
        return True

    for acc in acceptable:
        if pred_norm == normalize_text(acc):
            return True

    return False


def check_partial_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check if prediction contains any acceptable answer."""
    pred_norm = normalize_text(prediction)

    if normalize_text(gold) in pred_norm:
        return True

    for acc in acceptable:
        if normalize_text(acc) in pred_norm:
            return True

    return False


def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    """Load the Q&A benchmark from JSONL file."""
    questions = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


class OLMoRunner:
    """Wrapper for running OLMo inference."""

    def __init__(self, model_name: str = "allenai/OLMo-1B-hf", device: str = "auto"):
        """
        Initialize OLMo model.

        Args:
            model_name: HuggingFace model identifier
            device: "auto", "cuda", or "cpu"
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._memory_before = 0

    def load(self):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")

        # Track memory before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._memory_before = torch.cuda.memory_allocated()

        # Determine device
        if self.device == "auto":
            device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            device_map = self.device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model with appropriate settings
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {device_map}")

    def get_memory_usage(self) -> float:
        """Get peak GPU memory usage in GB."""
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            return peak / (1024 ** 3)
        return 0.0

    def format_prompt(self, question: str) -> str:
        """Format question as prompt for OLMo."""
        # Simple Esperanto Q&A prompt
        prompt = f"""Respondu la demandon koncize en Esperanto.

Demando: {question}
Respondo:"""
        return prompt

    def generate(self, question: str, max_new_tokens: int = 100) -> tuple:
        """
        Generate answer for a question.

        Returns:
            tuple: (answer_text, num_tokens_generated)
        """
        import torch

        prompt = self.format_prompt(question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to same device as model
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up answer
        answer = answer.strip()
        # Stop at first newline or question mark (end of answer)
        for stop in ['\n', '?', 'Demando:']:
            if stop in answer:
                answer = answer.split(stop)[0].strip()

        return answer, len(generated_ids)


def evaluate_with_olmo(
    questions: List[Dict[str, Any]],
    runner: OLMoRunner,
    verbose: bool = False
) -> OLMoEvaluationResults:
    """Evaluate all questions with OLMo."""
    results = OLMoEvaluationResults()
    results.model_name = runner.model_name
    results.total_questions = len(questions)

    # Category accumulators
    category_counts = defaultdict(int)
    category_exact = defaultdict(int)
    category_partial = defaultdict(int)
    category_f1 = defaultdict(float)

    total_latency = 0.0
    total_exact = 0
    total_partial = 0
    total_f1 = 0.0
    total_tokens = 0

    for i, question in enumerate(questions):
        if verbose:
            print(f"\r[{i+1}/{len(questions)}] Evaluating: {question['id']}...", end='', flush=True)

        q_id = question['id']
        q_text = question['question']
        gold = question['gold_answer']
        acceptable = question.get('acceptable_answers', [])
        category = question['category']

        # Time the generation
        start_time = time.time()
        try:
            predicted, tokens = runner.generate(q_text)
        except Exception as e:
            result = OLMoResult(
                question_id=q_id,
                question=q_text,
                category=category,
                gold_answer=gold,
                acceptable_answers=acceptable,
                predicted_answer="",
                error=str(e)
            )
            results.question_results.append(result)
            continue

        latency_ms = (time.time() - start_time) * 1000

        # Compute metrics
        exact = check_exact_match(predicted, gold, acceptable)
        partial = check_partial_match(predicted, gold, acceptable)
        f1 = compute_f1(predicted, gold)

        result = OLMoResult(
            question_id=q_id,
            question=q_text,
            category=category,
            gold_answer=gold,
            acceptable_answers=acceptable,
            predicted_answer=predicted,
            exact_match=exact,
            partial_match=partial,
            f1_score=f1,
            latency_ms=latency_ms,
            tokens_generated=tokens
        )
        results.question_results.append(result)

        # Aggregate
        category_counts[category] += 1
        if exact:
            total_exact += 1
            category_exact[category] += 1
        if partial:
            total_partial += 1
            category_partial[category] += 1

        total_f1 += f1
        category_f1[category] += f1
        total_latency += latency_ms
        total_tokens += tokens

    if verbose:
        print()  # Newline after progress

    # Compute final metrics
    n = results.total_questions
    results.exact_match = total_exact / n if n > 0 else 0
    results.partial_match = total_partial / n if n > 0 else 0
    results.f1_score = total_f1 / n if n > 0 else 0
    results.avg_latency_ms = total_latency / n if n > 0 else 0
    results.total_tokens = total_tokens
    results.memory_gb = runner.get_memory_usage()

    # By-category metrics
    for cat in category_counts:
        cat_n = category_counts[cat]
        results.by_category[cat] = {
            'count': cat_n,
            'exact_match': category_exact[cat] / cat_n if cat_n > 0 else 0,
            'partial_match': category_partial[cat] / cat_n if cat_n > 0 else 0,
            'f1_score': category_f1[cat] / cat_n if cat_n > 0 else 0,
        }

    return results


def print_results(results: OLMoEvaluationResults, show_errors: bool = False):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("OLMO 1B BASELINE EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nModel: {results.model_name}")
    print(f"Total Questions: {results.total_questions}")

    print(f"\nOverall Metrics:")
    print(f"  Exact Match:   {results.exact_match:.1%}")
    print(f"  Partial Match: {results.partial_match:.1%}")
    print(f"  F1 Score:      {results.f1_score:.3f}")
    print(f"  Avg Latency:   {results.avg_latency_ms:.0f}ms")
    print(f"  Total Tokens:  {results.total_tokens}")
    if results.memory_gb > 0:
        print(f"  Peak Memory:   {results.memory_gb:.2f} GB")

    print(f"\nBy Category:")
    print(f"  {'Category':<15} {'Count':>6} {'Exact':>8} {'Partial':>8} {'F1':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for cat, metrics in sorted(results.by_category.items()):
        print(f"  {cat:<15} {metrics['count']:>6} "
              f"{metrics['exact_match']:>7.1%} "
              f"{metrics['partial_match']:>7.1%} "
              f"{metrics['f1_score']:>7.3f}")

    # Show errors
    errors = [r for r in results.question_results if r.error]
    if errors:
        print(f"\nErrors: {len(errors)}")
        if show_errors:
            for r in errors:
                print(f"  {r.question_id}: {r.error}")

    # Show sample answers
    if show_errors:
        print(f"\nSample Answers (first 5):")
        for r in results.question_results[:5]:
            if not r.error:
                match_status = "EXACT" if r.exact_match else ("PARTIAL" if r.partial_match else "MISS")
                print(f"\n  [{r.question_id}] {match_status}")
                print(f"  Q: {r.question}")
                print(f"  Gold: {r.gold_answer}")
                print(f"  OLMo: {r.predicted_answer[:100]}{'...' if len(r.predicted_answer) > 100 else ''}")


def save_results(results: OLMoEvaluationResults, output_path: Path):
    """Save results to JSON file."""
    output = {
        'model_name': results.model_name,
        'total_questions': results.total_questions,
        'exact_match': results.exact_match,
        'partial_match': results.partial_match,
        'f1_score': results.f1_score,
        'avg_latency_ms': results.avg_latency_ms,
        'total_tokens': results.total_tokens,
        'memory_gb': results.memory_gb,
        'by_category': results.by_category,
        'question_results': [
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
                'tokens_generated': r.tokens_generated,
                'error': r.error,
            }
            for r in results.question_results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate OLMo 1B on Q&A benchmark')
    parser.add_argument('--benchmark', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'qa_benchmark_v1.jsonl',
                        help='Path to benchmark JSONL file')
    parser.add_argument('--model', type=str, default='allenai/OLMo-1B-hf',
                        help='HuggingFace model identifier')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show progress during evaluation')
    parser.add_argument('--show-errors', action='store_true',
                        help='Show error details and sample answers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show questions without evaluation')
    parser.add_argument('--category', type=str, default=None,
                        help='Only evaluate specific category')

    args = parser.parse_args()

    # Load benchmark
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    questions = load_benchmark(args.benchmark)
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    # Filter by category if specified
    if args.category:
        questions = [q for q in questions if q['category'] == args.category]
        print(f"Filtered to {len(questions)} questions in category '{args.category}'")

    # Limit questions if specified
    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to {len(questions)} questions")

    # Dry run - just show questions
    if args.dry_run:
        print("\n" + "=" * 60)
        print("BENCHMARK QUESTIONS (Dry Run)")
        print("=" * 60)
        for q in questions:
            print(f"\n[{q['id']}] ({q['category']}) {q['question']}")
            print(f"  Gold: {q['gold_answer']}")
        return

    # Load model
    runner = OLMoRunner(model_name=args.model, device=args.device)
    try:
        runner.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTip: Install transformers with: pip install transformers torch")
        sys.exit(1)

    # Run evaluation
    print(f"\nEvaluating {len(questions)} questions...")
    results = evaluate_with_olmo(questions, runner, verbose=args.verbose)

    # Print results
    print_results(results, show_errors=args.show_errors)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        default_output = PROJECT_ROOT / 'data' / 'benchmarks' / 'results_olmo.json'
        save_results(results, default_output)


if __name__ == '__main__':
    main()
