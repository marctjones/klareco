#!/usr/bin/env python3
"""
OLMo 1B Baseline Evaluation Script (CP3)

Evaluates OLMo 1B Instruct on the M1 Q&A benchmark for comparison with Klareco.
Uses the same evaluation metrics as evaluate_qa.py.

Now supports a translation gateway mode where:
- Esperanto questions are translated to English
- OLMo answers in English (its native language)
- English answers are translated back to Esperanto
- Comparison is done against Esperanto gold answers

Usage:
    python scripts/run_olmo_baseline.py                    # Run full evaluation
    python scripts/run_olmo_baseline.py --with-translation # Use translation gateway
    python scripts/run_olmo_baseline.py --dry-run          # Show questions only
    python scripts/run_olmo_baseline.py --limit 5          # Test with 5 questions
    python scripts/run_olmo_baseline.py --device cpu       # Force CPU mode
    python scripts/run_olmo_baseline.py --log-file olmo.log  # Custom log file

Logs are written to: logs/olmo_eval_YYYYMMDD_HHMMSS.log
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global logger
logger = logging.getLogger('olmo_eval')


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> Path:
    """
    Setup logging with both file and console handlers.

    Returns:
        Path to the log file
    """
    # Create logs directory
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp if not specified
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'olmo_eval_{timestamp}.log'
    elif not log_file.is_absolute():
        log_file = logs_dir / log_file

    # Configure root logger
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - always DEBUG level for full info
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Setup logger
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def estimate_eta(elapsed: float, completed: int, total: int) -> str:
    """Estimate time remaining."""
    if completed == 0:
        return "calculating..."
    rate = elapsed / completed
    remaining = (total - completed) * rate
    return format_duration(remaining)


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
    # Translation gateway fields
    translated_question: Optional[str] = None  # EO→EN question
    raw_english_answer: Optional[str] = None   # OLMo's EN response
    translation_time_ms: float = 0.0           # Time spent on translations


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


class TranslationGateway:
    """
    Translation gateway for EO<->EN using Helsinki-NLP MarianMT models.

    This allows OLMo to work in English (its native language) while
    the benchmark remains in Esperanto.

    Pipeline:
    1. EO question → EN (opus-mt-eo-en)
    2. OLMo answers in English
    3. EN answer → EO (opus-mt-en-eo)
    """

    EO_TO_EN_MODEL = 'Helsinki-NLP/opus-mt-eo-en'
    EN_TO_EO_MODEL = 'Helsinki-NLP/opus-mt-en-eo'

    def __init__(self):
        self.eo_en_model = None
        self.eo_en_tokenizer = None
        self.en_eo_model = None
        self.en_eo_tokenizer = None
        self._loaded = False

    def load(self):
        """Load both translation models."""
        from transformers import MarianMTModel, MarianTokenizer

        load_start = time.time()
        logger.info("Loading translation models...")

        # Load EO→EN
        logger.debug(f"Loading {self.EO_TO_EN_MODEL}...")
        self.eo_en_tokenizer = MarianTokenizer.from_pretrained(self.EO_TO_EN_MODEL)
        self.eo_en_model = MarianMTModel.from_pretrained(self.EO_TO_EN_MODEL)
        self.eo_en_model.eval()

        # Load EN→EO
        logger.debug(f"Loading {self.EN_TO_EO_MODEL}...")
        self.en_eo_tokenizer = MarianTokenizer.from_pretrained(self.EN_TO_EO_MODEL)
        self.en_eo_model = MarianMTModel.from_pretrained(self.EN_TO_EO_MODEL)
        self.en_eo_model.eval()

        self._loaded = True
        load_time = time.time() - load_start
        logger.info(f"Translation models loaded in {format_duration(load_time)}")

    def translate_eo_to_en(self, text: str) -> str:
        """Translate Esperanto to English."""
        if not self._loaded:
            raise RuntimeError("Translation models not loaded. Call load() first.")

        inputs = self.eo_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.eo_en_model.generate(**inputs, max_new_tokens=150)
        return self.eo_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_en_to_eo(self, text: str) -> str:
        """Translate English to Esperanto."""
        if not self._loaded:
            raise RuntimeError("Translation models not loaded. Call load() first.")

        inputs = self.en_eo_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.en_eo_model.generate(**inputs, max_new_tokens=150)
        return self.en_eo_tokenizer.decode(outputs[0], skip_special_tokens=True)


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

        load_start = time.time()
        logger.info(f"Loading model: {self.model_name}")
        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")

        # Track memory before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._memory_before = torch.cuda.memory_allocated()
            logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Determine device
        if self.device == "auto":
            device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            device_map = self.device

        logger.info(f"Device: {device_map}")

        # Load tokenizer
        logger.debug("Loading tokenizer...")
        tokenizer_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        logger.debug(f"Tokenizer loaded in {time.time() - tokenizer_start:.1f}s")

        # Load model with appropriate settings
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Loading model with dtype={dtype}...")

        model_start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.debug(f"Model loaded in {time.time() - model_start:.1f}s")

        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_time = time.time() - load_start
        logger.info(f"Model ready on {device_map} (load time: {format_duration(load_time)})")

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    def get_memory_usage(self) -> float:
        """Get peak GPU memory usage in GB."""
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            return peak / (1024 ** 3)
        return 0.0

    def format_prompt(self, question: str, language: str = "esperanto") -> str:
        """Format question as prompt for OLMo."""
        if language == "english":
            # English Q&A prompt (for use with translation gateway)
            prompt = f"""Answer the question concisely.

Question: {question}
Answer:"""
        else:
            # Esperanto Q&A prompt (original)
            prompt = f"""Respondu la demandon koncize en Esperanto.

Demando: {question}
Respondo:"""
        return prompt

    def generate(self, question: str, max_new_tokens: int = 100, language: str = "esperanto") -> tuple:
        """
        Generate answer for a question.

        Args:
            question: The question text
            max_new_tokens: Maximum tokens to generate
            language: "esperanto" or "english" for prompt format

        Returns:
            tuple: (answer_text, num_tokens_generated)
        """
        import torch

        prompt = self.format_prompt(question, language=language)

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
        stop_tokens = ['\n', '?', 'Demando:'] if language == "esperanto" else ['\n', '?', 'Question:']
        for stop in stop_tokens:
            if stop in answer:
                answer = answer.split(stop)[0].strip()

        return answer, len(generated_ids)


def evaluate_with_olmo(
    questions: List[Dict[str, Any]],
    runner: OLMoRunner,
    verbose: bool = False,
    translator: Optional[TranslationGateway] = None
) -> OLMoEvaluationResults:
    """
    Evaluate all questions with OLMo.

    Args:
        questions: List of benchmark questions
        runner: OLMoRunner instance
        verbose: Show detailed progress
        translator: Optional TranslationGateway for EO<->EN translation
                   If provided, questions are translated to English,
                   OLMo answers in English, then answers are translated back.
    """
    results = OLMoEvaluationResults()
    results.model_name = runner.model_name
    if translator:
        results.model_name += " (with translation)"
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

    eval_start_time = time.time()
    logger.info(f"Starting evaluation of {len(questions)} questions")
    logger.info("-" * 60)

    for i, question in enumerate(questions):
        q_id = question['id']
        q_text = question['question']
        gold = question['gold_answer']
        acceptable = question.get('acceptable_answers', [])
        category = question['category']

        # Progress with ETA
        elapsed = time.time() - eval_start_time
        eta = estimate_eta(elapsed, i, len(questions))
        progress_pct = (i + 1) / len(questions) * 100

        # Time the generation (includes translation if enabled)
        start_time = time.time()
        translated_q = None
        raw_en_answer = None
        translation_time = 0.0

        try:
            if translator:
                # Translation mode: EO → EN → OLMo → EN → EO
                trans_start = time.time()
                translated_q = translator.translate_eo_to_en(q_text)
                translation_time += (time.time() - trans_start) * 1000

                # Generate in English
                raw_en_answer, tokens = runner.generate(translated_q, language="english")

                # Translate answer back to Esperanto
                trans_start = time.time()
                predicted = translator.translate_en_to_eo(raw_en_answer)
                translation_time += (time.time() - trans_start) * 1000

                logger.debug(f"  EO Q: {q_text}")
                logger.debug(f"  EN Q: {translated_q}")
                logger.debug(f"  EN A: {raw_en_answer}")
                logger.debug(f"  EO A: {predicted}")
            else:
                # Direct Esperanto mode
                predicted, tokens = runner.generate(q_text, language="esperanto")
        except Exception as e:
            logger.error(f"[{q_id}] Generation error: {e}")
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
            tokens_generated=tokens,
            translated_question=translated_q,
            raw_english_answer=raw_en_answer,
            translation_time_ms=translation_time
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

        # Log progress
        match_status = "EXACT" if exact else ("PARTIAL" if partial else "MISS")
        logger.info(
            f"[{i+1:3d}/{len(questions)}] {q_id} ({category[:8]:8s}) "
            f"{match_status:7s} | {latency_ms:6.0f}ms | ETA: {eta}"
        )
        logger.debug(f"  Q: {q_text[:60]}...")
        logger.debug(f"  A: {predicted[:60]}...")

    # Log completion summary
    total_time = time.time() - eval_start_time
    logger.info("-" * 60)
    logger.info(f"Evaluation complete in {format_duration(total_time)}")

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
                # Translation fields (None if not using translation)
                'translated_question': r.translated_question,
                'raw_english_answer': r.raw_english_answer,
                'translation_time_ms': r.translation_time_ms,
            }
            for r in results.question_results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OLMo 1B on Q&A benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation with logging
    python scripts/run_olmo_baseline.py -v

    # Test with 5 questions
    python scripts/run_olmo_baseline.py --limit 5 -v

    # Custom log file
    python scripts/run_olmo_baseline.py --log-file my_eval.log -v

Logs are saved to: logs/olmo_eval_YYYYMMDD_HHMMSS.log
        """
    )
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
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Path to log file (default: logs/olmo_eval_TIMESTAMP.log)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress (DEBUG level logging)')
    parser.add_argument('--show-errors', action='store_true',
                        help='Show error details and sample answers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show questions without evaluation')
    parser.add_argument('--category', type=str, default=None,
                        help='Only evaluate specific category')
    parser.add_argument('--with-translation', action='store_true',
                        help='Use translation gateway: EO→EN→OLMo→EN→EO')

    args = parser.parse_args()

    # Setup logging first
    log_file = setup_logging(args.log_file, verbose=args.verbose)

    # Log session start
    logger.info("=" * 60)
    logger.info("OLMo 1B BASELINE EVALUATION (CP3)")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")

    # Load benchmark
    if not args.benchmark.exists():
        logger.error(f"Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    questions = load_benchmark(args.benchmark)
    logger.info(f"Loaded {len(questions)} questions from {args.benchmark.name}")

    # Filter by category if specified
    if args.category:
        questions = [q for q in questions if q['category'] == args.category]
        logger.info(f"Filtered to {len(questions)} questions in category '{args.category}'")

    # Limit questions if specified
    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {len(questions)} questions")

    # Dry run - just show questions
    if args.dry_run:
        logger.info("DRY RUN - showing questions only")
        for q in questions:
            logger.info(f"[{q['id']}] ({q['category']}) {q['question']}")
            logger.debug(f"  Gold: {q['gold_answer']}")
        return

    # Load translation gateway if requested
    translator = None
    if args.with_translation:
        logger.info("Translation mode enabled: EO→EN→OLMo→EN→EO")
        translator = TranslationGateway()
        try:
            translator.load()
        except Exception as e:
            logger.error(f"Error loading translation models: {e}")
            logger.error("Tip: Install transformers with: pip install transformers torch sentencepiece")
            sys.exit(1)

    # Load model
    runner = OLMoRunner(model_name=args.model, device=args.device)
    try:
        runner.load()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Tip: Install transformers with: pip install transformers torch accelerate")
        sys.exit(1)

    # Run evaluation
    results = evaluate_with_olmo(questions, runner, verbose=args.verbose, translator=translator)

    # Print results
    print_results(results, show_errors=args.show_errors)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Different output file for translation vs direct mode
        if args.with_translation:
            default_output = PROJECT_ROOT / 'data' / 'benchmarks' / 'results_olmo_translated.json'
        else:
            default_output = PROJECT_ROOT / 'data' / 'benchmarks' / 'results_olmo.json'
        save_results(results, default_output)

    # Log final summary
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Exact Match:   {results.exact_match:.1%}")
    logger.info(f"Partial Match: {results.partial_match:.1%}")
    logger.info(f"F1 Score:      {results.f1_score:.3f}")
    logger.info(f"Avg Latency:   {results.avg_latency_ms:.0f}ms")
    logger.info(f"Completed at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log saved to:  {log_file}")


if __name__ == '__main__':
    main()
