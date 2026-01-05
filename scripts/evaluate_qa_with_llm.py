#!/usr/bin/env python3
"""
LLM-based Q&A evaluation for retrievers.

Uses Claude (via API) to evaluate whether retrieved documents actually answer questions.
This is much more accurate than substring matching.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Anthropic for Claude API
try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_qa_benchmark(benchmark_file: Path) -> List[Dict]:
    """Load Q&A benchmark questions."""
    questions = []
    with open(benchmark_file) as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data)
    return questions


def evaluate_with_claude(
    question: str,
    retrieved_docs: List[str],
    gold_answer: str,
    acceptable_answers: List[str],
    client: anthropic.Anthropic
) -> Dict:
    """
    Evaluate retrieved documents using Claude Sonnet.

    Returns:
        {
            'best_rank': int or None (1-indexed rank of best answer, None if no answer found),
            'quality': str ('perfect', 'good', 'partial', 'none'),
            'reasoning': str (Claude's explanation)
        }
    """

    # Build prompt for Claude
    docs_text = ""
    for i, doc in enumerate(retrieved_docs[:10], 1):
        docs_text += f"\n\nDocument {i}:\n{doc[:500]}"  # Limit each doc to 500 chars to save tokens

    prompt = f"""You are evaluating a retrieval system for Esperanto question-answering.

QUESTION: {question}

EXPECTED ANSWER: {gold_answer}
ACCEPTABLE ANSWERS: {', '.join(acceptable_answers)}

Here are the top 10 documents retrieved by the system:
{docs_text}

TASK: Evaluate which document (if any) best answers the question.

Respond with a JSON object with this exact format:
{{
    "best_rank": <number 1-10, or null if no document answers the question>,
    "quality": "<perfect|good|partial|none>",
    "reasoning": "<brief explanation of your judgment>"
}}

Quality criteria:
- "perfect": The document directly and completely answers the question
- "good": The document contains the answer along with relevant context
- "partial": The document is related but doesn't fully answer the question
- "none": No document adequately answers the question

Respond ONLY with the JSON object, no other text."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,  # Deterministic for consistency
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        result = json.loads(response_text)

        return {
            'best_rank': result.get('best_rank'),
            'quality': result.get('quality', 'none'),
            'reasoning': result.get('reasoning', '')
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {response_text[:200]}")
        return {
            'best_rank': None,
            'quality': 'none',
            'reasoning': f'Parse error: {e}'
        }
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {
            'best_rank': None,
            'quality': 'none',
            'reasoning': f'API error: {e}'
        }


def evaluate_retriever(
    retriever,
    retriever_name: str,
    questions: List[Dict],
    top_k: int = 10,
    prefilter_n: int = 500,
    rerank_n: int = 100,
    claude_client: Optional[anthropic.Anthropic] = None
) -> Dict:
    """Evaluate a retriever on Q&A benchmark."""

    results = {
        'name': retriever_name,
        'perfect_in_top_1': 0,
        'perfect_in_top_5': 0,
        'perfect_in_top_10': 0,
        'good_in_top_1': 0,
        'good_in_top_5': 0,
        'good_in_top_10': 0,
        'any_answer_in_top_10': 0,
        'total_questions': len(questions),
        'avg_time_ms': 0,
        'questions': []
    }

    total_time = 0

    for qa in questions:
        question = qa['question']
        gold_answer = qa.get('gold_answer', '')
        acceptable_answers = qa.get('acceptable_answers', [])

        # Search
        start = time.time()
        try:
            if 'ScaNN' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    scann_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'HNSW' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    hnsw_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'Hybrid' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    faiss_top_n=prefilter_n
                )
            elif 'MultiFAISS' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    slot_top_n=rerank_n
                )
            else:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    rerank_top_n=prefilter_n
                )
        except Exception as e:
            logger.error(f"  ✗ Query failed: {e}")
            continue

        query_time = (time.time() - start) * 1000
        total_time += query_time

        # Get retrieved documents
        retrieved_docs = [doc_data['text'] for score, doc_data in search_results]

        # Evaluate with Claude
        if not claude_client:
            logger.error("Claude API client not initialized. Check ANTHROPIC_API_KEY.")
            continue

        eval_result = evaluate_with_claude(
            question,
            retrieved_docs,
            gold_answer,
            acceptable_answers,
            claude_client
        )

        logger.info(f"  Q{qa['id']}: {question[:50]}... -> Rank {eval_result['best_rank']}, Quality: {eval_result['quality']}")

        # Update metrics based on evaluation
        best_rank = eval_result['best_rank']
        quality = eval_result['quality']

        if best_rank:
            results['any_answer_in_top_10'] += 1

            if quality == 'perfect':
                if best_rank == 1:
                    results['perfect_in_top_1'] += 1
                if best_rank <= 5:
                    results['perfect_in_top_5'] += 1
                if best_rank <= 10:
                    results['perfect_in_top_10'] += 1

            if quality in ['perfect', 'good']:
                if best_rank == 1:
                    results['good_in_top_1'] += 1
                if best_rank <= 5:
                    results['good_in_top_5'] += 1
                if best_rank <= 10:
                    results['good_in_top_10'] += 1

        # Store per-question results
        results['questions'].append({
            'id': qa['id'],
            'question': question,
            'gold_answer': gold_answer,
            'best_rank': best_rank,
            'quality': quality,
            'reasoning': eval_result.get('reasoning'),
            'time_ms': query_time,
            'top_results': retrieved_docs[:3]  # Save top 3 for review
        })

    results['avg_time_ms'] = total_time / len(questions) if questions else 0
    return results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 120)
    print("LLM-EVALUATED Q&A RESULTS")
    print("=" * 120)
    print()

    total = results['total_questions']

    print(f"Retriever: {results['name']}")
    print(f"Total Questions: {total}")
    print(f"Average Latency: {results['avg_time_ms']:.1f}ms")
    print()

    print("Perfect Answers (directly answers the question):")
    print(f"  Top-1:  {results['perfect_in_top_1']}/{total} ({results['perfect_in_top_1']/total*100:.1f}%)")
    print(f"  Top-5:  {results['perfect_in_top_5']}/{total} ({results['perfect_in_top_5']/total*100:.1f}%)")
    print(f"  Top-10: {results['perfect_in_top_10']}/{total} ({results['perfect_in_top_10']/total*100:.1f}%)")
    print()

    print("Good Answers (perfect or good):")
    print(f"  Top-1:  {results['good_in_top_1']}/{total} ({results['good_in_top_1']/total*100:.1f}%)")
    print(f"  Top-5:  {results['good_in_top_5']}/{total} ({results['good_in_top_5']/total*100:.1f}%)")
    print(f"  Top-10: {results['good_in_top_10']}/{total} ({results['good_in_top_10']/total*100:.1f}%)")
    print()

    print(f"Any Answer in Top-10: {results['any_answer_in_top_10']}/{total} ({results['any_answer_in_top_10']/total*100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="LLM-based Q&A evaluation for retrievers")
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--benchmark',
        type=Path,
        default=Path('data/benchmarks/datasets/qa_benchmark_v1.jsonl'),
        help='Path to Q&A benchmark file'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        choices=['mmap', 'multifaiss', 'hybrid', 'hnsw', 'scann'],
        help='Retriever to evaluate'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    parser.add_argument(
        '--prefilter-n',
        type=int,
        default=500,
        help='Pre-filtering candidates (default: 500)'
    )
    parser.add_argument(
        '--rerank-n',
        type=int,
        default=100,
        help='Reranking candidates (default: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit to first N questions (for testing)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Initialize Claude API client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    logger.info("✓ Claude API client initialized")
    print()

    # Load Q&A benchmark
    logger.info(f"Loading Q&A benchmark from {args.benchmark}...")
    questions = load_qa_benchmark(args.benchmark)

    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"  Limited to first {args.limit} questions")

    logger.info(f"  Loaded {len(questions)} questions")
    print()

    # Load indexer
    logger.info("Loading embedding models...")
    indexer = SlotBasedIndexer(
        root_model_path=Path("models/root_embeddings/best_model.pt"),
        affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
        output_dir=args.index
    )
    logger.info("  ✓ Models loaded")
    print()

    # Load retriever
    RETRIEVERS = {
        'mmap': ('MemoryMapped', MemoryMappedSlotRetriever),
        'multifaiss': ('MultiFAISS', MultiFAISSSlotRetriever),
        'hybrid': ('Hybrid', HybridFAISSMmapRetriever),
        'hnsw': ('HNSW', HNSWSlotRetriever),
        'scann': ('ScaNN', ScaNNSlotRetriever),
    }

    name, retriever_class = RETRIEVERS[args.retriever]

    logger.info(f"Loading {name} retriever...")
    retriever = retriever_class(args.index, indexer)
    logger.info(f"  ✓ {name} loaded")
    print()

    # Run evaluation
    logger.info(f"Evaluating {name} on {len(questions)} questions using Claude Sonnet...")
    logger.info(f"  This will make ~{len(questions)} API calls")
    print()

    results = evaluate_retriever(
        retriever,
        name,
        questions,
        top_k=args.top_k,
        prefilter_n=args.prefilter_n,
        rerank_n=args.rerank_n,
        claude_client=client
    )

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Results saved")


if __name__ == '__main__':
    main()
