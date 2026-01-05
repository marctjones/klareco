#!/usr/bin/env python3
"""
Compare multiple slot-based retrievers on same queries.

This script runs all available retrievers (or selected ones) on the same
test queries and compares their performance in terms of:
- Speed (query latency)
- Memory usage
- CPU usage
- Result quality (overlap analysis)

Usage:
    # Compare all available retrievers (default)
    python scripts/compare_retrievers.py --index data/indexes/slot_full

    # Compare specific retrievers only
    python scripts/compare_retrievers.py --index data/indexes/slot_full --retrievers hnsw,scann

    # Use custom queries
    python scripts/compare_retrievers.py --index data/indexes/slot_full --queries queries.txt

    # Adjust number of results
    python scripts/compare_retrievers.py --index data/indexes/slot_full -k 10

Available retrievers: mmap, multifaiss, hybrid, hnsw, scann
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer

# Import all available retrievers
RETRIEVERS = {}

try:
    from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
    RETRIEVERS['mmap'] = ('MemoryMapped', MemoryMappedSlotRetriever)
except ImportError:
    pass

try:
    from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
    RETRIEVERS['multifaiss'] = ('MultiFAISS', MultiFAISSSlotRetriever)
except ImportError:
    pass

try:
    from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
    RETRIEVERS['hybrid'] = ('Hybrid', HybridFAISSMmapRetriever)
except ImportError:
    pass

try:
    from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
    RETRIEVERS['hnsw'] = ('HNSW', HNSWSlotRetriever)
except ImportError:
    pass

try:
    from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever
    RETRIEVERS['scann'] = ('ScaNN', ScaNNSlotRetriever)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_translator():
    """Load EO→EN translation model."""
    try:
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-eo-en"
        logger.info("Loading EO→EN translation model (this may take a minute)...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        logger.info("  ✓ Translation model loaded")
        return tokenizer, model
    except Exception as e:
        logger.warning(f"Failed to load translation model: {e}")
        logger.warning("Install with: pip install transformers sentencepiece")
        return None, None


def translate_text(text: str, tokenizer, model, max_length: int = 200) -> str:
    """Translate Esperanto text to English."""
    if not tokenizer or not model:
        return text

    try:
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        logger.debug(f"Translation failed: {e}")
        return text


def get_process_resources():
    """Get current process memory and CPU usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)
    return memory_mb, cpu_percent


def check_index_availability(index_path: Path) -> Dict[str, bool]:
    """Check which retrievers can be loaded based on available index files."""
    available = {}

    # MemoryMapped: needs mmap directory
    available['mmap'] = (index_path / "mmap").exists()

    # MultiFAISS: needs multifaiss directory
    available['multifaiss'] = (index_path / "multifaiss").exists()

    # Hybrid: needs faiss directory and mmap
    available['hybrid'] = (
        (index_path / "faiss").exists() and
        (index_path / "mmap").exists()
    )

    # HNSW: needs hnsw directory and mmap
    available['hnsw'] = (
        (index_path / "hnsw").exists() and
        (index_path / "mmap").exists()
    )

    # ScaNN: needs scann directory and mmap
    available['scann'] = (
        (index_path / "scann").exists() and
        (index_path / "mmap").exists()
    )

    return available


def load_retriever(retriever_type: str, index_path: Path, indexer: SlotBasedIndexer):
    """Load a specific retriever with progress updates."""
    if retriever_type not in RETRIEVERS:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    name, cls = RETRIEVERS[retriever_type]
    logger.info(f"Loading {name} retriever...")

    mem_before, _ = get_process_resources()
    start_time = time.time()

    retriever = cls(index_path, indexer)

    load_time = time.time() - start_time
    mem_after, _ = get_process_resources()
    mem_used = mem_after - mem_before

    logger.info(f"  ✓ {name} loaded in {load_time:.1f}s (memory: +{mem_used:.1f} MB)")

    return retriever, {'load_time': load_time, 'memory_overhead': mem_used}


def run_queries(
    retriever,
    retriever_name: str,
    queries: List[Tuple[str, str]],
    top_k: int = 10,
    prefilter_n: int = 500,
    rerank_n: int = 100,
) -> Dict:
    """Run queries on a retriever and collect metrics."""
    results = {
        'name': retriever_name,
        'queries': [],
        'total_time': 0,
        'avg_time': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'memory_peak': 0,
        'cpu_peak': 0,
    }

    logger.info(f"Running {len(queries)} queries on {retriever_name}...")

    # Progress bar for queries
    pbar = tqdm(queries, desc=f"{retriever_name:12}", unit="query", ncols=100)

    for query_eo, query_en in pbar:
        mem_before, _ = get_process_resources()
        start_time = time.time()

        # Run search with appropriate parameters
        try:
            if 'ScaNN' in retriever.__class__.__name__:
                search_results = retriever.search(
                    query_eo,
                    top_k=top_k,
                    scann_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'HNSW' in retriever.__class__.__name__:
                search_results = retriever.search(
                    query_eo,
                    top_k=top_k,
                    hnsw_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'Hybrid' in retriever.__class__.__name__:
                search_results = retriever.search(
                    query_eo,
                    top_k=top_k,
                    faiss_top_n=prefilter_n
                )
            elif 'MultiFAISS' in retriever.__class__.__name__:
                search_results = retriever.search(
                    query_eo,
                    top_k=top_k,
                    slot_top_n=rerank_n
                )
            else:
                search_results = retriever.search(
                    query_eo,
                    top_k=top_k,
                    rerank_top_n=prefilter_n
                )
        except Exception as e:
            logger.error(f"  ✗ Query failed: {e}")
            continue

        query_time = (time.time() - start_time) * 1000  # ms
        mem_after, cpu = get_process_resources()
        mem_used = mem_after - mem_before

        # Update progress bar with latest timing
        pbar.set_postfix({'last': f'{query_time:.1f}ms', 'mem': f'{mem_after:.0f}MB'})

        # Store results (full text, no truncation)
        top_results = []
        for score, doc_data in search_results:
            top_results.append({
                'score': float(score),
                'text': doc_data['text'],  # Full text, no truncation
                'doc_id': doc_data.get('doc_id', 'unknown')
            })

        query_result = {
            'query_eo': query_eo,
            'query_en': query_en,
            'time_ms': query_time,
            'memory_mb': mem_after,
            'cpu_percent': cpu,
            'num_results': len(search_results),
            'top_doc': search_results[0][1]['text'] if search_results else None,  # Full text for backward compat
            'results': top_results  # All results with full text
        }
        results['queries'].append(query_result)

        # Update aggregates
        results['total_time'] += query_time
        results['min_time'] = min(results['min_time'], query_time)
        results['max_time'] = max(results['max_time'], query_time)
        results['memory_peak'] = max(results['memory_peak'], mem_after)
        results['cpu_peak'] = max(results['cpu_peak'], cpu)

    pbar.close()

    # Calculate averages
    if results['queries']:
        results['avg_time'] = results['total_time'] / len(results['queries'])

    return results


def print_comparison_table(all_results: List[Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("RETRIEVER COMPARISON RESULTS")
    print("=" * 100)
    print()

    # Header
    print(f"{'Retriever':<15} {'Avg Time':<12} {'Min':<10} {'Max':<10} {'Memory':<12} {'CPU':<8} {'Queries':<8}")
    print("-" * 100)

    # Sort by average time
    sorted_results = sorted(all_results, key=lambda x: x['avg_time'])

    for result in sorted_results:
        print(
            f"{result['name']:<15} "
            f"{result['avg_time']:>10.1f}ms "
            f"{result['min_time']:>8.1f}ms "
            f"{result['max_time']:>8.1f}ms "
            f"{result['memory_peak']:>10.1f}MB "
            f"{result['cpu_peak']:>6.1f}% "
            f"{len(result['queries']):>7}"
        )

    print()
    print("Rankings:")
    print(f"  🥇 Fastest:       {sorted_results[0]['name']} ({sorted_results[0]['avg_time']:.1f}ms avg)")

    mem_sorted = sorted(all_results, key=lambda x: x['memory_peak'])
    print(f"  💾 Lowest Memory: {mem_sorted[0]['name']} ({mem_sorted[0]['memory_peak']:.1f}MB peak)")

    print()


def analyze_result_overlap(all_results: List[Dict], top_k: int = 10, translator=None, verbose: bool = False):
    """Analyze how much retrievers agree on top results."""
    print("=" * 100)
    print("RESULT OVERLAP ANALYSIS")
    if translator:
        print("(with English translations)")
    print("=" * 100)
    print()

    if len(all_results) < 2:
        print("Need at least 2 retrievers to compare overlap")
        return

    tokenizer, model = translator if translator else (None, None)

    # For each query, check overlap between retrievers
    num_queries = len(all_results[0]['queries'])

    for q_idx in range(min(3, num_queries)):  # Show first 3 queries
        query_eo = all_results[0]['queries'][q_idx]['query_eo']
        query_en = all_results[0]['queries'][q_idx]['query_en']

        print(f"Query {q_idx+1}: {query_eo}")
        if query_en:
            print(f"  EN: {query_en}")
        print()

        # Collect top docs from each retriever
        retriever_docs = {}
        for result in all_results:
            if q_idx < len(result['queries']):
                top_doc = result['queries'][q_idx]['top_doc']
                retriever_docs[result['name']] = top_doc

        # Show top result from each
        for name, doc in retriever_docs.items():
            if verbose or not doc:
                # Show full text
                preview_eo = doc
            else:
                # Truncate for readability
                preview_eo = doc[:80] + "..." if len(doc) > 80 else doc

            print(f"  {name:12}: {preview_eo}")

            # Add translation if requested
            if translator and doc:
                if verbose:
                    preview_en = translate_text(doc, tokenizer, model)
                else:
                    preview_en = translate_text(doc[:200], tokenizer, model)
                    if len(preview_en) > 80:
                        preview_en = preview_en[:80] + "..."
                print(f"              → {preview_en}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare slot-based retrievers on same queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--retrievers',
        type=str,
        help='Comma-separated list of retrievers to compare (default: all available). '
             'Options: mmap, multifaiss, hybrid, hnsw, scann'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    parser.add_argument(
        '--queries',
        type=Path,
        help='Path to file with queries (one per line, format: "eo_query | en_translation")'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--prefilter-n',
        type=int,
        default=500,
        help='Number of candidates from pre-filtering stage (hnsw_top_n, scann_top_n, faiss_top_n). Default: 500'
    )
    parser.add_argument(
        '--rerank-n',
        type=int,
        default=100,
        help='Number of candidates for slot reranking stage (slot_top_n). Default: 100'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Translate Esperanto results to English using Helsinki-NLP model'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full results in console output (no truncation)'
    )

    args = parser.parse_args()

    # Check index exists
    if not args.index.exists():
        logger.error(f"Index directory not found: {args.index}")
        sys.exit(1)

    # Determine which retrievers to test
    available = check_index_availability(args.index)

    if args.retrievers:
        requested = [r.strip() for r in args.retrievers.split(',')]
        to_test = []
        for r in requested:
            if r not in RETRIEVERS:
                logger.error(f"Unknown retriever: {r}")
                logger.info(f"Available: {', '.join(RETRIEVERS.keys())}")
                sys.exit(1)
            if not available.get(r, False):
                logger.warning(f"Retriever '{r}' requested but index files not available")
            else:
                to_test.append(r)
    else:
        # Test all available
        to_test = [r for r in RETRIEVERS.keys() if available.get(r, False)]

    if not to_test:
        logger.error("No retrievers available to test")
        logger.info("Available retrievers and their requirements:")
        for name, is_avail in available.items():
            status = "✓" if is_avail else "✗"
            logger.info(f"  {status} {name}")
        sys.exit(1)

    logger.info(f"Testing retrievers: {', '.join(to_test)}")
    print()

    # Load test queries
    if args.queries:
        queries = []
        query_path = Path(args.queries)

        # Check if JSONL format (benchmark file)
        if query_path.suffix == '.jsonl':
            with open(query_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        query_text = data.get('query', '')
                        if query_text:
                            queries.append((query_text, ""))
                    except json.JSONDecodeError:
                        continue
        else:
            # Text format with optional | separator
            with open(query_path) as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        eo, en = line.split('|', 1)
                        queries.append((eo.strip(), en.strip()))
                    else:
                        queries.append((line, ""))
    else:
        # Check for default benchmark file
        default_benchmark = args.index / 'benchmark_queries.jsonl'
        if default_benchmark.exists():
            logger.info(f"Using default benchmark queries: {default_benchmark}")
            queries = []
            with open(default_benchmark) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        query_text = data.get('query', '')
                        if query_text:
                            queries.append((query_text, ""))
                    except json.JSONDecodeError:
                        continue
        else:
            # Default test queries
            queries = [
                ("Kiu kreis Esperanton?", "Who created Esperanto?"),
                ("Kio estas Esperanto?", "What is Esperanto?"),
                ("Kiam Zamenhof kreis Esperanton?", "When did Zamenhof create Esperanto?"),
                ("Kie naskiĝis Zamenhof?", "Where was Zamenhof born?"),
            ]

    logger.info(f"Testing with {len(queries)} queries")
    print()

    # Load translator if requested
    translator = None
    if args.translate:
        tokenizer, model = load_translator()
        if tokenizer and model:
            translator = (tokenizer, model)
        print()

    # Load indexer
    logger.info("Loading embedding models...")
    indexer = SlotBasedIndexer(
        root_model_path=Path("models/root_embeddings/best_model.pt"),
        affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
        output_dir=args.index,  # Use index path as output_dir
    )
    logger.info("  ✓ Models loaded")
    print()

    # Test each retriever
    all_results = []

    for retriever_type in to_test:
        name, _ = RETRIEVERS[retriever_type]

        try:
            # Load retriever
            retriever, load_stats = load_retriever(retriever_type, args.index, indexer)

            # Run queries
            results = run_queries(
                retriever,
                name,
                queries,
                top_k=args.top_k,
                prefilter_n=args.prefilter_n,
                rerank_n=args.rerank_n,
            )
            results.update(load_stats)
            all_results.append(results)

            # Clean up
            del retriever
            gc.collect()

            print()

        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Print comparison
    if all_results:
        print_comparison_table(all_results)
        analyze_result_overlap(all_results, top_k=args.top_k, translator=translator, verbose=args.verbose)

        # Save to JSON if requested
        if args.output:
            logger.info(f"Saving detailed results to {args.output}")
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info("  ✓ Results saved")
    else:
        logger.error("No retrievers successfully tested")
        sys.exit(1)


if __name__ == '__main__':
    main()
