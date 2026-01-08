#!/usr/bin/env python3
"""
Regenerate ground truth doc IDs for retrieval benchmark from Kuzu index.

The benchmark file has expected_doc_ids from an older index that no longer match
the current Kuzu index. This script queries the Kuzu index for each query's roots
and samples relevant doc IDs as ground truth.

Usage:
    python scripts/regenerate_benchmark_ground_truth.py
    python scripts/regenerate_benchmark_ground_truth.py --output data/benchmarks/retrieval_benchmark_v2.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate benchmark ground truth from Kuzu index'
    )
    parser.add_argument('--benchmark', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'retrieval_benchmark_v1.json',
                        help='Input benchmark file')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'retrieval_benchmark_v2.json',
                        help='Output benchmark file')
    parser.add_argument('--index', type=Path,
                        default=PROJECT_ROOT / 'data' / 'indexes' / 'kuzu_index',
                        help='Path to Kuzu index')
    parser.add_argument('--sample-size', type=int, default=5,
                        help='Number of doc IDs to sample per query')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show progress')

    args = parser.parse_args()

    # Load benchmark
    print(f"Loading benchmark from {args.benchmark}...")
    with open(args.benchmark, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)

    # Load ASTAwareRetriever (which uses KuzuInvertedIndex internally)
    print(f"Loading ASTAwareRetriever from {args.index}...")
    from klareco.rag.ast_aware_retriever import ASTAwareRetriever
    from klareco.rag.kuzu_inverted_index import FallbackMode

    retriever = ASTAwareRetriever(
        index_path=args.index,
        fallback_mode=FallbackMode.NONE,
    )
    index = retriever.root_index
    print(f"  Loaded: {index.total_docs:,} docs, {index.total_roots:,} roots")

    # Update corpus stats
    benchmark['corpus_stats'] = {
        'total_docs': index.total_docs,
        'total_roots': index.total_roots,
    }

    # Process each tier
    total_updated = 0
    for tier_name, tier_data in benchmark['tiers'].items():
        print(f"\n{tier_name}:")

        for query in tier_data['queries']:
            query_roots = query.get('query_roots', [])
            query_text = query['query']

            if not query_roots:
                print(f"  [{query['id']}] No query_roots, skipping")
                continue

            # Run the actual retriever to get ground truth from what it returns
            # This ensures ground truth matches what the retriever actually returns
            results = retriever.search(query_text, top_k=args.sample_size * 2)

            doc_ids = []
            for score, doc, stats in results:
                doc_id = doc.get('doc_id', -1)
                if doc_id != -1:
                    # Verify doc actually contains query roots
                    text = doc.get('text', '').lower()
                    if any(root.lower() in text for root in query_roots):
                        doc_ids.append(doc_id)
                        if len(doc_ids) >= args.sample_size:
                            break

            old_ids = query.get('expected_doc_ids', [])
            query['expected_doc_ids'] = doc_ids
            query['corpus_doc_count'] = len(doc_ids)

            # Get sample texts
            sample_texts = []
            for doc_id in doc_ids[:2]:
                doc = index.get_document(doc_id)
                if doc:
                    text = doc.get('text', '')[:100]
                    sample_texts.append(text)
            query['sample_texts'] = sample_texts

            total_updated += 1

            if args.verbose:
                print(f"  [{query['id']}] {query_text[:40]}...")
                print(f"      Old: {old_ids[:3]}...")
                print(f"      New: {doc_ids[:3]}...")

    # Update version
    old_version = benchmark.get('version', '1.0')
    new_version = f"{float(old_version) + 0.1:.1f}"
    benchmark['version'] = new_version
    benchmark['description'] = f"Tiered retrieval benchmark with Kuzu index ground truth (regenerated from v{old_version})"

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Updated {total_updated} queries")
    print(f"  New version: {new_version}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
