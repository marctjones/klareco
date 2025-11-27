#!/usr/bin/env python3
"""
Enhanced retrieval with query expansion and entity-aware search.

Improves retrieval for entity queries like "Kiu estas Frodo?" by:
1. Detecting entity-focused queries (kiu, kio, kie, etc.)
2. Expanding queries with entity name
3. Combining structural + semantic results
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.retriever import create_retriever
from klareco.parser import parse
import re


def detect_query_type(query: str) -> str:
    """
    Detect the type of query.

    Returns:
        'who' - Kiu (who)
        'what' - Kio (what)
        'where' - Kie (where)
        'when' - Kiam (when)
        'why' - Kial (why)
        'how' - Kiel (how)
        'general' - Other queries
    """
    query_lower = query.lower().strip()

    if query_lower.startswith('kiu'):
        return 'who'
    elif query_lower.startswith('kio'):
        return 'what'
    elif query_lower.startswith('kie'):
        return 'where'
    elif query_lower.startswith('kiam'):
        return 'when'
    elif query_lower.startswith('kial'):
        return 'why'
    elif query_lower.startswith('kiel'):
        return 'how'
    else:
        return 'general'


def extract_entity(query: str) -> str:
    """
    Extract the entity from a query.

    Examples:
        "Kiu estas Frodo?" -> "Frodo"
        "Kio estas la ringo?" -> "ringo"
        "Kie loĝas la hobitoj?" -> "hobitoj"
    """
    # Remove question words and common patterns
    cleaned = query
    for pattern in [r'^Kiu\s+estas\s+', r'^Kio\s+estas\s+', r'^Kie\s+',
                    r'^Kiam\s+', r'^Kial\s+', r'^Kiel\s+']:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove articles
    cleaned = re.sub(r'\b(la|unu)\s+', '', cleaned, flags=re.IGNORECASE)

    # Remove punctuation
    cleaned = re.sub(r'[?.!,;]', '', cleaned)

    # Take first significant word (usually the entity)
    words = cleaned.strip().split()
    if words:
        return words[0]
    return ""


def expand_query_for_entity(query: str, entity: str, query_type: str) -> List[str]:
    """
    Generate expanded queries for better entity retrieval.

    Args:
        query: Original query
        entity: Extracted entity name
        query_type: Type of query (who, what, where, etc.)

    Returns:
        List of query variations
    """
    variations = [query]  # Always include original

    if query_type == 'who':
        # For "Kiu estas X?" add variations
        variations.append(entity)  # Just the name
        variations.append(f"{entity} estas")
        variations.append(f"pri {entity}")  # about X

    elif query_type == 'what':
        # For "Kio estas X?"
        variations.append(entity)
        variations.append(f"{entity} estas")
        variations.append(f"la {entity}")

    elif query_type == 'where':
        # For "Kie ...?"
        variations.append(f"{entity} loĝas")
        variations.append(f"{entity} troviĝas")

    return variations


def enhanced_retrieve(
    retriever,
    query: str,
    k: int = 5,
    expand_queries: bool = True,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with query expansion.

    Args:
        retriever: KlarecoRetriever instance
        query: Query string
        k: Number of results to return
        expand_queries: Use query expansion
        deduplicate: Remove duplicate results

    Returns:
        List of results with scores
    """
    if not expand_queries:
        # Simple retrieval
        return retriever.retrieve(query, k=k, return_scores=True)

    # Detect query type and extract entity
    query_type = detect_query_type(query)
    entity = extract_entity(query)

    print(f"📊 Query type: {query_type}, Entity: '{entity}'")

    # Generate query variations
    variations = expand_query_for_entity(query, entity, query_type)
    print(f"🔍 Searching with {len(variations)} variations")

    # Retrieve with each variation
    all_results = {}  # text -> (score, result)

    for i, var in enumerate(variations, 1):
        print(f"   {i}. \"{var}\"")
        try:
            results = retriever.retrieve(var, k=k*2, return_scores=True)

            for r in results:
                text = r['text']
                score = r.get('score', 0.0)

                # Boost score if entity appears in result
                if entity.lower() in text.lower():
                    score *= 1.2  # 20% boost for entity match

                # Keep best score for each text
                if text not in all_results or score > all_results[text][0]:
                    all_results[text] = (score, r)

        except Exception as e:
            print(f"   ⚠️  Error with variation '{var}': {e}")

    # Sort by score and take top k
    sorted_results = sorted(all_results.values(), key=lambda x: x[0], reverse=True)

    # Return top k with updated scores
    final_results = []
    for score, result in sorted_results[:k]:
        result = dict(result)  # Copy
        result['score'] = score
        final_results.append(result)

    return final_results


def demo():
    """Demo enhanced retrieval."""

    print("=" * 80)
    print("Enhanced Retrieval Demo - Query Expansion + Entity Boosting")
    print("=" * 80)
    print()

    # Load retriever
    print("📚 Loading index...")
    retriever = create_retriever(
        'data/corpus_index_v3',
        'models/tree_lstm/best_model.pt'
    )
    print("✅ Ready!")
    print()

    # Test queries
    queries = [
        "Kiu estas Frodo?",
        "Kio estas la Unu Ringo?",
        "Kie loĝas la hobitoj?",
    ]

    for query in queries:
        print("=" * 80)
        print(f"🔍 QUERY: {query}")
        print("=" * 80)
        print()

        # Standard retrieval
        print("📊 STANDARD RETRIEVAL:")
        standard = retriever.retrieve(query, k=3, return_scores=True)
        for i, r in enumerate(standard, 1):
            print(f"  {i}. [{r['score']:.2f}] {r['text'][:100]}...")
        print()

        # Enhanced retrieval
        print("✨ ENHANCED RETRIEVAL (with query expansion):")
        enhanced = enhanced_retrieve(retriever, query, k=3, expand_queries=True)
        for i, r in enumerate(enhanced, 1):
            print(f"  {i}. [{r['score']:.2f}] {r['text'][:100]}...")
        print()
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced retrieval with query expansion")
    parser.add_argument(
        'query',
        nargs='*',
        help='Query to run'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results (default: 5)'
    )

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.query:
        query = ' '.join(args.query)

        retriever = create_retriever(
            'data/corpus_index_v3',
            'models/tree_lstm/best_model.pt'
        )

        results = enhanced_retrieve(retriever, query, k=args.k)

        print(f"\n🔍 Query: {query}\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.2f}] {r['text'][:150]}...")
            print()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
