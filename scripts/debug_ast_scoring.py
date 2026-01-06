#!/usr/bin/env python3
"""
Debug AST scoring to understand why correct answers don't rank high.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.parser import parse

def main():
    index_path = Path("data/indexes/slot_full")

    print("Debugging AST Scoring...")
    print()

    retriever = ASTAwareRetriever(index_path, use_prefilter=True)

    query = "Kiam aperis la Fundamento de Esperanto?"
    expected = "1905"

    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print()

    # Get pre-filter results with larger n
    query_ast = parse(query)
    query_text = retriever._reconstruct_query(query_ast)

    print("Finding answer in pre-filter results...")
    prefilter_results = retriever.prefilter_retriever.search(
        query_text,
        top_k=2000,
        hnsw_top_n=2000,
        slot_top_n=2000
    )

    # Find the document with answer
    answer_doc = None
    answer_rank = None
    for i, (score, doc) in enumerate(prefilter_results, 1):
        if expected in doc['text']:
            answer_doc = doc
            answer_rank = i
            print(f"✓ Found answer at pre-filter rank {i}")
            print(f"  Text: {doc['text'][:200]}...")
            print()
            break

    if not answer_doc:
        print("❌ Answer not in top 2000 pre-filter results")
        return

    # Now run AST matching and see what score the answer gets
    print("Running AST pattern matching...")
    classification = retriever.question_classifier.classify(query, query_ast)
    target_slots = classification['target_slots']
    entity_type = classification['entity_type']

    # Parse answer document
    answer_ast = parse(answer_doc['text'])

    # Match pattern
    match_result = retriever.pattern_matcher.match(
        query_ast,
        answer_ast,
        target_slots,
        entity_type.value
    )

    print(f"AST Pattern Match Score: {match_result.score}")
    print(f"Matched slots: {match_result.matched_slots}")
    print(f"Explanation: {match_result.explanation}")
    print()

    # Compare with top AST results
    print("Top 5 AST-aware results:")
    results = retriever.search(query, top_k=5, strategy='auto', prefilter_n=2000)
    for i, (score, doc) in enumerate(results, 1):
        has_answer = "✓" if expected in doc['text'] else " "
        print(f"{i}. {has_answer} Score: {score:.3f} - {doc['text'][:100]}...")

if __name__ == '__main__':
    main()
