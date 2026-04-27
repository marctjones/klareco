#!/usr/bin/env python3
"""
Debug Retrieval System

Investigates why retrieval is failing for specific questions.
Shows detailed information about each retrieval step.

Usage:
    python scripts/debug_retrieval.py "Kio estas hundo?"
    python scripts/debug_retrieval.py "Kiu fondis Esperanton?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.importance_scorer import classify_question_type

def debug_retrieval(question: str, top_k: int = 20):
    """Debug retrieval for a single question."""
    print("=" * 80)
    print(f"DEBUGGING RETRIEVAL: {question}")
    print("=" * 80)

    # Step 1: Parse question
    print("\n[STEP 1] Parse question")
    print("-" * 80)
    query_ast = parse(question)
    question_type = classify_question_type(question)
    print(f"Question type: {question_type.value}")
    print(f"AST: {query_ast}")

    # Step 2: Extract roots
    print("\n[STEP 2] Extract roots from AST")
    print("-" * 80)
    roots = []
    def extract_roots(node):
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '')
                if root:
                    roots.append(root.lower())
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

    extract_roots(query_ast)
    print(f"Extracted roots: {roots}")

    # Step 3: Initialize retriever
    print("\n[STEP 3] Initialize Whoosh retriever")
    print("-" * 80)
    retriever = WhooshRetriever(
        whoosh_index_dir=Path('data/indexes/whoosh_fts'),
        kuzu_db_path=Path('data/indexes/v2.1_kuzu_index_full')
    )
    print("✓ Retriever initialized")

    # Step 4: Inspect Whoosh schema
    print("\n[STEP 4] Inspect Whoosh index schema")
    print("-" * 80)
    print(f"Whoosh index fields: {list(retriever.ix.schema.names())}")
    print(f"Schema details:")
    for field_name in retriever.ix.schema.names():
        field = retriever.ix.schema[field_name]
        print(f"  - {field_name}: {type(field).__name__}")

    # Step 5: Build Whoosh query (CORRECTED - use text_lower field)
    print("\n[STEP 5] Build Whoosh query")
    print("-" * 80)
    # OLD (WRONG): query_string = ' OR '.join([f'roots:{root}' for root in roots])
    # NEW (CORRECT): Search text_lower field directly
    query_string = ' OR '.join(roots)
    print(f"Whoosh query: {query_string}")
    print(f"Search field: text_lower")

    # Step 6: Test Whoosh search
    print("\n[STEP 6] Search Whoosh index")
    print("-" * 80)
    from whoosh.qparser import QueryParser

    with retriever.ix.searcher() as searcher:
        # CORRECTED: Use text_lower field instead of non-existent roots field
        parser_obj = QueryParser("text_lower", retriever.ix.schema)
        query_obj = parser_obj.parse(query_string)
        results = searcher.search(query_obj, limit=top_k * 10)

        print(f"Total results: {len(results)}")
        if len(results) > 0:
            print(f"\nTop {min(5, len(results))} results:")
            for i, result in enumerate(results[:5], 1):
                print(f"\n  [{i}] Score: {result.score:.3f}")
                print(f"      Sentence ID: {result['id']}")
                print(f"      Text: {result['text'][:150]}...")
        else:
            print("⚠️  NO RESULTS FOUND!")

    # Step 7: Test full retrieval pipeline (AST-first)
    print("\n[STEP 7] Test full retrieval pipeline (AST-first)")
    print("-" * 80)
    print("NOTE: Current WhooshRetriever is AST-first ONLY (no Whoosh fallback)")
    print("      It uses grammatical role queries via Kuzu, not text search")

    # Enable debug logging to see Kuzu queries
    import logging
    logging.getLogger('klareco.rag.whoosh_retriever').setLevel(logging.DEBUG)

    documents = retriever.retrieve(
        query_roots=roots,
        top_k=top_k,
        retrieval_limit=200,
        question_type=question_type.value,
        query_entity=None,
        query_ast=query_ast
    )

    print(f"Documents retrieved: {len(documents)}")
    if len(documents) > 0:
        print(f"\nTop {min(3, len(documents))} retrieved documents:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n  [{i}] Score: {doc.get('score', 0):.3f}")
            print(f"      Text: {doc.get('text', '')[:200]}...")
    else:
        print("⚠️  NO DOCUMENTS RETRIEVED!")

    # Step 8: Check if any results mention expected keywords
    print("\n[STEP 8] Check for expected keywords in results")
    print("-" * 80)

    # Hardcode expected keywords for known test questions
    expected_keywords = {
        "Kio estas hundo?": ["besto", "hundo", "best"],
        "Kiu fondis Esperanton?": ["Zamenhof", "zamenhof"],
        "Kio estas Esperanto?": ["planlingvo", "lingvo"],
        "Kio estas libro?": ["skribaĵo", "skrib"],
    }

    keywords = expected_keywords.get(question, [])
    if keywords:
        print(f"Expected keywords: {keywords}")

        if len(documents) > 0:
            found_in_docs = []
            for i, doc in enumerate(documents, 1):
                text = doc.get('text', '').lower()
                for keyword in keywords:
                    if keyword in text:
                        found_in_docs.append((i, keyword, doc.get('text', '')[:150]))

            if found_in_docs:
                print(f"\n✓ Found expected keywords in {len(found_in_docs)} documents:")
                for doc_num, keyword, text in found_in_docs[:3]:
                    print(f"  Doc #{doc_num}: keyword '{keyword}'")
                    print(f"    Text: {text}...")
            else:
                print("\n✗ Expected keywords NOT FOUND in any retrieved documents")
        else:
            print("\n✗ Cannot check keywords - no documents retrieved")
    else:
        print("(No expected keywords defined for this question)")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', help='Question to debug')
    parser.add_argument('--top-k', type=int, default=20, help='Number of documents to retrieve')

    args = parser.parse_args()

    debug_retrieval(args.question, args.top_k)
    return 0


if __name__ == '__main__':
    sys.exit(main())
