#!/usr/bin/env python3
"""
Debug Embedding Similarity Scores

Examines actual embedding similarity scores to understand why they hurt performance.

Usage:
    python scripts/debug_embedding_scores.py --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.fact_extractor import FactExtractor, Fact, RelationType
from klareco.rag.importance_scorer import FactImportanceScorer, classify_question_type
from klareco.rag.whoosh_retriever import WhooshRetriever
from scripts.demo_extractive_qa import extract_roots_from_ast, extract_query_entity


def analyze_question(question: str, expected: str, retriever, top_k: int = 30):
    """Analyze embedding scores for a single question."""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"{'='*80}\n")

    # Parse query
    query_ast = parse(question)
    roots = extract_roots_from_ast(query_ast)
    question_type = classify_question_type(question)
    query_entity = extract_query_entity(query_ast, question_type)

    # Strip endings to get root
    entity_root = None
    if query_entity:
        entity_root = query_entity.lower()
        if entity_root.endswith('jn'):
            entity_root = entity_root[:-2]
        elif entity_root.endswith('n') or entity_root.endswith('j'):
            entity_root = entity_root[:-1]
        if entity_root.endswith('o') or entity_root.endswith('a') or entity_root.endswith('e'):
            entity_root = entity_root[:-1]

    print(f"Query roots: {roots}")
    print(f"Query entity: {entity_root}")
    print()

    # Retrieve documents
    try:
        documents = retriever.retrieve(
            query_roots=list(roots),
            top_k=top_k,
            retrieval_limit=200,
            question_type=question_type.value,
            query_entity=entity_root,
            query_ast=query_ast
        )
    except Exception as e:
        print(f"Retrieval error: {e}")
        return

    # Check if answer in documents
    answer_in_docs = any(expected.lower() in doc.get('text', '').lower()
                         for doc in documents)

    if not answer_in_docs:
        print(f"⚠️  Answer NOT in retrieved documents (retrieval failure)")
        return

    print(f"✓ Answer found in retrieved documents\n")

    # Extract facts
    extractor = FactExtractor()
    all_facts = []
    for doc in documents[:10]:  # Top 10 docs
        doc_ast = doc.get('ast')
        if doc_ast:
            facts = extractor.extract(doc_ast, doc.get('text', ''))
            all_facts.extend([(f, doc) for f in facts])

    if not all_facts:
        print("No facts extracted")
        return

    print(f"Extracted {len(all_facts)} facts from top 10 documents\n")

    # Score facts with different configurations
    scorer_no_emb = FactImportanceScorer(use_embeddings=False)
    scorer_with_emb = FactImportanceScorer(use_embeddings=True)

    # Score all facts
    scored_facts_no_emb = []
    scored_facts_with_emb = []

    for fact, doc in all_facts:
        # Without embeddings
        score_no_emb = scorer_no_emb.score(
            fact, question_type, entity_root, None, {}
        )
        scored_facts_no_emb.append((fact, score_no_emb, doc))

        # With embeddings
        score_with_emb = scorer_with_emb.score(
            fact, question_type, entity_root, list(roots), {}
        )
        scored_facts_with_emb.append((fact, score_with_emb, doc))

    # Sort by score
    scored_facts_no_emb.sort(key=lambda x: x[1].final_score, reverse=True)
    scored_facts_with_emb.sort(key=lambda x: x[1].final_score, reverse=True)

    # Find facts containing expected answer
    def contains_answer(fact):
        return expected.lower() in str(fact).lower()

    correct_facts_no_emb = [(f, s, i) for i, (f, s, d) in enumerate(scored_facts_no_emb) if contains_answer(f)]
    correct_facts_with_emb = [(f, s, i) for i, (f, s, d) in enumerate(scored_facts_with_emb) if contains_answer(f)]

    print("WITHOUT EMBEDDINGS:")
    print("-" * 80)
    print(f"Top 3 facts:")
    for i, (fact, score, doc) in enumerate(scored_facts_no_emb[:3], 1):
        has_answer = "✓" if contains_answer(fact) else " "
        print(f"  {i}. [{has_answer}] {fact}")
        print(f"      Score: {score}")
        print()

    if correct_facts_no_emb:
        fact, score, rank = correct_facts_no_emb[0]
        print(f"First correct fact at rank {rank+1}: {score.final_score:.3f}")
    else:
        print("No correct facts found")
    print()

    print("WITH EMBEDDINGS:")
    print("-" * 80)
    print(f"Top 3 facts:")
    for i, (fact, score, doc) in enumerate(scored_facts_with_emb[:3], 1):
        has_answer = "✓" if contains_answer(fact) else " "
        print(f"  {i}. [{has_answer}] {fact}")
        print(f"      Score: {score}")
        print()

    if correct_facts_with_emb:
        fact, score, rank = correct_facts_with_emb[0]
        print(f"First correct fact at rank {rank+1}: {score.final_score:.3f}")
    else:
        print("No correct facts found")
    print()

    # Compare rankings
    print("RANKING COMPARISON:")
    print("-" * 80)

    if correct_facts_no_emb and correct_facts_with_emb:
        rank_no_emb = correct_facts_no_emb[0][2]
        rank_with_emb = correct_facts_with_emb[0][2]

        if rank_no_emb < rank_with_emb:
            print(f"❌ Embeddings HURT: correct fact ranked {rank_no_emb+1} → {rank_with_emb+1}")
        elif rank_no_emb > rank_with_emb:
            print(f"✓ Embeddings HELP: correct fact ranked {rank_no_emb+1} → {rank_with_emb+1}")
        else:
            print(f"= No change: correct fact at rank {rank_no_emb+1}")

        # Show score breakdown comparison
        score_no_emb = correct_facts_no_emb[0][1]
        score_with_emb = correct_facts_with_emb[0][1]

        print()
        print("Score breakdown for CORRECT fact:")
        print(f"  Without emb: {score_no_emb}")
        print(f"  With emb:    {score_with_emb}")
        print()
        print(f"  Embedding similarity: {score_with_emb.embedding_similarity:.3f}")
        print(f"  Effect on final score: {(score_with_emb.final_score - score_no_emb.final_score):.3f}")

    # Show embedding similarity distribution
    print()
    print("EMBEDDING SIMILARITY DISTRIBUTION:")
    print("-" * 80)

    emb_scores = [s.embedding_similarity for _, s, _ in scored_facts_with_emb]
    if emb_scores:
        print(f"  Min:  {min(emb_scores):.3f}")
        print(f"  Mean: {sum(emb_scores)/len(emb_scores):.3f}")
        print(f"  Max:  {max(emb_scores):.3f}")
        print(f"  Range: {max(emb_scores) - min(emb_scores):.3f}")

        # Count neutral scores (0.5)
        neutral_count = sum(1 for s in emb_scores if abs(s - 0.5) < 0.01)
        print(f"  Neutral (0.5): {neutral_count}/{len(emb_scores)} ({100*neutral_count/len(emb_scores):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--whoosh-index', type=Path, default=Path('data/indexes/whoosh_fts'))
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--limit', type=int, default=5, help='Number of questions to analyze')

    args = parser.parse_args()

    # Load test set
    test_questions = []
    with open(args.test_set) as f:
        for line in f:
            test_questions.append(json.loads(line))

    # Filter to questions where retrieval succeeds (for clearer analysis)
    test_questions = test_questions[:args.limit]

    # Initialize retriever
    print("Initializing retriever...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.db
    )
    print("✓ Initialized\n")

    # Analyze questions
    for item in test_questions:
        analyze_question(
            item['question'],
            item['expected_keywords'][0],
            retriever,
            args.top_k
        )


if __name__ == '__main__':
    main()
