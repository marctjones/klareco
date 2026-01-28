#!/usr/bin/env python3
"""
Validate Test Set Against Corpus

For each question:
1. Search corpus for expected answer
2. Find sentence containing answer
3. Analyze sentence complexity
4. Determine retrieval strategy needed

Usage:
    python scripts/validate_test_set.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.ast_aware_retriever import ASTAwareRetriever


def search_corpus_for_answer(
    retriever: ASTAwareRetriever,
    query: str,
    expected_keywords: List[str],
    top_k: int = 50
) -> Optional[Dict]:
    """
    Search corpus for sentence containing expected answer.

    Returns dict with:
    - found: bool
    - sentence: str (if found)
    - rank: int (position in results)
    - score: float (retrieval score)
    - match_type: str (which keyword matched)
    """
    results = retriever.search(query, top_k=top_k, use_m1_expansion=False)

    for rank, (score, doc, stats) in enumerate(results, 1):
        text = doc.get('text', '').lower()

        # Check if any expected keyword is in this sentence
        for keyword in expected_keywords:
            if keyword.lower() in text:
                return {
                    'found': True,
                    'sentence': doc.get('text', ''),
                    'rank': rank,
                    'score': score,
                    'match_type': keyword,
                    'full_doc': doc
                }

    # Not found in top-K
    return {
        'found': False,
        'sentence': None,
        'rank': None,
        'score': None,
        'match_type': None
    }


def analyze_sentence_complexity(sentence: str) -> Dict:
    """
    Analyze complexity of sentence containing answer.

    Returns:
    - length: int (character count)
    - word_count: int
    - parse_success: bool
    - num_entities: int (capitalized words)
    - has_numbers: bool
    - has_punctuation: bool
    - estimated_clauses: int
    - complexity_level: str (simple/medium/complex)
    """
    # Basic stats
    length = len(sentence)
    words = sentence.split()
    word_count = len(words)

    # Try to parse
    parse_success = False
    try:
        ast = parse(sentence)
        parse_success = (ast.get('parse_statistics', {}).get('success_rate', 0) > 0.8)
    except:
        pass

    # Count entities (capitalized words, not at sentence start)
    entities = [w for i, w in enumerate(words) if i > 0 and w and w[0].isupper()]
    num_entities = len(entities)

    # Check for numbers
    has_numbers = bool(re.search(r'\d', sentence))

    # Check punctuation (commas, semicolons indicate clauses)
    punctuation_count = sentence.count(',') + sentence.count(';') + sentence.count(':')
    has_punctuation = punctuation_count > 0

    # Estimate clause count (rough)
    estimated_clauses = 1 + punctuation_count

    # Complexity level
    if word_count < 15 and estimated_clauses == 1:
        complexity = 'simple'
    elif word_count < 30 and estimated_clauses <= 2:
        complexity = 'medium'
    else:
        complexity = 'complex'

    return {
        'length': length,
        'word_count': word_count,
        'parse_success': parse_success,
        'num_entities': num_entities,
        'has_numbers': has_numbers,
        'has_punctuation': has_punctuation,
        'estimated_clauses': estimated_clauses,
        'complexity_level': complexity,
        'entities': entities[:5]  # First 5 entities
    }


def determine_retrieval_strategy(
    query: str,
    sentence: str,
    match_type: str,
    complexity: Dict,
    rank: int
) -> Dict:
    """
    Determine what retrieval strategies would help find this sentence.

    Returns dict with:
    - keyword_match: bool (direct keyword in query)
    - needs_synonym_expansion: bool
    - needs_entity_recognition: bool
    - needs_semantic_understanding: bool
    - needs_reranking: bool (if found but low rank)
    - difficulty: str (easy/medium/hard)
    """
    query_lower = query.lower()
    sentence_lower = sentence.lower()

    # Check if query keywords appear directly
    query_words = set(query_lower.split())
    sentence_words = set(sentence_lower.split())
    keyword_match = len(query_words & sentence_words) > 2

    # Check if entity from query appears in sentence
    query_entities = [w for w in query.split() if w and w[0].isupper()]
    sentence_entities = complexity.get('entities', [])
    entity_match = any(qe.lower() in [se.lower() for se in sentence_entities] for qe in query_entities)

    # Determine needs
    needs_synonym_expansion = not keyword_match and rank and rank > 10
    needs_entity_recognition = len(query_entities) > 0 and not entity_match
    needs_semantic_understanding = complexity['complexity_level'] == 'complex' or not keyword_match
    needs_reranking = rank and rank > 5

    # Overall difficulty
    if rank and rank <= 3 and keyword_match:
        difficulty = 'easy'
    elif rank and rank <= 10 or (keyword_match and entity_match):
        difficulty = 'medium'
    else:
        difficulty = 'hard'

    return {
        'keyword_match': keyword_match,
        'entity_match': entity_match,
        'needs_synonym_expansion': needs_synonym_expansion,
        'needs_entity_recognition': needs_entity_recognition,
        'needs_semantic_understanding': needs_semantic_understanding,
        'needs_reranking': needs_reranking,
        'difficulty': difficulty,
        'retrieval_rank': rank
    }


def main():
    print("=" * 70)
    print("Test Set Validation & Complexity Analysis")
    print("=" * 70)

    # Load test set
    test_set_path = Path('data/test_sets/qa_test_set_50.json')
    if not test_set_path.exists():
        print(f"Error: Test set not found at {test_set_path}")
        return

    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    queries = test_set.get('queries', [])
    print(f"Loaded {len(queries)} questions\n")

    # Load retriever
    print("Loading retriever...")
    index_path = Path('data/indexes/kuzu_index')
    retriever = ASTAwareRetriever(index_path=index_path)
    print("✓ Retriever loaded\n")

    # Validate each question
    results = []
    found_count = 0

    for i, query_info in enumerate(queries, 1):
        query = query_info['query']
        expected_keywords = query_info['expected_keywords']
        question_type = query_info.get('question_type', 'UNKNOWN')

        print(f"\n[{i}/{len(queries)}] {question_type}: {query}")
        print(f"  Expected keywords: {expected_keywords}")

        # Search for answer
        search_result = search_corpus_for_answer(
            retriever,
            query,
            expected_keywords,
            top_k=100  # Search deeper
        )

        if search_result['found']:
            found_count += 1
            sentence = search_result['sentence']
            rank = search_result['rank']

            print(f"  ✓ FOUND at rank {rank}")
            print(f"    Match: '{search_result['match_type']}'")
            print(f"    Sentence: {sentence[:100]}...")

            # Analyze complexity
            complexity = analyze_sentence_complexity(sentence)
            print(f"    Complexity: {complexity['complexity_level']} ({complexity['word_count']} words, {complexity['estimated_clauses']} clauses)")

            # Determine strategy
            strategy = determine_retrieval_strategy(
                query,
                sentence,
                search_result['match_type'],
                complexity,
                rank
            )
            print(f"    Difficulty: {strategy['difficulty']}")

            if strategy['needs_synonym_expansion']:
                print(f"    ⚠️  Needs synonym expansion")
            if strategy['needs_entity_recognition']:
                print(f"    ⚠️  Needs entity recognition")
            if strategy['needs_reranking']:
                print(f"    ⚠️  Needs reranking (rank {rank} → top 5)")

            results.append({
                'question_id': query_info.get('id'),
                'query': query,
                'question_type': question_type,
                'found': True,
                'rank': rank,
                'score': search_result['score'],
                'sentence': sentence,
                'complexity': complexity,
                'strategy': strategy
            })
        else:
            print(f"  ✗ NOT FOUND in top-100")
            print(f"    Answer may not exist in corpus or needs different search terms")

            results.append({
                'question_id': query_info.get('id'),
                'query': query,
                'question_type': question_type,
                'found': False,
                'reason': 'not_in_top_100'
            })

    retriever.close()

    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(queries)}")
    print(f"Answers found in corpus: {found_count}/{len(queries)} ({found_count/len(queries)*100:.1f}%)")
    print(f"Answers NOT found: {len(queries) - found_count}/{len(queries)}")

    # Complexity breakdown
    if found_count > 0:
        complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        rank_ranges = {'1-5': 0, '6-10': 0, '11-20': 0, '21-50': 0, '51-100': 0}

        for r in results:
            if r['found']:
                complexity_counts[r['complexity']['complexity_level']] += 1
                difficulty_counts[r['strategy']['difficulty']] += 1

                rank = r['rank']
                if rank <= 5:
                    rank_ranges['1-5'] += 1
                elif rank <= 10:
                    rank_ranges['6-10'] += 1
                elif rank <= 20:
                    rank_ranges['11-20'] += 1
                elif rank <= 50:
                    rank_ranges['21-50'] += 1
                else:
                    rank_ranges['51-100'] += 1

        print("\nSentence Complexity:")
        for level, count in complexity_counts.items():
            print(f"  {level.capitalize()}: {count}/{found_count}")

        print("\nRetrieval Difficulty:")
        for level, count in difficulty_counts.items():
            print(f"  {level.capitalize()}: {count}/{found_count}")

        print("\nRank Distribution:")
        for range_str, count in rank_ranges.items():
            print(f"  Rank {range_str}: {count}/{found_count}")

        # Strategy needs
        needs_synonym = sum(1 for r in results if r['found'] and r['strategy']['needs_synonym_expansion'])
        needs_entity = sum(1 for r in results if r['found'] and r['strategy']['needs_entity_recognition'])
        needs_semantic = sum(1 for r in results if r['found'] and r['strategy']['needs_semantic_understanding'])
        needs_rerank = sum(1 for r in results if r['found'] and r['strategy']['needs_reranking'])

        print("\nRetrieval Strategies Needed:")
        print(f"  Synonym expansion: {needs_synonym}/{found_count}")
        print(f"  Entity recognition: {needs_entity}/{found_count}")
        print(f"  Semantic understanding: {needs_semantic}/{found_count}")
        print(f"  Reranking (rank > 5): {needs_rerank}/{found_count}")

    # Save detailed results
    output_file = Path('data/test_sets/qa_test_set_50_validation.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    validation_results = {
        'test_set_id': test_set.get('test_set_id'),
        'validation_date': '2026-01-27',
        'total_questions': len(queries),
        'found_count': found_count,
        'not_found_count': len(queries) - found_count,
        'results': results,
        'summary': {
            'complexity_distribution': complexity_counts if found_count > 0 else {},
            'difficulty_distribution': difficulty_counts if found_count > 0 else {},
            'rank_distribution': rank_ranges if found_count > 0 else {},
            'strategy_needs': {
                'synonym_expansion': needs_synonym if found_count > 0 else 0,
                'entity_recognition': needs_entity if found_count > 0 else 0,
                'semantic_understanding': needs_semantic if found_count > 0 else 0,
                'reranking': needs_rerank if found_count > 0 else 0
            }
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
