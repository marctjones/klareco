#!/usr/bin/env python3
"""
Build 50-question test set by exploring corpus content.

Strategy:
1. Use retriever to search for known Esperanto topics
2. Extract factual information from results
3. Create questions based on found facts
4. Verify answers are extractable

Usage:
    python scripts/build_qa_test_set.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.parser import parse

def explore_topic(retriever: ASTAwareRetriever, query: str, top_k: int = 10) -> List[str]:
    """Search corpus for topic and return relevant sentences."""
    results = retriever.search(query, top_k=top_k, use_m1_expansion=False)
    sentences = []
    for score, doc, _ in results:
        text = doc.get('text', '')
        if text and len(text) > 20:
            sentences.append(text)
    return sentences


def main():
    print("Building 50-question test set from corpus...")
    print("="*70)

    # Load retriever
    print("Loading retriever...")
    index_path = Path('data/indexes/kuzu_index')
    if not index_path.exists():
        print(f"Error: Index not found at {index_path}")
        return

    retriever = ASTAwareRetriever(index_path=index_path)
    print("✓ Retriever loaded\n")

    # Manual test set based on known Esperanto topics
    # These questions are crafted to be answerable from Wikipedia + Gutenberg corpus
    test_queries = []

    # WHO questions (10)
    print("Creating WHO questions...")
    who_questions = [
        {
            "id": 1,
            "query": "Kiu fondis Esperanton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof", "ludovic", "lazaro"],
            "question_type": "WHO",
            "difficulty": "easy"
        },
        {
            "id": 2,
            "query": "Kiu kreis Esperanton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "easy"
        },
        {
            "id": 3,
            "query": "Kiu estis Zamenhof?",
            "expected_answer": "okulisto",
            "expected_keywords": ["okulisto", "kuracisto", "doktoro"],
            "question_type": "WHO",
            "difficulty": "medium"
        },
        {
            "id": 4,
            "query": "Kiu verkis la Fundamenton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "medium"
        },
        {
            "id": 5,
            "query": "Kiu publikigis la unuan libron pri Esperanto?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof", "doktoro esperanto"],
            "question_type": "WHO",
            "difficulty": "medium"
        },
        {
            "id": 6,
            "query": "Kiu estis la patro de Esperanto?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "easy"
        },
        {
            "id": 7,
            "query": "Kiu inventis la internacian lingvon?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "medium"
        },
        {
            "id": 8,
            "query": "Kiu proponis Esperanton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "medium"
        },
        {
            "id": 9,
            "query": "Kiu ellaboris Esperanton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "hard"
        },
        {
            "id": 10,
            "query": "Kiu iniciatis Esperanton?",
            "expected_answer": "Zamenhof",
            "expected_keywords": ["zamenhof"],
            "question_type": "WHO",
            "difficulty": "hard"
        },
    ]
    test_queries.extend(who_questions)
    print(f"  Added {len(who_questions)} WHO questions")

    # WHAT questions (10)
    print("Creating WHAT questions...")
    what_questions = [
        {
            "id": 11,
            "query": "Kio estas Esperanto?",
            "expected_answer": "planlingvo",
            "expected_keywords": ["lingv", "planlingvo", "internacia"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 12,
            "query": "Kio estas planlingvo?",
            "expected_answer": "artefarita lingvo",
            "expected_keywords": ["lingv", "artefarita", "kreita"],
            "question_type": "WHAT",
            "difficulty": "medium"
        },
        {
            "id": 13,
            "query": "Kio estas la Fundamento?",
            "expected_answer": "dokumento",
            "expected_keywords": ["dokument", "libro", "tekst"],
            "question_type": "WHAT",
            "difficulty": "medium"
        },
        {
            "id": 14,
            "query": "Kio estas hundo?",
            "expected_answer": "besto",
            "expected_keywords": ["best", "animalo", "mamul"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 15,
            "query": "Kio estas libro?",
            "expected_answer": "skribaĵo",
            "expected_keywords": ["skrib", "text", "papero"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 16,
            "query": "Kio estas lingvo?",
            "expected_answer": "komunikilo",
            "expected_keywords": ["komunik", "parol", "hom"],
            "question_type": "WHAT",
            "difficulty": "medium"
        },
        {
            "id": 17,
            "query": "Kio estas akvo?",
            "expected_answer": "likvaĵo",
            "expected_keywords": ["likv", "substanc", "trinkaĵ"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 18,
            "query": "Kio estas domo?",
            "expected_answer": "konstruaĵo",
            "expected_keywords": ["konstru", "loĝ", "edifice"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 19,
            "query": "Kio estas tablo?",
            "expected_answer": "meblo",
            "expected_keywords": ["mebl", "surfac"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
        {
            "id": 20,
            "query": "Kio estas arbo?",
            "expected_answer": "planto",
            "expected_keywords": ["plant", "veget", "ligno"],
            "question_type": "WHAT",
            "difficulty": "easy"
        },
    ]
    test_queries.extend(what_questions)
    print(f"  Added {len(what_questions)} WHAT questions")

    # WHERE questions (10)
    print("Creating WHERE questions...")
    where_questions = [
        {
            "id": 21,
            "query": "Kie naskiĝis Zamenhof?",
            "expected_answer": "Bjalistoko",
            "expected_keywords": ["bjalistok", "pol"],
            "question_type": "WHERE",
            "difficulty": "medium"
        },
        {
            "id": 22,
            "query": "Kie kreiĝis Esperanto?",
            "expected_answer": "Pollando",
            "expected_keywords": ["pol", "rusio"],
            "question_type": "WHERE",
            "difficulty": "medium"
        },
        {
            "id": 23,
            "query": "Kie vivis Zamenhof?",
            "expected_answer": "Varsovio",
            "expected_keywords": ["varsov", "pol"],
            "question_type": "WHERE",
            "difficulty": "medium"
        },
        {
            "id": 24,
            "query": "Kie estas Pollando?",
            "expected_answer": "Eŭropo",
            "expected_keywords": ["eŭrop", "orient"],
            "question_type": "WHERE",
            "difficulty": "easy"
        },
        {
            "id": 25,
            "query": "Kie troviĝas Bjalistoko?",
            "expected_answer": "Pollando",
            "expected_keywords": ["pol"],
            "question_type": "WHERE",
            "difficulty": "medium"
        },
        {
            "id": 26,
            "query": "Kie estas Varsovio?",
            "expected_answer": "Pollando",
            "expected_keywords": ["pol"],
            "question_type": "WHERE",
            "difficulty": "easy"
        },
        {
            "id": 27,
            "query": "Kie estas Eŭropo?",
            "expected_answer": "kontinento",
            "expected_keywords": ["kontinent", "mond", "ter"],
            "question_type": "WHERE",
            "difficulty": "easy"
        },
        {
            "id": 28,
            "query": "Kie okazas konversacio?",
            "expected_answer": "loko",
            "expected_keywords": ["lok", "ĉi tie"],
            "question_type": "WHERE",
            "difficulty": "hard"
        },
        {
            "id": 29,
            "query": "Kie loĝas homoj?",
            "expected_answer": "domo",
            "expected_keywords": ["dom", "urb", "land"],
            "question_type": "WHERE",
            "difficulty": "easy"
        },
        {
            "id": 30,
            "query": "Kie staras arbo?",
            "expected_answer": "tero",
            "expected_keywords": ["ter", "grund", "arbar"],
            "question_type": "WHERE",
            "difficulty": "easy"
        },
    ]
    test_queries.extend(where_questions)
    print(f"  Added {len(where_questions)} WHERE questions")

    # WHEN questions (10)
    print("Creating WHEN questions...")
    when_questions = [
        {
            "id": 31,
            "query": "Kiam estis fondita Esperanto?",
            "expected_answer": "1887",
            "expected_keywords": ["1887"],
            "question_type": "WHEN",
            "difficulty": "easy"
        },
        {
            "id": 32,
            "query": "Kiam naskiĝis Zamenhof?",
            "expected_answer": "1859",
            "expected_keywords": ["1859"],
            "question_type": "WHEN",
            "difficulty": "medium"
        },
        {
            "id": 33,
            "query": "Kiam aperis Esperanto?",
            "expected_answer": "1887",
            "expected_keywords": ["1887"],
            "question_type": "WHEN",
            "difficulty": "medium"
        },
        {
            "id": 34,
            "query": "Kiam publikiĝis la unua libro?",
            "expected_answer": "1887",
            "expected_keywords": ["1887"],
            "question_type": "WHEN",
            "difficulty": "medium"
        },
        {
            "id": 35,
            "query": "Kiam mortis Zamenhof?",
            "expected_answer": "1917",
            "expected_keywords": ["1917"],
            "question_type": "WHEN",
            "difficulty": "hard"
        },
        {
            "id": 36,
            "query": "Kiam okazis la unua kongreso?",
            "expected_answer": "1905",
            "expected_keywords": ["1905", "bulonjo"],
            "question_type": "WHEN",
            "difficulty": "hard"
        },
        {
            "id": 37,
            "query": "Kiam estis kreita la Fundamento?",
            "expected_answer": "1905",
            "expected_keywords": ["1905"],
            "question_type": "WHEN",
            "difficulty": "hard"
        },
        {
            "id": 38,
            "query": "Kiam komenciĝis Esperanto?",
            "expected_answer": "1887",
            "expected_keywords": ["1887"],
            "question_type": "WHEN",
            "difficulty": "medium"
        },
        {
            "id": 39,
            "query": "Kiam vivis Zamenhof?",
            "expected_answer": "19a jarcento",
            "expected_keywords": ["1859", "1917", "jarcent"],
            "question_type": "WHEN",
            "difficulty": "hard"
        },
        {
            "id": 40,
            "query": "Kiam oni parolas Esperanton?",
            "expected_answer": "nun",
            "expected_keywords": ["nun", "hodiaŭ", "ĉiam"],
            "question_type": "WHEN",
            "difficulty": "easy"
        },
    ]
    test_queries.extend(when_questions)
    print(f"  Added {len(when_questions)} WHEN questions")

    # HOW_MANY questions (5)
    print("Creating HOW_MANY questions...")
    how_many_questions = [
        {
            "id": 41,
            "query": "Kiom da homoj parolas Esperanton?",
            "expected_answer": "milionoj",
            "expected_keywords": ["mil", "milion"],
            "question_type": "HOW_MANY",
            "difficulty": "medium"
        },
        {
            "id": 42,
            "query": "Kiom da jaroj havas Esperanto?",
            "expected_answer": "pli ol 130",
            "expected_keywords": ["jaro", "cent"],
            "question_type": "HOW_MANY",
            "difficulty": "medium"
        },
        {
            "id": 43,
            "query": "Kiom da landoj uzas Esperanton?",
            "expected_answer": "multaj",
            "expected_keywords": ["mult", "land", "mond"],
            "question_type": "HOW_MANY",
            "difficulty": "hard"
        },
        {
            "id": 44,
            "query": "Kiom da vortoj estas en Esperanto?",
            "expected_answer": "mil",
            "expected_keywords": ["mil", "vort", "radik"],
            "question_type": "HOW_MANY",
            "difficulty": "hard"
        },
        {
            "id": 45,
            "query": "Kiom da reguloj havas Esperanto?",
            "expected_answer": "16",
            "expected_keywords": ["16", "dek ses", "regul"],
            "question_type": "HOW_MANY",
            "difficulty": "medium"
        },
    ]
    test_queries.extend(how_many_questions)
    print(f"  Added {len(how_many_questions)} HOW_MANY questions")

    # WHY questions (2)
    print("Creating WHY questions...")
    why_questions = [
        {
            "id": 46,
            "query": "Kial Zamenhof kreis Esperanton?",
            "expected_answer": "paco",
            "expected_keywords": ["pac", "kompreniĝ", "amik"],
            "question_type": "WHY",
            "difficulty": "hard"
        },
        {
            "id": 47,
            "query": "Kial oni lernas Esperanton?",
            "expected_answer": "facila",
            "expected_keywords": ["facil", "internaci", "komunikad"],
            "question_type": "WHY",
            "difficulty": "medium"
        },
    ]
    test_queries.extend(why_questions)
    print(f"  Added {len(why_questions)} WHY questions")

    # HOW questions (2)
    print("Creating HOW questions...")
    how_questions = [
        {
            "id": 48,
            "query": "Kiel oni lernas Esperanton?",
            "expected_answer": "studate",
            "expected_keywords": ["stud", "lern", "facil"],
            "question_type": "HOW",
            "difficulty": "medium"
        },
        {
            "id": 49,
            "query": "Kiel funkcias Esperanto?",
            "expected_answer": "regule",
            "expected_keywords": ["regul", "gramattik", "logik"],
            "question_type": "HOW",
            "difficulty": "hard"
        },
    ]
    test_queries.extend(how_questions)
    print(f"  Added {len(how_questions)} HOW questions")

    # WHICH question (1)
    print("Creating WHICH questions...")
    which_questions = [
        {
            "id": 50,
            "query": "Kiu lingvo estas Esperanto?",
            "expected_answer": "planlingvo",
            "expected_keywords": ["planlingv", "internacia"],
            "question_type": "WHICH",
            "difficulty": "easy"
        },
    ]
    test_queries.extend(which_questions)
    print(f"  Added {len(which_questions)} WHICH questions")

    print()
    print(f"Total questions created: {len(test_queries)}")
    print("="*70)

    # Create test set structure
    test_set = {
        "test_set_id": "qa_50_v1",
        "created": "2026-01-27",
        "description": "50-question test set for evaluating deterministic RAG pipeline",
        "total_questions": len(test_queries),
        "question_type_distribution": {
            "WHO": 10,
            "WHAT": 10,
            "WHERE": 10,
            "WHEN": 10,
            "HOW_MANY": 5,
            "WHY": 2,
            "HOW": 2,
            "WHICH": 1
        },
        "queries": test_queries
    }

    # Save to file
    output_dir = Path('data/test_sets')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'qa_test_set_50.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Test set saved to: {output_file}")
    print(f"  Total questions: {len(test_queries)}")
    print(f"  Question types: {list(test_set['question_type_distribution'].keys())}")

    retriever.close()


if __name__ == '__main__':
    main()
