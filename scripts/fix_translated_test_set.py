#!/usr/bin/env python3
"""
Fix Translated Test Set - Convert English Keywords to Esperanto

The translated_qa_diverse.jsonl has good Esperanto questions but English keywords.
This script translates the English answer keywords to Esperanto keywords that would
actually appear in generated answers.

Strategy:
1. Extract name stems from English answers (e.g., "Harry Truman" → "truman")
2. Translate common words (president → prezidento, country names, etc.)
3. Extract keywords from the Esperanto question itself
4. Combine into short keyword stems (4-6 chars) for fuzzy matching

Usage:
    python scripts/fix_translated_test_set.py \
      --input data/test_sets/translated_qa_diverse.jsonl \
      --output data/test_sets/translated_qa_diverse_fixed.jsonl

    # Validate against corpus (check if keywords actually appear)
    python scripts/fix_translated_test_set.py \
      --input data/test_sets/translated_qa_diverse.jsonl \
      --output data/test_sets/translated_qa_diverse_fixed.jsonl \
      --validate-corpus data/corpus/unified_corpus.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Set
import sys

# Common English → Esperanto translations for keywords
COMMON_TRANSLATIONS = {
    # Countries
    'america': 'usona',
    'american': 'usona',
    'united states': 'usona',
    'usa': 'usona',
    'britain': 'brita',
    'british': 'brita',
    'england': 'angla',
    'english': 'angla',
    'france': 'franca',
    'french': 'franca',
    'germany': 'germa',
    'german': 'germa',
    'italy': 'itala',
    'italian': 'itala',
    'spain': 'hispana',
    'spanish': 'hispana',
    'china': 'ĉina',
    'chinese': 'ĉina',
    'japan': 'japana',
    'japanese': 'japana',
    'russia': 'rusa',
    'russian': 'rusa',

    # Titles/Roles
    'president': 'prezidento',
    'presidency': 'prezidento',
    'king': 'reĝo',
    'queen': 'reĝino',
    'prime minister': 'ĉefministro',
    'author': 'aŭtoro',
    'writer': 'verkisto',
    'actor': 'aktoro',
    'singer': 'kantisto',
    'painter': 'pentristo',
    'scientist': 'sciencisto',
    'inventor': 'inventisto',

    # Common words
    'book': 'libro',
    'war': 'milito',
    'battle': 'batalo',
    'city': 'urbo',
    'country': 'lando',
    'language': 'lingvo',
    'year': 'jaro',
    'century': 'jarcento',
}

def extract_name_stems(name: str) -> List[str]:
    """
    Extract searchable stems from a person/place name.

    Examples:
        "Harry S. Truman" → ["harry", "truman"]
        "New York" → ["york"]  (skip "new" - too common)
        "Leonardo da Vinci" → ["leonardo", "vinci"]
    """
    # Remove common prefixes/suffixes
    name = name.lower()

    # Remove punctuation
    name = re.sub(r'[.,\'\"]', '', name)

    # Split into words
    words = name.split()

    # Filter out very common words
    common = {'the', 'of', 'and', 'in', 'a', 'de', 'da', 'von', 'van', 'el', 'la'}
    words = [w for w in words if w not in common]

    # Keep words >= 4 chars (or single letter middle initials)
    stems = []
    for w in words:
        if len(w) >= 4:
            stems.append(w[:6])  # Use first 6 chars as stem

    return stems


def translate_phrase(phrase: str) -> List[str]:
    """
    Translate an English phrase to Esperanto keywords.

    Examples:
        "President of the United States" → ["prezidento", "usona"]
        "Harry Truman" → ["truman", "harry"]
        "World War II" → ["milito", "monda"]
    """
    phrase_lower = phrase.lower()
    keywords = []

    # Check for multi-word translations first
    for eng, eo in COMMON_TRANSLATIONS.items():
        if eng in phrase_lower:
            keywords.append(eo)
            phrase_lower = phrase_lower.replace(eng, '')  # Remove to avoid duplicates

    # Extract name stems
    name_stems = extract_name_stems(phrase)
    keywords.extend(name_stems)

    # Remove duplicates, keep order
    seen = set()
    result = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result


def extract_keywords_from_question(question: str) -> List[str]:
    """
    Extract potential answer keywords from the Esperanto question itself.

    Examples:
        "Kiu estis Prezidanto?" → ["prezidento"]
        "Kio estas lingvo?" → ["lingvo"]
    """
    keywords = []

    # Remove question words
    question = re.sub(r'\b(kiu|kio|kie|kiam|kial|kiel|kiom|kiun?|kion?)\b', '', question, flags=re.IGNORECASE)

    # Remove common verbs
    question = re.sub(r'\b(estas|estis|estos|estus)\b', '', question, flags=re.IGNORECASE)

    # Extract nouns (words ending in -o, -oj, -on, -ojn)
    nouns = re.findall(r'\b(\w+)o[jn]*\b', question, flags=re.IGNORECASE)

    # Get stems (remove 'o' ending)
    for noun in nouns:
        if len(noun) >= 3:
            keywords.append(noun.lower()[:6])

    return keywords


def translate_answer_to_esperanto(answer: str) -> str:
    """
    Translate an English answer to Esperanto.

    Strategy:
    - Keep proper names (Harry Truman → Harry Truman)
    - Translate titles/roles (President → Prezidanto)
    - Translate common words
    - Create readable Esperanto phrase

    Examples:
        "President Harry Truman" → "Prezidanto Harry Truman"
        "The United States" → "Usono" or "Unuiĝintaj Ŝtatoj"
        "World War II" → "Dua Mondmilito"
    """
    result = answer

    # Translate roles/titles at start of phrase
    if result.lower().startswith('president '):
        result = 'Prezidanto ' + result[10:]
    elif result.lower().startswith('presidency of '):
        result = 'Prezidanto ' + result[14:]
    elif result.lower().startswith('king '):
        result = 'Reĝo ' + result[5:]
    elif result.lower().startswith('queen '):
        result = 'Reĝino ' + result[6:]
    elif result.lower().startswith('prime minister '):
        result = 'Ĉefministro ' + result[15:]

    # Translate country names
    for eng, eo in COMMON_TRANSLATIONS.items():
        if len(eng.split()) > 1:  # Multi-word phrases
            pattern = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
            if pattern.search(result.lower()):
                # Capitalize if it's a proper noun (country, etc.)
                eo_capitalized = eo.capitalize() if eng in ['america', 'britain', 'france', 'germany', 'italy', 'spain', 'china', 'japan', 'russia'] else eo
                result = pattern.sub(eo_capitalized, result)

    # Remove common filler words at end
    result = re.sub(r',?\s+(the|a|an)$', '', result, flags=re.IGNORECASE)

    # If answer is just a role without a name, make it Esperanto noun
    if result.lower() in ['president', 'king', 'queen', 'author', 'writer']:
        return COMMON_TRANSLATIONS.get(result.lower(), result)

    return result.strip()


def fix_question(question_data: dict) -> dict:
    """Fix a single question by translating English answer and keywords to Esperanto."""

    # Get English answer variants
    answer_variants = question_data.get('answer_variants', [])
    if not answer_variants:
        answer_variants = [question_data.get('answer', '')]

    # Translate the main answer field
    original_answer = question_data.get('answer', '')
    esperanto_answer = translate_answer_to_esperanto(original_answer)

    # Translate each variant to keywords
    esperanto_keywords = set()

    for variant in answer_variants[:5]:  # Only use first 5 variants
        translated = translate_phrase(variant)
        esperanto_keywords.update(translated)

    # Also extract keywords from the Esperanto question
    question_text = question_data.get('question', '')
    question_keywords = extract_keywords_from_question(question_text)
    esperanto_keywords.update(question_keywords)

    # Convert to list, sort by length (longer = more specific)
    keywords = sorted(esperanto_keywords, key=len, reverse=True)

    # Limit to top 5 keywords
    keywords = keywords[:5]

    # Update question data
    fixed = question_data.copy()
    fixed['answer'] = esperanto_answer  # NEW: Translated answer
    fixed['expected_keywords'] = keywords
    fixed['original_english_answer'] = original_answer  # NEW: Keep original for reference
    fixed['original_english_keywords'] = question_data.get('expected_keywords', [])

    return fixed


def validate_against_corpus(keywords: List[str], corpus_path: Path, sample_size: int = 10000) -> dict:
    """
    Check if keywords actually appear in the Esperanto corpus.
    Returns stats on keyword coverage.
    """
    keyword_counts = {kw: 0 for kw in keywords}

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            data = json.loads(line)
            text = data.get('text', '').lower()

            for kw in keywords:
                if kw in text:
                    keyword_counts[kw] += 1

    return keyword_counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=Path, required=True,
                       help='Input translated test set (with English keywords)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output fixed test set (with Esperanto keywords)')
    parser.add_argument('--validate-corpus', type=Path,
                       help='Validate keywords against corpus')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show example translations')

    args = parser.parse_args()

    print("="*80)
    print("FIXING TRANSLATED TEST SET")
    print("="*80)
    print()
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()

    # Load questions
    questions = []
    with open(args.input) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions")
    print()

    # Fix each question
    fixed_questions = []
    examples_shown = 0

    for i, q in enumerate(questions, 1):
        fixed = fix_question(q)
        fixed_questions.append(fixed)

        # Show examples
        if args.show_examples and examples_shown < 10:
            print(f"Example {examples_shown + 1}:")
            print(f"  Question: {q['question']}")
            print(f"  English answer: {q.get('answer', 'N/A')}")
            print(f"  → Esperanto answer: {fixed['answer']}")
            print(f"  English keywords: {q.get('expected_keywords', [])[:3]}")
            print(f"  → Esperanto keywords: {fixed['expected_keywords']}")
            print()
            examples_shown += 1

        if i % 100 == 0:
            print(f"  Processed {i}/{len(questions)} questions...")

    print()
    print(f"✓ Fixed {len(fixed_questions)} questions")
    print()

    # Validate against corpus if requested
    if args.validate_corpus:
        print("Validating keywords against corpus...")

        # Collect all keywords
        all_keywords = set()
        for q in fixed_questions:
            all_keywords.update(q['expected_keywords'])

        counts = validate_against_corpus(list(all_keywords), args.validate_corpus)

        # Show stats
        found = sum(1 for c in counts.values() if c > 0)
        not_found = len(counts) - found

        print(f"  Keywords found in corpus: {found}/{len(counts)} ({found/len(counts)*100:.1f}%)")
        print(f"  Keywords not found: {not_found}")
        print()

        if not_found > 0:
            print("  Keywords not found in corpus:")
            for kw, count in sorted(counts.items(), key=lambda x: x[1]):
                if count == 0:
                    print(f"    - {kw}")
            print()

    # Save fixed questions
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        for q in fixed_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"✓ Saved fixed test set to: {args.output}")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Questions processed: {len(fixed_questions)}")
    print(f"Output file: {args.output}")
    print()
    print("Next steps:")
    print("  1. Review sample questions (--show-examples)")
    print("  2. Validate keywords (--validate-corpus)")
    print("  3. Use in evaluation:")
    print(f"     python scripts/evaluate_pipeline_comprehensive.py \\")
    print(f"       --test-set {args.output} \\")
    print(f"       --output results/translated_eval.json")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
