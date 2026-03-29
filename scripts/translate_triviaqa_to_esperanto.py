#!/usr/bin/env python3
"""
Translate TriviaQA Questions to Esperanto

Translates English trivia questions to Esperanto for evaluation.

Usage:
    python scripts/translate_triviaqa_to_esperanto.py --input data/external/triviaqa_sample_1000.jsonl --output data/test_sets/triviaqa_esperanto_100.jsonl --limit 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: This is a placeholder for translation logic
# In production, you would use:
# 1. Machine translation API (Google Translate, DeepL, etc.)
# 2. Manual translation for high-quality test set
# 3. Rule-based translation for simple question patterns

# For now, we'll create a simple manual translation mapping for common question patterns
QUESTION_PATTERNS = {
    # WHO questions
    r"Who was (.+)": "Kiu estis {0}",
    r"Who (.+)": "Kiu {0}",
    r"Which (.+) won": "Kiu {0} gajnis",

    # WHAT questions
    r"What is (.+)": "Kio estas {0}",
    r"What (.+)": "Kio {0}",

    # WHERE questions
    r"Where (.+)": "Kie {0}",
    r"In which (.+)": "En kiu {0}",

    # WHEN questions
    r"When (.+)": "Kiam {0}",
    r"In which year": "En kiu jaro",
    r"In which decade": "En kiu jardeko",

    # HOW MANY questions
    r"How many (.+)": "Kiom da {0}",
}


def classify_question_type(question: str) -> str:
    """Classify question type from English question."""
    question_lower = question.lower()

    if question_lower.startswith('who '):
        return 'WHO'
    elif question_lower.startswith('what '):
        return 'WHAT'
    elif question_lower.startswith('where ') or 'where ' in question_lower:
        return 'WHERE'
    elif question_lower.startswith('when ') or 'when ' in question_lower:
        return 'WHEN'
    elif question_lower.startswith('how many '):
        return 'HOW_MANY'
    elif question_lower.startswith('how '):
        return 'HOW'
    elif question_lower.startswith('why '):
        return 'WHY'
    elif question_lower.startswith('which '):
        return 'WHICH'
    else:
        return 'OTHER'


def extract_keywords_from_answers(answers: List[str]) -> List[str]:
    """
    Extract searchable keywords from answer variations.

    For evaluation, we need short keyword stems that can be found in Esperanto text.
    """
    keywords = set()

    for answer in answers:
        # Split on common separators
        parts = answer.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()

        # Extract significant words (>3 chars, not common words)
        common_words = {'the', 'of', 'and', 'in', 'to', 'a', 'an', 'for', 'on', 'at', 'by', 'with'}

        for part in parts:
            word = part.strip().lower()
            if len(word) > 3 and word not in common_words:
                # Take first 6 characters as keyword stem
                keywords.add(word[:6])

    return list(keywords)


def translate_question_simple(question: str) -> str:
    """
    Simple placeholder translation.

    In production, this would call a real translation service.
    For now, just mark as needing translation.
    """
    # This is a placeholder - in production you would:
    # 1. Call translation API
    # 2. Use pre-translated mapping
    # 3. Return manual translations

    return f"[TRANSLATE: {question}]"


def convert_triviaqa_to_esperanto_format(
    input_path: Path,
    output_path: Path,
    limit: int = None
) -> int:
    """
    Convert TriviaQA format to Esperanto QA format.

    TriviaQA format:
        {"question": "...", "answer": ["variant1", "variant2", ...]}

    Esperanto QA format:
        {"id": "...", "question": "...", "question_type": "...", "expected_keywords": ["..."]}

    Returns:
        Number of questions converted
    """
    print(f"Converting TriviaQA questions from {input_path}")
    print(f"Output will be saved to {output_path}")
    print()

    if limit:
        print(f"Limiting to first {limit} questions")
        print()

    converted = []
    skipped = 0

    with open(input_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line)
            question_en = data['question']
            answers_en = data['answer']

            # Classify question type
            question_type = classify_question_type(question_en)

            # Extract keywords from answers
            keywords = extract_keywords_from_answers(answers_en)

            if not keywords:
                print(f"⚠ Skipping question {i+1}: No keywords extracted from answers")
                skipped += 1
                continue

            # Translate question (placeholder - would use real translation)
            # For now, keep English and mark for manual translation
            question_eo = translate_question_simple(question_en)

            converted.append({
                'id': f'triviaqa_{i+1}',
                'question': question_eo,
                'question_en': question_en,  # Keep original for reference
                'question_type': question_type,
                'expected_keywords': keywords,
                'answer_variations_en': answers_en[:5],  # Keep top 5 variations for reference
            })

    # Save converted questions
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Converted {len(converted)} questions")
    print(f"✗ Skipped {skipped} questions (no keywords)")
    print(f"✓ Saved to {output_path}")
    print()
    print("NOTE: Questions are marked [TRANSLATE: ...] and need manual translation to Esperanto")
    print("      Use a human translator or translation API to complete the dataset")

    return len(converted)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=Path, default=Path('data/external/triviaqa_sample_1000.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/test_sets/triviaqa_esperanto_100.jsonl'))
    parser.add_argument('--limit', type=int, default=100, help='Number of questions to convert')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    convert_triviaqa_to_esperanto_format(args.input, args.output, args.limit)

    return 0


if __name__ == '__main__':
    sys.exit(main())
