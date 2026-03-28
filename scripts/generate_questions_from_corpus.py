#!/usr/bin/env python3
"""
Auto-Generate Questions from Corpus

VERSION: v2.1
COMPATIBLE WITH: v2.1 database, unified corpus
DEPENDENCIES: AST parser
STAGE: Data Generation

Description:
    Fully automated question generation from corpus sentences.
    Extracts factual SVO sentences and converts to WHO/WHAT/WHERE/WHEN questions.
    No manual review required.

Strategy:
    1. Parse corpus for high-quality SVO sentences
    2. Identify sentences with named entities, verbs, objects
    3. Generate questions by substituting with question words
    4. Validate questions parse correctly
    5. Output diverse question set with expected answers

Question Types:
    - WHO: "Kiu [VERB]is [OBJECT]?" (replace subject)
    - WHAT: "Kion [VERB]is [SUBJECT]?" (replace object)
    - WHERE: "Kie [VERB]is [SUBJECT]?" (if location modifier exists)
    - WHEN: "Kiam [VERB]is [SUBJECT]?" (if time modifier exists)

Usage:
    python scripts/generate_questions_from_corpus.py \\
        --corpus data/corpus/unified_corpus.jsonl \\
        --output data/test_sets/generated_questions.jsonl \\
        --num-questions 200 \\
        --min-parse-quality 0.9

Outputs:
    - JSONL with {'question', 'expected_keywords', 'source_sentence', 'question_type'}
    - Diverse question types
    - Validated parse quality

Last Updated: 2026-03-26
Author: Claude + Marc
Related Issues: #704
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate questions from corpus sentences."""

    def __init__(self, min_parse_quality: float = 0.9):
        """
        Initialize question generator.

        Args:
            min_parse_quality: Minimum parse success rate (0.0-1.0)
        """
        self.min_parse_quality = min_parse_quality
        self.question_words = {'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel', 'kiom'}

    def generate_questions(
        self,
        corpus_path: Path,
        num_questions: int = 200,
        max_sentences: int = 10000
    ) -> List[Dict]:
        """
        Generate questions from corpus.

        Args:
            corpus_path: Path to unified corpus JSONL
            num_questions: Target number of questions to generate
            max_sentences: Maximum sentences to process from corpus

        Returns:
            List of question dicts with 'question', 'expected_keywords', 'source_sentence'
        """
        logger.info(f"Loading corpus from {corpus_path}")

        # Load corpus sentences
        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_sentences:
                    break
                try:
                    sent = json.loads(line)
                    sentences.append(sent)
                except:
                    continue

        logger.info(f"Loaded {len(sentences)} sentences from corpus")

        # Filter for high-quality factual sentences
        logger.info("Filtering for high-quality SVO sentences...")
        factual_sentences = self._filter_factual_sentences(sentences)
        logger.info(f"Found {len(factual_sentences)} factual sentences")

        # Generate questions from factual sentences
        logger.info("Generating questions...")
        questions = []
        question_types = defaultdict(int)

        for sent in factual_sentences:
            # Try generating different question types
            generated = self._generate_from_sentence(sent)
            questions.extend(generated)

            for q in generated:
                question_types[q['question_type']] += 1

            # Stop if we have enough
            if len(questions) >= num_questions * 1.5:  # Generate extra for diversity
                break

        logger.info(f"Generated {len(questions)} questions")
        logger.info(f"Question type distribution: {dict(question_types)}")

        # Diversify and limit to target number
        diverse_questions = self._diversify_questions(questions, num_questions)

        logger.info(f"Selected {len(diverse_questions)} diverse questions")

        return diverse_questions

    def _filter_factual_sentences(self, sentences: List[Dict]) -> List[Dict]:
        """Filter for high-quality factual sentences suitable for questions."""
        factual = []

        for sent in sentences:
            # Parse if no AST exists
            if 'ast' not in sent or not sent['ast']:
                text = sent.get('text', '')
                if not text:
                    continue

                try:
                    ast = parse(text)
                    if not ast:
                        continue
                    sent['ast'] = ast
                except:
                    continue
            else:
                ast = sent['ast']

            # Must have good parse quality
            stats = ast.get('parse_statistics', {})
            success_rate = stats.get('sukcesoprocento', 0.0)  # Esperanto key name

            if success_rate < self.min_parse_quality:
                continue

            # Must have subject, verb, object (SVO)
            if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
                continue

            # Must not be a question already
            text = sent.get('text', '')
            if text.strip().endswith('?'):
                continue

            # Subject should be a named entity or specific noun (not "Mi", "Vi", "Ĝi")
            subj = ast.get('subjekto')
            if subj and subj.get('tipo') == 'vortgrupo':
                kerno = subj.get('kerno', {})
                if kerno:
                    root = kerno.get('radiko', '').lower()
                    # Skip generic pronouns
                    if root in ['mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni']:
                        continue

            # Verb should be past or present tense (factual statements)
            verb = ast.get('verbo')
            if not verb:
                continue
            tempo = verb.get('tempo')
            if tempo not in ['pasinteco', 'prezenco']:  # Past or present (facts)
                continue

            factual.append(sent)

        return factual

    def _generate_from_sentence(self, sent: Dict) -> List[Dict]:
        """Generate multiple question types from a single sentence."""
        questions = []
        text = sent.get('text', '')
        ast = sent.get('ast', {})

        # Extract components
        subj = ast.get('subjekto', {})
        verb = ast.get('verbo', {})
        obj = ast.get('objekto', {})

        # Extract verb root and form
        verb_root = verb.get('radiko', '')
        verb_tempo = verb.get('tempo', 'prezenco')
        # Convert tempo to ending: pasinteco→is, prezenco→as
        tempo_to_ending = {'pasinteco': 'is', 'prezenco': 'as', 'futuro': 'os'}
        verb_ending = tempo_to_ending.get(verb_tempo, 'as')
        verb_form = f"{verb_root}{verb_ending}"

        # Extract subject root
        subj_text = self._extract_phrase_text(subj)
        obj_text = self._extract_phrase_text(obj)

        if not verb_root or not subj_text or not obj_text:
            return questions

        # WHO question: "Kiu [VERB]is [OBJECT]?"
        who_question = f"Kiu {verb_form} {obj_text}?"
        who_keywords = [w.strip() for w in subj_text.lower().split() if len(w.strip()) > 2]
        questions.append({
            'question': who_question,
            'expected_keywords': who_keywords,
            'source_sentence': text,
            'question_type': 'who'
        })

        # WHAT question: "Kion [VERB]is [SUBJECT]?"
        # Only if object is not a person
        if not self._is_person_phrase(obj):
            what_question = f"Kion {verb_form} {subj_text}?"
            what_keywords = [w.strip() for w in obj_text.lower().split() if len(w.strip()) > 2]
            questions.append({
                'question': what_question,
                'expected_keywords': what_keywords,
                'source_sentence': text,
                'question_type': 'what'
            })

        # WHERE question: "Kie [VERB]is [SUBJECT]?"
        # Only if sentence has location modifier
        if self._has_location_modifier(ast):
            where_question = f"Kie {verb_form} {subj_text}?"
            # Extract location from modifiers
            location = self._extract_location(ast)
            where_keywords = [location] if location else []
            questions.append({
                'question': where_question,
                'expected_keywords': where_keywords,
                'source_sentence': text,
                'question_type': 'where'
            })

        # WHEN question: "Kiam [VERB]is [SUBJECT]?"
        # Only if sentence has time modifier
        if self._has_time_modifier(ast):
            when_question = f"Kiam {verb_form} {subj_text}?"
            # Extract time from modifiers
            time = self._extract_time(ast)
            when_keywords = [time] if time else []
            questions.append({
                'question': when_question,
                'expected_keywords': when_keywords,
                'source_sentence': text,
                'question_type': 'when'
            })

        # Validate all questions parse correctly
        valid_questions = []
        for q in questions:
            try:
                q_ast = parse(q['question'])
                if q_ast and q_ast.get('parse_statistics', {}).get('sukcesoprocento', 0) > 0.8:
                    valid_questions.append(q)
            except:
                continue

        return valid_questions

    def _extract_phrase_text(self, phrase: Dict) -> str:
        """Extract text representation of phrase."""
        if not phrase:
            return ""

        if phrase.get('tipo') == 'vorto':
            # Simple word
            root = phrase.get('radiko', '')
            prefix = phrase.get('prefikso', '')
            suffix = phrase.get('sufikso', '')
            ending = phrase.get('vortspeco_signo', '')

            # Reconstruct word
            word = prefix + root
            if suffix:
                word += suffix
            word += ending

            return word

        elif phrase.get('tipo') == 'vortgrupo':
            # Phrase with modifiers
            kerno = phrase.get('kerno', {})
            kerno_text = self._extract_phrase_text(kerno)

            # Add adjectives
            priskriboj = phrase.get('priskriboj', [])
            if priskriboj:
                adj_texts = [self._extract_phrase_text(p) for p in priskriboj]
                adj_texts = [a for a in adj_texts if a]
                if adj_texts:
                    return ' '.join(adj_texts) + ' ' + kerno_text

            return kerno_text

        return ""

    def _is_person_phrase(self, phrase: Dict) -> bool:
        """Check if phrase refers to a person."""
        if not phrase:
            return False

        # Check for person-related suffixes
        if phrase.get('tipo') == 'vorto':
            suffix = phrase.get('sufikso', '')
            if suffix in ['in', 'ist', 'ul']:
                return True

        # Check for capitalized proper names
        text = self._extract_phrase_text(phrase)
        if text and text[0].isupper():
            return True

        return False

    def _has_location_modifier(self, ast: Dict) -> bool:
        """Check if sentence has location modifier."""
        aliaj = ast.get('aliaj', [])
        for item in aliaj:
            if item.get('tipo') == 'prepozicia_frazo':
                prep = item.get('prepozicio', {})
                prep_root = prep.get('radiko', '').lower()
                if prep_root in ['en', 'sur', 'sub', 'apud', 'ĉe', 'ekster', 'inter', 'antaŭ', 'post']:
                    return True
        return False

    def _has_time_modifier(self, ast: Dict) -> bool:
        """Check if sentence has time modifier."""
        aliaj = ast.get('aliaj', [])
        for item in aliaj:
            if item.get('tipo') == 'prepozicia_frazo':
                prep = item.get('prepozicio', {})
                prep_root = prep.get('radiko', '').lower()
                if prep_root in ['dum', 'post', 'antaŭ']:
                    return True
            # Check for time words
            if item.get('tipo') == 'vorto':
                root = item.get('radiko', '').lower()
                if root in ['hieraŭ', 'hodiaŭ', 'morgaŭ', 'nun', 'jam', 'baldaŭ']:
                    return True
        return False

    def _extract_location(self, ast: Dict) -> str:
        """Extract location from sentence."""
        aliaj = ast.get('aliaj', [])
        for item in aliaj:
            if item.get('tipo') == 'prepozicia_frazo':
                prep = item.get('prepozicio', {})
                prep_root = prep.get('radiko', '').lower()
                if prep_root in ['en', 'sur', 'sub', 'apud', 'ĉe']:
                    obj = item.get('objekto', {})
                    return self._extract_phrase_text(obj)
        return ""

    def _extract_time(self, ast: Dict) -> str:
        """Extract time from sentence."""
        aliaj = ast.get('aliaj', [])
        for item in aliaj:
            if item.get('tipo') == 'prepozicia_frazo':
                prep = item.get('prepozicio', {})
                prep_root = prep.get('radiko', '').lower()
                if prep_root in ['dum', 'post', 'antaŭ']:
                    obj = item.get('objekto', {})
                    return self._extract_phrase_text(obj)
            if item.get('tipo') == 'vorto':
                root = item.get('radiko', '').lower()
                if root in ['hieraŭ', 'hodiaŭ', 'morgaŭ']:
                    return self._extract_phrase_text(item)
        return ""

    def _diversify_questions(self, questions: List[Dict], target: int) -> List[Dict]:
        """Select diverse subset of questions."""
        if not questions:
            return []

        # Group by question type
        by_type = defaultdict(list)
        for q in questions:
            by_type[q['question_type']].append(q)

        # Calculate target per type (balanced)
        num_types = len(by_type)
        if num_types == 0:
            return []

        per_type = target // num_types

        # Sample from each type
        diverse = []
        for qtype, qlist in by_type.items():
            # Shuffle and take up to per_type
            random.shuffle(qlist)
            diverse.extend(qlist[:per_type])

        # If we need more, add from remaining
        remaining = target - len(diverse)
        if remaining > 0:
            all_remaining = [q for q in questions if q not in diverse]
            random.shuffle(all_remaining)
            diverse.extend(all_remaining[:remaining])

        random.shuffle(diverse)
        return diverse[:target]


def main():
    parser = argparse.ArgumentParser(description="Auto-generate questions from corpus")
    parser.add_argument('--corpus', type=Path,
                       default=Path('data/corpus/unified_corpus.jsonl'),
                       help='Unified corpus JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL file for questions')
    parser.add_argument('--num-questions', type=int, default=200,
                       help='Number of questions to generate (default: 200)')
    parser.add_argument('--max-sentences', type=int, default=10000,
                       help='Maximum sentences to process from corpus (default: 10000)')
    parser.add_argument('--min-parse-quality', type=float, default=0.9,
                       help='Minimum parse quality (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Auto-Generate Questions from Corpus")
    logger.info(f"{'='*60}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Target questions: {args.num_questions}")
    logger.info(f"Min parse quality: {args.min_parse_quality}")

    # Initialize generator
    generator = QuestionGenerator(min_parse_quality=args.min_parse_quality)

    # Generate questions
    questions = generator.generate_questions(
        corpus_path=args.corpus,
        num_questions=args.num_questions,
        max_sentences=args.max_sentences
    )

    # Save to output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(questions)} questions")

    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info("Question Statistics")
    logger.info(f"{'='*60}")

    by_type = defaultdict(int)
    for q in questions:
        by_type[q['question_type']] += 1

    for qtype, count in sorted(by_type.items()):
        logger.info(f"  {qtype.upper()}: {count}")

    logger.info(f"\nTotal: {len(questions)}")
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
