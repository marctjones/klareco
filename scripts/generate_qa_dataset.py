#!/usr/bin/env python3
"""
Generate Question-Answer pairs from Esperanto corpus for QA Decoder training.

Two strategies:
1. Extract dialogue QA pairs: Find questions followed by answers in text
2. Generate synthetic QA pairs: Create questions from declarative sentences

This creates training data in format:
{
    "question": "Kiu estas Frodo?",
    "question_ast": {...},
    "answer": "Frodo estas hobito.",
    "answer_ast": {...},
    "context": ["Frodo estas hobito, kiu loĝas en Hobbiton.", ...],
    "context_asts": [{...}, ...]
}
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.logging_config import setup_logging

logger = logging.getLogger(__name__)


class QADatasetGenerator:
    """Generate QA pairs from Esperanto corpus."""

    def __init__(self):
        """Initialize generator."""
        # Question words (interrogative correlatives)
        self.question_words = {
            'kiu': 'who/which',
            'kio': 'what',
            'kia': 'what kind of',
            'kie': 'where',
            'kiam': 'when',
            'kiel': 'how',
            'kial': 'why',
            'kiom': 'how much/many'
        }

    def load_corpus(self, corpus_path: Path) -> List[Dict]:
        """
        Load corpus from JSONL file.

        Args:
            corpus_path: Path to corpus file

        Returns:
            List of corpus entries
        """
        logger.info(f"Loading corpus from {corpus_path}")
        entries = []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(entries)} corpus entries")
        return entries

    def is_question(self, sentence: str) -> bool:
        """
        Check if sentence is a question.

        Args:
            sentence: Input sentence

        Returns:
            True if sentence is a question
        """
        # Check for question mark
        if '?' not in sentence:
            return False

        # Check for question words
        sentence_lower = sentence.lower()
        return any(qword in sentence_lower for qword in self.question_words.keys())

    def extract_dialogue_pairs(
        self,
        entries: List[Dict],
        window_size: int = 5
    ) -> List[Dict]:
        """
        Extract QA pairs from dialogue in corpus.

        Strategy: Find questions, look for answers in next few sentences.

        Args:
            entries: Corpus entries
            window_size: How many sentences after question to search

        Returns:
            List of QA pairs
        """
        logger.info("Extracting dialogue QA pairs...")
        qa_pairs = []

        for i, entry in enumerate(tqdm(entries, desc="Scanning for questions")):
            sentence = entry.get('sentence', '').strip()

            if not sentence or not self.is_question(sentence):
                continue

            # Clean question
            question = self._clean_sentence(sentence)

            # Skip if too short or too long
            if len(question) < 10 or len(question) > 200:
                continue

            # Look for answer in next sentences
            context = []
            answer = None

            for j in range(i + 1, min(i + window_size + 1, len(entries))):
                next_sentence = entries[j].get('sentence', '').strip()
                next_sentence = self._clean_sentence(next_sentence)

                if not next_sentence or len(next_sentence) < 10:
                    continue

                context.append(next_sentence)

                # Use first substantial sentence as answer
                if answer is None and not self.is_question(next_sentence):
                    answer = next_sentence

            if answer and context:
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'context': context,  # Keep all context sentences (will be limited later)
                    'method': 'dialogue_extraction'
                })

        logger.info(f"Extracted {len(qa_pairs)} dialogue QA pairs")
        return qa_pairs

    def generate_synthetic_pairs(
        self,
        entries: List[Dict],
        max_pairs: int = 10000
    ) -> List[Dict]:
        """
        Generate synthetic QA pairs from declarative sentences.

        Strategy: Parse declarative sentences and generate questions.

        Args:
            entries: Corpus entries
            max_pairs: Maximum number of pairs to generate

        Returns:
            List of QA pairs
        """
        logger.info("Generating synthetic QA pairs...")
        qa_pairs = []

        for entry in tqdm(entries, desc="Generating from sentences"):
            if len(qa_pairs) >= max_pairs:
                break

            sentence = entry.get('sentence', '').strip()
            sentence = self._clean_sentence(sentence)

            # Skip questions, too short, or too long
            if not sentence or self.is_question(sentence):
                continue
            if len(sentence) < 20 or len(sentence) > 200:
                continue

            # Generate questions from this sentence
            pairs = self._generate_questions_from_sentence(sentence)
            qa_pairs.extend(pairs)

        logger.info(f"Generated {len(qa_pairs)} synthetic QA pairs")
        return qa_pairs

    def _clean_sentence(self, sentence: str) -> str:
        """
        Clean and normalize sentence.

        Args:
            sentence: Raw sentence

        Returns:
            Cleaned sentence
        """
        # Remove excessive whitespace
        sentence = re.sub(r'\s+', ' ', sentence)

        # Remove dialogue markers
        sentence = re.sub(r'^[—–-]\s*', '', sentence)

        # Strip
        sentence = sentence.strip()

        return sentence

    def _generate_questions_from_sentence(self, sentence: str) -> List[Dict]:
        """
        Generate questions from a declarative sentence.

        Examples:
        - "Frodo estas hobito" → "Kiu estas Frodo?" / "Kio estas Frodo?"
        - "La hobito loĝas en Hobbiton" → "Kie la hobito loĝas?"
        - "Gandalfo venis hieraŭ" → "Kiam Gandalfo venis?"

        Args:
            sentence: Declarative sentence

        Returns:
            List of QA pairs
        """
        pairs = []

        # Pattern 1: "X estas Y" → "Kiu/Kio estas X?"
        match = re.search(r'(\w+)\s+estas\s+(\w+)', sentence, re.IGNORECASE)
        if match:
            subject = match.group(1)
            predicate = match.group(2)

            # Generate "Kiu estas X?" or "Kio estas X?"
            question = f"Kiu estas {subject}?"
            pairs.append({
                'question': question,
                'answer': sentence,
                'context': [sentence],
                'method': 'synthetic_estas'
            })

            # Generate "Kio estas X?" as alternative
            if predicate.endswith('o'):  # noun
                question2 = f"Kio estas {subject}?"
                pairs.append({
                    'question': question2,
                    'answer': sentence,
                    'context': [sentence],
                    'method': 'synthetic_estas'
                })

        # Pattern 2: Contains location word → "Kie...?"
        location_words = ['loĝas', 'estas', 'troviĝas', 'staras', 'iras', 'venas']
        for loc_word in location_words:
            if loc_word in sentence.lower():
                # Extract subject if possible
                words = sentence.split()
                if len(words) > 2:
                    subject = words[0] if words[0][0].isupper() else "ĝi"
                    question = f"Kie {subject.lower()} {loc_word}?"
                    pairs.append({
                        'question': question,
                        'answer': sentence,
                        'context': [sentence],
                        'method': 'synthetic_location'
                    })
                    break

        # Pattern 3: Contains time word → "Kiam...?"
        time_words = ['hieraŭ', 'hodiaŭ', 'morgaŭ', 'antaŭe', 'nun', 'poste']
        for time_word in time_words:
            if time_word in sentence.lower():
                words = sentence.split()
                if len(words) > 2:
                    # Try to extract verb
                    for word in words:
                        if any(word.endswith(end) for end in ['is', 'as', 'os']):
                            question = f"Kiam tio okazis?"
                            pairs.append({
                                'question': question,
                                'answer': sentence,
                                'context': [sentence],
                                'method': 'synthetic_time'
                            })
                            break
                    break

        return pairs

    def parse_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Parse QA pairs into ASTs.

        Args:
            qa_pairs: Raw QA pairs

        Returns:
            QA pairs with ASTs
        """
        logger.info("Parsing QA pairs to ASTs...")
        parsed_pairs = []

        for pair in tqdm(qa_pairs, desc="Parsing"):
            try:
                # Parse question
                question_ast = parse(pair['question'])

                # Parse answer
                answer_ast = parse(pair['answer'])

                # Parse context
                context_asts = []
                for ctx in pair.get('context', []):
                    try:
                        ctx_ast = parse(ctx)
                        context_asts.append(ctx_ast)
                    except Exception:
                        continue

                # Add to parsed pairs
                parsed_pairs.append({
                    'question': pair['question'],
                    'question_ast': question_ast,
                    'answer': pair['answer'],
                    'answer_ast': answer_ast,
                    'context': pair.get('context', []),
                    'context_asts': context_asts,
                    'method': pair.get('method', 'unknown')
                })

            except Exception as e:
                logger.debug(f"Failed to parse pair: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_pairs)}/{len(qa_pairs)} pairs")
        return parsed_pairs

    def save_dataset(self, qa_pairs: List[Dict], output_path: Path):
        """
        Save QA dataset to file.

        Args:
            qa_pairs: QA pairs with ASTs
            output_path: Output file path
        """
        logger.info(f"Saving dataset to {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(qa_pairs)} QA pairs")


def main():
    """Generate QA dataset."""
    parser = argparse.ArgumentParser(description='Generate QA dataset from corpus')
    parser.add_argument('--corpus', type=str, default='data/corpus_sentences.jsonl',
                        help='Input corpus file')
    parser.add_argument('--output', type=str, default='data/qa_dataset.jsonl',
                        help='Output dataset file')
    parser.add_argument('--max-dialogue', type=int, default=5000,
                        help='Max dialogue pairs to extract')
    parser.add_argument('--max-synthetic', type=int, default=10000,
                        help='Max synthetic pairs to generate')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Total maximum QA pairs (overrides max-dialogue and max-synthetic)')
    parser.add_argument('--context-size', type=int, default=3,
                        help='Number of context sentences to include (default: 3)')
    parser.add_argument('--method', type=str, choices=['dialogue', 'synthetic', 'both'],
                        default='both', help='Generation method')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("QA DATASET GENERATION")
    logger.info("=" * 70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Method: {args.method}")
    logger.info("")

    try:
        # Create generator
        generator = QADatasetGenerator()

        # Load corpus
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        entries = generator.load_corpus(corpus_path)

        # Generate QA pairs
        qa_pairs = []

        if args.method in ['dialogue', 'both']:
            dialogue_pairs = generator.extract_dialogue_pairs(entries)
            qa_pairs.extend(dialogue_pairs[:args.max_dialogue])

        if args.method in ['synthetic', 'both']:
            synthetic_pairs = generator.generate_synthetic_pairs(
                entries,
                max_pairs=args.max_synthetic
            )
            qa_pairs.extend(synthetic_pairs)

        # Apply total max-pairs limit if specified
        if args.max_pairs is not None:
            qa_pairs = qa_pairs[:args.max_pairs]

        # Limit context size for each pair
        for pair in qa_pairs:
            if 'context' in pair and isinstance(pair['context'], list):
                pair['context'] = pair['context'][:args.context_size]

        logger.info(f"\nTotal QA pairs collected: {len(qa_pairs)}")
        logger.info(f"Context size per pair: {args.context_size} sentences")

        # Parse to ASTs
        parsed_pairs = generator.parse_qa_pairs(qa_pairs)

        # Save dataset
        output_path = Path(args.output)
        generator.save_dataset(parsed_pairs, output_path)

        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total pairs: {len(parsed_pairs)}")

        # Method breakdown
        method_counts = {}
        for pair in parsed_pairs:
            method = pair.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        for method, count in method_counts.items():
            logger.info(f"  {method}: {count}")

        logger.info(f"Output: {output_path}")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
