#!/usr/bin/env python3
"""
Translate and Filter QA Dataset

Takes English QA dataset, translates to Esperanto, keeps only
questions where answers are verifiable in corpus.

Usage:
    python scripts/translate_and_filter_qa.py \
        --input triviaqa_sample.json \
        --output data/test_sets/translated_qa_diverse_500.jsonl \
        --limit 1000
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from deep_translator import GoogleTranslator
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

from klareco.parser import parse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class QATranslator:
    def __init__(self, whoosh_index_dir: Path):
        self.translator = GoogleTranslator(source='en', target='eo')
        self.whoosh_idx = open_dir(str(whoosh_index_dir))

    def translate_question(self, english_q: str) -> str:
        """Translate English question to Esperanto."""
        try:
            return self.translator.translate(english_q)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None

    def verify_translation(self, eo_question: str) -> bool:
        """Check if Esperanto question parses correctly."""
        try:
            ast = parse(eo_question)
            return ast is not None and 'verbo' in ast
        except:
            return False

    def check_corpus_coverage(self, answer_variants: List[str]) -> Tuple[bool, List[str]]:
        """Check if answer exists in corpus."""
        with self.whoosh_idx.searcher() as searcher:
            for answer in answer_variants:
                # Clean answer for search
                answer_clean = answer.lower().strip()

                # Search in corpus
                query = QueryParser("text_lower", self.whoosh_idx.schema).parse(answer_clean)
                results = searcher.search(query, limit=3)

                if len(results) > 0:
                    sentences = [hit['text'] for hit in results]
                    return True, sentences

        return False, []

    def process_qa_pair(self, qa: Dict) -> Dict:
        """
        Process single Q&A pair: translate, verify, check coverage.

        Returns:
            Dict with status and translated Q&A, or None if rejected
        """
        english_q = qa['question']
        english_a = qa['answer']

        # Handle answer as list or string
        if isinstance(english_a, list):
            answer_variants = english_a
        else:
            answer_variants = [english_a]

        # Step 1: Translate question
        eo_question = self.translate_question(english_q)
        if not eo_question:
            return {'status': 'translation_failed'}

        # Step 2: Verify translation parses
        if not self.verify_translation(eo_question):
            return {'status': 'parse_failed', 'eo_question': eo_question}

        # Step 3: Check corpus coverage
        found, example_sentences = self.check_corpus_coverage(answer_variants)
        if not found:
            return {'status': 'not_in_corpus', 'eo_question': eo_question}

        # Success! Return translated Q&A
        return {
            'status': 'success',
            'question': eo_question,
            'answer': answer_variants[0],
            'answer_variants': answer_variants,
            'example_sentences': example_sentences,
            'original_english_question': english_q,
            'original_english_answer': english_a
        }

    def process_dataset(self, input_file: Path, output_file: Path, limit: int = None):
        """Process full dataset: translate, filter, save."""

        # Load input dataset
        with open(input_file) as f:
            if input_file.suffix == '.jsonl':
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)

        if limit:
            dataset = dataset[:limit]

        logger.info(f"Processing {len(dataset)} Q&A pairs...")

        # Process each Q&A pair
        results = {
            'success': [],
            'translation_failed': [],
            'parse_failed': [],
            'not_in_corpus': []
        }

        for i, qa in enumerate(dataset):
            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(dataset)}] Processed...")

            result = self.process_qa_pair(qa)
            status = result['status']
            results[status].append(result)

        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("RESULTS")
        logger.info("="*60)
        logger.info(f"Total processed: {len(dataset)}")
        logger.info(f"✓ Success: {len(results['success'])} ({len(results['success'])/len(dataset)*100:.1f}%)")
        logger.info(f"✗ Translation failed: {len(results['translation_failed'])}")
        logger.info(f"✗ Parse failed: {len(results['parse_failed'])}")
        logger.info(f"✗ Not in corpus: {len(results['not_in_corpus'])}")

        # Save successful translations
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for qa in results['success']:
                # Format for our test set
                output = {
                    'question': qa['question'],
                    'expected_keywords': [qa['answer']],  # Can be expanded
                    'answer': qa['answer'],
                    'answer_variants': qa['answer_variants'],
                    'source': 'translated',
                    'original_english': {
                        'question': qa['original_english_question'],
                        'answer': qa['original_english_answer']
                    }
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')

        logger.info(f"\nSaved {len(results['success'])} questions to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Translate and filter QA dataset')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input English QA dataset (JSON or JSONL)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output Esperanto QA dataset (JSONL)')
    parser.add_argument('--whoosh-index', type=Path,
                       default=Path('data/indexes/whoosh_fts'),
                       help='Whoosh index directory')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions to process')

    args = parser.parse_args()

    # Initialize translator
    translator = QATranslator(args.whoosh_index)

    # Process dataset
    results = translator.process_dataset(args.input, args.output, args.limit)


if __name__ == '__main__':
    main()
