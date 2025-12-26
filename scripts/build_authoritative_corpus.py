#!/usr/bin/env python3
"""
Build authoritative corpus from git-tracked texts.

This script creates properly annotated corpus entries from authoritative sources,
including source tier, weight, and citation information.

Usage:
    python scripts/build_authoritative_corpus.py \
        --output data/corpus/authoritative_corpus.jsonl

Output format:
{
    "text": "La patro amas la infanon.",
    "source": {
        "tier": 1,
        "name": "fundamento_ekzercaro",
        "citation": "fundamento:ekzercaro:§5:3",
        "author": "L. L. Zamenhof",
        "year": 1905,
        "weight": 10.0,
        "verified": true
    },
    "ast": { ... },
    "parse_statistics": { ... }
}
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base path for authoritative texts
TEXTS_BASE = Path(__file__).parent.parent / 'texts' / 'authoritative'


def load_metadata(text_dir: Path) -> Dict:
    """Load metadata.json from a text directory."""
    metadata_path = text_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text, handling Esperanto punctuation."""
    # Split on sentence-ending punctuation followed by space or newline
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up and filter
    result = []
    for sent in sentences:
        sent = sent.strip()
        # Skip empty, too short, or header-like lines
        if not sent or len(sent) < 10:
            continue
        # Skip lines that look like headers (all caps, section markers)
        if sent.isupper() or sent.startswith('§') or sent.startswith('ĈAPITRO'):
            continue
        # Skip lines that are just punctuation or special characters
        if not any(c.isalpha() for c in sent):
            continue
        result.append(sent)

    return result


def parse_sentence_with_stats(text: str) -> Tuple[Dict, Dict]:
    """Parse a sentence and return AST with statistics."""
    try:
        ast = parse(text)

        # Calculate parse statistics
        stats = ast.get('parse_statistics', {})
        total_words = stats.get('total_words', 0)
        success_rate = stats.get('success_rate', 0.0)

        return ast, {
            'total_words': total_words,
            'success_rate': success_rate,
            'successful_parses': int(total_words * success_rate)
        }
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return {}, {'total_words': 0, 'success_rate': 0.0, 'successful_parses': 0}


def process_fundamento_ekzercaro(output_file, metadata: Dict) -> int:
    """Process Fundamento Ekzercaro with exercise-level citations."""
    ekzercaro_path = TEXTS_BASE / 'fundamento' / 'ekzercaro.txt'

    if not ekzercaro_path.exists():
        logger.warning(f"Ekzercaro not found: {ekzercaro_path}")
        return 0

    with open(ekzercaro_path, 'r', encoding='utf-8') as f:
        content = f.read()

    count = 0
    current_exercise = 0
    sentence_in_exercise = 0
    in_vocabulary_section = False

    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Detect exercise number (§1, §2, etc.)
        if line.startswith('§'):
            match = re.match(r'§(\d+)', line)
            if match:
                current_exercise = int(match.group(1))
                sentence_in_exercise = 0
                in_vocabulary_section = False
                continue

        # Skip until we reach actual exercises (§1 or later)
        if current_exercise < 1:
            continue

        # Skip headers and empty lines
        if not line or line.isupper():
            continue

        # Vocabulary entries are single words (no spaces) followed by definitions with |
        # A single word line followed by a line with | indicates vocabulary section
        if '|' in line:
            in_vocabulary_section = True
            continue

        # Skip single words that are vocabulary entries
        if in_vocabulary_section and ' ' not in line and len(line) < 20:
            continue

        # Lines with ― or containing multiple sentences are actual content
        # Reset vocabulary section flag when we see actual content
        if '―' in line or (len(line) > 30 and ' ' in line):
            in_vocabulary_section = False

        if in_vocabulary_section:
            continue

        # Split on ― (em-dash) which separates sentences in Ekzercaro
        if '―' in line:
            sentence_parts = [s.strip() for s in line.split('―')]
        else:
            sentence_parts = [line]

        for part in sentence_parts:
            if not part or len(part) < 5:
                continue

            # Further split on sentence-ending punctuation
            sentences = extract_sentences(part)

            for sent in sentences:
                if len(sent) < 5:
                    continue

                sentence_in_exercise += 1
                ast, stats = parse_sentence_with_stats(sent)

                # Skip very low quality parses
                if stats['success_rate'] < 0.3:
                    continue

                entry = {
                    'text': sent,
                    'source': {
                        'tier': metadata.get('tier', 1),
                        'name': 'fundamento_ekzercaro',
                        'citation': f"fundamento:ekzercaro:§{current_exercise}:{sentence_in_exercise}",
                        'exercise_number': current_exercise,
                        'sentence_in_exercise': sentence_in_exercise,
                        'author': metadata.get('author', 'L. L. Zamenhof'),
                        'year': metadata.get('year', 1905),
                        'weight': metadata.get('weight', 10.0),
                        'verified': True
                    },
                    'ast': ast,
                    'parse_statistics': stats
                }

                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1

    return count


def process_krestomatio(output_file, metadata: Dict) -> int:
    """Process Fundamenta Krestomatio with section citations."""
    krestomatio_path = TEXTS_BASE / 'krestomatio' / 'fundamenta_krestomatio.txt'

    if not krestomatio_path.exists():
        logger.warning(f"Krestomatio not found: {krestomatio_path}")
        return 0

    with open(krestomatio_path, 'r', encoding='utf-8') as f:
        content = f.read()

    count = 0
    current_section = 'unknown'
    line_number = 0

    # Detect sections
    section_markers = {
        'ANTAŬPAROLO': 'antaŭparolo',
        'PROZO': 'prozo',
        'POEZIO': 'poezio',
        'TRADUKOJ': 'tradukoj',
        'ALDONO': 'aldono'
    }

    for line in content.split('\n'):
        line_number += 1
        line_stripped = line.strip()

        # Check for section markers
        for marker, section in section_markers.items():
            if marker in line_stripped.upper():
                current_section = section
                break

        # Skip empty lines and headers
        if not line_stripped or len(line_stripped) < 15:
            continue

        # Skip separator lines
        if line_stripped.startswith('---') or line_stripped.startswith('==='):
            continue

        # Extract sentences
        sentences = extract_sentences(line_stripped)

        for sent in sentences:
            ast, stats = parse_sentence_with_stats(sent)

            # Skip low-quality parses
            if stats['success_rate'] < 0.5:
                continue

            entry = {
                'text': sent,
                'source': {
                    'tier': metadata.get('tier', 2),
                    'name': 'fundamenta_krestomatio',
                    'citation': f"krestomatio:{current_section}:L{line_number}",
                    'section': current_section,
                    'line_number': line_number,
                    'author': metadata.get('author', 'L. L. Zamenhof'),
                    'year': metadata.get('year', 1903),
                    'weight': metadata.get('weight', 5.0),
                    'verified': True
                },
                'ast': ast,
                'parse_statistics': stats
            }

            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    return count


def process_gerda(output_file, metadata: Dict) -> int:
    """Process Gerda Malaperis with chapter citations."""
    gerda_path = TEXTS_BASE / 'gerda_malaperis' / 'gerda_malaperis.txt'

    if not gerda_path.exists():
        logger.warning(f"Gerda not found: {gerda_path}")
        return 0

    with open(gerda_path, 'r', encoding='utf-8') as f:
        content = f.read()

    count = 0
    current_chapter = 0
    line_in_chapter = 0
    is_dialogue = False
    speaker = None

    for line in content.split('\n'):
        line_stripped = line.strip()

        # Detect chapter
        if line_stripped.startswith('ĈAPITRO') or re.match(r'^ĈAPITRO \d+', line_stripped):
            match = re.search(r'(\d+)', line_stripped)
            if match:
                current_chapter = int(match.group(1))
                line_in_chapter = 0
                continue

        # Skip empty lines and headers
        if not line_stripped or len(line_stripped) < 10:
            continue

        # Detect dialogue (lines with speaker: format)
        dialogue_match = re.match(r'^([A-Za-zĉĝĥĵŝŭ]+):\s*(.+)', line_stripped)
        if dialogue_match:
            is_dialogue = True
            speaker = dialogue_match.group(1)
            text = dialogue_match.group(2)
        else:
            is_dialogue = False
            speaker = None
            text = line_stripped

        line_in_chapter += 1

        # Extract sentences
        sentences = extract_sentences(text)

        for sent in sentences:
            ast, stats = parse_sentence_with_stats(sent)

            # Skip low-quality parses
            if stats['success_rate'] < 0.5:
                continue

            entry = {
                'text': sent,
                'source': {
                    'tier': metadata.get('tier', 3),
                    'name': 'gerda_malaperis',
                    'citation': f"gerda:{current_chapter}:L{line_in_chapter}",
                    'chapter': current_chapter,
                    'line_in_chapter': line_in_chapter,
                    'is_dialogue': is_dialogue,
                    'speaker': speaker,
                    'author': metadata.get('author', 'Claude Piron'),
                    'year': metadata.get('year', 1983),
                    'weight': metadata.get('weight', 3.0),
                    'verified': True
                },
                'ast': ast,
                'parse_statistics': stats
            }

            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Build authoritative corpus')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output JSONL file')
    parser.add_argument('--sources', nargs='+',
                        default=['fundamento', 'krestomatio', 'gerda'],
                        choices=['fundamento', 'krestomatio', 'gerda'],
                        help='Which sources to include')

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building Authoritative Corpus")
    logger.info("=" * 60)

    total_sentences = 0

    with open(args.output, 'w', encoding='utf-8') as output_file:

        if 'fundamento' in args.sources:
            metadata = load_metadata(TEXTS_BASE / 'fundamento')
            logger.info(f"Processing Fundamento Ekzercaro (tier {metadata.get('tier', 1)}, weight {metadata.get('weight', 10.0)})")
            count = process_fundamento_ekzercaro(output_file, metadata)
            logger.info(f"  → {count} sentences")
            total_sentences += count

        if 'krestomatio' in args.sources:
            metadata = load_metadata(TEXTS_BASE / 'krestomatio')
            logger.info(f"Processing Fundamenta Krestomatio (tier {metadata.get('tier', 2)}, weight {metadata.get('weight', 5.0)})")
            count = process_krestomatio(output_file, metadata)
            logger.info(f"  → {count} sentences")
            total_sentences += count

        if 'gerda' in args.sources:
            metadata = load_metadata(TEXTS_BASE / 'gerda_malaperis')
            logger.info(f"Processing Gerda Malaperis (tier {metadata.get('tier', 3)}, weight {metadata.get('weight', 3.0)})")
            count = process_gerda(output_file, metadata)
            logger.info(f"  → {count} sentences")
            total_sentences += count

    logger.info("=" * 60)
    logger.info(f"Total: {total_sentences} sentences written to {args.output}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
