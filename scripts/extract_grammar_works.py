#!/usr/bin/env python3
"""
Extract sentences from Tier 0 grammar works (PMEG, PAG, Lingvaj Respondoj).

These contain:
1. Explanatory prose (grammar descriptions)
2. Example sentences (illustrating rules)

Both types are valuable for training but should be marked differently.

Output: JSONL with sentence + metadata (source, tier, sentence_type)
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Source metadata
GRAMMAR_WORKS = {
    'pmeg': {
        'title': 'Plena Manlibro de Esperanta Gramatiko (PMEG)',
        'author': 'Bertilo Wennergren',
        'year': 2024,
        'version': '15.5',
        'tier': 0,
        'quality': 'authoritative',
        'source_type': 'grammar_reference',
    },
    'pag': {
        'title': 'Plena Analiza Gramatiko (PAG)',
        'author': 'Kálmán Kalocsay & Gaston Waringhien',
        'year': 1985,
        'tier': 0,
        'quality': 'authoritative',
        'source_type': 'grammar_reference',
    },
    'lingvaj_respondoj': {
        'title': 'Lingvaj Respondoj',
        'author': 'L.L. Zamenhof',
        'year': 1908,
        'tier': 0,
        'quality': 'authoritative',
        'source_type': 'grammar_qa',
    },
}


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using Esperanto sentence boundaries.

    Same logic as literary extraction but tuned for grammar text.
    """
    # Common abbreviations in grammar texts
    abbreviations = {
        'D-ro': '<ABBR_DRO>',
        'S-ro': '<ABBR_SRO>',
        'k.t.p.': '<ABBR_KTP>',
        'k.a.': '<ABBR_KA>',
        'ktp.': '<ABBR_KTP2>',
        'ekz.': '<ABBR_EKZ>',
        'p.': '<ABBR_P>',
        'n-ro': '<ABBR_NRO>',
        'nr.': '<ABBR_NR>',
    }

    temp_text = text
    for abbr, placeholder in abbreviations.items():
        temp_text = temp_text.replace(abbr, placeholder)

    # Split on sentence boundaries
    sentences = re.split(r'([.!?])\s+(?=[A-ZĈĜĤĴŜŬ"\'\(])', temp_text)

    # Reconstruct sentences
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + sentences[i + 1]
        reconstructed.append(sentence.strip())

    # Add last sentence if no punctuation at end
    if len(sentences) % 2 == 1:
        reconstructed.append(sentences[-1].strip())

    # Restore abbreviations
    final_sentences = []
    for sent in reconstructed:
        for placeholder, abbr in {v: k for k, v in abbreviations.items()}.items():
            sent = sent.replace(placeholder, abbr)

        # Only keep sentences with actual content
        if len(sent) > 5 and any(c.isalpha() for c in sent):
            final_sentences.append(sent)

    return final_sentences


def classify_sentence_type(sentence: str) -> str:
    """
    Classify sentence as 'example' or 'explanation'.

    Example sentences often:
    - Are short and simple
    - Demonstrate a specific grammatical pattern
    - May be preceded by markers like "ekzemple:", "Ekz.:", bullet points

    Explanation sentences:
    - Discuss grammar rules
    - Contain metalanguage (vorto, radiko, sufikso, gramatiko, etc.)
    - Are longer and more complex
    """
    # Grammar metalanguage keywords
    metalanguage = [
        'vorto', 'radiko', 'sufikso', 'prefikso', 'finaĵo',
        'gramatiko', 'sintakso', 'morfologi',
        'substantivo', 'adjektivo', 'verbo', 'adverbo',
        'nominativo', 'akuzativo', 'genitivo',
        'singularo', 'pluralo', 'tempo', 'modo',
        'regulo', 'ekzemplo', 'uzo', 'formo',
    ]

    # Example markers
    example_markers = [
        'ekzemple', 'ekz.', 'ekz:', 'Ekz.', 'Ekz:',
        '•', '–', '—',  # Bullet points and dashes
    ]

    sentence_lower = sentence.lower()

    # Check for example markers at start
    for marker in example_markers:
        if sentence.startswith(marker) or sentence.startswith(marker.capitalize()):
            return 'example'

    # Check for metalanguage (indicates explanation)
    metalanguage_count = sum(1 for word in metalanguage if word in sentence_lower)

    if metalanguage_count >= 2:
        return 'explanation'
    elif metalanguage_count == 1 and len(sentence) > 80:
        return 'explanation'

    # Short sentences without metalanguage are likely examples
    if len(sentence) < 50 and metalanguage_count == 0:
        return 'example'

    # Default: explanation (conservative - most grammar text is explanatory)
    return 'explanation'


def extract_from_grammar_work(
    text_path: Path,
    work_id: str,
    output_path: Path
) -> Tuple[int, int]:
    """
    Extract sentences from grammar work.

    Returns (total_sentences, example_sentences)
    """
    if work_id not in GRAMMAR_WORKS:
        logger.error(f"Unknown work ID: {work_id}")
        return 0, 0

    metadata = GRAMMAR_WORKS[work_id]
    logger.info(f"Extracting from: {metadata['title']}")

    # Read text
    try:
        text = text_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to read {text_path}: {e}")
        return 0, 0

    logger.info(f"Text size: {len(text):,} characters")

    # Split into sentences
    sentences = split_into_sentences(text)
    logger.info(f"Found {len(sentences)} sentences")

    # Classify and write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    example_count = 0
    explanation_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences, 1):
            sentence_type = classify_sentence_type(sentence)

            if sentence_type == 'example':
                example_count += 1
            else:
                explanation_count += 1

            entry = {
                'sentence': sentence,
                'source': work_id,
                'source_title': metadata['title'],
                'author': metadata['author'],
                'year': metadata['year'],
                'tier': metadata['tier'],
                'quality': metadata['quality'],
                'source_type': metadata['source_type'],
                'sentence_type': sentence_type,
                'sentence_id': i,
            }

            # Add version for PMEG
            if work_id == 'pmeg':
                entry['version'] = metadata['version']

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    total = example_count + explanation_count
    logger.info(f"  Examples: {example_count} ({example_count/total*100:.1f}%)")
    logger.info(f"  Explanations: {explanation_count} ({explanation_count/total*100:.1f}%)")
    logger.info(f"✓ Extracted {total} sentences to {output_path}")

    return total, example_count


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Tuple[int, int]]:
    """Load checkpoint of already-processed works."""
    if not checkpoint_path.exists():
        return {}

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert list values back to tuples
            return {k: tuple(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {}


def save_checkpoint(checkpoint_path: Path, completed: Dict[str, Tuple[int, int]]):
    """Atomically save checkpoint."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)
        temp_path.rename(checkpoint_path)
        logger.debug(f"Checkpoint saved: {len(completed)} works completed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def extract_all_grammar(
    input_dir: Path,
    output_dir: Path,
    resume: bool = False,
    fresh: bool = False
) -> Dict[str, Tuple[int, int]]:
    """Extract sentences from all grammar works."""

    logger.info("=" * 70)
    logger.info("EXTRACT TIER 0 GRAMMAR WORKS")
    logger.info("=" * 70)

    # Checkpoint management
    checkpoint_path = output_dir / '.checkpoint.json'
    completed = {}

    if fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - ignoring checkpoint")
        checkpoint_path.unlink()
    elif resume:
        completed = load_checkpoint(checkpoint_path)
        if completed:
            logger.info(f"Resuming from checkpoint: {len(completed)} works already completed")
        else:
            logger.info("No checkpoint found - starting from beginning")

    results = {}

    # Map work IDs to input files
    # Note: Using cleaned files from data/cleaned/eo/tier0/
    input_files = {
        'pmeg': input_dir / 'pmeg.txt',
        # pag skipped - needs OCR first (0/12 diacritics)
        'lingvaj_respondoj': input_dir / 'lingvaj_respondoj.txt',
    }

    for work_id, input_path in input_files.items():
        # Skip if already completed (when resuming)
        if resume and work_id in completed:
            total, examples = completed[work_id]
            logger.info(f"Skipping {work_id} (already completed: {total} sentences)")
            results[work_id] = (total, examples)
            continue

        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            results[work_id] = (0, 0)
            continue

        logger.info("")
        output_path = output_dir / f"{work_id}_sentences.jsonl"
        total, examples = extract_from_grammar_work(input_path, work_id, output_path)
        results[work_id] = (total, examples)

        # Save checkpoint after each work
        completed[work_id] = (total, examples)
        save_checkpoint(checkpoint_path, completed)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)

    total_sentences = sum(total for total, _ in results.values())
    total_examples = sum(examples for _, examples in results.values())

    logger.info(f"Total sentences extracted: {total_sentences:,}")
    logger.info(f"  Example sentences: {total_examples:,}")
    logger.info(f"  Explanation sentences: {total_sentences - total_examples:,}")
    logger.info("")

    for work_id, (total, examples) in results.items():
        status = "✓" if total > 0 else "✗"
        logger.info(f"  {status} {work_id}: {total:,} sentences ({examples} examples)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract sentences from Tier 0 grammar works'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/cleaned/eo/tier0'),
        help='Input directory with cleaned texts'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/extracted/eo/tier0/grammar'),
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip already-processed works)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()

    # Validate flags
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh flags")
        return 1

    try:
        results = extract_all_grammar(
            args.input_dir,
            args.output_dir,
            resume=args.resume,
            fresh=args.fresh
        )

        # Exit with error if nothing extracted
        total = sum(total for total, _ in results.values())
        if total == 0:
            logger.error("No sentences extracted from any source")
            return 1

        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review sentence classification accuracy (example vs explanation)")
        logger.info("  2. Verify Esperanto quality in random sample")
        logger.info("  3. Integrate into unified corpus")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
