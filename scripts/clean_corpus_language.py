#!/usr/bin/env python3
"""
Clean Esperanto corpus by removing English-language contamination.

This script:
1. Detects language of each sentence using fastText
2. Removes predominantly English sentences (web scraping debris, UI text, etc.)
3. Keeps genuine Esperanto with incidental English (proper nouns, technical terms)
4. Removes web/HTML artifacts
5. Provides detailed statistics and reports

Usage:
    python scripts/clean_corpus_language.py
    python scripts/clean_corpus_language.py --min-confidence 0.7 --debug
    python scripts/clean_corpus_language.py --dry-run  # Preview only
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.logging_config import setup_logging
from lingua import LanguageDetectorBuilder, Language, IsoCode639_1

logger = logging.getLogger(__name__)


class CorpusCleaner:
    """Clean Esperanto corpus by removing English contamination."""

    def __init__(self, min_confidence: float = 0.6, min_length: int = 10):
        """
        Initialize corpus cleaner.

        Args:
            min_confidence: Minimum confidence for Esperanto detection
            min_length: Minimum sentence length (chars) to keep
        """
        self.min_confidence = min_confidence
        self.min_length = min_length

        # Initialize lingua language detector
        self.detector = LanguageDetectorBuilder.from_all_languages().build()

        # Statistics
        self.stats = {
            'total': 0,
            'kept': 0,
            'removed_english': 0,
            'removed_other_lang': 0,
            'removed_too_short': 0,
            'removed_web_artifact': 0,
            'removed_low_confidence': 0
        }

        # Track removed sentences by reason
        self.removed_samples = defaultdict(list)

    def is_web_artifact(self, text: str) -> bool:
        """
        Check if sentence is web scraping artifact.

        Args:
            text: Sentence text

        Returns:
            True if this is web UI/navigation text
        """
        # Common web UI patterns
        web_patterns = [
            r'\bicon\b.*\billustration\b',  # "icon An illustration of"
            r'\bskip to\b',                  # "Skip to main content"
            r'\bmenu that can be toggled\b', # UI descriptions
            r'\bwayback machine\b',          # Internet Archive
            r'\binternet archive\b',
            r'\bsign up\b.*\blog in\b',      # Login UI
            r'\bdonation\b.*\bheart shape\b',# Donate buttons
            r'\bsearch icon\b',
            r'\bupload icon\b',
            r'\buser icon\b',
            r'^\s*\n\s*\n',                  # Mostly whitespace
            r'^\s*[A-Z][a-z]+ icon\s*$',    # "Video icon", "Audio icon"
        ]

        text_lower = text.lower()

        for pattern in web_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check for excessive navigation/UI keywords
        ui_keywords = [
            'icon', 'illustration', 'upload', 'download', 'menu',
            'toggle', 'click', 'button', 'navigate', 'search',
            'archive', 'wayback', 'donate', 'sign up', 'log in'
        ]

        ui_count = sum(1 for kw in ui_keywords if kw in text_lower)
        if ui_count >= 3:  # 3+ UI keywords = likely artifact
            return True

        return False

    def has_esperanto_markers(self, text: str) -> bool:
        """
        Check if text has characteristic Esperanto markers.

        Args:
            text: Sentence text

        Returns:
            True if Esperanto markers present
        """
        # Esperanto-specific characters
        eo_chars = ['ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ']
        if any(char in text for char in eo_chars):
            return True

        # Common Esperanto words
        eo_words = [
            'estas', 'estis', 'estos',  # verb "to be"
            'kaj', 'sed', 'aŭ',         # conjunctions
            'la', 'de', 'en',           # articles/prepositions
            'tion', 'kion', 'kiu',      # correlatives
            'ĉiuj', 'ĉiu', 'tiu',
        ]

        words = text.lower().split()
        if any(word in eo_words for word in words):
            return True

        # Esperanto endings (-as, -is, -os, -us, -u, -i for verbs)
        verb_endings = [r'\w+(as|is|os|us|i)\b']
        # Noun/adjective endings (-o, -a, -oj, -aj, -on, -an, -ojn, -ajn)
        noun_endings = [r'\w+(o|a|oj|aj|on|an|ojn|ajn)\b']

        all_endings = verb_endings + noun_endings
        for pattern in all_endings:
            if re.search(pattern, text.lower()):
                return True

        return False

    def is_mostly_proper_nouns(self, text: str) -> bool:
        """
        Check if sentence is mostly proper nouns/names.

        Args:
            text: Sentence text

        Returns:
            True if majority is capitalized words (names)
        """
        words = re.findall(r'\b[A-ZÀ-ÿ][a-zà-ÿ]+\b', text)
        total_words = len(text.split())

        if total_words < 3:
            return False

        # If >60% capitalized, likely names/titles
        if len(words) / total_words > 0.6:
            return True

        return False

    def classify_sentence(self, text: str) -> Tuple[str, float, str]:
        """
        Classify sentence as keep/remove with reason.

        Args:
            text: Sentence text

        Returns:
            Tuple of (decision, confidence, reason)
            - decision: 'keep' or 'remove'
            - confidence: language detection confidence
            - reason: explanation
        """
        # Check length
        if len(text.strip()) < self.min_length:
            return ('remove', 0.0, 'too_short')

        # Check for web artifacts
        if self.is_web_artifact(text):
            return ('remove', 0.0, 'web_artifact')

        # Detect language with confidence
        try:
            confidence_values = self.detector.compute_language_confidence_values(text)

            if not confidence_values:
                lang, confidence = 'unknown', 0.0
            else:
                # Get top language
                top_result = confidence_values[0]
                lang_enum = top_result.language
                confidence = top_result.value

                # Convert to ISO code
                iso_code = lang_enum.iso_code_639_1
                lang = iso_code.name.lower()

        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            lang, confidence = 'unknown', 0.0

        # Handle Esperanto
        if lang == 'eo':
            if confidence >= self.min_confidence:
                return ('keep', confidence, f'esperanto_confident_{confidence:.2f}')
            else:
                # Low confidence - check for Esperanto markers
                if self.has_esperanto_markers(text):
                    return ('keep', confidence, f'esperanto_markers_{confidence:.2f}')
                else:
                    return ('remove', confidence, f'low_confidence_{confidence:.2f}')

        # Handle English
        elif lang == 'en':
            # Check if it's mostly proper nouns (names in Esperanto text)
            if self.is_mostly_proper_nouns(text):
                return ('keep', confidence, 'proper_nouns_in_eo_text')

            # Check if it has Esperanto markers despite English detection
            if self.has_esperanto_markers(text):
                return ('keep', confidence, f'eo_markers_despite_en_{confidence:.2f}')

            # Pure English - remove
            return ('remove', confidence, f'english_{confidence:.2f}')

        # Other languages
        else:
            # Check for Esperanto markers
            if self.has_esperanto_markers(text):
                return ('keep', confidence, f'eo_markers_despite_{lang}')

            return ('remove', confidence, f'other_language_{lang}_{confidence:.2f}')

    def clean_corpus(
        self,
        input_path: Path,
        output_path: Path,
        report_path: Optional[Path] = None
    ) -> Dict:
        """
        Clean corpus file.

        Args:
            input_path: Input corpus JSONL file
            output_path: Output cleaned corpus JSONL file
            report_path: Optional path for detailed report

        Returns:
            Statistics dictionary
        """
        logger.info(f"Cleaning corpus: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Min confidence: {self.min_confidence}")
        logger.info(f"Min length: {self.min_length}")
        logger.info("")

        # Read input corpus
        with open(input_path, 'r', encoding='utf-8') as f:
            corpus = [json.loads(line) for line in f]

        self.stats['total'] = len(corpus)
        logger.info(f"Total sentences: {self.stats['total']:,}")
        logger.info("")

        # Process each sentence
        kept_sentences = []
        removed_sentences = []

        for entry in tqdm(corpus, desc="Classifying sentences"):
            text = entry.get('sentence', '').strip()

            decision, confidence, reason = self.classify_sentence(text)

            if decision == 'keep':
                kept_sentences.append(entry)
                self.stats['kept'] += 1
            else:
                removed_sentences.append({
                    'entry': entry,
                    'confidence': confidence,
                    'reason': reason
                })

                # Update statistics
                if 'english' in reason:
                    self.stats['removed_english'] += 1
                elif 'other_language' in reason:
                    self.stats['removed_other_lang'] += 1
                elif reason == 'too_short':
                    self.stats['removed_too_short'] += 1
                elif reason == 'web_artifact':
                    self.stats['removed_web_artifact'] += 1
                elif 'low_confidence' in reason:
                    self.stats['removed_low_confidence'] += 1

                # Keep samples for report
                if len(self.removed_samples[reason]) < 10:
                    self.removed_samples[reason].append(text[:100])

        # Write cleaned corpus
        logger.info("")
        logger.info(f"Writing cleaned corpus to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in kept_sentences:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Wrote {len(kept_sentences):,} sentences")

        # Write detailed report
        if report_path:
            self._write_report(report_path, removed_sentences)

        return self.stats

    def _write_report(self, report_path: Path, removed_sentences: List[Dict]):
        """Write detailed cleaning report."""
        logger.info("")
        logger.info(f"Writing detailed report to {report_path}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ESPERANTO CORPUS CLEANING REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Statistics
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total sentences:          {self.stats['total']:,}\n")
            f.write(f"Kept (Esperanto):         {self.stats['kept']:,} ({self.stats['kept']/self.stats['total']*100:.1f}%)\n")
            f.write(f"Removed (total):          {self.stats['total'] - self.stats['kept']:,} ({(self.stats['total'] - self.stats['kept'])/self.stats['total']*100:.1f}%)\n")
            f.write("\n")

            f.write("Removal Breakdown:\n")
            f.write(f"  English contamination:  {self.stats['removed_english']:,}\n")
            f.write(f"  Web artifacts:          {self.stats['removed_web_artifact']:,}\n")
            f.write(f"  Other languages:        {self.stats['removed_other_lang']:,}\n")
            f.write(f"  Too short:              {self.stats['removed_too_short']:,}\n")
            f.write(f"  Low confidence:         {self.stats['removed_low_confidence']:,}\n")
            f.write("\n\n")

            # Samples by removal reason
            f.write("REMOVED SENTENCE SAMPLES\n")
            f.write("=" * 80 + "\n\n")

            for reason, samples in sorted(self.removed_samples.items()):
                f.write(f"{reason.upper()}\n")
                f.write("-" * 80 + "\n")
                for i, sample in enumerate(samples, 1):
                    f.write(f"{i}. {sample}...\n")
                f.write("\n")

            # Full removed sentences (first 100)
            f.write("\n" + "=" * 80 + "\n")
            f.write("FULL REMOVED SENTENCES (first 100)\n")
            f.write("=" * 80 + "\n\n")

            for i, item in enumerate(removed_sentences[:100], 1):
                entry = item['entry']
                f.write(f"{i}. [{item['reason']}] {entry.get('sentence', '')[:200]}\n")
                f.write(f"   Source: {entry.get('source_name', 'Unknown')}\n\n")

        logger.info("Report written")


def main():
    """Clean Esperanto corpus."""
    parser = argparse.ArgumentParser(
        description='Clean Esperanto corpus by removing English contamination'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/corpus_sentences.jsonl',
        help='Input corpus file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/corpus_sentences_cleaned.jsonl',
        help='Output cleaned corpus file'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='data/corpus_cleaning_report.txt',
        help='Detailed cleaning report file'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum language detection confidence (default: 0.6)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum sentence length in characters (default: 10)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview only, do not write output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    print("=" * 80)
    print("ESPERANTO CORPUS LANGUAGE CLEANING")
    print("=" * 80)
    print()
    print(f"Input:           {args.input}")
    print(f"Output:          {args.output}")
    print(f"Report:          {args.report}")
    print(f"Min confidence:  {args.min_confidence}")
    print(f"Min length:      {args.min_length} chars")
    print(f"Dry run:         {args.dry_run}")
    print()

    try:
        # Check input exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        # Create cleaner
        cleaner = CorpusCleaner(
            min_confidence=args.min_confidence,
            min_length=args.min_length
        )

        # Clean corpus (or preview)
        if args.dry_run:
            # Preview mode - just show statistics
            with open(input_path, 'r', encoding='utf-8') as f:
                corpus = [json.loads(line) for line in f]

            print("DRY RUN - Preview of classifications:")
            print()

            preview_count = 50
            for i, entry in enumerate(corpus[:preview_count], 1):
                text = entry.get('sentence', '').strip()
                decision, confidence, reason = cleaner.classify_sentence(text)

                status = "✓ KEEP" if decision == 'keep' else "✗ REMOVE"
                print(f"{i}. [{status}] {reason}")
                print(f"   {text[:80]}...")
                print()

            print(f"\nShowing first {preview_count} sentences.")
            print("Run without --dry-run to perform actual cleaning.")

        else:
            # Actual cleaning
            output_path = Path(args.output)
            report_path = Path(args.report)

            # Backup original if output would overwrite it
            if output_path == input_path:
                backup_path = input_path.with_suffix('.jsonl.backup')
                logger.warning(f"Output same as input - creating backup: {backup_path}")
                import shutil
                shutil.copy2(input_path, backup_path)

            # Clean
            stats = cleaner.clean_corpus(input_path, output_path, report_path)

            # Summary
            print()
            print("=" * 80)
            print("CLEANING COMPLETE")
            print("=" * 80)
            print()
            print(f"Total sentences:        {stats['total']:,}")
            print(f"Kept (Esperanto):       {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
            print(f"Removed:                {stats['total'] - stats['kept']:,} ({(stats['total'] - stats['kept'])/stats['total']*100:.1f}%)")
            print()
            print("Removal breakdown:")
            print(f"  English:              {stats['removed_english']:,}")
            print(f"  Web artifacts:        {stats['removed_web_artifact']:,}")
            print(f"  Other languages:      {stats['removed_other_lang']:,}")
            print(f"  Too short:            {stats['removed_too_short']:,}")
            print(f"  Low confidence:       {stats['removed_low_confidence']:,}")
            print()
            print(f"Cleaned corpus:         {output_path}")
            print(f"Detailed report:        {report_path}")
            print()

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
