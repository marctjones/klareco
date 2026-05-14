#!/usr/bin/env python3
"""
Validate corpus with token-efficient streaming processing.

Checks:
- JSON validity (can parse each line)
- Required fields present
- Quality labels correct
- Parse rates reasonable
- AST structure valid
- No duplicate sentences

Memory-efficient: Streams line-by-line, minimal storage.

Usage:
    # Fast validation (samples only)
    python scripts/validate/validate_corpus.py --quick

    # Full validation
    python scripts/validate/validate_corpus.py

    # Validate and fix issues
    python scripts/validate/validate_corpus.py --fix --output data/enhanced_corpus/corpus_fixed.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


class CorpusValidator:
    """Memory-efficient corpus validator."""

    def __init__(self, quick: bool = False, sample_rate: int = 100):
        self.quick = quick
        self.sample_rate = sample_rate

        # Statistics (small memory footprint)
        self.stats = {
            'total': 0,
            'valid': 0,
            'json_errors': 0,
            'missing_fields': 0,
            'invalid_quality': 0,
            'invalid_ast': 0,
            'duplicates': 0,
            'quality_counts': defaultdict(int),
            'parse_rate_sum': 0,
            'parse_rate_count': 0,
        }

        # For duplicate detection (use hash, not full text)
        self.seen_hashes = set()

        # Error samples (store max 100)
        self.error_samples = []
        self.max_error_samples = 100

    def add_error_sample(self, line_num: int, error_type: str, details: str, preview: str):
        """Store error sample if under limit."""
        if len(self.error_samples) < self.max_error_samples:
            self.error_samples.append({
                'line': line_num,
                'type': error_type,
                'details': details,
                'preview': preview[:200]
            })

    def validate_line(self, line: str, line_num: int) -> tuple:
        """
        Validate a single line.

        Returns: (is_valid, error_type, error_details, parsed_entry or None)
        """

        # Check JSON validity
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            return False, 'json_error', str(e), None

        # Check required fields
        required_fields = ['text', 'source', 'ast', 'parse_rate']
        for field in required_fields:
            if field not in entry:
                return False, 'missing_field', f"Missing field: {field}", entry

        # Check source has quality
        if 'quality' not in entry.get('source', {}):
            return False, 'missing_field', "Missing source.quality", entry

        # Validate quality label
        quality = entry['source']['quality']
        if quality not in ['GOLD', 'SILVER', 'BRONZE', 'COPPER', 'UNKNOWN']:
            return False, 'invalid_quality', f"Invalid quality: {quality}", entry

        # Validate parse rate
        parse_rate = entry.get('parse_rate')
        if not isinstance(parse_rate, (int, float)) or parse_rate < 0 or parse_rate > 1:
            return False, 'invalid_parse_rate', f"Invalid parse_rate: {parse_rate}", entry

        # Basic AST validation (just check it's a dict with tipo field)
        ast = entry.get('ast')
        if not isinstance(ast, dict) or 'tipo' not in ast:
            return False, 'invalid_ast', "AST missing or invalid", entry

        # Check for duplicates (use hash for memory efficiency)
        text = entry.get('text', '')
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return False, 'duplicate', f"Duplicate text hash: {text_hash[:16]}...", entry

        self.seen_hashes.add(text_hash)

        return True, None, None, entry

    def validate_corpus(self, corpus_path: Path, output_path: Path = None):
        """Validate entire corpus with streaming."""

        print(f"Validating corpus: {corpus_path}")
        print(f"Mode: {'Quick' if self.quick else 'Full'}")
        if self.quick:
            print(f"Sample rate: 1 in {self.sample_rate}")
        print()

        output_file = None
        if output_path:
            output_file = open(output_path, 'w', encoding='utf-8')
            print(f"Fixed output: {output_path}")
            print()

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total'] += 1

                # Quick mode: sample only
                if self.quick and line_num % self.sample_rate != 0:
                    continue

                # Validate
                is_valid, error_type, error_details, entry = self.validate_line(line, line_num)

                if is_valid:
                    self.stats['valid'] += 1

                    # Update statistics
                    quality = entry['source']['quality']
                    self.stats['quality_counts'][quality] += 1

                    parse_rate = entry['parse_rate']
                    self.stats['parse_rate_sum'] += parse_rate
                    self.stats['parse_rate_count'] += 1

                    # Write to output if fixing
                    if output_file:
                        output_file.write(line)

                else:
                    # Record error
                    if error_type == 'json_error':
                        self.stats['json_errors'] += 1
                    elif error_type == 'missing_field':
                        self.stats['missing_fields'] += 1
                    elif error_type == 'invalid_quality':
                        self.stats['invalid_quality'] += 1
                    elif error_type == 'invalid_ast':
                        self.stats['invalid_ast'] += 1
                    elif error_type == 'duplicate':
                        self.stats['duplicates'] += 1

                    self.add_error_sample(line_num, error_type, error_details, line)

                # Progress
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines... ({self.stats['valid']:,} valid)")

        if output_file:
            output_file.close()

        print()

    def print_report(self):
        """Print validation report."""

        print("=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print()

        total = self.stats['total']
        valid = self.stats['valid']
        invalid = total - valid

        print(f"Total lines: {total:,}")
        print(f"Valid: {valid:,} ({valid/total*100:.2f}%)")
        print(f"Invalid: {invalid:,} ({invalid/total*100:.2f}%)")
        print()

        if invalid > 0:
            print("Errors by type:")
            print(f"  JSON errors: {self.stats['json_errors']:,}")
            print(f"  Missing fields: {self.stats['missing_fields']:,}")
            print(f"  Invalid quality: {self.stats['invalid_quality']:,}")
            print(f"  Invalid AST: {self.stats['invalid_ast']:,}")
            print(f"  Duplicates: {self.stats['duplicates']:,}")
            print()

        if self.stats['parse_rate_count'] > 0:
            avg_parse_rate = self.stats['parse_rate_sum'] / self.stats['parse_rate_count']
            print(f"Average parse rate: {avg_parse_rate:.4f}")
            print()

        if self.stats['quality_counts']:
            print("Quality distribution:")
            for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER', 'UNKNOWN']:
                count = self.stats['quality_counts'].get(quality, 0)
                if count > 0:
                    pct = (count / valid * 100) if valid > 0 else 0
                    print(f"  {quality:8s}: {count:,} ({pct:.1f}%)")
            print()

        if self.error_samples:
            print(f"Error samples (showing first {len(self.error_samples)}):")
            for i, error in enumerate(self.error_samples[:10], 1):
                print(f"\n  {i}. Line {error['line']}: {error['type']}")
                print(f"     {error['details']}")
                print(f"     Preview: {error['preview'][:100]}...")

            if len(self.error_samples) > 10:
                print(f"\n  ... and {len(self.error_samples) - 10} more errors")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate corpus with memory-efficient streaming"
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Corpus file to validate'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation (sample every 100th line)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=100,
        help='Sample rate for quick mode (default: 100)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Write valid lines to output file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for fixed corpus (requires --fix)'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus not found: {args.corpus}")
        return 1

    if args.fix and not args.output:
        print("Error: --fix requires --output")
        return 1

    # Run validation
    validator = CorpusValidator(
        quick=args.quick,
        sample_rate=args.sample_rate
    )

    validator.validate_corpus(
        args.corpus,
        output_path=args.output if args.fix else None
    )

    validator.print_report()

    # Exit code: 0 if all valid, 1 if errors found
    if validator.stats['valid'] == validator.stats['total']:
        print("✓ Corpus is valid!")
        return 0
    else:
        print("⚠️  Corpus has errors")
        if args.fix:
            print(f"✓ Fixed corpus written to: {args.output}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
