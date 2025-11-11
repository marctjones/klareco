"""
Continuous Vocabulary Validation System

This script tracks vocabulary quality metrics over time:
1. Corpus coverage trends
2. Vocabulary size changes
3. Category distribution evolution
4. Frequency analysis
5. Test pass rates

Run periodically (e.g., after vocabulary updates) to track quality.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str], cwd: Path = None) -> tuple:
    """Run a command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except Exception as e:
        return "", str(e), -1


def get_vocabulary_stats() -> Dict:
    """Get current vocabulary statistics from parser."""
    try:
        from klareco.parser import (
            KNOWN_ROOTS, KNOWN_PREFIXES, KNOWN_SUFFIXES,
            KNOWN_CONJUNCTIONS, KNOWN_PREPOSITIONS,
            KNOWN_CORRELATIVES, KNOWN_PARTICLES, KNOWN_PRONOUNS
        )

        return {
            'roots': len(KNOWN_ROOTS),
            'prefixes': len(KNOWN_PREFIXES),
            'suffixes': len(KNOWN_SUFFIXES),
            'conjunctions': len(KNOWN_CONJUNCTIONS),
            'prepositions': len(KNOWN_PREPOSITIONS),
            'correlatives': len(KNOWN_CORRELATIVES),
            'particles': len(KNOWN_PARTICLES),
            'pronouns': len(KNOWN_PRONOUNS),
            'total': (len(KNOWN_ROOTS) + len(KNOWN_PREFIXES) +
                     len(KNOWN_SUFFIXES) + len(KNOWN_CONJUNCTIONS) +
                     len(KNOWN_PREPOSITIONS) + len(KNOWN_CORRELATIVES) +
                     len(KNOWN_PARTICLES) + len(KNOWN_PRONOUNS))
        }
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return {}


def get_enriched_vocab_stats(enriched_path: Path) -> Dict:
    """Get statistics from enriched vocabulary."""
    if not enriched_path.exists():
        return {}

    try:
        with open(enriched_path, 'r', encoding='utf-8') as f:
            enriched = json.load(f)

        categories = {}
        for root, data in enriched.items():
            cat = data.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        # Frequency tiers
        freq_tiers = {
            'common': 0,      # >= 10
            'moderate': 0,    # 5-9
            'rare': 0,        # 1-4
            'unused': 0       # 0
        }

        for root, data in enriched.items():
            freq = data.get('total_frequency', 0)
            if freq >= 10:
                freq_tiers['common'] += 1
            elif freq >= 5:
                freq_tiers['moderate'] += 1
            elif freq >= 1:
                freq_tiers['rare'] += 1
            else:
                freq_tiers['unused'] += 1

        return {
            'total_enriched': len(enriched),
            'categories': categories,
            'frequency_tiers': freq_tiers
        }
    except Exception as e:
        print(f"Error loading enriched vocabulary: {e}")
        return {}


def run_test_suite(project_root: Path) -> Dict:
    """Run integration tests and return results."""
    print("Running integration tests...")

    cmd = ['python', '-m', 'klareco', 'test', '--num-sentences', '20']
    stdout, stderr, returncode = run_command(cmd, cwd=project_root)

    # Parse output for pass/fail counts
    passed = 0
    failed = 0

    for line in stdout.split('\n') + stderr.split('\n'):
        if 'PASSED' in line and 'sentence' in line:
            passed += 1
        elif 'FAILED' in line and 'sentence' in line:
            failed += 1

    return {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'pass_rate': (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0,
        'returncode': returncode
    }


def run_vocabulary_validation(project_root: Path) -> Dict:
    """Run vocabulary validation script and parse results."""
    print("Running vocabulary validation...")

    cmd = ['python', 'scripts/validate_vocabulary.py']
    stdout, stderr, returncode = run_command(cmd, cwd=project_root)

    # Parse output for coverage stats
    corpus_coverage = None
    unique_roots = None
    covered_roots = None

    for line in stdout.split('\n'):
        if 'Coverage:' in line:
            match = line.split(':')[-1].strip().replace('%', '')
            try:
                corpus_coverage = float(match)
            except:
                pass
        elif 'Unique roots:' in line:
            try:
                unique_roots = int(line.split(':')[-1].strip())
            except:
                pass
        elif 'Covered roots:' in line:
            try:
                covered_roots = int(line.split(':')[-1].strip())
            except:
                pass

    return {
        'corpus_coverage': corpus_coverage,
        'unique_roots': unique_roots,
        'covered_roots': covered_roots,
        'returncode': returncode
    }


def generate_validation_report(project_root: Path, history_file: Path) -> Dict:
    """Generate comprehensive validation report and save to history."""

    timestamp = datetime.now().isoformat()
    print(f"\n=== Continuous Validation Report ===")
    print(f"Timestamp: {timestamp}\n")

    # 1. Vocabulary statistics
    print("1. Collecting vocabulary statistics...")
    vocab_stats = get_vocabulary_stats()
    print(f"   Total vocabulary: {vocab_stats.get('total', 0)} items")

    # 2. Enriched vocabulary statistics
    enriched_path = project_root / 'data' / 'enriched_vocabulary.json'
    print("\n2. Analyzing enriched vocabulary...")
    enriched_stats = get_enriched_vocab_stats(enriched_path)

    if enriched_stats:
        print(f"   Enriched roots: {enriched_stats.get('total_enriched', 0)}")
        freq_tiers = enriched_stats.get('frequency_tiers', {})
        print(f"   Common roots (>=10): {freq_tiers.get('common', 0)}")
        print(f"   Moderate roots (5-9): {freq_tiers.get('moderate', 0)}")
        print(f"   Rare roots (1-4): {freq_tiers.get('rare', 0)}")
        print(f"   Unused roots (0): {freq_tiers.get('unused', 0)}")

    # 3. Test suite results
    print("\n3. Running test suite...")
    test_results = run_test_suite(project_root)
    print(f"   Tests: {test_results['passed']}/{test_results['total']} passing")
    print(f"   Pass rate: {test_results['pass_rate']:.1f}%")

    # 4. Corpus coverage
    print("\n4. Validating corpus coverage...")
    validation_results = run_vocabulary_validation(project_root)
    if validation_results['corpus_coverage'] is not None:
        print(f"   Corpus coverage: {validation_results['corpus_coverage']:.1f}%")
        print(f"   Covered roots: {validation_results['covered_roots']}/{validation_results['unique_roots']}")

    # 5. Compile report
    report = {
        'timestamp': timestamp,
        'vocabulary': vocab_stats,
        'enriched': enriched_stats,
        'tests': test_results,
        'coverage': validation_results
    }

    # 6. Load history
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            pass

    # 7. Add to history
    history.append(report)

    # 8. Save history (keep last 100 reports)
    history = history[-100:]
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n5. Report saved to history: {history_file}")
    print(f"   History contains {len(history)} reports")

    # 9. Compare with previous report
    if len(history) > 1:
        print("\n6. Comparison with previous report:")
        prev = history[-2]

        # Vocabulary size change
        prev_total = prev.get('vocabulary', {}).get('total', 0)
        curr_total = vocab_stats.get('total', 0)
        vocab_change = curr_total - prev_total
        if vocab_change != 0:
            sign = '+' if vocab_change > 0 else ''
            print(f"   Vocabulary: {sign}{vocab_change} items")

        # Coverage change
        prev_cov = prev.get('coverage', {}).get('corpus_coverage')
        curr_cov = validation_results.get('corpus_coverage')
        if prev_cov is not None and curr_cov is not None:
            cov_change = curr_cov - prev_cov
            sign = '+' if cov_change > 0 else ''
            print(f"   Coverage: {sign}{cov_change:.1f}%")

        # Test pass rate change
        prev_pass = prev.get('tests', {}).get('pass_rate', 0)
        curr_pass = test_results.get('pass_rate', 0)
        pass_change = curr_pass - prev_pass
        if pass_change != 0:
            sign = '+' if pass_change > 0 else ''
            print(f"   Pass rate: {sign}{pass_change:.1f}%")

    print("\n=== End of Report ===\n")

    return report


def generate_trend_analysis(history_file: Path):
    """Analyze trends from validation history."""
    if not history_file.exists():
        print("No history file found. Run validation first.")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    if len(history) < 2:
        print("Need at least 2 reports to show trends.")
        return

    print("\n=== Trend Analysis ===\n")

    # Extract time series data
    timestamps = [r['timestamp'] for r in history]
    vocab_sizes = [r.get('vocabulary', {}).get('total', 0) for r in history]
    coverages = [r.get('coverage', {}).get('corpus_coverage', 0) for r in history]
    pass_rates = [r.get('tests', {}).get('pass_rate', 0) for r in history]

    print(f"Reports: {len(history)}")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")

    print(f"\nVocabulary size:")
    print(f"  First: {vocab_sizes[0]}")
    print(f"  Latest: {vocab_sizes[-1]}")
    print(f"  Change: {vocab_sizes[-1] - vocab_sizes[0]:+d}")

    print(f"\nCorpus coverage:")
    print(f"  First: {coverages[0]:.1f}%")
    print(f"  Latest: {coverages[-1]:.1f}%")
    print(f"  Change: {coverages[-1] - coverages[0]:+.1f}%")

    print(f"\nTest pass rate:")
    print(f"  First: {pass_rates[0]:.1f}%")
    print(f"  Latest: {pass_rates[-1]:.1f}%")
    print(f"  Change: {pass_rates[-1] - pass_rates[0]:+.1f}%")

    # Identify best/worst
    best_coverage_idx = coverages.index(max(coverages))
    print(f"\nBest coverage: {max(coverages):.1f}% on {timestamps[best_coverage_idx][:10]}")

    print("\n=== End of Trend Analysis ===\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Continuous validation for vocabulary quality'
    )
    parser.add_argument(
        '--history',
        default='data/validation_history.json',
        help='Path to validation history file'
    )
    parser.add_argument(
        '--trends',
        action='store_true',
        help='Show trend analysis from history'
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    history_file = project_root / args.history

    if args.trends:
        generate_trend_analysis(history_file)
    else:
        generate_validation_report(project_root, history_file)


if __name__ == '__main__':
    main()
