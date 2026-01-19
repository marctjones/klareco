#!/usr/bin/env python3
"""
Analyze coverage and quality of external semantic resources (ConceptNet, Wikidata).

Compares feasibility study results to determine best import strategy.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_coverage(
    conceptnet_report_path: Path,
    wikidata_report_path: Path,
    output_path: Path
) -> Dict:
    """Analyze and compare coverage from ConceptNet and Wikidata feasibility studies.

    Args:
        conceptnet_report_path: Path to ConceptNet feasibility report
        wikidata_report_path: Path to Wikidata feasibility report
        output_path: Path to save combined analysis

    Returns:
        Combined analysis report
    """
    # Load reports
    with open(conceptnet_report_path) as f:
        conceptnet_report = json.load(f)

    with open(wikidata_report_path) as f:
        wikidata_report = json.load(f)

    # Extract sample results
    cn_results = {r['root']: r for r in conceptnet_report['sample_results']}
    wd_results = {r['root']: r for r in wikidata_report['sample_results']}

    # Calculate coverage stats
    total_nouns = conceptnet_report['total_content_nouns']
    sample_size = conceptnet_report['sample_size']

    cn_found = conceptnet_report['coverage']['found']
    wd_found = wikidata_report['coverage']['found']

    # Calculate combined coverage (union)
    both_found = sum(
        1 for root in cn_results
        if cn_results[root]['found_in_conceptnet'] and wd_results[root]['found_in_wikidata']
    )

    either_found = sum(
        1 for root in cn_results
        if cn_results[root]['found_in_conceptnet'] or wd_results[root]['found_in_wikidata']
    )

    # Quality analysis: Manual inspection needed
    # For now, just identify which words were found in each source
    only_conceptnet = [
        root for root in cn_results
        if cn_results[root]['found_in_conceptnet'] and not wd_results[root]['found_in_wikidata']
    ]

    only_wikidata = [
        root for root in cn_results
        if not cn_results[root]['found_in_conceptnet'] and wd_results[root]['found_in_wikidata']
    ]

    in_both = [
        root for root in cn_results
        if cn_results[root]['found_in_conceptnet'] and wd_results[root]['found_in_wikidata']
    ]

    in_neither = [
        root for root in cn_results
        if not cn_results[root]['found_in_conceptnet'] and not wd_results[root]['found_in_wikidata']
    ]

    # Generate combined analysis
    analysis = {
        'sample_size': sample_size,
        'total_content_nouns': total_nouns,
        'conceptnet_coverage': {
            'found': cn_found,
            'percentage': cn_found / sample_size * 100,
            'avg_relations_per_word': conceptnet_report['relations']['avg_per_word']
        },
        'wikidata_coverage': {
            'found': wd_found,
            'percentage': wd_found / sample_size * 100,
            'avg_relations_per_word': wikidata_report['instance_relations']['avg_per_word']
        },
        'combined_coverage': {
            'union': {
                'found': either_found,
                'percentage': either_found / sample_size * 100
            },
            'intersection': {
                'found': both_found,
                'percentage': both_found / sample_size * 100
            }
        },
        'source_breakdown': {
            'only_conceptnet': {
                'count': len(only_conceptnet),
                'words': only_conceptnet
            },
            'only_wikidata': {
                'count': len(only_wikidata),
                'words': only_wikidata
            },
            'in_both': {
                'count': len(in_both),
                'words': in_both
            },
            'in_neither': {
                'count': len(in_neither),
                'words': in_neither
            }
        },
        'projected_coverage': {
            'conceptnet_only': {
                'estimated_words': int(total_nouns * (cn_found / sample_size)),
                'percentage': cn_found / sample_size * 100
            },
            'wikidata_only': {
                'estimated_words': int(total_nouns * (wd_found / sample_size)),
                'percentage': wd_found / sample_size * 100
            },
            'combined': {
                'estimated_words': int(total_nouns * (either_found / sample_size)),
                'percentage': either_found / sample_size * 100
            }
        },
        'recommendation': None  # Will be set below
    }

    # Generate recommendation
    combined_pct = analysis['combined_coverage']['union']['percentage']

    if combined_pct >= 70:
        recommendation = "PROCEED with full import from both sources (combined coverage >= 70%)"
        proceed = True
    elif combined_pct >= 50:
        recommendation = "PROCEED with caution - consider manual categorization for remainder"
        proceed = True
    else:
        recommendation = "DO NOT PROCEED - coverage too low, recommend manual categorization (Issue #488)"
        proceed = False

    analysis['recommendation'] = recommendation
    analysis['proceed_with_import'] = proceed

    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EXTERNAL RESOURCE COVERAGE ANALYSIS")
    logger.info("="*70)
    logger.info(f"\nSample size: {sample_size} nouns")
    logger.info(f"Total uncategorized content nouns: {total_nouns}")
    logger.info(f"\nConceptNet coverage: {cn_found}/{sample_size} ({analysis['conceptnet_coverage']['percentage']:.1f}%)")
    logger.info(f"  Avg relations per word: {analysis['conceptnet_coverage']['avg_relations_per_word']:.2f}")
    logger.info(f"\nWikidata coverage: {wd_found}/{sample_size} ({analysis['wikidata_coverage']['percentage']:.1f}%)")
    logger.info(f"  Avg relations per word: {analysis['wikidata_coverage']['avg_relations_per_word']:.2f}")
    logger.info(f"\nCombined coverage (union): {either_found}/{sample_size} ({combined_pct:.1f}%)")
    logger.info(f"  Both sources: {both_found}/{sample_size} ({analysis['combined_coverage']['intersection']['percentage']:.1f}%)")
    logger.info(f"\nSource breakdown:")
    logger.info(f"  Only ConceptNet: {len(only_conceptnet)}")
    logger.info(f"  Only Wikidata: {len(only_wikidata)}")
    logger.info(f"  In both: {len(in_both)}")
    logger.info(f"  In neither: {len(in_neither)}")
    logger.info(f"\nProjected full coverage:")
    logger.info(f"  ConceptNet: ~{analysis['projected_coverage']['conceptnet_only']['estimated_words']} nouns")
    logger.info(f"  Wikidata: ~{analysis['projected_coverage']['wikidata_only']['estimated_words']} nouns")
    logger.info(f"  Combined: ~{analysis['projected_coverage']['combined']['estimated_words']} nouns ({analysis['projected_coverage']['combined']['percentage']:.1f}%)")
    logger.info(f"\n{'='*70}")
    logger.info(f"RECOMMENDATION: {recommendation}")
    logger.info(f"{'='*70}")
    logger.info(f"\nAnalysis saved to: {output_path}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Analyze coverage from ConceptNet and Wikidata feasibility studies'
    )
    parser.add_argument(
        '--conceptnet-report',
        type=Path,
        default=Path('data/vocabularies/external/conceptnet_feasibility_report.json'),
        help='Path to ConceptNet feasibility report'
    )
    parser.add_argument(
        '--wikidata-report',
        type=Path,
        default=Path('data/vocabularies/external/wikidata_feasibility_report.json'),
        help='Path to Wikidata feasibility report'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/external/coverage_analysis.json'),
        help='Output path for combined analysis'
    )

    args = parser.parse_args()

    # Check that feasibility reports exist
    if not args.conceptnet_report.exists():
        logger.error(f"ConceptNet report not found: {args.conceptnet_report}")
        logger.error("Run: python scripts/query_conceptnet.py --sample-size 50")
        return

    if not args.wikidata_report.exists():
        logger.error(f"Wikidata report not found: {args.wikidata_report}")
        logger.error("Run: python scripts/query_wikidata.py --sample-size 50")
        return

    # Run analysis
    analysis = analyze_coverage(
        conceptnet_report_path=args.conceptnet_report,
        wikidata_report_path=args.wikidata_report,
        output_path=args.output
    )

    logger.info("\n✓ Coverage analysis complete!")

    # Exit code based on recommendation
    if not analysis['proceed_with_import']:
        logger.warning("\n⚠ Coverage too low - consider manual categorization instead")
        exit(1)


if __name__ == '__main__':
    main()
