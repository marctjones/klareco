#!/usr/bin/env python3
"""
Validate ReVo semantic relations extracted from database.

This script performs quality checks on the extracted semantic relations:
1. Checks for symmetric consistency (if A syn B, then B syn A expected)
2. Validates antonym pairs (A ant B should have B ant A)
3. Checks hypernym/hyponym consistency (if A super B, then B sub A)
4. Detects cycles in hierarchical relations
5. Validates root existence in corpus

Usage:
    python scripts/validate_revo_relations.py
    python scripts/validate_revo_relations.py --relations path/to/revo_semantic_relations.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RevoValidator:
    """Validate ReVo semantic relations."""

    def __init__(self, relations_file: Path, corpus_index: Path = None):
        self.relations_file = Path(relations_file)
        self.corpus_index = Path(corpus_index) if corpus_index else None

        self.relations: Dict[str, List[Dict]] = {}
        self.metadata: Dict = {}

        # Validation results
        self.issues = {
            'asymmetric_synonyms': [],
            'asymmetric_antonyms': [],
            'inconsistent_hypernyms': [],
            'cycles': [],
            'missing_in_corpus': [],
        }

    def load_relations(self):
        """Load relations from JSON file."""
        logger.info(f"Loading relations from: {self.relations_file}")

        if not self.relations_file.exists():
            logger.error(f"Relations file not found: {self.relations_file}")
            sys.exit(1)

        with open(self.relations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})
        self.relations = data.get('relations', {})

        total = sum(len(rels) for rels in self.relations.values())
        logger.info(f"  Loaded {total:,} relations")
        for rel_type, rels in self.relations.items():
            logger.info(f"    {rel_type}: {len(rels):,}")

    def load_corpus_roots(self) -> Set[str]:
        """Load roots from Kuzu corpus index (if available)."""
        if not self.corpus_index or not self.corpus_index.exists():
            logger.info("No corpus index provided, skipping corpus validation")
            return set()

        logger.info(f"Loading corpus roots from: {self.corpus_index}")

        try:
            import kuzu
            db = kuzu.Database(str(self.corpus_index))
            conn = kuzu.Connection(db)

            roots = set()
            result = conn.execute("MATCH (r:Root) RETURN r.root")
            while result.has_next():
                roots.add(result.get_next()[0])

            logger.info(f"  Loaded {len(roots):,} corpus roots")
            return roots
        except Exception as e:
            logger.warning(f"Failed to load corpus roots: {e}")
            return set()

    def check_symmetric_relations(self):
        """Check if symmetric relations are truly symmetric."""
        logger.info("")
        logger.info("Checking symmetric relations...")

        # Build bidirectional maps
        synonym_map = defaultdict(set)
        antonym_map = defaultdict(set)

        for rel in self.relations.get('synonym', []):
            synonym_map[rel['source']].add(rel['target'])

        for rel in self.relations.get('antonym', []):
            antonym_map[rel['source']].add(rel['target'])

        # Check synonyms
        for source, targets in synonym_map.items():
            for target in targets:
                if source not in synonym_map.get(target, set()):
                    self.issues['asymmetric_synonyms'].append((source, target))

        # Check antonyms
        for source, targets in antonym_map.items():
            for target in targets:
                if source not in antonym_map.get(target, set()):
                    self.issues['asymmetric_antonyms'].append((source, target))

        logger.info(f"  Asymmetric synonyms: {len(self.issues['asymmetric_synonyms']):,}")
        logger.info(f"  Asymmetric antonyms: {len(self.issues['asymmetric_antonyms']):,}")

    def check_hypernym_consistency(self):
        """Check if hypernym/hyponym relations are consistent."""
        logger.info("")
        logger.info("Checking hypernym/hyponym consistency...")

        hypernym_map = defaultdict(set)
        hyponym_map = defaultdict(set)

        for rel in self.relations.get('hypernym', []):
            # A hypernym B means A is-a B (A is subtype of B)
            # So B should have A as hyponym
            hypernym_map[rel['source']].add(rel['target'])

        for rel in self.relations.get('hyponym', []):
            # A hyponym B means A has-subtype B
            # So B should have A as hypernym
            hyponym_map[rel['source']].add(rel['target'])

        # Check consistency: if A hypernym B, then B should have A as hyponym
        for source, targets in hypernym_map.items():
            for target in targets:
                if source not in hyponym_map.get(target, set()):
                    self.issues['inconsistent_hypernyms'].append((source, 'hypernym', target))

    def check_cycles(self):
        """Check for cycles in hierarchical relations (hypernym/hyponym)."""
        logger.info("")
        logger.info("Checking for cycles...")

        # Build directed graph
        graph = defaultdict(set)
        for rel in self.relations.get('hypernym', []):
            graph[rel['source']].add(rel['target'])

        # DFS to detect cycles
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    self.issues['cycles'].append((node, neighbor))
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for node in graph.keys():
            if node not in visited:
                has_cycle(node, visited, set())

        logger.info(f"  Cycles found: {len(self.issues['cycles']):,}")

    def check_corpus_coverage(self, corpus_roots: Set[str]):
        """Check how many relation roots exist in corpus."""
        if not corpus_roots:
            logger.info("")
            logger.info("Skipping corpus coverage check (no corpus roots loaded)")
            return

        logger.info("")
        logger.info("Checking corpus coverage...")

        all_roots = set()
        for rel_type, rels in self.relations.items():
            for rel in rels:
                all_roots.add(rel['source'])
                all_roots.add(rel['target'])

        missing = all_roots - corpus_roots
        self.issues['missing_in_corpus'] = list(missing)

        coverage_pct = (len(all_roots & corpus_roots) / len(all_roots) * 100) if all_roots else 0

        logger.info(f"  Total unique roots in relations: {len(all_roots):,}")
        logger.info(f"  Found in corpus: {len(all_roots & corpus_roots):,} ({coverage_pct:.1f}%)")
        logger.info(f"  Missing from corpus: {len(missing):,}")

        if len(missing) > 0 and len(missing) <= 20:
            logger.info(f"  Missing roots: {', '.join(sorted(missing)[:20])}")

    def print_summary(self):
        """Print validation summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info("")

        total_issues = sum(len(issues) for issues in self.issues.values())

        if total_issues == 0:
            logger.info("✓ All checks passed! No issues found.")
        else:
            logger.info(f"Found {total_issues:,} potential issues:")
            logger.info("")
            for issue_type, issues in self.issues.items():
                if len(issues) > 0:
                    logger.info(f"  {issue_type}: {len(issues):,}")

                    # Show first few examples
                    if issue_type in ['asymmetric_synonyms', 'asymmetric_antonyms']:
                        for src, tgt in issues[:5]:
                            logger.info(f"    - {src} → {tgt} (missing reverse)")
                    elif issue_type == 'cycles':
                        for src, tgt in issues[:5]:
                            logger.info(f"    - {src} → {tgt}")

        logger.info("")
        logger.info("Note: Some asymmetry is expected and not necessarily errors.")
        logger.info("ReVo may intentionally have one-way relations for clarity.")

    def save_report(self, output_path: Path):
        """Save validation report to JSON."""
        logger.info("")
        logger.info(f"Saving validation report to: {output_path}")

        report = {
            'metadata': self.metadata,
            'validation_date': '2026-01-18',
            'total_issues': sum(len(issues) for issues in self.issues.values()),
            'issues': {
                k: v[:100]  # Limit to first 100 of each type
                for k, v in self.issues.items()
            },
            'statistics': {
                'asymmetric_synonyms': len(self.issues['asymmetric_synonyms']),
                'asymmetric_antonyms': len(self.issues['asymmetric_antonyms']),
                'inconsistent_hypernyms': len(self.issues['inconsistent_hypernyms']),
                'cycles': len(self.issues['cycles']),
                'missing_in_corpus': len(self.issues['missing_in_corpus']),
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Validate ReVo semantic relations'
    )
    parser.add_argument(
        '--relations',
        type=Path,
        default=Path('data/raw/eo/dictionaries/revo/revo_semantic_relations.json'),
        help='Path to extracted relations JSON'
    )
    parser.add_argument(
        '--corpus-index',
        type=Path,
        default=Path('data/indexes/kuzu_index/kuzu.db'),
        help='Path to Kuzu corpus index (for coverage check)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/eo/dictionaries/revo/revo_validation_report.json'),
        help='Output validation report'
    )

    args = parser.parse_args()

    validator = RevoValidator(args.relations, args.corpus_index)

    try:
        validator.load_relations()
        corpus_roots = validator.load_corpus_roots()
        validator.check_symmetric_relations()
        validator.check_hypernym_consistency()
        validator.check_cycles()
        validator.check_corpus_coverage(corpus_roots)
        validator.print_summary()
        validator.save_report(args.output)
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
