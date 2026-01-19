#!/usr/bin/env python3
"""
Extract semantic relations from ReVo SQLite database.

ReVo (Reta Vortaro) is the comprehensive Esperanto dictionary.
This script extracts semantic relations from the SQLite database:
- Synonyms (sin)
- Antonyms (ant)
- Hypernyms (super)
- Hyponyms (sub)
- Part-of relations (prt)

Output: JSON file with cleaned semantic relations for Kuzu loading.

Usage:
    python scripts/extract_revo_semantic_relations.py
    python scripts/extract_revo_semantic_relations.py --fresh
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import parser to validate roots
try:
    from klareco.parser import parse_word
except ImportError:
    print("Error: Cannot import klareco.parser")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# X-notation to Unicode mapping
X_TO_UNICODE = {
    'cx': 'ĉ', 'Cx': 'Ĉ', 'CX': 'Ĉ',
    'gx': 'ĝ', 'Gx': 'Ĝ', 'GX': 'Ĝ',
    'hx': 'ĥ', 'Hx': 'Ĥ', 'HX': 'Ĥ',
    'jx': 'ĵ', 'Jx': 'Ĵ', 'JX': 'Ĵ',
    'sx': 'ŝ', 'Sx': 'Ŝ', 'SX': 'Ŝ',
    'ux': 'ŭ', 'Ux': 'Ŭ', 'UX': 'Ŭ',
}

# Valid Esperanto characters
VALID_EO_CHARS = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭ')

# Semantic relation types to extract
SEMANTIC_RELATION_TYPES = {
    'sin': 'synonym',
    'ant': 'antonym',
    'super': 'hypernym',
    'sub': 'hyponym',
    'prt': 'part_of',
}


def normalize_x_notation(text: str) -> str:
    """Convert x-notation to proper Unicode."""
    result = text
    for x_form, unicode_form in X_TO_UNICODE.items():
        result = result.replace(x_form, unicode_form)
    return result


def extract_root_from_marker(mrk: str) -> Optional[str]:
    """
    Extract root from ReVo marker.

    Examples:
        'bronz.0o' -> 'bronz'
        'akcipitro.0' -> 'akcipitr'
        'efemer1.0oj' -> 'efemer'

    Returns None if invalid.
    """
    if not mrk:
        return None

    # Split on first dot
    parts = mrk.split('.', 1)
    if not parts:
        return None

    root = parts[0]

    # Normalize x-notation
    root = normalize_x_notation(root)

    # Strip homograph numbers (efemer1 -> efemer)
    root = re.sub(r'\d+$', '', root)

    # Must be at least 2 chars
    if len(root) < 2:
        return None

    # Must contain only valid Esperanto characters
    root_lower = root.lower()
    if not all(c in VALID_EO_CHARS for c in root_lower):
        return None

    return root_lower


def validate_root_with_parser(root: str) -> bool:
    """
    Validate that root can be parsed by Klareco parser.

    This ensures we only include roots that are recognized by our system.
    """
    if not root or len(root) < 2:
        return False

    # Try parsing as a noun (most common)
    try:
        # Try with -o ending
        ast = parse_word(root + 'o')
        if ast and ast.get('radiko') == root:
            return True
    except Exception:
        pass

    # Try as verb
    try:
        ast = parse_word(root + 'i')
        if ast and ast.get('radiko') == root:
            return True
    except Exception:
        pass

    # Try as adjective
    try:
        ast = parse_word(root + 'a')
        if ast and ast.get('radiko') == root:
            return True
    except Exception:
        pass

    return False


class RevoExtractor:
    """Extract semantic relations from ReVo SQLite database."""

    def __init__(self, db_path: Path, output_path: Path):
        self.db_path = Path(db_path)
        self.output_path = Path(output_path)

        self.conn: Optional[sqlite3.Connection] = None

        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_references': 0,
            'valid_roots': 0,
            'invalid_roots': 0,
            'synonym': 0,
            'antonym': 0,
            'hypernym': 0,
            'hyponym': 0,
            'part_of': 0,
            'skipped_self_reference': 0,
            'skipped_invalid_root': 0,
        }

        # Cache for validated roots
        self.valid_roots: Set[str] = set()
        self.invalid_roots: Set[str] = set()

    def connect(self):
        """Connect to ReVo SQLite database."""
        logger.info(f"Connecting to ReVo database: {self.db_path}")

        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            sys.exit(1)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Get counts
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodo")
        self.stats['total_nodes'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM referenco")
        self.stats['total_references'] = cursor.fetchone()[0]

        logger.info(f"  Nodes: {self.stats['total_nodes']:,}")
        logger.info(f"  References: {self.stats['total_references']:,}")

    def is_valid_root(self, root: str) -> bool:
        """Check if root is valid (with caching)."""
        if root in self.valid_roots:
            return True
        if root in self.invalid_roots:
            return False

        # Validate
        is_valid = validate_root_with_parser(root)

        if is_valid:
            self.valid_roots.add(root)
            self.stats['valid_roots'] += 1
        else:
            self.invalid_roots.add(root)
            self.stats['invalid_roots'] += 1

        return is_valid

    def extract_relations(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Extract semantic relations from database.

        Returns:
            Dict mapping relation type to list of (root1, root2, weight) tuples
        """
        logger.info("")
        logger.info("Extracting semantic relations...")

        relations: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

        cursor = self.conn.cursor()

        # Query all semantic references
        query = """
            SELECT r.mrk, r.cel, r.tip, n1.kap as source_kap, n2.kap as target_kap
            FROM referenco r
            LEFT JOIN nodo n1 ON r.mrk = n1.mrk
            LEFT JOIN nodo n2 ON r.cel = n2.mrk
            WHERE r.tip IN ('sin', 'ant', 'super', 'sub', 'prt')
        """

        cursor.execute(query)

        processed = 0
        for row in cursor:
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"  Processed {processed:,} references...")

            rel_type = row['tip']
            rel_name = SEMANTIC_RELATION_TYPES.get(rel_type)
            if not rel_name:
                continue

            # Extract roots from markers
            source_root = extract_root_from_marker(row['mrk'])
            target_root = extract_root_from_marker(row['cel'])

            if not source_root or not target_root:
                self.stats['skipped_invalid_root'] += 1
                continue

            # Skip self-references
            if source_root == target_root:
                self.stats['skipped_self_reference'] += 1
                continue

            # Validate both roots with parser
            if not self.is_valid_root(source_root):
                self.stats['skipped_invalid_root'] += 1
                continue

            if not self.is_valid_root(target_root):
                self.stats['skipped_invalid_root'] += 1
                continue

            # Add relation (weight = 2.0 for ReVo - higher than ConceptNet)
            relations[rel_name].append((source_root, target_root, 2.0))
            self.stats[rel_name] += 1

        logger.info(f"  Processed {processed:,} references")
        logger.info("")

        return dict(relations)

    def save_relations(self, relations: Dict[str, List[Tuple[str, str, float]]]):
        """Save relations to JSON file."""
        logger.info(f"Saving relations to: {self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        output = {
            'metadata': {
                'source': 'ReVo (Reta Vortaro)',
                'version': 'revo.db (2017-12-15)',
                'extraction_date': '2026-01-18',
                'total_relations': sum(len(rels) for rels in relations.values()),
                'statistics': self.stats,
            },
            'relations': {
                rel_type: [
                    {'source': src, 'target': tgt, 'weight': wt}
                    for src, tgt, wt in rel_list
                ]
                for rel_type, rel_list in relations.items()
            }
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved {output['metadata']['total_relations']:,} relations")

    def print_summary(self, relations: Dict[str, List[Tuple[str, str, float]]]):
        """Print extraction summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Relations extracted:")
        for rel_type, rel_list in sorted(relations.items()):
            logger.info(f"  {rel_type}: {len(rel_list):,}")
        logger.info("")
        logger.info(f"Total relations: {sum(len(r) for r in relations.values()):,}")
        logger.info("")
        logger.info("Root validation:")
        logger.info(f"  Valid roots: {self.stats['valid_roots']:,}")
        logger.info(f"  Invalid roots: {self.stats['invalid_roots']:,}")
        logger.info("")
        logger.info("Skipped:")
        logger.info(f"  Self-references: {self.stats['skipped_self_reference']:,}")
        logger.info(f"  Invalid roots: {self.stats['skipped_invalid_root']:,}")
        logger.info("")
        logger.info(f"Output: {self.output_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract semantic relations from ReVo dictionary'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/raw/eo/dictionaries/revo/revo.db'),
        help='Path to ReVo SQLite database'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/eo/dictionaries/revo/revo_semantic_relations.json'),
        help='Output JSON file'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force re-extraction even if output exists'
    )

    args = parser.parse_args()

    if args.output.exists() and not args.fresh:
        logger.info(f"Output already exists: {args.output}")
        logger.info("Use --fresh to re-extract")
        return 0

    if not args.db.exists():
        logger.error(f"ReVo database not found: {args.db}")
        return 1

    extractor = RevoExtractor(args.db, args.output)

    try:
        extractor.connect()
        relations = extractor.extract_relations()
        extractor.save_relations(relations)
        extractor.print_summary(relations)
        return 0
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1
    finally:
        extractor.close()


if __name__ == '__main__':
    sys.exit(main())
