#!/usr/bin/env python3
"""
Validate Kuzu v2.1 database integrity and data quality.

Two modes:
- Quick mode (~30s): Smoke tests, schema validation, basic counts
- Thorough mode (~5-10m): Full integrity checks, relationship validation, invariants

Usage:
    python scripts/validate/validate_kuzu_v2.1.py --quick
    python scripts/validate/validate_kuzu_v2.1.py --thorough
    python scripts/validate/validate_kuzu_v2.1.py  # Default: quick
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KuzuValidator:
    """Validate Kuzu v2.1 database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db = None
        self.conn = None
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def connect(self):
        """Connect to database."""
        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            sys.exit(1)

        logger.info(f"Connecting to: {self.db_path}")
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

    def query_single(self, query: str) -> any:
        """Execute query and return single value."""
        result = self.conn.execute(query)
        if result.has_next():
            return result.get_next()[0]
        return None

    def query_count(self, node_type: str) -> int:
        """Count nodes of given type."""
        return self.query_single(f"MATCH (n:{node_type}) RETURN count(n)") or 0

    def query_rel_count(self, rel_type: str) -> int:
        """Count relationships of given type."""
        return self.query_single(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)") or 0

    # ==================== QUICK MODE CHECKS (~30s) ====================

    def check_schema(self) -> bool:
        """Check that all expected tables exist."""
        logger.info("\n=== Schema Validation ===")

        expected_nodes = [
            'Fontaro', 'Dokumento', 'Sekcio', 'Paragrafo', 'Frazoteksto',
            'AST', 'Frazo', 'Vortgrupo', 'Vorto', 'Radiko'
        ]

        expected_rels = [
            'EN_FONTARO', 'EN_DOKUMENTO', 'EN_SEKCIO', 'EN_PARAGRAFO',
            'GEPATRA_SEKCIO', 'SEKVA_SEKCIO', 'SEKVA_PARAGRAFO', 'SEKVA_FRAZOTEKSTO',
            'FRAZOTEKSTO_HAVAS_AST', 'AST_HAVAS_FRAZON',
            'HAVAS_SUBJEKTON_VORTGRUPO', 'HAVAS_SUBJEKTON_VORTO',
            'HAVAS_VERBON', 'HAVAS_OBJEKTON_VORTGRUPO', 'HAVAS_OBJEKTON_VORTO',
            'HAVAS_ALIAJN', 'HAVAS_KERNON', 'HAVAS_PRISKRIBON', 'HAVAS_RADIKON'
        ]

        all_exist = True

        # Check nodes
        for node in expected_nodes:
            try:
                count = self.query_count(node)
                if count == 0:
                    self.warnings.append(f"Node type {node} exists but has 0 rows")
                logger.info(f"  ✓ {node}: {count:,} nodes")
            except Exception as e:
                self.errors.append(f"Node type {node} missing: {e}")
                logger.error(f"  ✗ {node}: MISSING")
                all_exist = False

        # Check relationships (sample)
        for rel in expected_rels[:5]:  # Quick mode: check first 5
            try:
                count = self.query_rel_count(rel)
                logger.info(f"  ✓ {rel}: {count:,} edges")
            except Exception as e:
                self.errors.append(f"Relationship {rel} missing: {e}")
                logger.error(f"  ✗ {rel}: MISSING")
                all_exist = False

        return all_exist

    def check_basic_counts(self) -> bool:
        """Check basic node counts are reasonable."""
        logger.info("\n=== Basic Count Validation ===")

        checks = [
            ('Fontaro', 1, 100),  # Should have a few sources
            ('Dokumento', 100, 1000000),  # Many documents
            ('Frazoteksto', 10000, 10000000),  # Many sentences
            ('Vorto', 10000, 100000000),  # Many words
            ('Radiko', 1000, 50000),  # Thousands of roots
        ]

        all_valid = True

        for node_type, min_count, max_count in checks:
            count = self.query_count(node_type)
            if count < min_count:
                self.errors.append(f"{node_type} count too low: {count} < {min_count}")
                logger.error(f"  ✗ {node_type}: {count:,} (expected {min_count:,}+)")
                all_valid = False
            elif count > max_count:
                self.warnings.append(f"{node_type} count very high: {count} > {max_count}")
                logger.warning(f"  ⚠ {node_type}: {count:,} (expected < {max_count:,})")
            else:
                logger.info(f"  ✓ {node_type}: {count:,}")

        return all_valid

    def check_critical_samples(self) -> bool:
        """Sample a few nodes to verify data quality."""
        logger.info("\n=== Critical Sample Checks ===")

        all_valid = True

        # Check: All Frazoteksto have text
        count_no_text = self.query_single("""
            MATCH (f:Frazoteksto)
            WHERE f.teksto IS NULL OR f.teksto = ''
            RETURN count(f)
        """)

        if count_no_text > 0:
            self.errors.append(f"Found {count_no_text} Frazoteksto nodes with no text")
            logger.error(f"  ✗ Frazoteksto with missing text: {count_no_text}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Frazoteksto have text")

        # Check: All Vorto have radiko
        count_no_radiko = self.query_single("""
            MATCH (v:Vorto)
            WHERE v.radiko IS NULL OR v.radiko = ''
            RETURN count(v)
        """)

        if count_no_radiko > 0:
            self.errors.append(f"Found {count_no_radiko} Vorto nodes with no radiko")
            logger.error(f"  ✗ Vorto with missing radiko: {count_no_radiko}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Vorto have radiko")

        # Check: Radiko nodes have reasonable statistics
        result = self.conn.execute("""
            MATCH (r:Radiko)
            WHERE r.nombro_da_vortoj > 0
            RETURN count(r)
        """)
        radiko_with_stats = result.get_next()[0] if result.has_next() else 0
        total_radiko = self.query_count('Radiko')

        if radiko_with_stats < total_radiko * 0.8:
            self.warnings.append(f"Only {radiko_with_stats}/{total_radiko} Radiko have statistics")
            logger.warning(f"  ⚠ Radiko with stats: {radiko_with_stats}/{total_radiko} ({radiko_with_stats/total_radiko*100:.1f}%)")
        else:
            logger.info(f"  ✓ Radiko with stats: {radiko_with_stats}/{total_radiko} ({radiko_with_stats/total_radiko*100:.1f}%)")

        return all_valid

    # ==================== THOROUGH MODE CHECKS (~5-10m) ====================

    def check_hierarchy_integrity(self) -> bool:
        """Validate document → section → paragraph → sentence hierarchy."""
        logger.info("\n=== Hierarchy Integrity ===")

        all_valid = True

        # Check: All Sekcio belong to a Dokumento
        orphan_sekcio = self.query_single("""
            MATCH (s:Sekcio)
            WHERE NOT EXISTS { MATCH (s)-[:EN_DOKUMENTO]->(:Dokumento) }
            RETURN count(s)
        """)

        if orphan_sekcio > 0:
            self.errors.append(f"Found {orphan_sekcio} Sekcio not linked to Dokumento")
            logger.error(f"  ✗ Orphan Sekcio: {orphan_sekcio}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Sekcio linked to Dokumento")

        # Check: All Paragrafo belong to a Sekcio
        orphan_paragrafo = self.query_single("""
            MATCH (p:Paragrafo)
            WHERE NOT EXISTS { MATCH (p)-[:EN_SEKCIO]->(:Sekcio) }
            RETURN count(p)
        """)

        if orphan_paragrafo > 0:
            self.errors.append(f"Found {orphan_paragrafo} Paragrafo not linked to Sekcio")
            logger.error(f"  ✗ Orphan Paragrafo: {orphan_paragrafo}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Paragrafo linked to Sekcio")

        # Check: All Frazoteksto belong to a Paragrafo
        orphan_frazoteksto = self.query_single("""
            MATCH (f:Frazoteksto)
            WHERE NOT EXISTS { MATCH (f)-[:EN_PARAGRAFO]->(:Paragrafo) }
            RETURN count(f)
        """)

        if orphan_frazoteksto > 0:
            self.errors.append(f"Found {orphan_frazoteksto} Frazoteksto not linked to Paragrafo")
            logger.error(f"  ✗ Orphan Frazoteksto: {orphan_frazoteksto}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Frazoteksto linked to Paragrafo")

        return all_valid

    def check_ast_completeness(self) -> bool:
        """Check AST structure completeness."""
        logger.info("\n=== AST Completeness ===")

        all_valid = True

        # Check: All Frazoteksto have AST
        count_no_ast = self.query_single("""
            MATCH (f:Frazoteksto)
            WHERE NOT EXISTS { MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(:AST) }
            RETURN count(f)
        """)

        if count_no_ast > 0:
            self.errors.append(f"Found {count_no_ast} Frazoteksto without AST")
            logger.error(f"  ✗ Frazoteksto without AST: {count_no_ast}")
            all_valid = False
        else:
            logger.info(f"  ✓ All Frazoteksto have AST")

        # Check: All AST have Frazo
        count_no_frazo = self.query_single("""
            MATCH (a:AST)
            WHERE NOT EXISTS { MATCH (a)-[:AST_HAVAS_FRAZON]->(:Frazo) }
            RETURN count(a)
        """)

        if count_no_frazo > 0:
            self.errors.append(f"Found {count_no_frazo} AST without Frazo")
            logger.error(f"  ✗ AST without Frazo: {count_no_frazo}")
            all_valid = False
        else:
            logger.info(f"  ✓ All AST have Frazo")

        # Check: All Frazo have Verbo
        count_no_verbo = self.query_single("""
            MATCH (f:Frazo)
            WHERE NOT EXISTS { MATCH (f)-[:HAVAS_VERBON]->(:Vorto) }
            RETURN count(f)
        """)

        total_frazo = self.query_count('Frazo')
        if count_no_verbo > total_frazo * 0.1:  # Allow 10% without verb (fragments)
            self.warnings.append(f"{count_no_verbo}/{total_frazo} Frazo without Verbo")
            logger.warning(f"  ⚠ Frazo without Verbo: {count_no_verbo}/{total_frazo} ({count_no_verbo/total_frazo*100:.1f}%)")
        else:
            logger.info(f"  ✓ Frazo without Verbo: {count_no_verbo}/{total_frazo} ({count_no_verbo/total_frazo*100:.1f}%)")

        return all_valid

    def check_radiko_relationships(self) -> bool:
        """Check Vorto → Radiko relationships."""
        logger.info("\n=== Radiko Relationships ===")

        all_valid = True

        # Check: All Vorto have HAVAS_RADIKON edge
        count_no_radikon = self.query_single("""
            MATCH (v:Vorto)
            WHERE NOT EXISTS { MATCH (v)-[:HAVAS_RADIKON]->(:Radiko) }
            RETURN count(v)
        """)

        total_vorto = self.query_count('Vorto')
        if count_no_radikon > total_vorto * 0.05:  # Allow 5% missing (unknown roots)
            self.errors.append(f"{count_no_radikon}/{total_vorto} Vorto without HAVAS_RADIKON")
            logger.error(f"  ✗ Vorto without HAVAS_RADIKON: {count_no_radikon}/{total_vorto} ({count_no_radikon/total_vorto*100:.1f}%)")
            all_valid = False
        else:
            logger.info(f"  ✓ Vorto without HAVAS_RADIKON: {count_no_radikon}/{total_vorto} ({count_no_radikon/total_vorto*100:.1f}%)")

        # Check: Radiko statistics consistency
        result = self.conn.execute("""
            MATCH (r:Radiko)
            WHERE r.nombro_da_vortoj > 0 AND r.nombro_da_frazoj > r.nombro_da_vortoj
            RETURN count(r)
        """)
        inconsistent_stats = result.get_next()[0] if result.has_next() else 0

        if inconsistent_stats > 0:
            self.warnings.append(f"{inconsistent_stats} Radiko have inconsistent statistics")
            logger.warning(f"  ⚠ Radiko with inconsistent stats: {inconsistent_stats}")
        else:
            logger.info(f"  ✓ All Radiko statistics consistent")

        return all_valid

    def run_quick(self) -> bool:
        """Run quick validation (~30s)."""
        logger.info("\n" + "="*70)
        logger.info("QUICK VALIDATION MODE (~30s)")
        logger.info("="*70)

        start = datetime.now()

        valid = True
        valid &= self.check_schema()
        valid &= self.check_basic_counts()
        valid &= self.check_critical_samples()

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"\nQuick validation completed in {elapsed:.1f}s")

        return valid

    def run_thorough(self) -> bool:
        """Run thorough validation (~5-10m)."""
        logger.info("\n" + "="*70)
        logger.info("THOROUGH VALIDATION MODE (~5-10 minutes)")
        logger.info("="*70)

        start = datetime.now()

        # Run all quick checks first
        valid = self.run_quick()

        # Then run thorough checks
        valid &= self.check_hierarchy_integrity()
        valid &= self.check_ast_completeness()
        valid &= self.check_radiko_relationships()

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"\nThorough validation completed in {elapsed:.1f}s")

        return valid

    def print_summary(self, valid: bool):
        """Print validation summary."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)

        if len(self.errors) == 0 and len(self.warnings) == 0:
            logger.info("\n✓ ALL CHECKS PASSED")
        else:
            if len(self.errors) > 0:
                logger.error(f"\n✗ ERRORS: {len(self.errors)}")
                for error in self.errors:
                    logger.error(f"  - {error}")

            if len(self.warnings) > 0:
                logger.warning(f"\n⚠ WARNINGS: {len(self.warnings)}")
                for warning in self.warnings:
                    logger.warning(f"  - {warning}")

        logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Validate Kuzu v2.1 database')
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'),
                        help='Path to Kuzu database')
    parser.add_argument('--quick', action='store_true', help='Quick validation (~30s)')
    parser.add_argument('--thorough', action='store_true', help='Thorough validation (~5-10m)')

    args = parser.parse_args()

    # Default to quick if neither specified
    if not args.quick and not args.thorough:
        args.quick = True

    validator = KuzuValidator(args.db)

    try:
        validator.connect()

        if args.thorough:
            valid = validator.run_thorough()
        else:
            valid = validator.run_quick()

        validator.print_summary(valid)

        return 0 if valid else 1

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
