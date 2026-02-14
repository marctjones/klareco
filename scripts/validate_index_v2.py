#!/usr/bin/env python3
"""
Validate v2.0 Kuzu index structure and integrity.

Checks:
- All required node types exist
- Relationships are properly linked
- AST structure is complete
- Root index is built
- Data can be queried
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexValidator:
    """Validate v2.0 Kuzu index."""

    def __init__(self, db_path: Path):
        """Initialize validator."""
        self.db_path = db_path
        self.db = None
        self.conn = None
        self.errors = []
        self.warnings = []

    def connect(self):
        """Connect to database."""
        logger.info(f"Opening database at {self.db_path}")
        self.db = kuzu.Database(str(self.db_path), read_only=True)
        self.conn = kuzu.Connection(self.db)

    def check_node_types(self) -> Dict[str, int]:
        """Check all required node types exist."""
        logger.info("Checking node types...")

        node_types = [
            'SourceCollection', 'Document', 'Sentence',
            'AST', 'Frazo', 'Vortgrupo', 'Vorto', 'Root'
        ]

        counts = {}
        for node_type in node_types:
            result = self.conn.execute(f"MATCH (n:{node_type}) RETURN count(n)")
            count = result.get_next()[0]
            counts[node_type] = count

            if count == 0:
                self.errors.append(f"No {node_type} nodes found")
            else:
                logger.info(f"  {node_type}: {count} nodes")

        return counts

    def check_hierarchy_relationships(self):
        """Check document hierarchy relationships."""
        logger.info("Checking hierarchy relationships...")

        # Document -> Collection
        result = self.conn.execute("""
            MATCH (d:Document)-[:IN_COLLECTION]->(c:SourceCollection)
            RETURN count(*)
        """)
        count = result.get_next()[0]
        logger.info(f"  Document->Collection links: {count}")

        if count == 0:
            self.errors.append("No Document->Collection links found")

    def check_ast_relationships(self):
        """Check AST structure relationships."""
        logger.info("Checking AST relationships...")

        checks = [
            ("Sentence->AST", "MATCH (s:Sentence)-[:SENTENCE_HAS_AST]->(ast:AST) RETURN count(*)"),
            ("AST->Frazo", "MATCH (ast:AST)-[:AST_HAS_FRAZO]->(f:Frazo) RETURN count(*)"),
            ("Frazo->Subjekto", "MATCH (f:Frazo)-[:HAS_SUBJEKTO_VORTGRUPO|HAS_SUBJEKTO_VORTO]->() RETURN count(*)"),
            ("Frazo->Verbo", "MATCH (f:Frazo)-[:HAS_VERBO]->(v:Vorto) RETURN count(*)"),
            ("Vortgrupo->Kerno", "MATCH (vg:Vortgrupo)-[:HAS_KERNO]->(v:Vorto) RETURN count(*)"),
            ("Vorto->Root", "MATCH (v:Vorto)-[:HAS_ROOT]->(r:Root) RETURN count(*)"),
        ]

        for name, query in checks:
            result = self.conn.execute(query)
            count = result.get_next()[0]
            logger.info(f"  {name}: {count}")

            if count == 0 and "Subjekto" not in name:  # Subjekto can be missing
                self.warnings.append(f"No {name} links found")

    def check_ast_completeness(self):
        """Check that ASTs have complete structure."""
        logger.info("Checking AST completeness...")

        # Check each AST has a Frazo
        result = self.conn.execute("""
            MATCH (ast:AST)
            WHERE NOT (ast)-[:AST_HAS_FRAZO]->(:Frazo)
            RETURN count(*)
        """)
        count = result.get_next()[0]
        if count > 0:
            self.errors.append(f"{count} AST nodes without Frazo")
        else:
            logger.info("  All ASTs have Frazo nodes")

        # Check each Vorto is linked to a Root (should be)
        result = self.conn.execute("""
            MATCH (v:Vorto)
            WHERE NOT (v)-[:HAS_ROOT]->(:Root)
            RETURN count(*)
        """)
        count = result.get_next()[0]
        if count > 0:
            self.warnings.append(f"{count} Vorto nodes not linked to Root")
        else:
            logger.info("  All Vorto nodes linked to Roots")

    def check_root_index(self):
        """Check root index integrity."""
        logger.info("Checking root index...")

        # Check roots have frequency data
        result = self.conn.execute("""
            MATCH (r:Root)
            WHERE r.total_freq IS NULL OR r.doc_freq IS NULL
            RETURN count(*)
        """)
        count = result.get_next()[0]
        if count > 0:
            self.errors.append(f"{count} Root nodes missing frequency data")
        else:
            logger.info("  All roots have frequency data")

        # Check top roots by frequency
        result = self.conn.execute("""
            MATCH (r:Root)
            WHERE r.total_freq > 0
            RETURN r.root, r.total_freq, r.doc_freq
            ORDER BY r.total_freq DESC
            LIMIT 10
        """)

        logger.info("  Top 10 roots by frequency:")
        while result.has_next():
            root, total_freq, doc_freq = result.get_next()
            logger.info(f"    {root}: total={total_freq}, docs={doc_freq}")

    def test_queries(self):
        """Test sample queries."""
        logger.info("Testing sample queries...")

        # Query 1: Find all sentences containing a specific root
        result = self.conn.execute("""
            MATCH (s:Sentence)-[:SENTENCE_HAS_AST]->(:AST)-[:AST_HAS_FRAZO]->()-[*]->(v:Vorto)-[:HAS_ROOT]->(r:Root {root: 'vid'})
            RETURN s.text
            LIMIT 5
        """)

        count = 0
        logger.info("  Sentences with root 'vid':")
        while result.has_next():
            text = result.get_next()[0]
            logger.info(f"    {text[:80]}...")
            count += 1

        if count == 0:
            self.warnings.append("No sentences found with root 'vid'")

        # Query 2: Find all words with a specific vortspeco
        result = self.conn.execute("""
            MATCH (v:Vorto {vortspeco: 'verbo'})
            RETURN v.plena_vorto, v.radiko
            LIMIT 5
        """)

        count = 0
        logger.info("  Sample verbs:")
        while result.has_next():
            plena, radiko = result.get_next()
            logger.info(f"    {plena} (root: {radiko})")
            count += 1

        if count == 0:
            self.warnings.append("No verbs found")

        # Query 3: Find ASTs with specific sentence type
        result = self.conn.execute("""
            MATCH (ast:AST {fraztipo: 'demando'})
            RETURN count(*)
        """)
        count = result.get_next()[0]
        logger.info(f"  Questions (fraztipo='demando'): {count}")

    def validate(self):
        """Run all validation checks."""
        logger.info("Starting validation...")

        self.check_node_types()
        self.check_hierarchy_relationships()
        self.check_ast_relationships()
        self.check_ast_completeness()
        self.check_root_index()
        self.test_queries()

        logger.info("\n=== Validation Summary ===")

        if self.errors:
            logger.error(f"ERRORS: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if not self.errors and not self.warnings:
            logger.info("✓ All validation checks passed!")
            return True
        elif not self.errors:
            logger.info("✓ Validation passed with warnings")
            return True
        else:
            logger.error("✗ Validation failed")
            return False


def main():
    parser = argparse.ArgumentParser(description='Validate v2.0 Kuzu index')
    parser.add_argument('--db', type=Path, required=True, help='Path to Kuzu database')

    args = parser.parse_args()

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    validator = IndexValidator(args.db)
    validator.connect()

    success = validator.validate()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
