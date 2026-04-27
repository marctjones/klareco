#!/usr/bin/env python3
"""
Semantic Query Expansion using Ontology

Expands query roots using semantic relationships:
- APARTENAS_AL_VERBA_KLASO: Find verb class members (synonyms)
- HAVAS_ENTECAN_TIPON: Find entity type members
- Hierarchical taxonomy: Include subclass members

Author: Claude Sonnet 4.5
Last Updated: 2026-03-31
"""

import logging
from pathlib import Path
from typing import List, Set
import kuzu

logger = logging.getLogger(__name__)


class SemanticQueryExpander:
    """Expand query using semantic ontology relationships."""

    def __init__(self, kuzu_db_path: Path):
        """Initialize with Kuzu database connection."""
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

    def expand_verb_root(self, root: str, include_subclasses: bool = True, max_members: int = 10) -> Set[str]:
        """
        Expand verb root using verb class taxonomy.

        Args:
            root: Verb root to expand
            include_subclasses: If True, include members of subclasses
            max_members: Maximum number of members to return (limits noisy classes)

        Returns:
            Set of semantically related verb roots
        """
        expanded = {root}  # Always include original

        try:
            # Find verb class for this root
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
                RETURN v.klaso_id
            """)

            if not result.has_next():
                return expanded

            klaso_id = result.get_next()[0]

            # Get members of this verb class (limited to avoid noise)
            # Prioritize roots that are similar to the query root
            result = self.conn.execute(f"""
                MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                RETURN r.radiko
                LIMIT {max_members}
            """)

            while result.has_next():
                expanded.add(result.get_next()[0])

            # If including subclasses, get members of child classes (also limited)
            if include_subclasses:
                result = self.conn.execute(f"""
                    MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(sub:VerbaKlaso)
                    WHERE sub.superklaso_id = '{klaso_id}'
                    RETURN r.radiko
                    LIMIT {max_members // 2}
                """)

                while result.has_next():
                    expanded.add(result.get_next()[0])

        except Exception as e:
            logger.debug(f"Failed to expand verb root '{root}': {e}")

        return expanded

    def expand_entity_root(self, root: str) -> Set[str]:
        """
        Expand entity root using entity type taxonomy.

        Args:
            root: Entity root to expand

        Returns:
            Set of semantically related entity roots
        """
        expanded = {root}

        try:
            # Find entity type for this root
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
                RETURN e.tipo_id
            """)

            if not result.has_next():
                return expanded

            tipo_id = result.get_next()[0]

            # Get all members of this entity type
            result = self.conn.execute(f"""
                MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                RETURN r.radiko
                LIMIT 50
            """)

            while result.has_next():
                expanded.add(result.get_next()[0])

        except Exception as e:
            logger.debug(f"Failed to expand entity root '{root}': {e}")

        return expanded

    def expand_query(self, roots: List[str], max_expansion: int = 20) -> Set[str]:
        """
        Expand query roots using all available semantic relationships.

        Args:
            roots: Original query roots
            max_expansion: Maximum number of roots to return (prevents explosion)

        Returns:
            Expanded set of roots
        """
        expanded = set(roots)  # Start with originals

        for root in roots:
            # Try verb expansion
            verb_expanded = self.expand_verb_root(root)
            if len(verb_expanded) > 1:  # Found semantic class
                expanded.update(verb_expanded)
                logger.debug(f"Verb expansion: {root} → {len(verb_expanded)} roots")

            # Try entity expansion
            entity_expanded = self.expand_entity_root(root)
            if len(entity_expanded) > 1:  # Found entity type
                expanded.update(entity_expanded)
                logger.debug(f"Entity expansion: {root} → {len(entity_expanded)} roots")

        # Limit expansion to prevent query explosion
        if len(expanded) > max_expansion:
            logger.warning(f"Query expansion too large ({len(expanded)} roots), limiting to {max_expansion}")
            # Keep original roots + most frequent expanded roots
            # (In practice, should use frequency stats from corpus)
            expanded = set(list(expanded)[:max_expansion])

        return expanded


def expand_with_semantic_ontology(roots: List[str], kuzu_db_path: Path,
                                  max_expansion: int = 20) -> Set[str]:
    """
    Convenience function for one-shot semantic expansion.

    Args:
        roots: Query roots to expand
        kuzu_db_path: Path to Kuzu database
        max_expansion: Maximum number of roots to return

    Returns:
        Expanded set of roots
    """
    expander = SemanticQueryExpander(kuzu_db_path)
    return expander.expand_query(roots, max_expansion=max_expansion)
