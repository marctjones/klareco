"""
Semantic Relation Database for Esperanto.

Loads and provides access to semantic relations (synonyms, antonyms, etc.)
from the ReVo thesaurus.

Phase 1: Basic synonym/antonym lookup
Phase 2: Hypernym/hyponym traversal
Phase 3: Corpus-based expansion
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SemanticRelationDB:
    """
    Database of Esperanto semantic relations.

    Provides root-level synonym/antonym lookups for AST-aware retrieval.
    """

    def __init__(self, revo_path: Optional[Path] = None):
        """
        Initialize semantic database.

        Args:
            revo_path: Path to revo_semantic_relations.json
                       (default: data/raw/eo/dictionaries/revo/revo_semantic_relations.json)
        """
        if revo_path is None:
            # Default path
            revo_path = Path(__file__).parent.parent.parent / \
                'data' / 'raw' / 'eo' / 'dictionaries' / 'revo' / 'revo_semantic_relations.json'

        self.revo_path = Path(revo_path)

        # Symmetric relation dictionaries: root → set of related roots
        self.synonyms: Dict[str, Set[str]] = {}
        self.antonyms: Dict[str, Set[str]] = {}

        # Asymmetric relations (for Phase 2)
        self.hypernyms: Dict[str, Set[str]] = {}  # root → more general roots
        self.hyponyms: Dict[str, Set[str]] = {}   # root → more specific roots

        # Load relations
        self._load_relations()

    def _load_relations(self):
        """Load semantic relations from ReVo."""
        if not self.revo_path.exists():
            logger.warning(
                f"ReVo semantic relations not found at {self.revo_path}. "
                "Semantic relation database will be empty."
            )
            return

        logger.info(f"Loading semantic relations from {self.revo_path}")

        with open(self.revo_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        stats = metadata.get('statistics', {})

        # Load synonyms (symmetric)
        for root1, root2 in data.get('relations', {}).get('synonym', []):
            root1 = root1.lower()
            root2 = root2.lower()

            # Add bidirectional mapping
            if root1 not in self.synonyms:
                self.synonyms[root1] = set()
            if root2 not in self.synonyms:
                self.synonyms[root2] = set()

            self.synonyms[root1].add(root2)
            self.synonyms[root2].add(root1)

        # Load antonyms (symmetric)
        for root1, root2 in data.get('relations', {}).get('antonym', []):
            root1 = root1.lower()
            root2 = root2.lower()

            # Add bidirectional mapping
            if root1 not in self.antonyms:
                self.antonyms[root1] = set()
            if root2 not in self.antonyms:
                self.antonyms[root2] = set()

            self.antonyms[root1].add(root2)
            self.antonyms[root2].add(root1)

        # Load hypernyms (asymmetric: more general)
        for specific, general in data.get('relations', {}).get('hypernym', []):
            specific = specific.lower()
            general = general.lower()

            if specific not in self.hypernyms:
                self.hypernyms[specific] = set()

            self.hypernyms[specific].add(general)

        # Load hyponyms (asymmetric: more specific)
        for general, specific in data.get('relations', {}).get('hyponym', []):
            general = general.lower()
            specific = specific.lower()

            if general not in self.hyponyms:
                self.hyponyms[general] = set()

            self.hyponyms[general].add(specific)

        logger.info(
            f"Loaded semantic relations: "
            f"{len(self.synonyms)} synonym roots, "
            f"{len(self.antonyms)} antonym roots, "
            f"{len(self.hypernyms)} hypernym roots, "
            f"{len(self.hyponyms)} hyponym roots"
        )
        logger.info(f"  ReVo statistics: {stats}")

    def get_synonyms(self, root: str) -> Set[str]:
        """
        Get all synonyms for a root.

        Args:
            root: Esperanto root word (lowercase)

        Returns:
            Set of synonym roots (may be empty)
        """
        return self.synonyms.get(root.lower(), set()).copy()

    def get_antonyms(self, root: str) -> Set[str]:
        """
        Get all antonyms for a root.

        Args:
            root: Esperanto root word (lowercase)

        Returns:
            Set of antonym roots (may be empty)
        """
        return self.antonyms.get(root.lower(), set()).copy()

    def get_hypernyms(self, root: str) -> Set[str]:
        """
        Get all hypernyms (more general terms) for a root.

        Example: "hundo" → "besto" (dog → animal)

        Args:
            root: Esperanto root word (lowercase)

        Returns:
            Set of hypernym roots (may be empty)
        """
        return self.hypernyms.get(root.lower(), set()).copy()

    def get_hyponyms(self, root: str) -> Set[str]:
        """
        Get all hyponyms (more specific terms) for a root.

        Example: "besto" → "hundo", "kato", etc. (animal → dog, cat, etc.)

        Args:
            root: Esperanto root word (lowercase)

        Returns:
            Set of hyponym roots (may be empty)
        """
        return self.hyponyms.get(root.lower(), set()).copy()

    def are_synonyms(self, root1: str, root2: str) -> bool:
        """Check if two roots are synonyms."""
        root1 = root1.lower()
        root2 = root2.lower()

        return root2 in self.synonyms.get(root1, set())

    def are_antonyms(self, root1: str, root2: str) -> bool:
        """Check if two roots are antonyms."""
        root1 = root1.lower()
        root2 = root2.lower()

        return root2 in self.antonyms.get(root1, set())

    def expand_with_synonyms(self, roots: Set[str]) -> Set[str]:
        """
        Expand a set of roots with their synonyms.

        Args:
            roots: Set of root words

        Returns:
            Expanded set including original roots and all synonyms
        """
        expanded = set(roots)

        for root in roots:
            expanded.update(self.get_synonyms(root))

        return expanded

    def get_semantic_similarity(self, root1: str, root2: str) -> float:
        """
        Compute semantic similarity between two roots.

        Uses ReVo relations to estimate similarity:
        - Synonyms: 1.0
        - Antonyms: -1.0
        - Hypernym/hyponym: 0.5
        - No relation: 0.0

        Args:
            root1: First root
            root2: Second root

        Returns:
            Similarity score in [-1.0, 1.0]
        """
        root1 = root1.lower()
        root2 = root2.lower()

        # Exact match
        if root1 == root2:
            return 1.0

        # Synonyms
        if self.are_synonyms(root1, root2):
            return 1.0

        # Antonyms
        if self.are_antonyms(root1, root2):
            return -1.0

        # Hypernym/hyponym (related but not identical)
        if root2 in self.get_hypernyms(root1) or root2 in self.get_hyponyms(root1):
            return 0.5

        # No known relation
        return 0.0

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            'synonym_roots': len(self.synonyms),
            'antonym_roots': len(self.antonyms),
            'hypernym_roots': len(self.hypernyms),
            'hyponym_roots': len(self.hyponyms),
            'total_synonym_pairs': sum(len(syns) for syns in self.synonyms.values()) // 2,
            'total_antonym_pairs': sum(len(ants) for ants in self.antonyms.values()) // 2,
        }
