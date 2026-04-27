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

    def __init__(self, revo_path: Optional[Path] = None, curated_path: Optional[Path] = None):
        """
        Initialize semantic database.

        Args:
            revo_path: Path to revo_semantic_relations.json
                       (default: data/raw/eo/dictionaries/revo/revo_semantic_relations.json)
            curated_path: Path to curated_synonyms.json
                          (default: data/semantic_relations/curated_synonyms.json)
        """
        if revo_path is None:
            # Default path
            revo_path = Path(__file__).parent.parent.parent / \
                'data' / 'raw' / 'eo' / 'dictionaries' / 'revo' / 'revo_semantic_relations.json'

        if curated_path is None:
            # Default curated path
            curated_path = Path(__file__).parent.parent.parent / \
                'data' / 'semantic_relations' / 'curated_synonyms.json'

        self.revo_path = Path(revo_path)
        self.curated_path = Path(curated_path)

        # Symmetric relation dictionaries: root → set of related roots
        self.synonyms: Dict[str, Set[str]] = {}
        self.antonyms: Dict[str, Set[str]] = {}

        # Asymmetric relations (for Phase 2)
        self.hypernyms: Dict[str, Set[str]] = {}  # root → more general roots
        self.hyponyms: Dict[str, Set[str]] = {}   # root → more specific roots

        # Agent noun mappings (verb root → agent nouns)
        self.agent_nouns: Dict[str, Set[str]] = {}

        # Related words (weaker than synonyms)
        self.related: Dict[str, Set[str]] = {}

        # Load relations
        self._load_relations()
        self._load_curated_relations()

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
        for entry in data.get('relations', {}).get('synonym', []):
            if isinstance(entry, dict):
                root1, root2 = entry['source'], entry['target']
            else:
                root1, root2 = entry
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
        for entry in data.get('relations', {}).get('antonym', []):
            if isinstance(entry, dict):
                root1, root2 = entry['source'], entry['target']
            else:
                root1, root2 = entry
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
        for entry in data.get('relations', {}).get('hypernym', []):
            if isinstance(entry, dict):
                specific, general = entry['source'], entry['target']
            else:
                specific, general = entry
            specific = specific.lower()
            general = general.lower()

            if specific not in self.hypernyms:
                self.hypernyms[specific] = set()

            self.hypernyms[specific].add(general)

        # Load hyponyms (asymmetric: more specific)
        for entry in data.get('relations', {}).get('hyponym', []):
            if isinstance(entry, dict):
                general, specific = entry['source'], entry['target']
            else:
                general, specific = entry
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

    def _load_curated_relations(self):
        """Load manually curated semantic relations."""
        if not self.curated_path.exists():
            logger.debug(f"No curated relations at {self.curated_path}")
            return

        logger.info(f"Loading curated relations from {self.curated_path}")

        with open(self.curated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        curated_count = 0

        # Load verb synonyms
        for root, info in data.get('verb_synonyms', {}).items():
            root = root.lower()

            # Add synonyms
            for syn in info.get('synonyms', []):
                syn = syn.lower()
                if root not in self.synonyms:
                    self.synonyms[root] = set()
                if syn not in self.synonyms:
                    self.synonyms[syn] = set()

                self.synonyms[root].add(syn)
                self.synonyms[syn].add(root)
                curated_count += 1

            # Add related (weaker relationship)
            for rel in info.get('related', []):
                rel = rel.lower()
                if root not in self.related:
                    self.related[root] = set()
                if rel not in self.related:
                    self.related[rel] = set()

                self.related[root].add(rel)
                self.related[rel].add(root)

        # Load noun synonyms
        for root, info in data.get('noun_synonyms', {}).items():
            root = root.lower()

            for syn in info.get('synonyms', []):
                syn = syn.lower()
                if root not in self.synonyms:
                    self.synonyms[root] = set()
                if syn not in self.synonyms:
                    self.synonyms[syn] = set()

                self.synonyms[root].add(syn)
                self.synonyms[syn].add(root)
                curated_count += 1

            for rel in info.get('related', []):
                rel = rel.lower()
                if root not in self.related:
                    self.related[root] = set()
                if rel not in self.related:
                    self.related[rel] = set()

                self.related[root].add(rel)
                self.related[rel].add(root)

        # Load agent nouns
        for verb_root, agents in data.get('agent_nouns', {}).items():
            if verb_root == 'comment':
                continue
            verb_root = verb_root.lower()
            if verb_root not in self.agent_nouns:
                self.agent_nouns[verb_root] = set()
            for agent in agents:
                self.agent_nouns[verb_root].add(agent.lower())

        logger.info(f"  Added {curated_count} curated synonym pairs")
        logger.info(f"  Agent noun mappings: {len(self.agent_nouns)} verbs")

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

    def get_related(self, root: str) -> Set[str]:
        """
        Get related words (weaker than synonyms).

        Args:
            root: Esperanto root word (lowercase)

        Returns:
            Set of related roots (may be empty)
        """
        return self.related.get(root.lower(), set()).copy()

    def get_agent_nouns(self, verb_root: str) -> Set[str]:
        """
        Get agent nouns for a verb root.

        Uses Esperanto morphology rules to SYSTEMATICALLY generate agent nouns:
        - verb + -into = past active participle (one who did X)
        - verb + -anto = present active participle (one who does X)
        - verb + -onto = future active participle (one who will do X)
        - verb + -isto = professional/habitual doer

        Also includes curated mappings for irregular/semantic equivalents
        (e.g., "fond" → "aŭtoro" which isn't morphologically derived).

        Example: "fond" → {"fondinto", "fondanto", "fondonto", "fondisto", "aŭtoro", ...}

        Args:
            verb_root: Verb root (lowercase)

        Returns:
            Set of agent noun forms (both generated and curated)
        """
        verb_root = verb_root.lower()
        result = set()

        # Systematic generation using Esperanto morphology
        # These suffixes create agent nouns from verb roots
        agent_suffixes = [
            'into',   # Past active participle: fondinto = one who founded
            'anto',   # Present active participle: fondanto = one who founds
            'onto',   # Future active participle: fondonto = one who will found
            'isto',   # Professional/habitual: fondisto = founder (as profession)
        ]

        for suffix in agent_suffixes:
            result.add(f"{verb_root}{suffix}")

        # Also add curated mappings for semantic equivalents that aren't
        # morphologically derived (e.g., fond → aŭtoro)
        curated = self.agent_nouns.get(verb_root, set())
        result.update(curated)

        return result

    def get_patient_nouns(self, verb_root: str) -> Set[str]:
        """
        Get patient nouns (passive participles) for a verb root.

        Uses Esperanto morphology rules:
        - verb + -ito = past passive participle (one who was X-ed)
        - verb + -ato = present passive participle (one who is being X-ed)
        - verb + -oto = future passive participle (one who will be X-ed)

        Example: "kre" → {"kreito", "kreato", "kreoto"}
                 (kreito = something that was created)

        Args:
            verb_root: Verb root (lowercase)

        Returns:
            Set of patient noun forms
        """
        verb_root = verb_root.lower()
        result = set()

        patient_suffixes = [
            'ito',   # Past passive: kreito = something created
            'ato',   # Present passive: kreato = something being created
            'oto',   # Future passive: kreoto = something to be created
        ]

        for suffix in patient_suffixes:
            result.add(f"{verb_root}{suffix}")

        return result

    def extract_verb_root_from_participle(self, word: str) -> Optional[str]:
        """
        Extract the verb root from an agent/patient noun (participle).

        Uses Esperanto morphology to reverse-derive the verb root.

        Examples:
            "fondinto" → "fond"
            "verkisto" → "verk"
            "kreato" → "kre"

        Args:
            word: Participle or agent noun (lowercase)

        Returns:
            Verb root, or None if not a recognizable participle
        """
        word = word.lower()

        # Agent noun suffixes (active participles + -isto)
        agent_suffixes = ['into', 'anto', 'onto', 'isto']
        # Patient noun suffixes (passive participles)
        patient_suffixes = ['ito', 'ato', 'oto']

        all_suffixes = agent_suffixes + patient_suffixes

        for suffix in sorted(all_suffixes, key=len, reverse=True):  # Try longest first
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]

        return None

    def get_all_related(self, root: str) -> Set[str]:
        """
        Get all semantically related words (synonyms + related + agent nouns).

        Args:
            root: Esperanto root word

        Returns:
            Combined set of related words
        """
        root = root.lower()
        result = set()

        # Add synonyms
        result.update(self.get_synonyms(root))

        # Add related
        result.update(self.get_related(root))

        # Add agent nouns
        result.update(self.get_agent_nouns(root))

        return result

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            'synonym_roots': len(self.synonyms),
            'antonym_roots': len(self.antonyms),
            'hypernym_roots': len(self.hypernyms),
            'hyponym_roots': len(self.hyponyms),
            'related_roots': len(self.related),
            'agent_noun_verbs': len(self.agent_nouns),
            'total_synonym_pairs': sum(len(syns) for syns in self.synonyms.values()) // 2,
            'total_antonym_pairs': sum(len(ants) for ants in self.antonyms.values()) // 2,
        }
