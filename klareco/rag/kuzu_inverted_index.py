"""
Kuzu-Backed Inverted Index for AST-First Retrieval.

This uses Kuzu graph database for root-based inverted index with
integrated semantic relations. Features:

- O(1) root lookups (hash-based)
- Native graph traversal for synonyms/hypernyms
- Sentence adjacency for context retrieval
- Role-aware pattern matching via Cypher queries

Usage:
    index = KuzuInvertedIndex("data/indexes/kuzu_index")

    # Basic search (same interface as SQLite version)
    results, stats = index.search(query_ast, max_results=10)

    # Graph-enhanced features
    synonyms = index.get_synonyms_transitive("fond", max_hops=2)
    context = index.get_sentence_context(sent_id, window=2)
    patterns = index.search_by_role(verb="fond", obj="esperant")
"""

import json
import logging
import mmap
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

try:
    import kuzu
except ImportError:
    kuzu = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

logger = logging.getLogger(__name__)


class FallbackMode(Enum):
    """Controls which fallback mechanisms are enabled."""
    NONE = auto()       # Pure deterministic - no fallbacks
    EMBEDDING = auto()  # Use embeddings for OOV root similarity
    RERANK = auto()     # Use embeddings to rerank after deterministic retrieval
    FULL = auto()       # All fallbacks enabled


@dataclass
class SemanticConcept:
    """
    A semantic concept represented by multiple equivalent roots.

    All roots in a concept express the same meaning and should score equally.
    """
    original_root: str
    equivalent_roots: Set[str] = field(default_factory=set)
    weight: float = 1.0

    def __post_init__(self):
        self.equivalent_roots.add(self.original_root)


@dataclass
class RetrievalStats:
    """Statistics from a retrieval operation for debugging/testing."""
    query_roots: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    expanded_roots: List[str] = field(default_factory=list)
    semantic_db_synonyms: List[str] = field(default_factory=list)
    embedding_synonyms: List[str] = field(default_factory=list)
    roots_found_in_index: List[str] = field(default_factory=list)
    roots_not_found: List[str] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    candidate_count_before_scoring: int = 0
    candidate_count_after_scoring: int = 0
    scoring_method: str = "bm25"
    graph_expansions: int = 0  # New: track graph-based expansions
    # Predicate-first retrieval stats
    predicate_query: Optional[str] = None  # The predicate pattern searched
    predicate_matches: int = 0  # Number of predicate-matched docs
    predicate_boost_applied: bool = False

    def to_dict(self) -> Dict:
        return {
            "query_roots": self.query_roots,
            "concepts": self.concepts,
            "expanded_roots": self.expanded_roots,
            "semantic_db_synonyms": self.semantic_db_synonyms,
            "embedding_synonyms": self.embedding_synonyms,
            "roots_found": self.roots_found_in_index,
            "roots_not_found": self.roots_not_found,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "candidates_before_scoring": self.candidate_count_before_scoring,
            "candidates_after_scoring": self.candidate_count_after_scoring,
            "scoring_method": self.scoring_method,
            "graph_expansions": self.graph_expansions,
            "predicate_query": self.predicate_query,
            "predicate_matches": self.predicate_matches,
            "predicate_boost_applied": self.predicate_boost_applied,
        }


@dataclass
class RootOccurrence:
    """A single occurrence of a root in a document."""
    doc_id: int
    role: str  # subjekto, verbo, objekto, aliaj, predikato
    grammar: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result with explanation."""
    doc_id: int
    score: float
    text: str
    matched_roots: List[str]
    grammar_matches: Dict[str, bool]
    source: Dict[str, Any] = field(default_factory=dict)


class KuzuInvertedIndex:
    """
    Kuzu-backed inverted index for AST-first retrieval.

    Provides root-based inverted index plus graph features.
    """

    def __init__(
        self,
        index_path: Path,
        fallback_mode: FallbackMode = FallbackMode.NONE,
    ):
        """
        Initialize Kuzu inverted index.

        Args:
            index_path: Path to Kuzu index directory
            fallback_mode: Which fallbacks to enable (default: NONE for pure deterministic)
        """
        if kuzu is None:
            raise ImportError("kuzu package not installed. Run: pip install kuzu")

        self.index_path = Path(index_path)
        self.fallback_mode = fallback_mode

        # Kuzu database
        self.db_path = self.index_path / "kuzu.db"
        self._db: Optional[kuzu.Database] = None
        self._conn: Optional[kuzu.Connection] = None

        # Document storage (memory-mapped for O(1) access)
        self.doc_offsets: Optional[np.ndarray] = None
        self._doc_file: Optional[Any] = None
        self._doc_mmap: Optional[mmap.mmap] = None

        # Cached stats
        self._all_roots: Optional[Set[str]] = None
        self._doc_freq_cache: Dict[str, int] = {}
        self.total_docs = 0
        self.total_roots = 0

        # BM25 parameters
        # k1 reduced from 1.2 to 0.7 to reduce term frequency saturation
        # (helps concise factual sentences compete with verbose documents)
        self.bm25_k1 = 0.7
        # b increased from 0.75 to 0.9 for more aggressive length normalization
        # (penalizes long documents more heavily)
        self.bm25_b = 0.9
        # Average document length (will be computed on load)
        self.avg_doc_length = 20.0  # Default estimate (words per sentence)

        # AI models for semantic constraints (#588, #589)
        self.root_embeddings: Optional[torch.Tensor] = None
        self.root_to_idx: Optional[Dict[str, int]] = None
        self.embedding_dim: int = 64

        # Load if exists
        if self.index_path.exists():
            self._load_index()
            self._load_semantic_models()

    def _load_index(self):
        """Load Kuzu index from disk with optimized configuration."""
        if not self.db_path.exists():
            logger.warning(f"Kuzu database not found at {self.db_path}")
            return

        logger.info(f"Loading Kuzu index from {self.index_path}")

        # Open Kuzu database with default configuration
        # Testing showed that custom configuration (buffer_pool_size, threads, read_only)
        # did not improve performance - Kuzu's defaults are already well-tuned
        self._db = kuzu.Database(str(self.db_path))
        self._conn = kuzu.Connection(self._db)

        # Get statistics
        result = self._conn.execute("MATCH (r:Root) RETURN count(r)")
        self.total_roots = result.get_next()[0]
        logger.info(f"  Loaded {self.total_roots:,} roots")

        result = self._conn.execute("MATCH (s:Sentence) RETURN count(s)")
        self.total_docs = result.get_next()[0]
        logger.info(f"  Loaded {self.total_docs:,} sentences")

        # Estimate average document length (sample 1000 docs for efficiency)
        logger.info("  Computing average document length...")
        sample_size = min(1000, self.total_docs)
        result = self._conn.execute(
            f"MATCH (s:Sentence) RETURN s.text LIMIT {sample_size}"
        )
        total_length = 0
        count = 0
        while result.has_next():
            text = result.get_next()[0]
            # Approximate word count (split on whitespace)
            total_length += len(text.split())
            count += 1
        if count > 0:
            self.avg_doc_length = total_length / count
            logger.info(f"  Average document length: {self.avg_doc_length:.1f} words (sample={count})")
        else:
            self.avg_doc_length = 20.0  # Fallback
            logger.info(f"  Average document length: {self.avg_doc_length:.1f} words (default)")

        # Cache all root names for fast lookup
        logger.info("  Caching root names...")
        self._all_roots = set()
        result = self._conn.execute("MATCH (r:Root) RETURN r.root")
        while result.has_next():
            self._all_roots.add(result.get_next()[0])
        logger.info(f"  Cached {len(self._all_roots):,} root keys")

        # Load document offsets for O(1) access
        offsets_file = self.index_path / "doc_offsets.npy"
        docs_file = self.index_path / "documents.jsonl"

        if offsets_file.exists():
            self.doc_offsets = np.load(offsets_file)
            logger.info(f"  Document offsets: {len(self.doc_offsets):,}")

            if docs_file.exists():
                self._doc_file = open(docs_file, 'rb')
                self._doc_mmap = mmap.mmap(
                    self._doc_file.fileno(), 0, access=mmap.ACCESS_READ
                )

    def _load_semantic_models(self):
        """Load AI models for semantic constraints (#588, #589)."""
        if torch is None:
            logger.warning("PyTorch not available, skipping semantic model loading")
            return

        # Load root embeddings (#589)
        model_path = Path("models/root_embeddings/best_model.pt")
        if not model_path.exists():
            logger.warning(f"Root embeddings not found at {model_path}")
            return

        logger.info(f"Loading root embeddings from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract embeddings and vocabulary mapping
            self.embedding_dim = checkpoint.get('embedding_dim', 64)
            self.root_to_idx = checkpoint.get('root_to_idx', {})

            # Get embeddings tensor from model state dict
            model_state = checkpoint.get('model_state_dict', {})
            if 'embeddings.weight' in model_state:
                self.root_embeddings = model_state['embeddings.weight']
                logger.info(f"  Loaded {self.root_embeddings.shape[0]:,} root embeddings ({self.embedding_dim}d)")
            else:
                logger.warning("  No embeddings.weight found in checkpoint")

        except Exception as e:
            logger.error(f"Failed to load root embeddings: {e}")
            self.root_embeddings = None
            self.root_to_idx = None

    def has_root(self, root: str) -> bool:
        """Check if a root exists in the index (O(1) using cached set)."""
        if self._all_roots is not None:
            return root.lower() in self._all_roots
        return False

    def get_all_roots(self) -> Set[str]:
        """Get all roots in the index."""
        return self._all_roots or set()

    def get_occurrences(self, root: str) -> List[RootOccurrence]:
        """
        Get all occurrences of a root from Kuzu.

        Uses Cypher query to find all sentences containing this root.
        """
        if self._conn is None:
            return []

        root = root.lower()
        occurrences = []

        result = self._conn.execute(
            """
            MATCH (s:Sentence)-[e:HAS_ROOT]->(r:Root {root: $root})
            RETURN s.id, e.role, e.grammar
            """,
            {"root": root}
        )

        while result.has_next():
            row = result.get_next()
            doc_id = row[0]
            role = row[1] or 'unknown'
            grammar_str = row[2] or '{}'
            try:
                grammar = json.loads(grammar_str)
            except json.JSONDecodeError:
                grammar = {}

            occurrences.append(RootOccurrence(
                doc_id=doc_id,
                role=role,
                grammar=grammar,
            ))

        return occurrences

    def get_doc_frequency(self, root: str) -> int:
        """Get document frequency for a root (for IDF calculation)."""
        root = root.lower()

        if root in self._doc_freq_cache:
            return self._doc_freq_cache[root]

        if self._conn is None:
            return 0

        result = self._conn.execute(
            "MATCH (r:Root {root: $root}) RETURN r.doc_freq",
            {"root": root}
        )

        if result.has_next():
            freq = result.get_next()[0] or 0
        else:
            freq = 0

        self._doc_freq_cache[root] = freq
        return freq

    def compute_idf(self, doc_freq: int) -> float:
        """Compute IDF (Inverse Document Frequency) for BM25."""
        import math
        if doc_freq == 0:
            return 0.0
        N = self.total_docs
        return math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get document by ID using memory-mapped access."""
        if self.doc_offsets is None or self._doc_mmap is None:
            return None
        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            return None

        offset = int(self.doc_offsets[doc_id])
        self._doc_mmap.seek(offset)
        line = self._doc_mmap.readline()
        return json.loads(line.decode('utf-8'))

    # =========================================================================
    # Graph-Enhanced Features (New in Kuzu version)
    # =========================================================================

    def get_synonyms(self, root: str) -> Set[str]:
        """
        Get direct synonyms from Kuzu graph (1-hop).

        This replaces SemanticRelationDB.get_synonyms().
        """
        if self._conn is None:
            return set()

        root = root.lower()
        synonyms = set()

        result = self._conn.execute(
            """
            MATCH (r:Root {root: $root})-[:IS_SYNONYM]->(syn:Root)
            RETURN syn.root
            """,
            {"root": root}
        )

        while result.has_next():
            synonyms.add(result.get_next()[0])

        return synonyms

    def get_synonyms_transitive(self, root: str, max_hops: int = 2) -> Set[str]:
        """
        Get transitive synonyms (synonyms of synonyms) from Kuzu graph.

        This enables finding "fond" → "kre" → "establ" chains.
        """
        if self._conn is None:
            return set()

        root = root.lower()
        synonyms = set()

        # Use variable-length path matching
        result = self._conn.execute(
            f"""
            MATCH (r:Root {{root: $root}})-[:IS_SYNONYM*1..{max_hops}]->(syn:Root)
            RETURN DISTINCT syn.root
            """,
            {"root": root}
        )

        while result.has_next():
            synonyms.add(result.get_next()[0])

        return synonyms

    def get_hypernyms(self, root: str) -> Set[str]:
        """Get hypernyms (more general terms) for a root."""
        if self._conn is None:
            return set()

        root = root.lower()
        hypernyms = set()

        result = self._conn.execute(
            """
            MATCH (r:Root {root: $root})-[:IS_HYPERNYM]->(hyper:Root)
            RETURN hyper.root
            """,
            {"root": root}
        )

        while result.has_next():
            hypernyms.add(result.get_next()[0])

        return hypernyms

    def get_conceptnet_relations(
        self,
        root: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 1
    ) -> Set[str]:
        """
        Get related roots via ConceptNet relations.

        Args:
            root: Source root to expand
            relation_types: List of relation types to follow. If None, uses default set.
                           Examples: CN_SYNONYM, CN_IS_A, CN_SIMILAR_TO, CN_PART_OF
            max_hops: Maximum traversal depth (default: 1 for direct relations only)

        Returns:
            Set of related roots from ConceptNet

        Example:
            >>> index.get_conceptnet_relations("hund", ["CN_SYNONYM", "CN_IS_A"])
            {"kanid", "best", "mamul"}
        """
        if self._conn is None:
            return set()

        root = root.lower()
        related = set()

        # Default relation types: use semantic similarity relations
        if relation_types is None:
            relation_types = [
                "CN_SYNONYM",      # Direct synonyms
                "CN_IS_A",         # Taxonomic relations (dog is-a animal)
                "CN_SIMILAR_TO",   # Similar concepts
            ]

        # Build relation type pattern for Cypher
        # For single hop: -[:CN_SYNONYM|CN_IS_A]->
        # For multi-hop: -[:CN_SYNONYM|CN_IS_A*1..2]->
        relation_pattern = "|".join(relation_types)

        if max_hops == 1:
            path_spec = f"[:{relation_pattern}]"
        else:
            path_spec = f"[:{relation_pattern}*1..{max_hops}]"

        # Query both Root→Root and Root→Concept→Root paths
        # ConceptNet has both internal Esperanto relations and external concept links
        result = self._conn.execute(
            f"""
            MATCH (r:Root {{root: $root}})-{path_spec}->(related:Root)
            RETURN DISTINCT related.root
            LIMIT 100
            """,
            {"root": root}
        )

        while result.has_next():
            related.add(result.get_next()[0])

        return related

    def get_hypernym_chain(self, root: str, max_depth: int = 5) -> List[str]:
        """
        Get hypernym chain (root → parent → grandparent → ...).

        Example: "hundo" → ["besto", "vivaĵo", "aĵo"]
        """
        if self._conn is None:
            return []

        root = root.lower()
        chain = []

        result = self._conn.execute(
            f"""
            MATCH path = (r:Root {{root: $root}})-[:IS_HYPERNYM*1..{max_depth}]->(ancestor:Root)
            RETURN ancestor.root, length(path)
            ORDER BY length(path)
            """,
            {"root": root}
        )

        while result.has_next():
            row = result.get_next()
            if row[0] not in chain:
                chain.append(row[0])

        return chain

    def get_sentence_context(
        self,
        sent_id: int,
        window: int = 2,
    ) -> List[Dict]:
        """
        Get surrounding sentences for context.

        Returns sentences within `window` hops via NEXT_SENTENCE edges.
        """
        if self._conn is None:
            return []

        context = []

        # Get preceding sentences
        result = self._conn.execute(
            f"""
            MATCH path = (prev:Sentence)-[:NEXT_SENTENCE*1..{window}]->(s:Sentence {{id: $sent_id}})
            RETURN prev.id, prev.text, length(path) as dist
            ORDER BY dist DESC
            """,
            {"sent_id": sent_id}
        )

        preceding = []
        while result.has_next():
            row = result.get_next()
            preceding.append({
                'id': row[0],
                'text': row[1],
                'distance': -row[2],  # Negative for preceding
            })

        context.extend(reversed(preceding))

        # Add current sentence
        result = self._conn.execute(
            "MATCH (s:Sentence {id: $sent_id}) RETURN s.text",
            {"sent_id": sent_id}
        )
        if result.has_next():
            context.append({
                'id': sent_id,
                'text': result.get_next()[0],
                'distance': 0,
            })

        # Get following sentences
        result = self._conn.execute(
            f"""
            MATCH path = (s:Sentence {{id: $sent_id}})-[:NEXT_SENTENCE*1..{window}]->(next:Sentence)
            RETURN next.id, next.text, length(path) as dist
            ORDER BY dist
            """,
            {"sent_id": sent_id}
        )

        while result.has_next():
            row = result.get_next()
            context.append({
                'id': row[0],
                'text': row[1],
                'distance': row[2],
            })

        return context

    # =========================================================================
    # Predicate-First Retrieval (Phase 1.3)
    # =========================================================================

    def _extract_predicate_from_ast(
        self,
        ast: Dict,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract predicate (verb, subj, obj) from query AST.

        Returns the main predicate triple for predicate-first lookup.

        Args:
            ast: Parsed query AST

        Returns:
            Tuple of (verb, subj, obj) roots, any can be None
        """
        if not ast or ast.get('tipo') != 'frazo':
            return (None, None, None)

        skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

        def get_root(node: Optional[Dict]) -> Optional[str]:
            """Get root from a word or word group, skipping function words."""
            if not node or not isinstance(node, dict):
                return None

            tipo = node.get('tipo')

            if tipo == 'vorto':
                vortspeco = node.get('vortspeco', '')
                if vortspeco in skip_vortspeco:
                    return None
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    return root.lower()
                return None

            elif tipo == 'vortgrupo':
                return get_root(node.get('kerno'))

            return None

        verb = get_root(ast.get('verbo'))
        subj = get_root(ast.get('subjekto'))
        obj = get_root(ast.get('objekto'))

        return (verb, subj, obj)

    def search_by_predicate(
        self,
        verb: Optional[str] = None,
        subj: Optional[str] = None,
        obj: Optional[str] = None,
        expand_synonyms: bool = True,
        max_results: int = 1000,
    ) -> List[int]:
        """
        Search for sentences matching a predicate pattern via HAS_PREDICATE edges.

        This is the predicate-first retrieval from Issue #255. It uses the
        Predicate table (from import_predicates_kuzu.py) for O(1) structural lookup.

        Args:
            verb: Verb root (required for meaningful search)
            subj: Subject root (optional)
            obj: Object root (optional)
            expand_synonyms: Expand roots to synonyms via Kuzu graph
            max_results: Maximum sentence IDs to return

        Returns:
            List of sentence IDs matching the predicate pattern
        """
        if self._conn is None:
            return []

        if not verb:
            # Without a verb, predicate search isn't meaningful
            return []

        # Build list of roots for each role (with optional synonym expansion)
        verb_roots = {verb.lower()}
        subj_roots = {subj.lower()} if subj else None
        obj_roots = {obj.lower()} if obj else None

        if expand_synonyms:
            verb_roots.update(self.get_synonyms_transitive(verb, max_hops=2))
            if subj_roots:
                subj_roots.update(self.get_synonyms_transitive(subj, max_hops=2))
            if obj_roots:
                obj_roots.update(self.get_synonyms_transitive(obj, max_hops=2))

        # Build Cypher query for HAS_PREDICATE edges
        where_clauses = ["p.verb IN $verb_roots"]
        params = {'verb_roots': list(verb_roots)}

        if subj_roots:
            where_clauses.append("p.subj IN $subj_roots")
            params['subj_roots'] = list(subj_roots)

        if obj_roots:
            where_clauses.append("p.obj IN $obj_roots")
            params['obj_roots'] = list(obj_roots)

        query = f"""
            MATCH (s:Sentence)-[:HAS_PREDICATE]->(p:Predicate)
            WHERE {' AND '.join(where_clauses)}
            RETURN DISTINCT s.id
            LIMIT {max_results}
        """

        try:
            result = self._conn.execute(query, params)
            sent_ids = []
            while result.has_next():
                sent_ids.append(result.get_next()[0])
            return sent_ids
        except Exception as e:
            # HAS_PREDICATE table may not exist yet
            logger.debug(f"Predicate search failed (table may not exist): {e}")
            return []

    def has_predicate_table(self) -> bool:
        """Check if the Predicate table exists in Kuzu."""
        if self._conn is None:
            return False
        try:
            result = self._conn.execute("MATCH (p:Predicate) RETURN count(p) LIMIT 1")
            result.get_next()
            return True
        except Exception:
            return False

    def search_by_role(
        self,
        verb: Optional[str] = None,
        subj: Optional[str] = None,
        obj: Optional[str] = None,
        expand_synonyms: bool = True,
        max_results: int = 100,
    ) -> List[Tuple[int, str]]:
        """
        Search for sentences matching specific role patterns.

        This is AST role-aware search - much more precise than keyword matching.

        Example:
            search_by_role(verb="fond", obj="esperant")
            → Find sentences where "fond" is VERB and "esperant" is OBJECT
        """
        if self._conn is None:
            return []

        # Build list of roots for each role (with optional synonym expansion)
        verb_roots = {verb.lower()} if verb else None
        subj_roots = {subj.lower()} if subj else None
        obj_roots = {obj.lower()} if obj else None

        if expand_synonyms:
            if verb_roots:
                verb_roots.update(self.get_synonyms_transitive(verb, max_hops=2))
            if subj_roots:
                subj_roots.update(self.get_synonyms_transitive(subj, max_hops=2))
            if obj_roots:
                obj_roots.update(self.get_synonyms_transitive(obj, max_hops=2))

        # Build Cypher query dynamically
        match_clauses = []
        where_clauses = []
        params = {}

        if verb_roots:
            match_clauses.append("(s)-[:HAS_ROOT {role: 'verbo'}]->(v:Root)")
            where_clauses.append(f"v.root IN $verb_roots")
            params['verb_roots'] = list(verb_roots)

        if subj_roots:
            match_clauses.append("(s)-[:HAS_ROOT {role: 'subjekto'}]->(subj:Root)")
            where_clauses.append(f"subj.root IN $subj_roots")
            params['subj_roots'] = list(subj_roots)

        if obj_roots:
            match_clauses.append("(s)-[:HAS_ROOT {role: 'objekto'}]->(obj:Root)")
            where_clauses.append(f"obj.root IN $obj_roots")
            params['obj_roots'] = list(obj_roots)

        if not match_clauses:
            return []

        query = f"""
            MATCH (s:Sentence), {', '.join(match_clauses)}
            WHERE {' AND '.join(where_clauses)}
            RETURN DISTINCT s.id, s.text
            LIMIT {max_results}
        """

        results = []
        result = self._conn.execute(query, params)

        while result.has_next():
            row = result.get_next()
            results.append((row[0], row[1]))

        return results

    # =========================================================================
    # Main Search Interface
    # =========================================================================

    # Predicate boost factor for documents matching predicate structure
    PREDICATE_BOOST = 1.5

    def search(
        self,
        query_ast: Dict,
        max_results: int = 10,
        fallback_mode: Optional[FallbackMode] = None,
        require_all_roots: bool = False,
        use_graph_expansion: bool = True,
        use_conceptnet_expansion: bool = False,  # Disabled by default - data quality issues
        use_predicate_boost: bool = True,
    ) -> Tuple[List[SearchResult], RetrievalStats]:
        """
        Search for documents matching query AST.

        This is the main search interface with predicate-first retrieval.

        Args:
            query_ast: Parsed query AST
            max_results: Maximum results to return
            fallback_mode: Override instance fallback mode
            require_all_roots: If True, only return docs with ALL query roots
            use_graph_expansion: Use Kuzu graph for ReVo synonym expansion
            use_conceptnet_expansion: Use ConceptNet for semantic expansion
            use_predicate_boost: Apply boost to predicate-matched documents

        Returns:
            Tuple of (results, stats)
        """
        mode = fallback_mode if fallback_mode is not None else self.fallback_mode
        stats = RetrievalStats()

        # 1. Extract roots from query AST
        query_roots = self._extract_roots(query_ast)
        query_grammar = self._extract_grammar(query_ast)
        stats.query_roots = list(query_roots.keys())

        if not query_roots:
            logger.warning("No roots extracted from query")
            return [], stats

        # 1b. Extract predicate for predicate-first retrieval
        predicate_matched_docs: Set[int] = set()
        if use_predicate_boost:
            verb, subj, obj = self._extract_predicate_from_ast(query_ast)
            if verb:
                # Format predicate for stats
                pred_parts = [verb]
                if subj:
                    pred_parts.append(f"subj={subj}")
                if obj:
                    pred_parts.append(f"obj={obj}")
                stats.predicate_query = f"({', '.join(pred_parts)})"

                # Do predicate lookup
                # OPTIMIZATION: Limit to 100 results (sufficient for top-50 reranking)
                # Reduces graph traversal and scoring overhead
                predicate_matched_docs = set(self.search_by_predicate(
                    verb=verb,
                    subj=subj,
                    obj=obj,
                    expand_synonyms=use_graph_expansion,
                    max_results=100,
                ))
                stats.predicate_matches = len(predicate_matched_docs)

                if predicate_matched_docs:
                    stats.predicate_boost_applied = True
                    logger.debug(
                        f"Predicate search found {len(predicate_matched_docs)} docs "
                        f"for {stats.predicate_query}"
                    )

        # 2. Build semantic concepts using Kuzu graph (ReVo + ConceptNet)
        concepts = self._build_concepts_from_graph(
            query_roots,
            use_graph=use_graph_expansion,
            use_conceptnet=use_conceptnet_expansion,
            stats=stats,
        )

        # Flatten concepts for index lookup
        expanded_roots = {}
        for concept in concepts:
            for eq_root in concept.equivalent_roots:
                expanded_roots[eq_root] = max(expanded_roots.get(eq_root, 0), concept.weight)
        stats.expanded_roots = list(expanded_roots.keys())

        # 3. Look up in graph
        candidates = self._lookup_candidates(
            expanded_roots,
            require_all=require_all_roots,
            stats=stats,
        )
        stats.candidate_count_before_scoring = len(candidates)

        # 4. Score using BM25 with predicate boost
        scored = self._score_with_bm25(
            candidates,
            concepts,
            query_grammar,
            stats,
            predicate_matched_docs=predicate_matched_docs,
        )
        stats.candidate_count_after_scoring = len(scored)

        # 5. Build results
        results = []
        for doc_id, score, matched_roots, grammar_matches in scored[:max_results]:
            doc = self.get_document(doc_id)
            if doc:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    text=doc.get("text", ""),
                    matched_roots=matched_roots,
                    grammar_matches=grammar_matches,
                    source=doc.get("source", {}),
                ))

        return results, stats

    # High-frequency roots that appear in too many documents to be useful for retrieval.
    # These are typically auxiliary verbs and common function-like roots.
    # Filtering these prevents generating millions of candidates.
    STOPWORD_ROOTS = frozenset({
        # Auxiliary/copula verbs (appear in 1M+ documents)
        'est',      # to be (1.7M docs)
        # Common verbs that are too general
        'hav',      # to have
        'far',      # to do/make
        'pov',      # to be able
        'dev',      # must/should
        'vol',      # to want
        'ir',       # to go
        'ven',      # to come
        'don',      # to give
        'pren',     # to take
        'dir',      # to say
        'vid',      # to see
        'sci',      # to know
        'trov',     # to find
        # Common modifiers
        'bon',      # good
        'grand',    # big
        'nov',      # new
        'ali',      # other
        'mult',     # many
        'sam',      # same
    })

    def _extract_roots(self, ast: Dict) -> Dict[str, float]:
        """Extract roots from AST with weights based on role and part of speech.

        Weighting strategy:
        - NOUNS (substantivo) get highest weight - they're the content of the query
        - VERBS get lower weight - they're often paraphrased in answers
          e.g., "Kiam aperis X?" may be answered by "X estis fondita en..."
        - Named entities (proper nouns) get highest weight
        """
        roots = {}
        skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

        def extract(node, role: str = "aliaj", weight: float = 1.0):
            if not node or not isinstance(node, dict):
                return

            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco', '')
                if vortspeco in skip_vortspeco:
                    return

                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    root = root.lower()
                    # Skip high-frequency stopword roots
                    if root in KuzuInvertedIndex.STOPWORD_ROOTS:
                        return

                    # Part-of-speech weights: nouns > adjectives > verbs
                    # Verbs are often paraphrased in answers
                    pos_weights = {
                        'substantivo': 1.5,  # Nouns are key content
                        'adjektivo': 1.2,    # Adjectives describe entities
                        'verbo': 0.8,        # Verbs often paraphrased
                        'adverbo': 0.7,      # Adverbs less critical
                        'numero': 1.3,       # Numbers (dates, quantities) important
                    }

                    # Role weights (less extreme than before)
                    role_weights = {
                        'verbo': 0.9,        # Verb role (reduced from 1.5)
                        'objekto': 1.2,      # Object is usually important
                        'subjekto': 1.1,     # Subject is important
                        'predikato': 1.0,
                        'aliaj': 0.9,
                    }

                    pos_weight = pos_weights.get(vortspeco, 1.0)
                    role_weight = role_weights.get(role, 1.0)
                    final_weight = weight * pos_weight * role_weight
                    roots[root] = max(roots.get(root, 0), final_weight)

            elif node.get('tipo') == 'vortgrupo':
                extract(node.get('kerno'), role, weight)
                for p in node.get('priskriboj', []):
                    extract(p, role, weight * 0.8)

            elif node.get('tipo') == 'frazo':
                extract(node.get('subjekto'), 'subjekto', weight)
                extract(node.get('verbo'), 'verbo', weight)
                extract(node.get('objekto'), 'objekto', weight)
                for a in node.get('aliaj', []):
                    extract(a, 'aliaj', weight)

        extract(ast)
        return roots

    def _extract_grammar(self, ast: Dict) -> Dict[str, Any]:
        """Extract grammatical features from query AST."""
        grammar = {}
        verbo = ast.get('verbo', {})
        if verbo:
            if verbo.get('tempo'):
                grammar['tempo'] = verbo['tempo']
            if verbo.get('modo'):
                grammar['modo'] = verbo['modo']
        if ast.get('negita'):
            grammar['negita'] = True
        if ast.get('fraztipo'):
            grammar['fraztipo'] = ast['fraztipo']

        # Detect WHO questions (kiu = who/which person)
        # Check subjekto for "kiu" or korelativo_sufikso "u"
        subjekto = ast.get('subjekto', {})
        if isinstance(subjekto, dict):
            kerno = subjekto.get('kerno', {})
            if kerno.get('radiko') == 'kiu' or kerno.get('korelativo_sufikso') == 'u':
                grammar['is_who_question'] = True

        return grammar

    def _build_concepts_from_graph(
        self,
        roots: Dict[str, float],
        use_graph: bool = True,
        use_conceptnet: bool = True,
        stats: Optional[RetrievalStats] = None,
    ) -> List[SemanticConcept]:
        """
        Build semantic concepts using Kuzu graph for synonym expansion.

        Expands query roots using:
        1. ReVo synonyms (IS_SYNONYM edges)
        2. ConceptNet relations (CN_SYNONYM, CN_IS_A, CN_SIMILAR_TO)

        Args:
            roots: Query roots with weights
            use_graph: Enable ReVo synonym expansion
            use_conceptnet: Enable ConceptNet semantic expansion
            stats: RetrievalStats to update

        Returns:
            List of semantic concepts with expanded equivalent roots
        """
        concepts = []
        index_roots = self.get_all_roots()

        for root, weight in roots.items():
            concept = SemanticConcept(original_root=root, weight=weight)

            if use_graph and self._conn:
                # 1. Get synonyms from ReVo (transitive, up to 2 hops)
                synonyms = self.get_synonyms_transitive(root, max_hops=2)

                for syn in synonyms:
                    if syn in index_roots and syn not in self.STOPWORD_ROOTS:
                        concept.equivalent_roots.add(syn)
                        if stats:
                            stats.semantic_db_synonyms.append(f"{root}→{syn}")
                            stats.graph_expansions += 1

                # 2. Expand via ConceptNet relations
                if use_conceptnet:
                    # Query ConceptNet graph for semantically related roots
                    conceptnet_related = self.get_conceptnet_relations(
                        root,
                        relation_types=["CN_SYNONYM", "CN_IS_A", "CN_SIMILAR_TO"],
                        max_hops=1
                    )

                    for related in conceptnet_related:
                        if related in index_roots:
                            concept.equivalent_roots.add(related)
                            if stats:
                                stats.graph_expansions += 1
                                # Track ConceptNet expansions separately
                                if not hasattr(stats, 'conceptnet_expansions'):
                                    stats.conceptnet_expansions = []
                                stats.conceptnet_expansions.append(f"{root}→{related}")

            concepts.append(concept)

            if stats:
                stats.concepts.append(
                    f"{root}[{len(concept.equivalent_roots)}]: {','.join(list(concept.equivalent_roots)[:5])}"
                )

        return concepts

    def _lookup_candidates(
        self,
        roots: Dict[str, float],
        require_all: bool,
        stats: RetrievalStats,
    ) -> Dict[int, Tuple[float, List[str], List[RootOccurrence]]]:
        """Look up documents containing query roots."""
        candidates: Dict[int, Tuple[float, List[str], List[RootOccurrence]]] = {}

        for root, weight in roots.items():
            if self.has_root(root):
                stats.roots_found_in_index.append(root)

                for occ in self.get_occurrences(root):
                    if occ.doc_id not in candidates:
                        candidates[occ.doc_id] = (0.0, [], [])

                    score, matched, occs = candidates[occ.doc_id]
                    candidates[occ.doc_id] = (
                        score + weight,
                        matched + [root],
                        occs + [occ],
                    )
            else:
                stats.roots_not_found.append(root)

        if require_all and stats.query_roots:
            query_root_set = set(stats.query_roots)
            candidates = {
                doc_id: data
                for doc_id, data in candidates.items()
                if query_root_set.issubset(set(data[1]))
            }

        return candidates

    def _score_with_bm25(
        self,
        candidates: Dict[int, Tuple[float, List[str], List[RootOccurrence]]],
        concepts: List[SemanticConcept],
        query_grammar: Dict[str, Any],
        stats: RetrievalStats,
        predicate_matched_docs: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float, List[str], Dict[str, bool]]]:
        """
        Score candidates using BM25 with concept-based IDF.

        Args:
            candidates: Dict mapping doc_id to (score, matched_roots, occurrences)
            concepts: List of semantic concepts from query
            query_grammar: Grammar features from query AST
            stats: RetrievalStats to update
            predicate_matched_docs: Set of doc_ids that match predicate structure

        Returns:
            List of (doc_id, score, matched_roots, grammar_matches) sorted by score
        """
        import math

        stats.scoring_method = "bm25"
        scored = []
        predicate_matched_docs = predicate_matched_docs or set()

        # Build concept lookup
        root_to_concept: Dict[str, SemanticConcept] = {}
        for concept in concepts:
            for eq_root in concept.equivalent_roots:
                root_to_concept[eq_root] = concept

        # Compute IDF for each concept
        concept_idf: Dict[str, float] = {}
        for concept in concepts:
            max_freq = 0
            for eq_root in concept.equivalent_roots:
                freq = self.get_doc_frequency(eq_root)
                max_freq = max(max_freq, freq)
            idf = self.compute_idf(max_freq)
            concept_idf[concept.original_root] = idf

        for doc_id, (_, matched_roots, occurrences) in candidates.items():
            bm25_score = 0.0
            grammar_matches = {}
            grammar_bonus = 0.0

            # Get document for length normalization and quality detection
            doc = self.get_document(doc_id)
            doc_length = len(doc['text'].split()) if doc else 20.0

            # Track matched concepts
            matched_concepts: Dict[str, int] = {}
            for root in matched_roots:
                concept = root_to_concept.get(root)
                if concept:
                    matched_concepts[concept.original_root] = (
                        matched_concepts.get(concept.original_root, 0) + 1
                    )

            # Compute BM25 score with length normalization
            for concept_root, tf in matched_concepts.items():
                idf = concept_idf.get(concept_root, 0.0)

                # BM25 with length normalization: tf / (tf + k1 * (1 - b + b * (dl / avgdl)))
                length_norm = (1 - self.bm25_b) + self.bm25_b * (doc_length / self.avg_doc_length)
                tf_component = tf / (tf + self.bm25_k1 * length_norm)

                concept = next((c for c in concepts if c.original_root == concept_root), None)
                weight = concept.weight if concept else 1.0
                bm25_score += idf * tf_component * weight

            # AST-based quality detection: Penalize fragments (titles, labels, incomplete sentences)
            # Fragments lack verbs - they're noun phrases like "Fondaĵo pri Esperantaj Studoj"
            # Complete sentences have verbs - even short ones like "Zamenhof kreis Esperanton"
            if doc:
                from klareco.parser import parse
                ast = parse(doc['text'])
                if ast:
                    has_verb = bool(ast.get('verbo'))
                    if not has_verb:
                        # Fragment penalty: titles, labels, metadata get heavily penalized
                        fragment_penalty = 0.3  # 70% penalty for fragments without verbs
                        bm25_score *= fragment_penalty
                        grammar_matches['fragment'] = True

            # Option 4: Proper noun boost for WHO questions
            # Use query_grammar to detect WHO questions
            # Boost documents containing proper nouns (capitalized words like names)
            is_who_question = query_grammar.get('is_who_question', False)

            if is_who_question and doc:
                # Count proper nouns (capitalized words that aren't sentence-initial)
                words = doc['text'].split()
                proper_noun_count = 0
                for i, word in enumerate(words):
                    # Skip first word (sentence-initial caps don't count)
                    if i > 0 and word and word[0].isupper():
                        proper_noun_count += 1

                if proper_noun_count > 0:
                    # Boost by 50% for each proper noun (up to 2x total)
                    proper_noun_boost = min(1.0 + (proper_noun_count * 0.5), 2.0)
                    bm25_score *= proper_noun_boost
                    grammar_matches['proper_nouns'] = True

            # WHO-Question Subject Boosting (#586)
            # For WHO questions, boost sentences where PERSON NAME is the subject
            # This promotes "Zamenhof kreis..." over "li kreis..." or passive "kreiĝis..."
            if is_who_question and ast:
                subj_node = ast.get('subjekto', {})
                if subj_node:
                    kerno = subj_node.get('kerno', {})
                    subj_word = kerno.get('plena_vorto', '')
                    subj_root = kerno.get('radiko', '')

                    # Check if subject is a PERSON name (proper noun, not pronoun)
                    if subj_word and subj_word[0].isupper() and subj_root not in ['li', 'ŝi', 'ili', 'mi', 'vi', 'ni']:
                        # Named person as subject → direct answer to WHO question
                        who_answer_boost = 1.4
                        bm25_score *= who_answer_boost
                        grammar_matches['who_answer'] = True

            # Grammar matching
            for occ in occurrences:
                if 'tempo' in query_grammar and occ.grammar.get('tempo'):
                    if occ.grammar['tempo'] == query_grammar['tempo']:
                        grammar_matches['tempo'] = True
                        grammar_bonus += 0.1

            # Apply predicate boost with object role analysis (#585) and POS constraint (#587)
            predicate_boost = 1.0
            if doc_id in predicate_matched_docs:
                # Part-of-Speech Constraint (#587)
                # Check if verb is actually VERB form vs NOUN form (kreinto, fondinto)
                verb_pos_multiplier = 1.0
                if ast and ast.get('verbo'):
                    verb_node = ast.get('verbo')
                    vortspeco = verb_node.get('vortspeco')
                    if vortspeco == 'verbo':
                        verb_pos_multiplier = 1.0  # Full boost for verb form
                        grammar_matches['verb_form'] = True
                    elif vortspeco in ['substantivo', 'adjektivo']:
                        verb_pos_multiplier = 0.8  # Partial boost for noun/adj form (kreinto)
                        grammar_matches['noun_form'] = True

                # Object Role Analysis (#585)
                # Check if object is MAIN thing being acted upon vs MODIFIER
                # Example: "kreis Esperanton" (main object) vs "kreis esperantan grupon" (modifier)
                if ast and ast.get('objekto'):
                    obj_node = ast.get('objekto')
                    kerno = obj_node.get('kerno', {})
                    obj_word = kerno.get('plena_vorto', '').lower()

                    # Check if object word is the actual noun vs an adjective modifier
                    # "esperanton" (accusative noun) vs "esperantan" (adjective)
                    if obj_word in ['esperanton', 'esperanto']:
                        # Main object is the language itself - STRONG boost
                        predicate_boost = 2.0 * verb_pos_multiplier
                        grammar_matches['predicate'] = True
                        grammar_matches['object_main'] = True
                    elif 'esperant' in obj_word and obj_word not in ['esperanton', 'esperanto']:
                        # It's a modifier (esperantan grupon, esperantistan socion) - PENALTY
                        predicate_boost = 0.5 * verb_pos_multiplier
                        grammar_matches['predicate'] = True
                        grammar_matches['object_modifier'] = True
                    else:
                        # Neutral - predicate matches but object doesn't match query
                        predicate_boost = self.PREDICATE_BOOST * verb_pos_multiplier
                        grammar_matches['predicate'] = True
                else:
                    # No AST or no object - use default predicate boost with POS multiplier
                    predicate_boost = self.PREDICATE_BOOST * verb_pos_multiplier
                    grammar_matches['predicate'] = True

            # Semantic Root Similarity (#589)
            # Use root embeddings to compute semantic similarity between query and doc roots
            semantic_sim_boost = 1.0
            if self.root_embeddings is not None and self.root_to_idx is not None and torch is not None:
                # Get embeddings for query roots
                query_root_embeddings = []
                for concept in concepts:
                    root = concept.original_root
                    if root in self.root_to_idx:
                        idx = self.root_to_idx[root]
                        query_root_embeddings.append(self.root_embeddings[idx])

                # Get embeddings for document roots
                doc_root_embeddings = []
                for root in matched_roots:
                    if root in self.root_to_idx:
                        idx = self.root_to_idx[root]
                        doc_root_embeddings.append(self.root_embeddings[idx])

                # Compute max cosine similarity between any query root and any doc root
                if query_root_embeddings and doc_root_embeddings:
                    max_sim = 0.0
                    for q_emb in query_root_embeddings:
                        for d_emb in doc_root_embeddings:
                            # Cosine similarity
                            sim = F.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
                            max_sim = max(max_sim, sim)

                    # Apply boost if similarity exceeds threshold
                    if max_sim > 0.7:
                        semantic_sim_boost = 1.0 + (max_sim - 0.7) * 0.5  # Up to 1.15x boost for sim=1.0
                        grammar_matches['semantic_sim'] = True

            final_score = (bm25_score + grammar_bonus) * predicate_boost * semantic_sim_boost
            scored.append((doc_id, final_score, matched_roots, grammar_matches))

        scored.sort(key=lambda x: -x[1])
        return scored

    def close(self):
        """Clean up resources."""
        if self._doc_mmap:
            self._doc_mmap.close()
            self._doc_mmap = None
        if self._doc_file:
            self._doc_file.close()
            self._doc_file = None
        if self._conn:
            self._conn = None
        if self._db:
            self._db = None

    def __del__(self):
        self.close()
