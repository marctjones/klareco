"""
AST-Aware Retriever using Kuzu Graph Database.

Features:
1. Root-based inverted index lookup (O(1))
2. Transitive synonym expansion via graph traversal
3. Hypernym chain traversal
4. Grammar-aware scoring
5. Question type classification for slot weighting
6. Sentence context retrieval (adjacent sentences)
7. Role-aware pattern matching

This is a DETERMINISTIC retriever that leverages Klareco's unique advantage:
fully parsed AST annotations for both queries and corpus.

MEMORY EFFICIENT:
- Kuzu-backed root lookups (~50MB RAM instead of 8GB)
- Memory-mapped document access
- No HNSW index needed (root index provides fast lookup)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from klareco.parser import parse
from klareco.rag.question_classifier import QuestionClassifier, QuestionType
from klareco.rag.entity_recognizer import EntityRecognizer
from klareco.rag.kuzu_inverted_index import KuzuInvertedIndex, FallbackMode, RetrievalStats

# Try to import HybridEmbeddings for fallback
try:
    from klareco.embeddings.hybrid_embeddings import HybridEmbeddings
    HYBRID_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HYBRID_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ASTAwareRetriever:
    """
    AST-aware retriever using Kuzu graph database backend.

    Memory efficient: Uses Kuzu-backed root lookups and mmap document access.
    No HNSW index needed - root inverted index provides O(1) root lookup.

    Features:
    - Transitive synonym expansion (2+ hops)
    - Hypernym chain traversal
    - Sentence context retrieval (adjacent sentences)
    - Role-aware pattern matching
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        fallback_mode: FallbackMode = FallbackMode.NONE,
    ):
        """
        Initialize AST-aware retriever.

        Args:
            index_path: Path to Kuzu index directory.
                        Defaults to data/indexes/kuzu_index.
            fallback_mode: Fallback mode (default: NONE for pure deterministic)
        """
        # Set default path
        if index_path is None:
            index_path = Path("data/indexes/kuzu_index")
        self.index_path = Path(index_path)

        logger.info("Initializing AST-aware retriever (Kuzu backend)...")

        # Initialize question classifier
        self.question_classifier = QuestionClassifier()
        logger.info("  ✓ QuestionClassifier initialized")

        # Initialize entity recognizer
        self.entity_recognizer = EntityRecognizer()
        logger.info("  ✓ EntityRecognizer initialized")

        # Try to load hybrid embeddings for fallback mode
        self.hybrid_embedder = None
        if fallback_mode != FallbackMode.NONE and HYBRID_EMBEDDINGS_AVAILABLE:
            self.hybrid_embedder = self._load_hybrid_embedder()

        # Initialize Kuzu inverted index
        self.root_index = KuzuInvertedIndex(
            index_path=self.index_path,
            hybrid_embedder=self.hybrid_embedder,
            fallback_mode=fallback_mode,
        )

        if self.root_index.total_docs > 0:
            logger.info(f"  ✓ KuzuInvertedIndex: {self.root_index.total_docs:,} docs, "
                       f"{self.root_index.total_roots:,} roots")
        else:
            logger.warning(f"  ! KuzuInvertedIndex not loaded from {self.index_path}")

        logger.info("AST-aware retriever initialized")

    def _load_hybrid_embedder(self) -> Optional['HybridEmbeddings']:
        """Load hybrid embeddings for fallback mode."""
        try:
            root_model = Path("models/root_embeddings/best_model.pt")
            topical_model = Path("models/topical_embeddings/best_model.pt")

            if root_model.exists() and topical_model.exists():
                embedder = HybridEmbeddings.from_checkpoints(
                    linguistic_checkpoint=root_model,
                    topical_checkpoint=topical_model,
                    pad_missing=True,
                    default_mode='hybrid'
                )
                logger.info("  ✓ HybridEmbeddings loaded (for fallback)")
                return embedder
            elif root_model.exists():
                from klareco.embeddings.linguistic_embeddings import LinguisticEmbeddings
                embedder = LinguisticEmbeddings.from_checkpoint(root_model)
                logger.info("  ✓ LinguisticEmbeddings loaded (for fallback)")
                return embedder
        except Exception as e:
            logger.warning(f"  Failed to load embeddings: {e}")
        return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        fallback_mode: Optional[FallbackMode] = None,
    ) -> List[Tuple[float, Dict, RetrievalStats]]:
        """
        Search for relevant documents using AST-aware root-based retrieval.

        Args:
            query: Query string (Esperanto)
            top_k: Number of results to return
            fallback_mode: Override default fallback mode for this search

        Returns:
            List of (score, document, stats) tuples sorted by relevance
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Classify question for grammar-aware scoring
        classification = self.question_classifier.classify(query, query_ast)
        question_type = classification['question_type']

        logger.info(f"Query: \"{query}\"")
        logger.info(f"  Question type: {question_type.value}")

        # Search using root inverted index
        mode = fallback_mode if fallback_mode is not None else self.root_index.fallback_mode
        results, stats = self.root_index.search(
            query_ast=query_ast,
            max_results=top_k,
            fallback_mode=mode,
        )

        # Log stats
        logger.info(f"  Roots found: {len(stats.roots_found_in_index)}")
        if stats.roots_not_found:
            logger.info(f"  Roots not found: {stats.roots_not_found}")
        if stats.graph_expansions:
            logger.info(f"  Graph expansions: {stats.graph_expansions}")
        if stats.embedding_synonyms:
            logger.info(f"  Synonyms (Embedding): {stats.embedding_synonyms[:3]}...")
        logger.info(f"  Results: {len(results)}")

        # Convert to expected format (score, doc, stats)
        output = []
        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if doc:
                # Add doc_id to the document for benchmarking/evaluation
                doc['doc_id'] = result.doc_id
                output.append((result.score, doc, stats))

        return output

    def _get_grammar_hints(
        self,
        question_type: QuestionType,
        query_ast: Dict,
    ) -> Dict[str, str]:
        """
        Get grammar hints based on question type for scoring.

        Different question types expect answers with different grammatical features:
        - WHO: Look for subject role (nominative)
        - WHEN: Look for temporal markers (pasinteco, prezenco, futuro)
        - WHERE: Look for locative expressions
        """
        hints = {}

        # Role hints based on question type
        if question_type == QuestionType.WHO:
            hints['preferred_role'] = 'subjekto'
        elif question_type == QuestionType.WHAT:
            # For "Kio estas X?" - look for predicate nominative
            verb = query_ast.get('verbo', {})
            if verb.get('radiko') == 'est':
                hints['preferred_role'] = 'subjekto'  # Definition questions
            else:
                hints['preferred_role'] = 'objekto'
        elif question_type == QuestionType.WHERE:
            hints['preferred_role'] = 'aliaj'  # Location in modifiers
        elif question_type == QuestionType.WHEN:
            hints['preferred_role'] = 'aliaj'  # Time in modifiers
            # Also look for tense matches
            verb = query_ast.get('verbo', {})
            if verb.get('tempo'):
                hints['tempo'] = verb['tempo']

        return hints

    def search_simple(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """
        Simplified search returning just (score, document) tuples.

        Convenience wrapper around search() that drops stats.
        """
        results = self.search(query, top_k)
        return [(score, doc) for score, doc, _ in results]

    def explain_retrieval(self, query: str, doc_id: int) -> Dict:
        """
        Explain why a document was retrieved for a query.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            Explanation dict with roots, synonyms, and scoring details
        """
        # Parse query
        query_ast = parse(query)

        # Classify
        classification = self.question_classifier.classify(query, query_ast)

        # Get document
        doc = self.root_index.get_document(doc_id)
        if not doc:
            return {'error': f'Document {doc_id} not found'}

        # Extract query roots
        query_roots = self._extract_roots(query_ast)

        # Get synonyms for each root (using Kuzu graph traversal)
        synonyms = {}
        for root in query_roots:
            syns = self.root_index.get_synonyms_transitive(root, max_hops=2)
            if syns:
                synonyms[root] = list(syns)

        # Check which roots are in the document
        doc_text = doc.get('text', '').lower()
        roots_in_doc = [r for r in query_roots if r in doc_text]
        synonyms_in_doc = []
        for root, syns in synonyms.items():
            for syn in syns:
                if syn in doc_text:
                    synonyms_in_doc.append(f"{root}→{syn}")

        # Add hypernym chains for query roots
        hypernym_chains = {}
        for root in query_roots:
            chain = self.root_index.get_hypernym_chain(root, max_depth=3)
            if chain:
                hypernym_chains[root] = chain

        result = {
            'query': query,
            'document': doc.get('text', ''),
            'doc_id': doc_id,
            'question_type': classification['question_type'].value,
            'query_roots': query_roots,
            'roots_found_in_doc': roots_in_doc,
            'synonyms': synonyms,
            'synonyms_found_in_doc': synonyms_in_doc,
        }

        if hypernym_chains:
            result['hypernym_chains'] = hypernym_chains

        return result

    def get_sentence_context(
        self,
        doc_id: int,
        window: int = 2,
    ) -> List[Dict]:
        """
        Get surrounding sentences for a document/sentence.

        Args:
            doc_id: Sentence/document ID
            window: Number of sentences before/after to retrieve

        Returns:
            List of context sentences with distance info
        """
        return self.root_index.get_sentence_context(doc_id, window)

    def search_by_role(
        self,
        verb: Optional[str] = None,
        subj: Optional[str] = None,
        obj: Optional[str] = None,
        expand_synonyms: bool = True,
        max_results: int = 100,
    ) -> List[Tuple[int, str]]:
        """
        Search for sentences matching specific grammatical role patterns.

        Example:
            search_by_role(verb="fond", obj="esperant")
            → Find sentences where "fond" is the VERB and "esperant" is the OBJECT

        Args:
            verb: Root that should appear as verb
            subj: Root that should appear as subject
            obj: Root that should appear as object
            expand_synonyms: Whether to expand roots with synonyms
            max_results: Maximum results to return

        Returns:
            List of (doc_id, text) tuples
        """
        return self.root_index.search_by_role(
            verb=verb,
            subj=subj,
            obj=obj,
            expand_synonyms=expand_synonyms,
            max_results=max_results,
        )

    def get_hypernym_chain(self, root: str, max_depth: int = 5) -> List[str]:
        """
        Get hypernym chain for a root word.

        Example: "hundo" → ["besto", "vivaĵo", "aĵo"]

        Args:
            root: Root word to look up
            max_depth: Maximum chain depth

        Returns:
            List of hypernyms from specific to general
        """
        return self.root_index.get_hypernym_chain(root, max_depth)

    def get_synonyms_transitive(self, root: str, max_hops: int = 2) -> set:
        """
        Get transitive synonyms (synonyms of synonyms).

        Args:
            root: Root word to look up
            max_hops: Maximum synonym chain length

        Returns:
            Set of transitive synonyms
        """
        return self.root_index.get_synonyms_transitive(root, max_hops)

    def _extract_roots(self, ast: Dict) -> List[str]:
        """Extract content word roots from AST."""
        roots = []
        skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

        def extract(node):
            if not node or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco', '')
                if vortspeco not in skip_vortspeco:
                    root = node.get('radiko', '')
                    if root and len(root) >= 2:
                        roots.append(root.lower())
            elif node.get('tipo') == 'vortgrupo':
                extract(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract(p)
            elif node.get('tipo') == 'frazo':
                extract(node.get('subjekto'))
                extract(node.get('verbo'))
                extract(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract(a)

        extract(ast)
        return roots

    def close(self):
        """Clean up resources."""
        if self.root_index:
            self.root_index.close()

    def __del__(self):
        self.close()
