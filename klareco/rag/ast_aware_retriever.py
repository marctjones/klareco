"""
Multi-Strategy AST-Aware Retriever.

Combines all AST-aware components into a unified retrieval system:
1. Question Type Classifier - Understand question intent
2. Entity Recognizer - Extract named entities
3. AST Pattern Matcher - Match structural patterns
4. Semantic Relations - Expand with synonyms

This is a DETERMINISTIC retriever that leverages Klareco's unique advantage:
fully parsed AST annotations for both queries and corpus.

Expected performance: 60-70% accuracy (6x improvement from baseline 10-12%)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from klareco.parser import parse
from klareco.rag.question_classifier import QuestionClassifier, QuestionType, EntityType as QEntityType
from klareco.rag.entity_recognizer import EntityRecognizer, EntityType as EEntityType
from klareco.rag.ast_pattern_matcher import ASTPatternMatcher, MatchResult
from klareco.rag.semantic_db import SemanticRelationDB

logger = logging.getLogger(__name__)


class ASTAwareRetriever:
    """
    Multi-strategy AST-aware retriever.

    Combines deterministic AST analysis with semantic relations to achieve
    high accuracy question answering without learned parameters.
    """

    def __init__(
        self,
        index_path: Path,
        revo_path: Optional[Path] = None,
        use_prefilter: bool = True,
        prefilter_retriever: Optional = None,
    ):
        """
        Initialize AST-aware retriever.

        Args:
            index_path: Path to slot-based index directory
            revo_path: Path to ReVo semantic relations (optional)
            use_prefilter: Whether to use embedding-based pre-filtering (recommended)
            prefilter_retriever: Optional pre-filtering retriever (HNSW, FAISS, etc.)
                                If None and use_prefilter=True, will try to load HNSW
        """
        self.index_path = Path(index_path)
        self.use_prefilter = use_prefilter

        # Initialize components
        logger.info("Initializing AST-aware retrieval components...")

        self.question_classifier = QuestionClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.semantic_db = SemanticRelationDB(revo_path=revo_path)

        # Initialize pattern matcher with semantic DB
        synonym_dict = {
            root: synonyms
            for root, synonyms in self.semantic_db.synonyms.items()
        }
        antonym_dict = {
            root: antonyms
            for root, antonyms in self.semantic_db.antonyms.items()
        }

        self.pattern_matcher = ASTPatternMatcher(
            synonym_db=synonym_dict,
            antonym_db=antonym_dict,
        )

        # Load index metadata
        self._load_index()

        # Initialize pre-filter retriever if requested
        self.prefilter_retriever = None
        if use_prefilter:
            if prefilter_retriever:
                self.prefilter_retriever = prefilter_retriever
                logger.info("  Using provided pre-filter retriever")
            else:
                # Try to load HNSW as default pre-filter
                self._load_prefilter()

        logger.info("AST-aware retriever initialized")

    def _load_index(self):
        """Load slot-based index metadata."""
        index_file = self.index_path / "slot_index.jsonl"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Slot index not found: {index_file}\n"
                f"Run: python scripts/index_slot_based.py --corpus <corpus> --index {self.index_path}"
            )

        # Build document offset index for lazy loading
        self.doc_offsets = []
        with open(index_file, 'rb') as f:
            offset = 0
            for line in f:
                self.doc_offsets.append(offset)
                offset += len(line)

        logger.info(f"Loaded index with {len(self.doc_offsets):,} documents")

    def _load_prefilter(self):
        """Load HNSW pre-filter retriever if available."""
        hnsw_dir = self.index_path / "hnsw"

        if not hnsw_dir.exists():
            logger.warning("  No HNSW index found - will use brute-force search (slow for large corpora)")
            logger.warning(f"  Run: python scripts/build_hnsw_index.sh {self.index_path}")
            return

        try:
            from klareco.rag.slot_indexer import SlotBasedIndexer
            from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever

            # Load indexer for HNSW
            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=self.index_path
            )

            self.prefilter_retriever = HNSWSlotRetriever(self.index_path, indexer)
            logger.info("  ✓ HNSW pre-filter loaded (fast retrieval enabled)")
        except Exception as e:
            logger.warning(f"  Failed to load HNSW pre-filter: {e}")
            logger.warning("  Will use brute-force search (slow for large corpora)")

    def _get_document(self, doc_id: int) -> Dict:
        """Load a document by ID from index."""
        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            raise IndexError(f"Document ID {doc_id} out of range")

        import json

        index_file = self.index_path / "slot_index.jsonl"
        with open(index_file, 'rb') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            return json.loads(line)

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = 'auto',
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Search for relevant documents using AST-aware strategies.

        Args:
            query: Query string (Esperanto)
            top_k: Number of results to return
            strategy: Search strategy:
                - 'auto': Automatically choose based on question type
                - 'pattern': Pure pattern matching
                - 'entity': Entity-focused matching
                - 'hybrid': Combine pattern + entity
            prefilter_n: Number of candidates from pre-filter (default: 500)
                        Increase for better recall at cost of speed

        Returns:
            List of (score, document) tuples sorted by relevance
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Classify question
        classification = self.question_classifier.classify(query, query_ast)
        question_type = classification['question_type']
        entity_type = classification['entity_type']
        target_slots = classification['target_slots']

        logger.info(f"Query classified as {question_type.value}, entity: {entity_type.value}")

        # Extract entities from query
        query_entities = self.entity_recognizer.recognize_entities(query_ast)

        # Choose strategy based on question type
        if strategy == 'auto':
            strategy = self._select_strategy(question_type, query_entities)

        logger.info(f"Using strategy: {strategy}")

        # Execute search with chosen strategy
        if strategy == 'entity':
            return self._search_entity_focused(
                query_ast, query_entities, target_slots, entity_type, top_k, prefilter_n
            )
        elif strategy == 'pattern':
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )
        else:  # hybrid
            return self._search_hybrid(
                query_ast, query_entities, target_slots, entity_type, top_k, prefilter_n
            )

    def _select_strategy(
        self,
        question_type: QuestionType,
        query_entities: List,
    ) -> str:
        """
        Automatically select search strategy based on question type.

        Rules:
        - WHO questions with entities → entity-focused
        - WHERE questions → entity-focused (places)
        - WHEN questions → entity-focused (times)
        - WHAT/HOW/WHY questions → pattern matching
        - Questions with multiple entities → hybrid
        """
        if len(query_entities) >= 2:
            return 'hybrid'

        if question_type == QuestionType.WHO:
            return 'entity'
        elif question_type == QuestionType.WHERE:
            return 'entity'
        elif question_type == QuestionType.WHEN:
            return 'entity'
        else:
            return 'pattern'

    def _reconstruct_query(self, query_ast: Dict) -> str:
        """
        Reconstruct query text from AST for pre-filtering.

        Simple reconstruction that extracts words from AST nodes.
        """
        words = []

        def extract_words(node):
            if not node:
                return

            if isinstance(node, dict):
                # Extract word if present
                if node.get('tipo') == 'vorto' and 'plena_vorto' in node:
                    words.append(node['plena_vorto'])

                # Recurse into structure
                if node.get('tipo') == 'frazo':
                    extract_words(node.get('subjekto'))
                    extract_words(node.get('verbo'))
                    extract_words(node.get('objekto'))
                    for item in node.get('aliaj', []):
                        extract_words(item)
                elif node.get('tipo') == 'vortgrupo':
                    extract_words(node.get('kerno'))
                    for item in node.get('priskriboj', []):
                        extract_words(item)
            elif isinstance(node, list):
                for item in node:
                    extract_words(item)

        extract_words(query_ast)
        return ' '.join(words)

    def _search_pattern_matching(
        self,
        query_ast: Dict,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Pure pattern matching strategy.

        Uses AST pattern matcher to find structurally similar sentences.

        Args:
            query_ast: Parsed query AST
            target_slots: Priority slots from classifier
            entity_type: Expected entity type
            top_k: Final number of results
            prefilter_n: Number of candidates from pre-filter (if available)
        """
        results = []

        # Stage 1: Pre-filter candidates using embeddings (if available)
        if self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            # Use HNSW to get top candidates
            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            # Extract document IDs from pre-filter results
            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Fallback: Brute-force scan (slow!)
            logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
            scan_limit = min(10000, len(self.doc_offsets))
            candidate_docs = []
            for doc_id in range(scan_limit):
                doc = self._get_document(doc_id)
                candidate_docs.append((0.0, doc))  # No pre-filter score

        # Stage 2: AST pattern matching on candidates
        logger.debug(f"  AST pattern matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Match patterns
            match_result = self.pattern_matcher.match(
                query_ast,
                doc_ast,
                target_slots,
                entity_type.value,
            )

            if match_result.score > 0:
                results.append((match_result.score, doc))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def _search_entity_focused(
        self,
        query_ast: Dict,
        query_entities: List,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Entity-focused strategy.

        Prioritizes documents containing the same entities as the query.
        Then uses pattern matching for ranking.
        """
        if not query_entities:
            # Fall back to pattern matching
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )

        # Get entity texts and roots
        query_entity_texts = self.entity_recognizer.get_entity_texts(query_entities)
        query_entity_roots = self.entity_recognizer.get_entity_roots(query_entities)

        results = []

        # Stage 1: Pre-filter candidates using embeddings (if available)
        if self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Fallback: Brute-force scan (slow!)
            logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
            scan_limit = min(10000, len(self.doc_offsets))
            candidate_docs = []
            for doc_id in range(scan_limit):
                doc = self._get_document(doc_id)
                candidate_docs.append((0.0, doc))

        # Stage 2: Entity matching + pattern matching on candidates
        logger.debug(f"  Entity + pattern matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Extract entities from document
            doc_entities = self.entity_recognizer.recognize_entities(doc_ast)
            doc_entity_texts = self.entity_recognizer.get_entity_texts(doc_entities)
            doc_entity_roots = self.entity_recognizer.get_entity_roots(doc_entities)

            # Check for entity overlap
            text_overlap = len(query_entity_texts & doc_entity_texts)
            root_overlap = len(query_entity_roots & doc_entity_roots)

            entity_score = (text_overlap * 0.6 + root_overlap * 0.4)

            if entity_score > 0:
                # Also compute pattern match score
                pattern_result = self.pattern_matcher.match(
                    query_ast, doc_ast, target_slots, entity_type.value
                )

                # Combine scores: entity match + pattern match
                combined_score = entity_score * 0.6 + pattern_result.score * 0.4

                results.append((combined_score, doc))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def _search_hybrid(
        self,
        query_ast: Dict,
        query_entities: List,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Hybrid strategy combining entity and pattern matching.

        Balances entity matching with structural pattern matching.
        """
        # For now, same as entity-focused but with equal weighting
        if not query_entities:
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )

        query_entity_texts = self.entity_recognizer.get_entity_texts(query_entities)
        query_entity_roots = self.entity_recognizer.get_entity_roots(query_entities)

        results = []

        # Stage 1: Pre-filter candidates using embeddings (if available)
        if self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Fallback: Brute-force scan (slow!)
            logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
            scan_limit = min(10000, len(self.doc_offsets))
            candidate_docs = []
            for doc_id in range(scan_limit):
                doc = self._get_document(doc_id)
                candidate_docs.append((0.0, doc))

        # Stage 2: Entity + pattern matching on candidates
        logger.debug(f"  Hybrid matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Entity matching
            doc_entities = self.entity_recognizer.recognize_entities(doc_ast)
            doc_entity_texts = self.entity_recognizer.get_entity_texts(doc_entities)
            doc_entity_roots = self.entity_recognizer.get_entity_roots(doc_entities)

            text_overlap = len(query_entity_texts & doc_entity_texts)
            root_overlap = len(query_entity_roots & doc_entity_roots)

            entity_score = (text_overlap * 0.6 + root_overlap * 0.4)

            # Pattern matching
            pattern_result = self.pattern_matcher.match(
                query_ast, doc_ast, target_slots, entity_type.value
            )

            # Equal weighting for hybrid
            combined_score = entity_score * 0.5 + pattern_result.score * 0.5

            if combined_score > 0:
                results.append((combined_score, doc))

        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def explain_retrieval(self, query: str, doc: Dict) -> Dict:
        """
        Explain why a document was retrieved for a query.

        Args:
            query: Query string
            doc: Document dict

        Returns:
            Explanation dict with classification, entities, and pattern match
        """
        # Parse query
        query_ast = parse(query)

        # Classify
        classification = self.question_classifier.classify(query, query_ast)

        # Extract entities
        query_entities = self.entity_recognizer.recognize_entities(query_ast)
        doc_entities = self.entity_recognizer.recognize_entities(doc['ast'])

        # Pattern match
        pattern_result = self.pattern_matcher.match(
            query_ast,
            doc['ast'],
            classification['target_slots'],
            classification['entity_type'].value,
        )

        return {
            'query': query,
            'document': doc['text'],
            'classification': {
                'question_type': classification['question_type'].value,
                'entity_type': classification['entity_type'].value,
                'focus': classification['focus'],
                'target_slots': classification['target_slots'],
            },
            'query_entities': [
                {'text': e.text, 'type': e.entity_type.value, 'slot': e.slot}
                for e in query_entities
            ],
            'doc_entities': [
                {'text': e.text, 'type': e.entity_type.value, 'slot': e.slot}
                for e in doc_entities
            ],
            'pattern_match': {
                'score': pattern_result.score,
                'matched_slots': list(pattern_result.matched_slots),
                'transformations': pattern_result.transformations,
                'explanation': pattern_result.explanation,
            },
        }
