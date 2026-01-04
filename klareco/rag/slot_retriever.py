"""
Slot-based retriever for AST-aware semantic search.

Two-stage retrieval:
1. Fast slot-based filtering (structural matching)
2. Full embedding reranking (semantic matching)

This enables partial query support and role-aware matching.

Memory optimization:
- P0: Lazy document loading with offset index (saves 36.6 GB for 4.2M docs)
- For small indexes (<100K), loads all docs for speed
- For large indexes (>100K), uses lazy loading
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)


class SlotBasedRetriever:
    """Retrieve documents using slot-based matching."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize retriever.

        Args:
            index_path: Path to slot_index.jsonl
            indexer: SlotBasedIndexer instance for query embedding
            slot_weights: Weights for each slot (default: equal)
        """
        self.index_path = index_path
        self.indexer = indexer

        # Default slot weights (can be tuned)
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }

        # Load index (lazy or eager based on size)
        self.documents = []
        self.doc_offsets = None
        self.use_lazy_loading = False
        self._load_index()

    def _load_index(self):
        """Load slot-based index from file.

        P0: For small indexes (<100K), loads all docs for speed.
        For large indexes (>100K), uses lazy loading to save memory.
        """
        logger.info(f"Loading slot index from {self.index_path}")

        # Count documents to decide loading strategy
        with open(self.index_path) as f:
            num_docs = sum(1 for _ in f)

        if num_docs > 100000:
            # Large index: use lazy loading
            logger.info(f"  Large index ({num_docs:,} docs) - using lazy loading")
            self.use_lazy_loading = True
            self._build_offset_index()
            logger.info(f"  Built offset index for {num_docs:,} documents")
        else:
            # Small index: load all for speed
            logger.info(f"  Small index ({num_docs:,} docs) - loading all documents")
            with open(self.index_path) as f:
                for line in f:
                    doc = json.loads(line)
                    # Convert slot embeddings back to numpy
                    doc['slots_np'] = {
                        k: np.array(v) if v is not None else None
                        for k, v in doc['slots'].items()
                    }
                    doc['full_embedding_np'] = np.array(doc['full_embedding'])
                    self.documents.append(doc)
            logger.info(f"  Loaded {len(self.documents):,} documents")

    def _build_offset_index(self):
        """Build byte offset index for O(1) document lookup.

        P0: Enables lazy loading instead of loading all 4.2M docs into memory.
        """
        self.doc_offsets = []
        with open(self.index_path, 'rb') as f:
            offset = 0
            doc_count = 0
            for line in f:
                self.doc_offsets.append(offset)
                offset += len(line)
                doc_count += 1

                # Progress logging every 500K docs
                if doc_count % 500000 == 0:
                    logger.info(f"    Indexed {doc_count:,} document offsets...")

    def _get_document(self, doc_id: int) -> Dict:
        """Load single document by ID (lazy loading mode only).

        P0: Only loads the specific document needed (~1ms disk I/O).
        """
        if not self.use_lazy_loading:
            return self.documents[doc_id]

        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            raise IndexError(f"Document ID {doc_id} out of range [0, {len(self.doc_offsets)})")

        with open(self.index_path, 'rb') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            doc = json.loads(line)

            # Add numpy conversions
            doc['slots_np'] = {
                k: np.array(v, dtype=np.float32) if v is not None else None
                for k, v in doc['slots'].items()
            }
            doc['full_embedding_np'] = np.array(doc['full_embedding'], dtype=np.float32)

            return doc

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc_slots: Dict[str, Optional[np.ndarray]],
        is_question: bool = False,
    ) -> float:
        """
        Compute weighted slot similarity.

        Handles partial queries: missing slots get a partial match bonus.

        Args:
            query_slots: Query slot embeddings
            doc_slots: Document slot embeddings
            is_question: Whether query is a question (affects partial bonus)
        """
        score = 0.0
        matched_slots = 0

        # Bug #2 fix: Higher partial bonus for questions
        # Questions with missing slots likely indicate the answer we seek
        partial_bonus = 0.8 if is_question else 0.5

        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)
            doc_emb = doc_slots.get(slot)

            if query_emb is not None and doc_emb is not None:
                # Both have this slot: compute similarity
                sim = self.cosine_similarity(query_emb, doc_emb)
                score += weight * sim
                matched_slots += 1

            elif query_emb is None and doc_emb is not None:
                # Query missing this slot: partial match bonus
                # (e.g., "Kiu X?" doesn't specify subject)
                score += weight * partial_bonus
                matched_slots += 1

            # If doc missing slot but query has it: no score (mismatch)

        # Normalize by number of matched slots
        if matched_slots > 0:
            return score / matched_slots
        else:
            return 0.0

    def feature_similarity(
        self,
        query_features: Dict,
        doc_features: Dict,
    ) -> float:
        """
        Compute feature matching bonus.

        Returns multiplicative factor based on grammar matches.
        """
        bonus = 1.0

        # Negation match
        if query_features.get('negita') == doc_features.get('negita'):
            bonus *= 1.1

        # Tense match (less important)
        if query_features.get('tempo') == doc_features.get('tempo'):
            bonus *= 1.05

        # Sentence type match
        if query_features.get('fraztipo') == doc_features.get('fraztipo'):
            bonus *= 1.05

        return bonus

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 100,
        slot_weight: float = 0.6,
        full_weight: float = 0.4,
    ) -> List[Tuple[float, Dict]]:
        """
        Two-stage retrieval: slot filtering + full embedding reranking.

        Args:
            query: Query text in Esperanto
            top_k: Number of results to return
            rerank_top_n: Number of candidates for stage 2 reranking
            slot_weight: Weight for slot similarity in final score
            full_weight: Weight for full embedding in final score

        Returns:
            List of (score, document) tuples sorted by score descending
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Extract query slots and features
        query_slots = self.indexer.extract_slots(query_ast)
        query_features = self.indexer.extract_features(query_ast)

        # Detect if query is a question (Bug #2 fix)
        is_question = query.strip().endswith('?') or query_ast.get('fraztipo') == 'demando'

        # Compute query full embedding
        query_word_embs = [emb for emb in query_slots.values() if emb is not None]
        if not query_word_embs:
            logger.warning(f"No content words in query: {query}")
            return []

        query_full_emb = np.mean(query_word_embs, axis=0)
        query_full_emb = query_full_emb / np.linalg.norm(query_full_emb)

        # Stage 1: Slot-based filtering
        num_docs = len(self.doc_offsets) if self.use_lazy_loading else len(self.documents)
        logger.info(f"Stage 1: Slot-based filtering ({num_docs:,} docs)")

        candidates = []
        if self.use_lazy_loading:
            # P0: Lazy loading mode - load each doc on demand
            for doc_id in range(len(self.doc_offsets)):
                doc = self._get_document(doc_id)

                # Compute slot similarity
                slot_sim = self.slot_similarity(query_slots, doc['slots_np'], is_question=is_question)

                # Apply feature matching bonus
                feature_bonus = self.feature_similarity(query_features, doc['features'])
                slot_score = slot_sim * feature_bonus

                candidates.append((slot_score, doc))
        else:
            # Eager mode - all docs already loaded
            for doc in self.documents:
                # Compute slot similarity
                slot_sim = self.slot_similarity(query_slots, doc['slots_np'], is_question=is_question)

                # Apply feature matching bonus
                feature_bonus = self.feature_similarity(query_features, doc['features'])
                slot_score = slot_sim * feature_bonus

                candidates.append((slot_score, doc))

        # Sort by slot score and take top-N for reranking
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:rerank_top_n]

        logger.info(f"  Top slot score: {top_candidates[0][0]:.3f}")
        logger.info(f"  {len(top_candidates)} candidates for stage 2")

        # Stage 2: Full embedding reranking
        logger.info(f"Stage 2: Full embedding reranking")

        final_results = []
        for slot_score, doc in top_candidates:
            # Compute full embedding similarity
            full_sim = self.cosine_similarity(query_full_emb, doc['full_embedding_np'])

            # Combine scores
            final_score = slot_weight * slot_score + full_weight * full_sim

            final_results.append((final_score, doc))

        # Sort by final score
        final_results.sort(key=lambda x: x[0], reverse=True)

        logger.info(f"  Top final score: {final_results[0][0]:.3f}")

        return final_results[:top_k]

    def explain_match(
        self,
        query: str,
        doc: Dict,
    ) -> Dict:
        """
        Explain why a document matched the query.

        Shows which slots matched and their similarity scores.
        """
        # Parse query
        query_ast = parse(query)
        query_slots = self.indexer.extract_slots(query_ast)

        # Compute slot similarities
        explanation = {
            'query': query,
            'document': doc['text'],
            'slot_matches': {},
        }

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            query_emb = query_slots.get(slot)
            doc_emb = doc['slots_np'].get(slot)

            if query_emb is not None and doc_emb is not None:
                sim = self.cosine_similarity(query_emb, doc_emb)
                explanation['slot_matches'][slot] = {
                    'similarity': float(sim),
                    'status': 'matched',
                }
            elif query_emb is None:
                explanation['slot_matches'][slot] = {
                    'status': 'query_missing',
                }
            elif doc_emb is None:
                explanation['slot_matches'][slot] = {
                    'status': 'doc_missing',
                }

        return explanation
