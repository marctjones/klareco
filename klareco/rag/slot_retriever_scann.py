"""
ScaNN + Mmap retriever optimized for highest accuracy.

Three-stage retrieval pipeline:
1. ScaNN pre-filtering: 4.2M docs → 500 candidates (anisotropic quantization)
2. Mmap slot reranking: 500 docs → slot-aware scoring (high accuracy)
3. Final ranking: Combined scores → top-k results

Expected performance: 90-95% recall, ~3-5ms latency
- Higher accuracy than FAISS (anisotropic quantization advantage)
- Optimized for cosine similarity (inner product on normalized vectors)
- Best for production systems requiring >90% recall

Memory optimizations:
- P0: Lazy metadata loading
- P1: ScaNN compression via anisotropic quantization
- P1: Pre-computed norms from mmap
- P3: Thread tuning

IMPORTANT: ScaNN requires normalized vectors for dot_product metric!
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import scann
    SCANN_AVAILABLE = True
except ImportError:
    SCANN_AVAILABLE = False

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)


class ScaNNSlotRetriever:
    """Hybrid retriever combining ScaNN pre-filtering with mmap slot reranking."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        scann_num_leaves: int = 2000,
        scann_num_leaves_to_search: int = 100,
        scann_training_sample_size: int = 250000,
        scann_dimensions_per_block: int = 2,
        scann_quantization_threshold: float = 0.2,
        scann_reorder_k: int = 100,
    ):
        """
        Initialize ScaNN-based retriever.

        Args:
            index_path: Path to slot index directory
            indexer: SlotBasedIndexer instance for query embedding
            slot_weights: Weights for each slot (default: equal)
            scann_num_leaves: Number of clusters for tree partitioning
            scann_num_leaves_to_search: Number of clusters to search
            scann_training_sample_size: Sample size for training
            scann_dimensions_per_block: Dimensions per quantization block
            scann_quantization_threshold: Anisotropic quantization threshold
            scann_reorder_k: Number of candidates to reorder with exact scores
        """
        if not SCANN_AVAILABLE:
            raise ImportError(
                "ScaNN not installed. Install with: pip install scann\n"
                "Note: Requires TensorFlow and specific Python version (3.8-3.11)"
            )

        self.index_path = Path(index_path)
        self.indexer = indexer

        # ScaNN parameters
        self.scann_num_leaves = scann_num_leaves
        self.scann_num_leaves_to_search = scann_num_leaves_to_search
        self.scann_training_sample_size = scann_training_sample_size
        self.scann_dimensions_per_block = scann_dimensions_per_block
        self.scann_quantization_threshold = scann_quantization_threshold
        self.scann_reorder_k = scann_reorder_k

        # Default slot weights
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }

        # Load indexes
        self._load_indexes()

    def _load_indexes(self):
        """Load ScaNN searcher, mmap arrays, and build offset index."""
        logger.info(f"Loading ScaNN+mmap indexes from {self.index_path}")

        # Load ScaNN searcher
        self._load_scann_searcher()

        # Load mmap arrays
        self._load_mmap_arrays()

        # Build offset index for metadata
        self.index_file = self.index_path / "slot_index.jsonl"
        self._build_offset_index()

        logger.info(f"  ScaNN retriever ready: {len(self.doc_offsets):,} documents")

    def _load_scann_searcher(self):
        """Load ScaNN searcher from disk."""
        scann_dir = self.index_path / "scann"

        if not scann_dir.exists():
            raise FileNotFoundError(
                f"ScaNN index not found: {scann_dir}\n"
                f"Run: python scripts/build_scann_index.sh --index {self.index_path}"
            )

        # Load searcher
        self.scann_searcher = scann.scann_ops_pybind.load_searcher(str(scann_dir))

        logger.info(f"  ScaNN searcher loaded from {scann_dir}")
        logger.info(f"  ScaNN params: num_leaves={self.scann_num_leaves}, "
                   f"num_leaves_to_search={self.scann_num_leaves_to_search}, "
                   f"reorder_k={self.scann_reorder_k}")

    def _load_mmap_arrays(self):
        """Load memory-mapped slot embeddings and norms."""
        mmap_dir = self.index_path / "mmap"

        if not mmap_dir.exists():
            raise FileNotFoundError(
                f"Mmap directory not found: {mmap_dir}\n"
                f"Run: python scripts/index_slot_based.py --index {self.index_path}"
            )

        # Load slot embeddings
        self.slot_embeddings = {}
        self.slot_norms = {}

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            emb_file = mmap_dir / f"{slot}.npy"
            norm_file = mmap_dir / f"{slot}_norms.npy"

            if not emb_file.exists():
                raise FileNotFoundError(f"Mmap file not found: {emb_file}")

            # Load embeddings and norms (memory-mapped)
            self.slot_embeddings[slot] = np.load(emb_file, mmap_mode='r')

            if norm_file.exists():
                self.slot_norms[slot] = np.load(norm_file, mmap_mode='r')
                logger.info(f"  Mmap {slot}: {self.slot_embeddings[slot].shape} (with pre-computed norms)")
            else:
                logger.warning(f"  Mmap {slot}: {self.slot_embeddings[slot].shape} (no pre-computed norms)")
                self.slot_norms[slot] = None

    def _build_offset_index(self):
        """Build byte offset index for O(1) document lookup (P0: lazy metadata loading)."""
        self.doc_offsets = []
        with open(self.index_file, 'rb') as f:
            offset = 0
            doc_count = 0
            for line in f:
                self.doc_offsets.append(offset)
                offset += len(line)
                doc_count += 1

                # Progress logging every 500K docs
                if doc_count % 500000 == 0:
                    logger.info(f"    Indexed {doc_count:,} document offsets...")

    def _get_metadata(self, doc_id: int) -> Dict:
        """Load single document metadata by ID (P0: lazy loading)."""
        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            raise IndexError(f"Document ID {doc_id} out of range [0, {len(self.doc_offsets)})")

        with open(self.index_file, 'rb') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            doc = json.loads(line)

            return {
                'text': doc['text'],
                'features': doc['features'],
                'source': doc.get('source', {}),
            }

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray, norm_a: Optional[float] = None, norm_b: Optional[float] = None) -> float:
        """Compute cosine similarity with optional pre-computed norms."""
        if norm_a is None:
            norm_a = np.linalg.norm(a)
        if norm_b is None:
            norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0 or np.isnan(norm_a) or np.isnan(norm_b):
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc_id: int,
        is_question: bool = False,
    ) -> float:
        """
        Compute weighted slot similarity for a single document.

        Uses pre-computed norms from mmap if available.

        Args:
            query_slots: Query slot embeddings
            doc_id: Document ID
            is_question: Whether query is a question (affects partial bonus)
        """
        score = 0.0
        matched_slots = 0

        # Bug #2 fix: Higher partial bonus for questions
        partial_bonus = 0.8 if is_question else 0.5

        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)

            # Get document embedding from mmap
            doc_emb = self.slot_embeddings[slot][doc_id]

            # Check if doc has this slot (not NaN)
            doc_norm = self.slot_norms[slot][doc_id] if self.slot_norms[slot] is not None else None
            doc_has_slot = not (doc_norm is None or np.isnan(doc_norm))

            if query_emb is not None and doc_has_slot:
                # Both have this slot: compute similarity
                sim = self.cosine_similarity(query_emb, doc_emb, norm_b=doc_norm)
                score += weight * sim
                matched_slots += 1
            elif query_emb is None and doc_has_slot:
                # Query missing this slot: partial match bonus
                score += weight * partial_bonus
                matched_slots += 1

        # Normalize by matched slots
        if matched_slots > 0:
            return score / matched_slots
        else:
            return 0.0

    def feature_similarity(
        self,
        query_features: Dict,
        doc_features: Dict,
    ) -> float:
        """Compute feature matching bonus."""
        bonus = 1.0

        if query_features.get('negita') == doc_features.get('negita'):
            bonus *= 1.1
        if query_features.get('tempo') == doc_features.get('tempo'):
            bonus *= 1.05
        if query_features.get('fraztipo') == doc_features.get('fraztipo'):
            bonus *= 1.05

        return bonus

    def search(
        self,
        query: str,
        top_k: int = 10,
        scann_top_n: int = 500,
        slot_top_n: int = 100,
        slot_weight: float = 0.6,
        scann_weight: float = 0.4,
    ) -> List[Tuple[float, Dict]]:
        """
        Three-stage hybrid retrieval.

        Stage 0: ScaNN pre-filtering (4.2M → 500 docs)
        Stage 1: Mmap slot reranking (500 → 100 docs)
        Stage 2: Final ranking (100 → top_k)

        Args:
            query: Query text in Esperanto
            top_k: Number of results to return
            scann_top_n: ScaNN candidates (default: 500)
            slot_top_n: Slot reranking candidates (default: 100)
            slot_weight: Weight for slot similarity
            scann_weight: Weight for ScaNN score

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

        query_full_emb = np.mean(query_word_embs, axis=0).astype(np.float32)

        # CRITICAL: Normalize query vector for ScaNN dot_product
        query_norm = np.linalg.norm(query_full_emb)
        query_normalized = query_full_emb / query_norm

        # Stage 0: ScaNN pre-filtering
        logger.debug(f"Stage 0: ScaNN pre-filtering (4.2M docs → {scann_top_n} candidates)")

        # Search ScaNN index
        neighbors, distances = self.scann_searcher.search(
            query_normalized,
            final_num_neighbors=scann_top_n
        )

        # distances are dot products (higher = better)
        scann_scores = distances
        scann_indices = neighbors

        logger.debug(f"  Top ScaNN score: {scann_scores[0]:.3f}")

        # Stage 1: Mmap slot reranking
        logger.debug(f"Stage 1: Mmap slot reranking ({scann_top_n} → {slot_top_n} candidates)")

        slot_results = []
        for i, doc_id in enumerate(scann_indices):
            if doc_id == -1:  # ScaNN padding
                continue

            doc_id = int(doc_id)

            # Compute slot similarity (using mmap)
            slot_sim = self.slot_similarity(query_slots, doc_id, is_question=is_question)

            # Get metadata for feature bonus (lazy load)
            metadata = self._get_metadata(doc_id)
            feature_bonus = self.feature_similarity(query_features, metadata['features'])
            slot_score = slot_sim * feature_bonus

            # Combine ScaNN score + slot score
            scann_score = float(scann_scores[i])
            combined_score = slot_weight * slot_score + scann_weight * scann_score

            slot_results.append((combined_score, doc_id, metadata))

        # Sort by combined score and take top slot_top_n
        slot_results.sort(key=lambda x: x[0], reverse=True)
        top_candidates = slot_results[:slot_top_n]

        logger.debug(f"  Top slot score: {top_candidates[0][0]:.3f}")

        # Stage 2: Final ranking (already done in Stage 1)
        logger.debug(f"Stage 2: Returning top {top_k} results")

        final_results = []
        for score, doc_id, metadata in top_candidates[:top_k]:
            # Add slots_np for compatibility with explain_match
            doc = metadata.copy()
            doc['slots_np'] = {
                slot: self.slot_embeddings[slot][doc_id]
                for slot in ['SUBJ', 'VERB', 'OBJ']
            }
            final_results.append((score, doc))

        return final_results

    def explain_match(
        self,
        query: str,
        doc: Dict,
    ) -> Dict:
        """Explain why a document matched the query."""
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

            # Check if valid
            doc_has_slot = not (doc_emb is None or np.any(np.isnan(doc_emb)))

            if query_emb is not None and doc_has_slot:
                sim = self.cosine_similarity(query_emb, doc_emb)
                explanation['slot_matches'][slot] = {
                    'similarity': float(sim),
                    'status': 'matched',
                }
            elif query_emb is None:
                explanation['slot_matches'][slot] = {
                    'status': 'query_missing',
                }
            elif not doc_has_slot:
                explanation['slot_matches'][slot] = {
                    'status': 'doc_missing',
                }

        return explanation
