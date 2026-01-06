"""
Solution 1: Memory-mapped slot-based retriever.

Uses memory-mapped numpy arrays and lazy loading to reduce RAM usage.
Trades speed for memory efficiency.

Memory optimizations:
- P0: Lazy metadata loading with offset index (saves 36.6 GB for 4.2M docs)
- Mmap for embeddings (disk-backed)
"""

import json
import logging
import mmap
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)


class MemoryMappedSlotRetriever:
    """Slot-based retriever using memory-mapped files for embeddings."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 1000,
    ):
        """
        Initialize memory-mapped retriever.

        Args:
            index_path: Path to slot index directory
            indexer: SlotBasedIndexer instance for query embedding
            slot_weights: Weights for each slot (default: equal)
            batch_size: Number of documents to process per batch
        """
        self.index_path = Path(index_path)
        self.indexer = indexer
        self.batch_size = batch_size

        # Default slot weights
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }

        # Load index metadata and create memory maps
        self._load_index()

    def _load_index(self):
        """Load index and create memory-mapped arrays.

        P0: Uses lazy metadata loading instead of loading all 4.2M docs.
        Only builds offset index (~70 MB) instead of full metadata (36.6 GB).
        """
        logger.info(f"Loading memory-mapped index from {self.index_path}")

        self.index_file = self.index_path / "slot_index.jsonl"

        # P0: Build offset index for lazy metadata loading
        logger.info("  Building document offset index for lazy metadata loading...")
        self._build_offset_index()

        self.num_docs = len(self.doc_offsets)
        self.embedding_dim = self.indexer.embedding_dim

        logger.info(f"  Built offset index for {self.num_docs:,} documents")

        # Create memory-mapped arrays for embeddings
        mmap_dir = self.index_path / "mmap"
        mmap_dir.mkdir(exist_ok=True)

        # Check if mmap files exist
        mmap_files_exist = all([
            (mmap_dir / f"{slot}.npy").exists()
            for slot in ['SUBJ', 'VERB', 'OBJ', 'full']
        ])

        # P1: Check if norm files exist
        norm_files_exist = all([
            (mmap_dir / f"{slot}_norms.npy").exists()
            for slot in ['SUBJ', 'VERB', 'OBJ', 'full']
        ])

        if not mmap_files_exist or not norm_files_exist:
            logger.info("  Creating memory-mapped arrays...")
            self._create_mmap_arrays(self.index_file, mmap_dir)

        # Load memory-mapped arrays
        self.embeddings = {}
        self.norms = {}  # P1: Pre-computed norms
        for slot in ['SUBJ', 'VERB', 'OBJ', 'full']:
            arr = np.load(mmap_dir / f"{slot}.npy", mmap_mode='r')
            self.embeddings[slot] = arr

            # P1: Load pre-computed norms
            norms = np.load(mmap_dir / f"{slot}_norms.npy", mmap_mode='r')
            self.norms[slot] = norms

            logger.info(f"    {slot}: {arr.shape}")

    def _build_offset_index(self):
        """Build byte offset index for O(1) document lookup.

        P0: Enables lazy loading instead of loading all 4.2M docs into memory.
        """
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

    def _get_metadata(self, doc_idx: int) -> Dict:
        """Load single document metadata by index (lazy loading).

        P0: Only loads the specific document needed (~1ms disk I/O).

        Args:
            doc_idx: Document index (0-based)

        Returns:
            Metadata dict with 'text', 'features', 'source'
        """
        if doc_idx < 0 or doc_idx >= len(self.doc_offsets):
            raise IndexError(f"Document index {doc_idx} out of range [0, {len(self.doc_offsets)})")

        with open(self.index_file, 'rb') as f:
            f.seek(self.doc_offsets[doc_idx])
            line = f.readline()
            doc = json.loads(line)

            return {
                'text': doc['text'],
                'features': doc['features'],
                'source': doc.get('source', {}),
            }

    def _create_mmap_arrays(self, index_file: Path, mmap_dir: Path):
        """Create memory-mapped arrays from index file.

        P1: Pre-computes and stores norms for 20% speedup.
        """

        # Allocate arrays (fill with NaN for missing slots)
        slot_arrays = {
            'SUBJ': np.full((self.num_docs, self.embedding_dim), np.nan, dtype=np.float32),
            'VERB': np.full((self.num_docs, self.embedding_dim), np.nan, dtype=np.float32),
            'OBJ': np.full((self.num_docs, self.embedding_dim), np.nan, dtype=np.float32),
            'full': np.zeros((self.num_docs, self.embedding_dim), dtype=np.float32),
        }

        # P1: Pre-allocate norm arrays
        norm_arrays = {
            'SUBJ': np.full(self.num_docs, np.nan, dtype=np.float32),
            'VERB': np.full(self.num_docs, np.nan, dtype=np.float32),
            'OBJ': np.full(self.num_docs, np.nan, dtype=np.float32),
            'full': np.zeros(self.num_docs, dtype=np.float32),
        }

        # Fill arrays
        with open(index_file) as f:
            for i, line in enumerate(f):
                doc = json.loads(line)

                for slot in ['SUBJ', 'VERB', 'OBJ']:
                    if doc['slots'][slot] is not None:
                        emb = np.array(doc['slots'][slot], dtype=np.float32)
                        slot_arrays[slot][i] = emb
                        norm_arrays[slot][i] = np.linalg.norm(emb)

                full_emb = np.array(doc['full_embedding'], dtype=np.float32)
                slot_arrays['full'][i] = full_emb
                norm_arrays['full'][i] = np.linalg.norm(full_emb)

                if (i + 1) % 10000 == 0:
                    logger.info(f"      Processed {i+1:,} documents")

        # Save as memory-mapped files
        for slot, arr in slot_arrays.items():
            np.save(mmap_dir / f"{slot}.npy", arr)

        # P1: Save pre-computed norms
        for slot, norms in norm_arrays.items():
            np.save(mmap_dir / f"{slot}_norms.npy", norms)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0 or np.isnan(norm_a) or np.isnan(norm_b):
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def slot_similarity_batch(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        batch_start: int,
        batch_end: int,
        is_question: bool = False,
    ) -> np.ndarray:
        """
        Compute slot similarity for a batch of documents.

        P1: Vectorized implementation using numpy batch operations (2-3x faster).
        P1: Uses pre-computed norms instead of recomputing (20% speedup).

        Args:
            query_slots: Query slot embeddings
            batch_start: Batch start index
            batch_end: Batch end index
            is_question: Whether query is a question (affects partial bonus)

        Returns:
            Array of similarity scores for documents [batch_start:batch_end]
        """
        batch_size = batch_end - batch_start
        # Get actual batch size from first slot to handle edge cases
        actual_batch_size = len(self.embeddings['SUBJ'][batch_start:batch_end])
        scores = np.zeros(actual_batch_size, dtype=np.float32)
        matched_counts = np.zeros(actual_batch_size, dtype=np.float32)

        # Bug #2 fix: Higher partial bonus for questions
        partial_bonus = 0.8 if is_question else 0.5

        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)

            # Get batch of document embeddings for this slot
            doc_embs = self.embeddings[slot][batch_start:batch_end]  # (batch_size, dim)

            # P1: Get pre-computed norms instead of computing
            doc_norms = self.norms[slot][batch_start:batch_end]  # (batch_size,)

            # Check which docs have this slot (not NaN)
            has_slot = ~np.isnan(doc_norms)  # (batch_size,)

            if query_emb is not None:
                # Vectorized cosine similarity: dot(query, docs) / (norm(query) * norm(docs))
                query_norm = np.linalg.norm(query_emb)
                if query_norm > 0:
                    # Dot products for all docs in batch
                    dots = np.dot(doc_embs, query_emb)  # (batch_size,)

                    # Cosine similarities (vectorized, using pre-computed norms)
                    sims = np.zeros(actual_batch_size, dtype=np.float32)
                    valid = (doc_norms > 0) & has_slot & ~np.isnan(dots)
                    sims[valid] = dots[valid] / (query_norm * doc_norms[valid])

                    # Add weighted similarities
                    scores += weight * sims
                    matched_counts[has_slot] += 1
            else:
                # Query missing this slot: partial match bonus
                scores[has_slot] += weight * partial_bonus
                matched_counts[has_slot] += 1

        # Normalize by matched slots (avoid division by zero)
        valid_matches = matched_counts > 0
        scores[valid_matches] /= matched_counts[valid_matches]

        return scores

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
        rerank_top_n: int = 100,
        slot_weight: float = 0.6,
        full_weight: float = 0.4,
    ) -> List[Tuple[float, Dict]]:
        """
        Two-stage retrieval with batched processing.

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

        # Stage 1: Slot-based filtering (batched)
        # WARNING: This still processes ALL 4.2M docs (12.6M similarity ops)
        # See issue #208 for FAISS pre-filtering optimization
        logger.debug(f"Stage 1: Batched slot filtering ({self.num_docs:,} docs)")

        all_scores = []
        for batch_start in range(0, self.num_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_docs)

            # Compute slot similarities for batch
            batch_scores = self.slot_similarity_batch(query_slots, batch_start, batch_end, is_question=is_question)

            # Store raw scores without feature bonus (defer metadata loading)
            for i, doc_idx in enumerate(range(batch_start, batch_end)):
                all_scores.append((batch_scores[i], doc_idx))

        # Sort and take top-N for reranking
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates_pre_feature = all_scores[:rerank_top_n * 2]  # Get 2x for feature filtering

        # Apply feature bonuses only to top candidates
        top_candidates = []
        for slot_score, doc_idx in top_candidates_pre_feature:
            # P0: Lazy load metadata only for top candidates
            metadata = self._get_metadata(doc_idx)
            feature_bonus = self.feature_similarity(query_features, metadata['features'])
            final_slot_score = slot_score * feature_bonus
            top_candidates.append((final_slot_score, doc_idx))

        # Re-sort after feature bonuses
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = top_candidates[:rerank_top_n]

        logger.debug(f"  Top slot score: {top_candidates[0][0]:.3f}")

        # Stage 2: Full embedding reranking
        logger.debug(f"Stage 2: Full embedding reranking")

        final_results = []
        for slot_score, doc_idx in top_candidates:
            # Get full embedding
            full_emb = self.embeddings['full'][doc_idx]
            full_sim = self.cosine_similarity(query_full_emb, full_emb)

            # Combine scores
            final_score = slot_weight * slot_score + full_weight * full_sim

            # P0: Lazy load metadata only for top candidates
            metadata = self._get_metadata(doc_idx)
            doc = {
                'text': metadata['text'],
                'features': metadata['features'],
                'source': metadata['source'],
            }

            final_results.append((final_score, doc))

        # Sort by final score
        final_results.sort(key=lambda x: x[0], reverse=True)

        return final_results[:top_k]

    def explain_match(
        self,
        query: str,
        doc: Dict,
    ) -> Dict:
        """Explain why a document matched the query."""
        # Find document index (linear search through lazy loading)
        # Note: This is slow for explain_match, but rarely used
        doc_idx = None
        for i in range(self.num_docs):
            metadata = self._get_metadata(i)
            if metadata['text'] == doc['text']:
                doc_idx = i
                break

        if doc_idx is None:
            return {'error': 'Document not found'}

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
            doc_emb = self.embeddings[slot][doc_idx]
            doc_has_slot = not np.any(np.isnan(doc_emb))

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
