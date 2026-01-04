"""
Solution 2: FAISS pre-filtering + slot reranking.

Uses FAISS index for fast approximate nearest neighbor search on full embeddings,
then applies slot-based scoring on top candidates.

Memory optimizations:
- P0: Lazy document loading with offset index (saves 36.6 GB)
- P1: FAISS memory mapping (saves 2 GB)
- P3: Thread tuning for 20% speed gain
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

# P3: Thread tuning - set before importing faiss
# Use physical cores only (not hyperthreads) for 20% speed gain
physical_cores = psutil.cpu_count(logical=False) or 8
os.environ['OMP_NUM_THREADS'] = str(physical_cores)

import faiss
import numpy as np

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)
logger.info(f"FAISS thread count set to {physical_cores} (physical cores)")


class FAISSSlotRetriever:
    """Slot-based retriever using FAISS for pre-filtering."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize FAISS-based retriever.

        Args:
            index_path: Path to slot index directory
            indexer: SlotBasedIndexer instance for query embedding
            slot_weights: Weights for each slot (default: equal)
            use_gpu: Use GPU for FAISS (requires faiss-gpu)
        """
        self.index_path = Path(index_path)
        self.indexer = indexer
        self.use_gpu = use_gpu

        # Default slot weights
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }

        # Load index
        self._load_index()

    def _load_index(self):
        """Load FAISS index and metadata."""
        logger.info(f"Loading FAISS index from {self.index_path}")

        faiss_dir = self.index_path / "faiss"

        # Check if FAISS index exists
        faiss_index_file = faiss_dir / "full_embeddings.index"
        if not faiss_index_file.exists():
            logger.info("  FAISS index not found, creating...")
            self._create_faiss_index(faiss_dir)

        # P1: Load FAISS index with memory mapping to save RAM
        self.faiss_index = faiss.read_index(str(faiss_index_file), faiss.IO_FLAG_MMAP)

        # Set nprobe for IVF indexes
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = 32  # Good balance for 4M docs
            logger.info(f"  IVF index with nprobe={self.faiss_index.nprobe}")

        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("  Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        logger.info(f"  FAISS index: {self.faiss_index.ntotal:,} vectors (memory-mapped)")

        # P0: Build offset index for lazy document loading
        # Instead of loading all 4.2M docs (36.6 GB), build offset index (~70 MB)
        logger.info("  Building document offset index for lazy loading...")
        self.index_file = self.index_path / "slot_index.jsonl"
        self._build_offset_index()
        logger.info(f"  Built offset index for {len(self.doc_offsets):,} documents")

    def _build_offset_index(self):
        """Build byte offset index for O(1) document lookup.

        P0: This allows lazy loading of documents instead of loading all 4.2M
        into memory (saving 36.6 GB RAM).
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

    def _get_document(self, doc_id: int) -> Dict:
        """Load single document by ID.

        P0: Lazy loading - only loads the specific document needed.
        Trade-off: ~1ms disk I/O per document vs 36.6 GB RAM savings.

        Args:
            doc_id: Document index (0-based)

        Returns:
            Document dict with 'slots_np' added
        """
        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            raise IndexError(f"Document ID {doc_id} out of range [0, {len(self.doc_offsets)})")

        with open(self.index_file, 'rb') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            doc = json.loads(line)

            # Add numpy slots for this document
            doc['slots_np'] = {
                k: np.array(v, dtype=np.float32) if v is not None else None
                for k, v in doc['slots'].items()
            }

            return doc

    def _create_faiss_index(self, faiss_dir: Path):
        """Create FAISS index from slot index."""
        faiss_dir.mkdir(exist_ok=True)

        # Load full embeddings
        index_file = self.index_path / "slot_index.jsonl"

        logger.info("  Loading embeddings for index creation...")
        embeddings = []
        doc_count = 0
        with open(index_file) as f:
            for line in f:
                doc = json.loads(line)
                embeddings.append(doc['full_embedding'])
                doc_count += 1

                # Progress logging every 500K docs
                if doc_count % 500000 == 0:
                    logger.info(f"    Loaded {doc_count:,} embeddings...")

        embeddings = np.array(embeddings, dtype=np.float32)

        logger.info(f"  Creating FAISS index for {embeddings.shape[0]:,} vectors")

        # Create index (IVF for large datasets, Flat for small)
        dim = embeddings.shape[1]

        if embeddings.shape[0] > 10000:
            # IVF index for large datasets
            nlist = int(np.sqrt(embeddings.shape[0]))  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (normalized)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train index
            logger.info(f"  Training IVF index with {nlist} clusters...")
            index.train(embeddings)
        else:
            # Flat index for small datasets
            index = faiss.IndexFlatIP(dim)

        # Add vectors
        logger.info("  Adding vectors to index...")
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(faiss_dir / "full_embeddings.index"))
        logger.info(f"  Saved FAISS index to {faiss_dir / 'full_embeddings.index'}")

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
        faiss_top_n: int = 500,
        slot_weight: float = 0.6,
        full_weight: float = 0.4,
    ) -> List[Tuple[float, Dict]]:
        """
        Two-stage retrieval: FAISS pre-filtering + slot reranking.

        Args:
            query: Query text in Esperanto
            top_k: Number of results to return
            faiss_top_n: Number of candidates from FAISS for slot reranking
            slot_weight: Weight for slot similarity in final score
            full_weight: Weight for full embedding (FAISS score) in final score

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
        query_full_emb = query_full_emb / np.linalg.norm(query_full_emb)

        # Stage 1: FAISS search
        logger.debug(f"Stage 1: FAISS search (retrieving {faiss_top_n} candidates)")

        # Query FAISS (needs to be 2D array)
        query_vector = query_full_emb.reshape(1, -1)

        # Set nprobe for IVF indexes
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = min(32, self.faiss_index.nlist)

        faiss_scores, faiss_indices = self.faiss_index.search(query_vector, faiss_top_n)
        faiss_scores = faiss_scores[0]  # Unwrap batch dimension
        faiss_indices = faiss_indices[0]

        logger.debug(f"  Top FAISS score: {faiss_scores[0]:.3f}")

        # Stage 2: Slot-based reranking
        logger.debug(f"Stage 2: Slot-based reranking")

        final_results = []
        for i, doc_idx in enumerate(faiss_indices):
            if doc_idx == -1:  # FAISS returns -1 for padding
                continue

            # P0: Lazy loading - only load document when needed
            doc = self._get_document(int(doc_idx))

            # Compute slot similarity
            slot_sim = self.slot_similarity(query_slots, doc['slots_np'], is_question=is_question)

            # Apply feature bonus
            feature_bonus = self.feature_similarity(query_features, doc['features'])
            slot_score = slot_sim * feature_bonus

            # FAISS score (already normalized by FAISS as inner product)
            faiss_score = float(faiss_scores[i])

            # Combine scores
            final_score = slot_weight * slot_score + full_weight * faiss_score

            final_results.append((final_score, doc))

        # Sort by final score
        final_results.sort(key=lambda x: x[0], reverse=True)

        logger.debug(f"  Top final score: {final_results[0][0]:.3f}")

        return final_results[:top_k]

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
