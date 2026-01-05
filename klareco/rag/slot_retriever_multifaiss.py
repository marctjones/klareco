"""
Solution 3: Hierarchical multi-slot FAISS indexes.

Builds separate FAISS indexes for each slot (SUBJ, VERB, OBJ),
queries each independently, and merges results using intersection + slot scoring.

Memory optimizations:
- P0: Lazy document loading with offset index (saves 36.6 GB)
- P1: FAISS memory mapping (saves 2 GB)
- P3: Thread tuning for 20% speed gain
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


class MultiFAISSSlotRetriever:
    """Slot-based retriever using separate FAISS indexes per slot."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize multi-FAISS retriever.

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

        # Load indexes
        self._load_indexes()

    def _load_indexes(self):
        """Load separate FAISS indexes for each slot."""
        logger.info(f"Loading multi-FAISS indexes from {self.index_path}")

        multifaiss_dir = self.index_path / "multifaiss"

        # Check if indexes exist
        if not multifaiss_dir.exists():
            logger.info("  Multi-FAISS indexes not found, creating...")
            self._create_multifaiss_indexes(multifaiss_dir)

        # Load each slot index
        self.slot_indexes = {}
        self.slot_doc_ids = {}  # Maps slot -> list of valid doc IDs

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            index_file = multifaiss_dir / f"{slot}.index"
            id_file = multifaiss_dir / f"{slot}_ids.npy"

            if index_file.exists():
                # P1: Load with memory mapping to save RAM
                index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP)

                # Set nprobe for IVF indexes
                if hasattr(index, 'nprobe'):
                    index.nprobe = 32  # Good balance for 4M docs
                    logger.info(f"  {slot}: IVF index with nprobe={index.nprobe}")

                if self.use_gpu and faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)

                self.slot_indexes[slot] = index
                self.slot_doc_ids[slot] = np.load(id_file)

                logger.info(f"  {slot}: {index.ntotal:,} vectors (memory-mapped)")
            else:
                logger.warning(f"  {slot}: index not found")
                self.slot_indexes[slot] = None
                self.slot_doc_ids[slot] = np.array([])

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

    def _create_multifaiss_indexes(self, multifaiss_dir: Path):
        """Create separate FAISS indexes for each slot."""
        multifaiss_dir.mkdir(exist_ok=True)

        # Load slot embeddings
        index_file = self.index_path / "slot_index.jsonl"

        slot_embeddings = {slot: [] for slot in ['SUBJ', 'VERB', 'OBJ']}
        slot_doc_ids = {slot: [] for slot in ['SUBJ', 'VERB', 'OBJ']}

        with open(index_file) as f:
            for doc_id, line in enumerate(f):
                doc = json.loads(line)

                for slot in ['SUBJ', 'VERB', 'OBJ']:
                    if doc['slots'][slot] is not None:
                        slot_embeddings[slot].append(doc['slots'][slot])
                        slot_doc_ids[slot].append(doc_id)

        # Create index for each slot
        dim = self.indexer.embedding_dim

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            if not slot_embeddings[slot]:
                logger.warning(f"  {slot}: no embeddings found, skipping")
                continue

            embeddings = np.array(slot_embeddings[slot], dtype=np.float32)
            doc_ids = np.array(slot_doc_ids[slot], dtype=np.int32)

            logger.info(f"  Creating {slot} index: {embeddings.shape[0]:,} vectors")

            # Create index (IVF for large, Flat for small)
            if embeddings.shape[0] > 10000:
                nlist = int(np.sqrt(embeddings.shape[0]))
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

                logger.info(f"    Training IVF with {nlist} clusters...")
                index.train(embeddings)
            else:
                index = faiss.IndexFlatIP(dim)

            # Add vectors
            index.add(embeddings)

            # Save index and ID mapping
            faiss.write_index(index, str(multifaiss_dir / f"{slot}.index"))
            np.save(multifaiss_dir / f"{slot}_ids.npy", doc_ids)

        logger.info(f"  Saved multi-FAISS indexes to {multifaiss_dir}")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

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
        slot_top_n: int = 200,
        merge_strategy: str = 'union',
    ) -> List[Tuple[float, Dict]]:
        """
        Multi-slot retrieval with FAISS indexes.

        Args:
            query: Query text in Esperanto
            top_k: Number of results to return
            slot_top_n: Number of candidates to retrieve per slot
            merge_strategy: 'union' or 'intersection' for merging slot results

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

        # Stage 1: Query each slot index
        logger.debug(f"Stage 1: Querying slot indexes")

        slot_candidates = {}  # slot -> set of doc IDs
        slot_scores = {}      # slot -> {doc_id -> score}

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            query_emb = query_slots.get(slot)
            index = self.slot_indexes.get(slot)

            if query_emb is None or index is None:
                slot_candidates[slot] = set()
                slot_scores[slot] = {}
                continue

            # Normalize query embedding
            query_vector = query_emb.astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector)
            query_vector = query_vector.reshape(1, -1)

            # Set nprobe for IVF
            if hasattr(index, 'nprobe'):
                index.nprobe = min(32, index.nlist)

            # Search
            scores, indices = index.search(query_vector, slot_top_n)
            scores = scores[0]
            indices = indices[0]

            # Map back to document IDs
            doc_ids = set()
            doc_scores = {}
            for i, idx in enumerate(indices):
                if idx != -1:
                    doc_id = int(self.slot_doc_ids[slot][idx])
                    doc_ids.add(doc_id)
                    doc_scores[doc_id] = float(scores[i])

            slot_candidates[slot] = doc_ids
            slot_scores[slot] = doc_scores

            logger.debug(f"  {slot}: {len(doc_ids)} candidates")

        # Stage 2: Merge candidates
        logger.debug(f"Stage 2: Merging with '{merge_strategy}' strategy")

        if merge_strategy == 'intersection':
            # Only documents that appear in ALL queried slots
            queried_slots = [s for s in ['SUBJ', 'VERB', 'OBJ'] if query_slots.get(s) is not None]
            if queried_slots:
                candidate_doc_ids = slot_candidates[queried_slots[0]]
                for slot in queried_slots[1:]:
                    candidate_doc_ids = candidate_doc_ids.intersection(slot_candidates[slot])
            else:
                candidate_doc_ids = set()
        else:  # union
            # Documents from any slot
            candidate_doc_ids = set()
            for doc_ids in slot_candidates.values():
                candidate_doc_ids.update(doc_ids)

        logger.debug(f"  {len(candidate_doc_ids)} merged candidates")

        # Stage 3: Compute final scores
        logger.debug(f"Stage 3: Computing final scores")

        final_results = []
        for doc_id in candidate_doc_ids:
            # P0: Lazy loading - only load document when needed
            doc = self._get_document(doc_id)

            # Compute weighted slot score
            total_score = 0.0
            total_weight = 0.0

            for slot, weight in self.slot_weights.items():
                if doc_id in slot_scores[slot]:
                    total_score += weight * slot_scores[slot][doc_id]
                    total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                slot_score = total_score / total_weight
            else:
                slot_score = 0.0

            # Apply feature bonus
            feature_bonus = self.feature_similarity(query_features, doc['features'])
            final_score = slot_score * feature_bonus

            final_results.append((final_score, doc))

        # Sort by final score
        final_results.sort(key=lambda x: x[0], reverse=True)

        if final_results:
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
