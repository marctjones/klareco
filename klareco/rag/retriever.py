"""
Retriever: Semantic search using SemanticPipeline and FAISS index.

This module provides retrieval using the EnrichedAST pipeline with lazy enrichment:
- Query embedding computed via SemanticPipeline (Stage 1 only for speed)
- Pre-built FAISS index for corpus search
- Optional lazy enrichment of retrieved results

Usage:
    retriever = Retriever.load()
    results = retriever.search("La hundo kuras rapide.", top_k=10)

    # With lazy enrichment (applies Stage 1+ to results)
    enriched_results = retriever.search_and_enrich("Kio estas Esperanto?", top_k=5)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from klareco.enriched_ast import EnrichedAST
from klareco.semantic_pipeline import SemanticPipeline, SemanticModel

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_INDEX_DIR = Path("data/corpus_index_compositional")
DEFAULT_ROOT_MODEL = Path("models/root_embeddings/best_model.pt")
DEFAULT_AFFIX_MODEL = Path("models/affix_transforms_v2/best_model.pt")


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    text: str
    score: float
    index: int
    source: Optional[str] = None
    tier: Optional[int] = None
    enriched_ast: Optional[EnrichedAST] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'text': self.text,
            'score': self.score,
            'index': self.index,
        }
        if self.source:
            result['source'] = self.source
        if self.tier is not None:
            result['tier'] = self.tier
        return result


class Retriever:
    """
    Semantic retriever using SemanticPipeline and FAISS index.

    This retriever:
    1. Uses SemanticPipeline to compute query embeddings
    2. Searches a pre-built FAISS index
    3. Optionally enriches results with full AST information

    The retriever supports lazy enrichment: results are returned quickly
    with just text and score, but can be enriched on-demand with full
    EnrichedAST processing.
    """

    def __init__(
        self,
        pipeline: SemanticPipeline,
        faiss_index: Any,  # faiss.Index
        metadata: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize retriever with pipeline and index.

        Args:
            pipeline: SemanticPipeline for query embedding
            faiss_index: FAISS index for similarity search
            metadata: List of metadata dicts for each indexed sentence
            embeddings: Optional embeddings array (for reranking)
        """
        self.pipeline = pipeline
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embeddings = embeddings

    @classmethod
    def load(
        cls,
        index_dir: Path = DEFAULT_INDEX_DIR,
        root_model_path: Path = DEFAULT_ROOT_MODEL,
        affix_model_path: Optional[Path] = DEFAULT_AFFIX_MODEL,
    ) -> 'Retriever':
        """
        Load retriever from index directory and model paths.

        Args:
            index_dir: Directory containing FAISS index and metadata
            root_model_path: Path to root embeddings model
            affix_model_path: Path to affix transforms model (optional)

        Returns:
            Retriever ready for search
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required: pip install faiss-cpu")

        logger.info(f"Loading retriever from {index_dir}")

        # Load FAISS index
        index_path = index_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        faiss_index = faiss.read_index(str(index_path))
        logger.info(f"  Loaded FAISS index: {faiss_index.ntotal} vectors")

        # Load metadata
        metadata_path = index_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        metadata = []
        with open(metadata_path) as f:
            for line in f:
                metadata.append(json.loads(line))
        logger.info(f"  Loaded metadata: {len(metadata)} entries")

        # Optionally load embeddings (for reranking)
        embeddings = None
        embeddings_path = index_dir / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            logger.info(f"  Loaded embeddings: {embeddings.shape}")

        # Load SemanticPipeline
        pipeline = SemanticPipeline.load(root_model_path, affix_model_path)

        return cls(
            pipeline=pipeline,
            faiss_index=faiss_index,
            metadata=metadata,
            embeddings=embeddings,
        )

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Embed a query using the SemanticPipeline.

        Args:
            query: Esperanto query text

        Returns:
            Query embedding vector (normalized) or None if embedding fails
        """
        # Use pipeline for retrieval (Stage 1 only)
        enriched = self.pipeline.for_retrieval(query)

        if enriched.sentence_embedding is None:
            return None

        # Convert to numpy and normalize
        emb = enriched.sentence_embedding.detach().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb.astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Search for similar sentences.

        Args:
            query: Esperanto query text
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        query_emb = self.embed_query(query)

        if query_emb is None:
            logger.warning(f"Could not embed query: {query[:50]}...")
            return []

        # Search FAISS
        query_emb = query_emb.reshape(1, -1)
        scores, indices = self.faiss_index.search(query_emb, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append(RetrievalResult(
                text=meta.get('text', ''),
                score=float(score),
                index=int(idx),
                source=meta.get('source'),
                tier=meta.get('tier'),
            ))

        return results

    def search_and_enrich(
        self,
        query: str,
        top_k: int = 10,
        stages: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Search and enrich results with full AST processing.

        This performs lazy enrichment: results are first retrieved quickly,
        then enriched with full pipeline processing.

        Args:
            query: Esperanto query text
            top_k: Number of results to return
            stages: Which stages to apply ('semantic', 'grammatical', 'discourse')
                   Defaults to ['semantic']

        Returns:
            List of RetrievalResult objects with enriched_ast populated
        """
        if stages is None:
            stages = ['semantic']

        # First, get raw results
        results = self.search(query, top_k)

        # Enrich each result
        for result in results:
            try:
                enriched = self.pipeline(result.text, stages=stages)
                result.enriched_ast = enriched
            except Exception as e:
                logger.warning(f"Could not enrich result: {e}")

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[RetrievalResult]]:
        """
        Search for multiple queries.

        Args:
            queries: List of Esperanto query texts
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        return [self.search(q, top_k) for q in queries]

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self.metadata)

    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings."""
        return self.faiss_index.d

    def __repr__(self) -> str:
        return (
            f"Retriever("
            f"documents={self.num_documents:,}, "
            f"dim={self.embedding_dim})"
        )
