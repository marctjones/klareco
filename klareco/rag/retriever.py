"""
RAG Retriever for Klareco

Retrieves semantically similar sentences from indexed corpus using Tree-LSTM embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import faiss
import torch

from klareco.parser import parse
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter


logger = logging.getLogger(__name__)


class KlarecoRetriever:
    """
    Semantic retriever using Tree-LSTM embeddings and FAISS index.

    Supports both Tree-LSTM (GNN) and baseline encoders for comparison.
    """

    def __init__(
        self,
        index_dir: str,
        model_path: str,
        mode: str = 'tree_lstm',
        device: str = 'cpu'
    ):
        """
        Initialize retriever.

        Args:
            index_dir: Directory containing FAISS index and metadata
            model_path: Path to Tree-LSTM checkpoint
            mode: 'tree_lstm' or 'baseline' encoder
            device: 'cpu' or 'cuda'
        """
        self.index_dir = Path(index_dir)
        self.model_path = Path(model_path)
        self.mode = mode
        self.device = device

        # Load components
        logger.info(f"Loading retriever components from {index_dir}")
        self._load_index()
        self._load_metadata()
        self._load_embeddings()  # Load raw embeddings for hybrid retrieval

        if mode == 'tree_lstm':
            self._load_tree_lstm()
        elif mode == 'baseline':
            self._load_baseline()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'tree_lstm' or 'baseline'")

        logger.info(f"Retriever initialized in {mode} mode")
        logger.info(f"  Corpus size: {len(self.metadata):,} sentences")
        logger.info(f"  Embedding dim: {self.index.d}")

    def _load_index(self):
        """Load FAISS index."""
        index_path = self.index_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        logger.debug(f"Loaded FAISS index: {self.index.ntotal:,} vectors")

    def _load_metadata(self):
        """Load sentence metadata."""
        metadata_path = self.index_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))

        logger.debug(f"Loaded {len(self.metadata):,} metadata entries")

    def _load_embeddings(self):
        """Load raw embeddings for hybrid retrieval."""
        embeddings_path = self.index_dir / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        self.embeddings = np.load(embeddings_path)
        logger.debug(f"Loaded {self.embeddings.shape[0]:,} embeddings of dimension {self.embeddings.shape[1]}")

    def _load_tree_lstm(self):
        """Load Tree-LSTM encoder."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Infer hyperparameters from model state
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        vocab_size = state_dict["embed.weight"].shape[0]
        embed_dim = state_dict["embed.weight"].shape[1]
        hidden_dim = state_dict["tree_lstm.cell.W_i.weight"].shape[0]
        output_dim = state_dict["tree_lstm.output_proj.weight"].shape[0]

        # Initialize model
        self.encoder = TreeLSTMEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        self.encoder.load_state_dict(state_dict)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Initialize AST converter
        self.converter = ASTToGraphConverter(embed_dim=embed_dim)

        logger.debug(f"Loaded Tree-LSTM: vocab={vocab_size}, embed={embed_dim}, "
                    f"hidden={hidden_dim}, output={output_dim}")

    def _load_baseline(self):
        """Load baseline sentence encoder."""
        # Placeholder for baseline encoder (e.g., sentence-transformers)
        # This would load a pre-trained model like distiluse-base-multilingual
        raise NotImplementedError("Baseline encoder not yet implemented")

    def _encode_ast(self, ast: Dict[str, Any]) -> np.ndarray:
        """
        Encode AST to embedding vector.

        Args:
            ast: Parsed query AST

        Returns:
            Embedding vector (output_dim,)
        """
        if self.mode == 'tree_lstm':
            # Convert AST to graph
            graph_data = self.converter.ast_to_graph(ast)
            graph_data = graph_data.to(self.device)

            # Encode with Tree-LSTM
            with torch.no_grad():
                embedding = self.encoder(graph_data)
                embedding = embedding.cpu().numpy()

            return embedding
        else:
            raise NotImplementedError(f"Encoding not implemented for mode: {self.mode}")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar sentences for text query.

        Args:
            query: Query sentence in Esperanto
            k: Number of results to return
            return_scores: Include similarity scores in results

        Returns:
            List of result dictionaries with keys:
                - text: Sentence text
                - index: Corpus index
                - score: Similarity score (if return_scores=True)
        """
        # Parse query to AST
        try:
            ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query '{query}': {e}")
            return []

        # Retrieve from AST
        return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

    def retrieve_from_ast(
        self,
        ast: Dict[str, Any],
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar sentences for AST query.

        Args:
            ast: Parsed query AST
            k: Number of results to return
            return_scores: Include similarity scores in results

        Returns:
            List of result dictionaries
        """
        # Encode query
        query_embedding = self._encode_ast(ast)

        # Ensure 2D array for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search FAISS index (using inner product = cosine similarity for normalized vectors)
        scores, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            # Start with all metadata fields
            result = dict(self.metadata[idx])

            # Rename 'sentence' to 'text' for consistency
            if 'sentence' in result:
                result['text'] = result.pop('sentence')

            # Add retrieval-specific fields
            result['index'] = int(idx)
            result['rank'] = i + 1

            if return_scores:
                result['score'] = float(score)

            results.append(result)

        return results

    def _extract_keywords_from_ast(self, ast: Dict[str, Any]) -> List[str]:
        """
        Extract important keywords from AST for hybrid retrieval.

        Prioritizes:
        - Proper nouns (names, places) - EXCEPT question words
        - Content words (nouns, verbs, adjectives)
        - Ignores function words (articles, pronouns, conjunctions, question words)

        Args:
            ast: Parsed query AST

        Returns:
            List of keyword strings (lowercased)
        """
        # Question words to exclude (too common, not useful for filtering)
        QUESTION_WORDS = {'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel', 'kiom', 'kies'}

        keywords = []

        def extract_from_node(node):
            if not isinstance(node, dict):
                return

            # Extract from word nodes
            if node.get('tipo') == 'vorto':
                radiko = node.get('radiko', '').lower()
                vortspeco = node.get('vortspeco', '')

                # Skip question words (even if classified as proper nouns)
                if radiko in QUESTION_WORDS:
                    return

                # Always include proper nouns (HIGHEST PRIORITY)
                if vortspeco == 'nomo':
                    keywords.append(radiko)
                # Include content words (nouns, verbs, adjectives)
                elif vortspeco in ['substantivo', 'verbo', 'adjektivo']:
                    # Skip very common verbs
                    if radiko not in ['est', 'hav', 'far']:
                        keywords.append(radiko)

            # Recursively process nested structures
            for value in node.values():
                if isinstance(value, dict):
                    extract_from_node(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            extract_from_node(item)

        extract_from_node(ast)
        return keywords

    def retrieve_hybrid(
        self,
        ast: Dict[str, Any],
        k: int = 20,
        keyword_candidates: int = 200,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: keyword filter + semantic rerank.

        Stage 1: Find candidates mentioning query keywords (BM25-like)
        Stage 2: Re-rank candidates by semantic similarity (Tree-LSTM)

        This combines:
        - Lexical matching (finds sentences with "Gandalf")
        - Semantic matching (ranks by relevance using AST structure)

        Args:
            ast: Parsed query AST
            k: Final number of results to return
            keyword_candidates: Number of keyword candidates to consider
            return_scores: Include similarity scores

        Returns:
            List of result dictionaries (keyword-filtered + semantically ranked)
        """
        # Extract keywords from AST
        keywords = self._extract_keywords_from_ast(ast)

        if not keywords:
            # Fallback to pure semantic search
            logger.debug("No keywords extracted, using pure semantic search")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        logger.debug(f"Extracted keywords: {keywords}")

        # Stage 1: Keyword filtering
        # Find all sentences mentioning ANY keyword
        candidate_indices = set()
        for keyword in keywords:
            for idx, metadata in enumerate(self.metadata):
                text = metadata.get('sentence', '').lower()
                if keyword in text:
                    candidate_indices.add(idx)

        if not candidate_indices:
            # No keyword matches, fallback to semantic search
            logger.debug(f"No keyword matches for {keywords}, using semantic search")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        logger.debug(f"Found {len(candidate_indices)} keyword candidates")

        # Limit candidates for efficiency
        candidate_indices = list(candidate_indices)[:keyword_candidates]

        # Stage 2: Semantic re-ranking
        # Encode query once
        query_embedding = self._encode_ast(ast).reshape(1, -1).astype('float32')

        # Get embeddings for candidates
        candidate_embeddings = self.embeddings[candidate_indices]

        # Compute similarity scores
        import numpy as np
        scores = np.dot(candidate_embeddings, query_embedding.T).flatten()

        # Sort by score (descending)
        sorted_pairs = sorted(
            zip(candidate_indices, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Build results
        results = []
        for rank, (idx, score) in enumerate(sorted_pairs, 1):
            # Start with all metadata fields
            result = dict(self.metadata[idx])

            # Rename 'sentence' to 'text' for consistency
            if 'sentence' in result:
                result['text'] = result.pop('sentence')

            # Add retrieval-specific fields
            result['index'] = int(idx)
            result['rank'] = rank

            if return_scores:
                result['score'] = float(score)

            results.append(result)

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        return_scores: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries in batch.

        Args:
            queries: List of query sentences
            k: Number of results per query
            return_scores: Include similarity scores

        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, k=k, return_scores=return_scores))
        return results


def create_retriever(
    index_dir: str = "data/corpus_index",
    model_path: str = "models/tree_lstm/checkpoint_epoch_12.pt",
    mode: str = 'tree_lstm',
    device: str = 'cpu'
) -> KlarecoRetriever:
    """
    Convenience function to create retriever with default paths.

    Args:
        index_dir: Directory with indexed corpus
        model_path: Path to encoder checkpoint
        mode: 'tree_lstm' or 'baseline'
        device: 'cpu' or 'cuda'

    Returns:
        Initialized KlarecoRetriever
    """
    return KlarecoRetriever(
        index_dir=index_dir,
        model_path=model_path,
        mode=mode,
        device=device
    )
