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
from klareco.structural_index import rank_candidates_by_slot_overlap
from klareco.canonicalizer import canonicalize_sentence
from klareco.semantic_signatures import extract_signature, match_signature


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
        device: str = 'cpu',
        semantic_index_dir: Optional[str] = None,
    ):
        """
        Initialize retriever.

        Args:
            index_dir: Directory containing FAISS index and metadata
            model_path: Path to Tree-LSTM checkpoint
            mode: 'tree_lstm' or 'baseline' encoder
            device: 'cpu' or 'cuda'
            semantic_index_dir: Optional path to semantic signature index
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

        # Load semantic index if available
        self.semantic_index = None
        if semantic_index_dir:
            self._load_semantic_index(semantic_index_dir)
        else:
            # Try default location
            default_semantic_dir = Path(index_dir).parent / "semantic_index"
            if default_semantic_dir.exists():
                self._load_semantic_index(str(default_semantic_dir))

        if mode == 'tree_lstm':
            self._load_tree_lstm()
        elif mode == 'baseline':
            self._load_baseline()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'tree_lstm' or 'baseline'")

        logger.info(f"Retriever initialized in {mode} mode")
        logger.info(f"  Corpus size: {len(self.metadata):,} sentences")
        logger.info(f"  Embedding dim: {self.index.d}")
        if self.semantic_index:
            logger.info(f"  Semantic index: {len(self.semantic_index.signatures):,} signatures")

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

    def _load_semantic_index(self, index_dir: str):
        """Load semantic signature index for role-based filtering."""
        from klareco.semantic_search import SemanticIndex

        try:
            self.semantic_index = SemanticIndex(Path(index_dir))
            logger.debug(f"Loaded semantic index: {len(self.semantic_index.signatures):,} signatures")
        except Exception as e:
            logger.warning(f"Could not load semantic index: {e}")
            self.semantic_index = None

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
        return_scores: bool = True,
        structural_candidates: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar sentences for AST query.

        Args:
            ast: Parsed query AST
            k: Number of results to return
            return_scores: Include similarity scores in results
            structural_candidates: Max candidates to consider from structural filter

        Returns:
            List of result dictionaries
        """
        # Structural filter (Stage 1)
        candidate_indices = None
        try:
            query_slots = canonicalize_sentence(ast)
            query_slot_roots = {role: slot.root for role, slot in query_slots.items() if slot and slot.root}
            has_slot_roots = any(isinstance(m, dict) and m.get('slot_roots') for m in self.metadata[:10])
            if has_slot_roots and query_slot_roots:
                candidate_indices = rank_candidates_by_slot_overlap(
                    query_slot_roots,
                    self.metadata,
                    limit=structural_candidates,
                )
                if candidate_indices:
                    logger.debug("Structural filter kept %d candidates", len(candidate_indices))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Structural filter skipped: %s", exc)

        # Encode query
        query_embedding = self._encode_ast(ast)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        if candidate_indices:
            # Manual rerank on candidate embeddings
            candidate_embeddings = self.embeddings[candidate_indices]
            scores = np.dot(candidate_embeddings, query_embedding.T).flatten()
            sorted_pairs = sorted(
                zip(candidate_indices, scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            indices_scores = [(idx, score) for idx, score in sorted_pairs]
        else:
            # Full FAISS search
            scores, indices = self.index.search(query_embedding, k)
            indices_scores = list(zip(indices[0], scores[0]))

        # Build results
        results = []
        for i, (idx, score) in enumerate(indices_scores, 1):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            if 'sentence' in result:
                result['text'] = result.pop('sentence')
            result['index'] = int(idx)
            result['rank'] = i
            if return_scores:
                result['score'] = float(score)
            results.append(result)

        return results

    def retrieve_semantic(
        self,
        query: str,
        k: int = 10,
        use_neural_rerank: bool = True,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using semantic role filtering.

        Uses (agent, action, patient) signatures to find sentences
        where entities play the same semantic roles as in the query.

        Example:
            Query: "Kiu vidas la katon?" (Who sees the cat?)
            Finds sentences where cat is PATIENT (being seen),
            not where cat is AGENT (doing the seeing).

        Args:
            query: Query sentence in Esperanto
            k: Number of results to return
            use_neural_rerank: Whether to rerank with Tree-LSTM (default True)
            return_scores: Include similarity scores

        Returns:
            List of result dictionaries
        """
        # Parse query
        try:
            ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query '{query}': {e}")
            return []

        return self.retrieve_semantic_from_ast(
            ast, k=k, use_neural_rerank=use_neural_rerank, return_scores=return_scores
        )

    def retrieve_semantic_from_ast(
        self,
        ast: Dict[str, Any],
        k: int = 10,
        use_neural_rerank: bool = True,
        return_scores: bool = True,
        semantic_candidates: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using semantic role filtering from AST.

        Three-stage retrieval:
        1. Semantic filter: Find sentences with matching (agent, action, patient) roles
        2. Neural rerank: Score candidates with Tree-LSTM (optional)
        3. Return top-k results

        Args:
            ast: Parsed query AST
            k: Number of results to return
            use_neural_rerank: Whether to rerank with Tree-LSTM
            return_scores: Include similarity scores
            semantic_candidates: Max candidates from semantic filter

        Returns:
            List of result dictionaries
        """
        if not self.semantic_index:
            logger.warning("No semantic index loaded, falling back to standard retrieval")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        # Extract query signature
        query_sig = extract_signature(ast)

        if not any(query_sig):
            logger.debug("No semantic signature extracted, falling back to standard retrieval")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        logger.debug(f"Query signature: {query_sig}")

        # Stage 1: Semantic filtering
        semantic_results = self.semantic_index.search(query_sig, k=semantic_candidates)

        if not semantic_results:
            logger.debug("No semantic matches, falling back to standard retrieval")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        logger.debug(f"Semantic filter returned {len(semantic_results)} candidates")

        # Get candidate indices (map from semantic index IDs to our metadata IDs)
        # Note: semantic index uses corpus line numbers as IDs
        candidate_indices = [r['sentence_id'] for r in semantic_results]

        # Filter to valid indices
        valid_candidates = [idx for idx in candidate_indices if idx < len(self.metadata)]

        if not valid_candidates:
            logger.warning("No valid candidates after semantic filter")
            return self.retrieve_from_ast(ast, k=k, return_scores=return_scores)

        if use_neural_rerank:
            # Stage 2: Neural reranking
            query_embedding = self._encode_ast(ast).reshape(1, -1).astype('float32')
            candidate_embeddings = self.embeddings[valid_candidates]
            scores = np.dot(candidate_embeddings, query_embedding.T).flatten()

            # Sort by score
            sorted_pairs = sorted(
                zip(valid_candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]
        else:
            # Use semantic match scores directly
            idx_to_score = {r['sentence_id']: r['score'] for r in semantic_results}
            sorted_pairs = sorted(
                [(idx, idx_to_score.get(idx, 0)) for idx in valid_candidates],
                key=lambda x: x[1],
                reverse=True
            )[:k]

        # Build results
        results = []
        for rank, (idx, score) in enumerate(sorted_pairs, 1):
            result = dict(self.metadata[idx])
            if 'sentence' in result:
                result['text'] = result.pop('sentence')
            result['index'] = int(idx)
            result['rank'] = rank
            result['semantic_signature'] = query_sig
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
                plena_vorto = node.get('plena_vorto', '').lower()

                # Skip question words (even if classified as proper nouns)
                if radiko in QUESTION_WORDS:
                    return

                # Always include proper nouns (HIGHEST PRIORITY)
                # Check both vortspeco and category for proper names
                if (vortspeco in ['nomo', 'propra_nomo'] or
                    node.get('category') in ['proper_name', 'proper_name_esperantized']):
                    # Use full word for proper names (not root)
                    keywords.append(plena_vorto if plena_vorto else radiko)
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
        return_scores: bool = True,
        return_stage1_info: bool = False
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
            return_stage1_info: Return dict with stage1 stats and results

        Returns:
            If return_stage1_info=False:
                List of result dictionaries (keyword-filtered + semantically ranked)
            If return_stage1_info=True:
                Dict with 'results' (stage 2) and 'stage1' (stage 1 info)
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

        # Store stage1 info before limiting
        stage1_total_candidates = len(candidate_indices)
        stage1_candidates = []
        if return_stage1_info:
            # Build stage1 results (limited for display)
            for idx in list(candidate_indices)[:min(20, len(candidate_indices))]:
                result = dict(self.metadata[idx])
                if 'sentence' in result:
                    result['text'] = result.pop('sentence')
                result['index'] = int(idx)
                stage1_candidates.append(result)

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

        # Return with or without stage1 info
        if return_stage1_info:
            return {
                'results': results,
                'stage1': {
                    'keywords': keywords,
                    'total_candidates': stage1_total_candidates,
                    'candidates_shown': stage1_candidates,
                    'candidates_reranked': len(candidate_indices)
                }
            }
        else:
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
