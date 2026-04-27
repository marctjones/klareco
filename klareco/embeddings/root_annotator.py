"""
Root Embeddings Annotator - Example Implementation of ASTAnnotator Interface

This module shows how the Root Embeddings model implements the ASTAnnotator
protocol to add root embedding annotations to ASTs.

Example:
    >>> parser = Parser()
    >>> root_annotator = RootEmbeddingsAnnotator(
    ...     model_path='models/root_embeddings/best_model.pt'
    ... )
    >>>
    >>> ast = parser.parse("La hundo kuras.")
    >>> # AST has deterministic features: radiko='hund', kazo='nominativo', etc.
    >>>
    >>> ast = root_annotator.annotate(ast)
    >>> # AST now has: annotations['root_embedding']['hund'] = [...64d vector...]
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from klareco.ast_annotator import ASTAnnotator

logger = logging.getLogger(__name__)


class RootEmbeddingsAnnotator(ASTAnnotator):
    """
    Annotates AST with root embeddings (64d vectors for content word roots).

    This model:
    - Reads 'radiko' (root) from word ASTs (deterministic feature from M0)
    - Looks up root in embedding table (learned)
    - Adds 'root_embedding' annotation to AST
    - NEVER re-parses or modifies grammatical features

    Training:
    - Trained on ~9,800 content word roots (tier1a+1b+2)
    - Excludes tier0 function words (187 words: mi, kaj, la, de, mal, iĝ, etc.)
    - Uses semantic similarity objective
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize Root Embeddings Annotator.

        Args:
            model_path: Path to trained model checkpoint (best_model.pt)
            vocab_path: Path to vocabulary JSON (root_to_idx.json)
            device: Device to run model on ('cpu' or 'cuda')
        """
        # Initialize attributes BEFORE calling super().__init__()
        # because parent calls _validate_setup() which checks these
        self.model_name = "RootEmbeddings"  # Set early so _load_model can log
        self.device = torch.device(device)
        self.model_path = model_path
        self.vocab_path = vocab_path

        # Load model and vocabulary
        self.embedding_table = None
        self.root_to_idx = None
        self.idx_to_root = None
        self.embed_dim = 64

        if model_path is not None:
            self._load_model(model_path)

        if vocab_path is not None:
            self._load_vocabulary(vocab_path)

        # Call parent __init__ AFTER initializing attributes
        super().__init__(model_name="RootEmbeddings")

    def _validate_setup(self):
        """Validate model and vocabulary are loaded."""
        if self.embedding_table is None:
            logger.warning(f"{self.model_name}: No model loaded. Call _load_model() first.")

        if self.root_to_idx is None:
            logger.warning(f"{self.model_name}: No vocabulary loaded. Call _load_vocabulary() first.")

    def _load_model(self, model_path: str):
        """Load trained embedding table from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'embeddings.weight' in checkpoint['model_state_dict']:
            # Compositional model format
            self.embedding_table = checkpoint['model_state_dict']['embeddings.weight']
        elif 'model_state_dict' in checkpoint:
            # Direct embedding table
            self.embedding_table = checkpoint['model_state_dict']
        else:
            raise ValueError(f"Unknown checkpoint format: {list(checkpoint.keys())}")

        self.embed_dim = self.embedding_table.shape[1]
        logger.info(f"{self.model_name}: Loaded {self.embedding_table.shape[0]} embeddings "
                   f"({self.embed_dim}d) from {model_path}")

    def _load_vocabulary(self, vocab_path: str):
        """Load root vocabulary (root -> idx mapping)."""
        with open(vocab_path) as f:
            self.root_to_idx = json.load(f)

        self.idx_to_root = {idx: root for root, idx in self.root_to_idx.items()}
        logger.info(f"{self.model_name}: Loaded vocabulary with {len(self.root_to_idx)} roots")

    def annotate(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add root embedding annotations to AST.

        For each word in the AST:
        1. Read 'radiko' (root) from deterministic features (computed by M0)
        2. Look up root in embedding table
        3. Add 'root_embedding' to annotations

        Args:
            ast: AST from M0 parser with structure:
                {
                    'tipo': 'frazo',
                    'subjekto': {'tipo': 'vorto', 'radiko': 'hund', ...},
                    'verbo': {'tipo': 'vorto', 'radiko': 'kur', ...},
                    ...
                }
            context: Unused (root embeddings are context-independent)

        Returns:
            AST with root embeddings added:
                {
                    'tipo': 'frazo',
                    'subjekto': {
                        'tipo': 'vorto',
                        'radiko': 'hund',
                        'annotations': {
                            'root_embedding': [...64d vector...]
                        }
                    },
                    ...
                }
        """
        if self.embedding_table is None or self.root_to_idx is None:
            raise RuntimeError(f"{self.model_name}: Model or vocabulary not loaded. "
                             f"Call _load_model() and _load_vocabulary() first.")

        # Recursively annotate all words in AST
        ast = self._ensure_annotations_dict(ast)
        ast = self._annotate_node(ast)

        return ast

    def _annotate_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively annotate a node and its children.

        Args:
            node: AST node (frazo, vortgrupo, or vorto)

        Returns:
            Node with root embeddings added
        """
        node_type = node.get('tipo', 'unknown')

        if node_type == 'vorto':
            # Leaf node: add root embedding
            node = self._annotate_word(node)

        elif node_type == 'vortgrupo':
            # Word group: annotate kerno (head) and priskriboj (modifiers)
            if 'kerno' in node:
                node['kerno'] = self._annotate_node(node['kerno'])
            if 'priskriboj' in node:
                node['priskriboj'] = [self._annotate_node(p) for p in node['priskriboj']]

        elif node_type == 'frazo':
            # Sentence: annotate subjekto, verbo, objekto, aliaj
            if 'subjekto' in node and node['subjekto'] is not None:
                node['subjekto'] = self._annotate_node(node['subjekto'])
            if 'verbo' in node and node['verbo'] is not None:
                node['verbo'] = self._annotate_node(node['verbo'])
            if 'objekto' in node and node['objekto'] is not None:
                node['objekto'] = self._annotate_node(node['objekto'])
            if 'aliaj' in node:
                node['aliaj'] = [self._annotate_node(a) for a in node['aliaj']]

        return node

    def _annotate_word(self, word_ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add root embedding to a word AST.

        Args:
            word_ast: Word-level AST with 'radiko' feature

        Returns:
            Word AST with 'root_embedding' annotation added
        """
        # Read deterministic feature 'radiko' (computed by M0 parser)
        try:
            root = self._read_deterministic_feature(word_ast, 'radiko')
        except KeyError:
            logger.warning(f"{self.model_name}: Word AST missing 'radiko' field: {word_ast}")
            return word_ast

        # Look up root embedding
        if root in self.root_to_idx:
            idx = self.root_to_idx[root]
            embedding = self.embedding_table[idx].cpu().numpy().tolist()
        else:
            # OOV root: use zero vector
            logger.debug(f"{self.model_name}: OOV root '{root}', using zero vector")
            embedding = [0.0] * self.embed_dim

        # Add annotation
        word_ast = self._ensure_annotations_dict(word_ast)
        word_ast = self._add_annotation(word_ast, 'root_embedding', embedding)

        return word_ast

    def get_root_embedding(self, root: str) -> Optional[torch.Tensor]:
        """
        Get embedding vector for a root word.

        Args:
            root: Root word (e.g., 'hund', 'kur', 'bel')

        Returns:
            Embedding tensor (64d) or None if OOV
        """
        if root not in self.root_to_idx:
            return None

        idx = self.root_to_idx[root]
        return self.embedding_table[idx]

    def get_similar_roots(self, root: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Find most similar roots by cosine similarity.

        Args:
            root: Root word
            top_k: Number of similar roots to return

        Returns:
            List of (root, similarity) tuples, sorted by similarity (descending)
        """
        if root not in self.root_to_idx:
            return []

        idx = self.root_to_idx[root]
        query_vec = self.embedding_table[idx]

        # Compute cosine similarity with all roots
        similarities = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(0),
            self.embedding_table,
            dim=1
        )

        # Get top-k (excluding self)
        top_indices = torch.argsort(similarities, descending=True)[1:top_k+1]
        top_similarities = similarities[top_indices]

        return [
            (self.idx_to_root[idx.item()], sim.item())
            for idx, sim in zip(top_indices, top_similarities)
        ]
