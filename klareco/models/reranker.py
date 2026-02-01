"""
Query-Document Relevance Reranker for Klareco.

A lightweight MLP that scores (query, document) pairs by relevance.
Uses frozen CompositionalEmbedding for encoding, learns only interaction patterns.

Architecture:
- Input: Query AST + Document AST
- Encoding: CompositionalEmbedding (frozen, 320K params)
- Interaction: Element-wise product, difference, structural features
- Scoring: 3-layer MLP (~180K params)
- Output: Relevance score [0, 1]

Total trainable params: ~180K (minimal learned capacity!)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from klareco.embeddings import CompositionalEmbedding
from klareco.parser import parse

logger = logging.getLogger(__name__)


class RelevanceScorer(nn.Module):
    """
    MLP-based relevance scorer for (query, document) pairs.

    Features:
    - Query embedding (128d)
    - Document embedding (128d)
    - Interaction features: element-wise product (128d), absolute difference (128d)
    - Structural features: question type, root overlap, pattern matching

    Total input: ~520d
    Output: Single relevance score [0, 1]
    """

    def __init__(
        self,
        feature_dim: int = 520,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.2,
    ):
        """
        Initialize relevance scorer.

        Args:
            feature_dim: Input feature dimension (default: 520)
            hidden_dims: Hidden layer dimensions (default: [256, 256, 128])
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final scoring layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.scoring_net = nn.Sequential(*layers)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"RelevanceScorer initialized: {total_params:,} parameters")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Score relevance given feature vector.

        Args:
            features: Tensor of shape (batch_size, feature_dim)

        Returns:
            Relevance scores of shape (batch_size, 1) in range [0, 1]
        """
        return self.scoring_net(features)


class ASTReranker(nn.Module):
    """
    Full reranker module that encodes ASTs and scores relevance.

    Combines:
    - CompositionalEmbedding (frozen) for encoding
    - Feature extraction from AST structure
    - RelevanceScorer MLP for scoring
    """

    def __init__(
        self,
        compositional_embedding: CompositionalEmbedding,
        freeze_embedding: bool = True,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.2,
    ):
        """
        Initialize AST reranker.

        Args:
            compositional_embedding: Pre-trained CompositionalEmbedding model
            freeze_embedding: If True, freeze embedding parameters (recommended)
            hidden_dims: Hidden layer dimensions for scorer
            dropout: Dropout rate
        """
        super().__init__()

        self.compositional_emb = compositional_embedding

        # Freeze embedding if requested
        if freeze_embedding:
            for param in self.compositional_emb.parameters():
                param.requires_grad = False
            logger.info("CompositionalEmbedding frozen (not trainable)")

        # Get embedding dimension
        self.embed_dim = compositional_embedding.embed_dim

        # Calculate feature dimension
        # query_emb (128) + doc_emb (128) + interaction (128) + diff (128) + structural (8)
        self.feature_dim = self.embed_dim * 4 + 8

        # Initialize scorer
        self.scorer = RelevanceScorer(
            feature_dim=self.feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def encode_ast(self, ast: Dict) -> torch.Tensor:
        """
        Encode AST to embedding using compositional model.

        Args:
            ast: Parsed AST dictionary

        Returns:
            Embedding tensor of shape (embed_dim,)
        """
        # Extract roots with weights
        roots = self._extract_roots(ast)

        if not roots:
            # Return zero embedding for empty AST
            return torch.zeros(self.embed_dim)

        # Average embeddings of roots (weighted by role importance)
        embeddings = []
        weights = []

        for root, weight in roots.items():
            # Get root embedding
            emb = self.compositional_emb.get_root_embedding(root)
            if emb is not None:
                embeddings.append(emb)
                weights.append(weight)

        if not embeddings:
            return torch.zeros(self.embed_dim)

        # Weighted average
        embeddings = torch.stack(embeddings)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize

        return (embeddings * weights.unsqueeze(1)).sum(dim=0)

    def _extract_roots(self, ast: Dict) -> Dict[str, float]:
        """
        Extract roots from AST with role-based weights.

        STRENGTHENED WEIGHTS (Issue #576):
        - OBJECT/SUBJECT: 3.0x boost (main entities)
        - VERB: 2.0x boost (action/relation)
        - ALIAJ: 0.3x penalty (modifiers, less important)
        - COMPOUND MODIFIERS: 0.2x heavy penalty

        Args:
            ast: Parsed AST

        Returns:
            Dictionary mapping root → weight
        """
        roots = {}

        def extract(node, role='', weight=1.0):
            if node is None:
                return

            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    root = node.get('radiko', '')
                    if root:
                        # STRENGTHENED role-based weights (Issue #576)
                        role_weights = {
                            'verbo': 2.0,      # 2x boost (was 1.0)
                            'objekto': 3.0,    # 3x boost (was 1.2)
                            'subjekto': 2.5,   # 2.5x boost (was 1.1)
                            'aliaj': 0.3,      # 0.3x penalty (was 0.9)
                        }
                        final_weight = weight * role_weights.get(role, 1.0)

                        # NEW: Compound modifier detection (Issue #576)
                        # If this word has compound modifiers, extract them with heavy penalty
                        # Example: "Esperanto-grupon" → radiko='grup', kunmetajhoj=[{radiko: 'esperant'}]
                        kunmetajhoj = node.get('kunmetajhoj', [])

                        # Also extract compound modifier roots with heavy penalty
                        for kunmetajho in kunmetajhoj:
                            if isinstance(kunmetajho, dict):
                                modifier_root = kunmetajho.get('radiko', '').lower()
                                if modifier_root:
                                    # Compound modifier penalty: 0.2x (heavy penalty)
                                    compound_weight = weight * 0.2
                                    roots[modifier_root] = max(roots.get(modifier_root, 0), compound_weight)

                        # Main root gets full role weight
                        roots[root.lower()] = max(roots.get(root.lower(), 0), final_weight)

                elif node.get('tipo') == 'vortgrupo':
                    extract(node.get('kerno'), role, weight)
                    for p in node.get('priskriboj', []):
                        extract(p, role, weight * 0.8)

                elif node.get('tipo') == 'frazo':
                    extract(node.get('subjekto'), 'subjekto', weight)
                    extract(node.get('verbo'), 'verbo', weight)
                    extract(node.get('objekto'), 'objekto', weight)
                    for a in node.get('aliaj', []):
                        extract(a, 'aliaj', weight)

        extract(ast)
        return roots

    def build_features(
        self,
        query_ast: Dict,
        doc_ast: Dict,
    ) -> torch.Tensor:
        """
        Build feature vector for (query, document) pair.

        Features:
        - Query embedding (embed_dim)
        - Document embedding (embed_dim)
        - Element-wise product (embed_dim)
        - Absolute difference (embed_dim)
        - Structural features (8d):
            - Has shared verb
            - Has shared subject
            - Root overlap ratio
            - Document has definition pattern ("X estas Y")
            - Question type encoding (one-hot: 4d)

        Args:
            query_ast: Query AST
            doc_ast: Document AST

        Returns:
            Feature tensor of shape (feature_dim,)
        """
        # Encode ASTs
        query_emb = self.encode_ast(query_ast)
        doc_emb = self.encode_ast(doc_ast)

        # Interaction features
        product = query_emb * doc_emb
        diff = torch.abs(query_emb - doc_emb)

        # Structural features
        structural = self._extract_structural_features(query_ast, doc_ast)

        # Concatenate all features
        features = torch.cat([
            query_emb,
            doc_emb,
            product,
            diff,
            structural,
        ])

        return features

    def _extract_structural_features(
        self,
        query_ast: Dict,
        doc_ast: Dict,
    ) -> torch.Tensor:
        """
        Extract structural features from AST pair.

        Returns:
            Tensor of shape (8,) containing:
            - has_shared_verb (1d)
            - has_shared_subject (1d)
            - root_overlap_ratio (1d)
            - has_definition_pattern (1d)
            - question_type_onehot (4d): WHO/WHAT/WHERE/OTHER
        """
        features = []

        # Extract roots
        query_roots = set(self._extract_roots(query_ast).keys())
        doc_roots = set(self._extract_roots(doc_ast).keys())

        # Shared verb
        query_verb = (query_ast.get('verbo') or {}).get('radiko', '').lower()
        doc_verb = (doc_ast.get('verbo') or {}).get('radiko', '').lower()
        has_shared_verb = float(bool(query_verb) and bool(doc_verb) and query_verb == doc_verb)
        features.append(has_shared_verb)

        # Shared subject
        query_subj_root = ''
        if query_ast.get('subjekto'):
            subj = query_ast['subjekto']
            if subj.get('tipo') == 'vortgrupo':
                query_subj_root = (subj.get('kerno') or {}).get('radiko', '').lower()
            elif subj.get('tipo') == 'vorto':
                query_subj_root = subj.get('radiko', '').lower()

        doc_subj_root = ''
        if doc_ast.get('subjekto'):
            subj = doc_ast['subjekto']
            if subj.get('tipo') == 'vortgrupo':
                doc_subj_root = (subj.get('kerno') or {}).get('radiko', '').lower()
            elif subj.get('tipo') == 'vorto':
                doc_subj_root = subj.get('radiko', '').lower()

        has_shared_subject = float(bool(query_subj_root) and bool(doc_subj_root) and query_subj_root == doc_subj_root)
        features.append(has_shared_subject)

        # Root overlap ratio
        if query_roots:
            overlap_ratio = len(query_roots & doc_roots) / len(query_roots)
        else:
            overlap_ratio = 0.0
        features.append(overlap_ratio)

        # Definition pattern: "X estas Y" for "Kio estas X?" queries
        has_def_pattern = float(
            doc_verb == 'est' and
            query_ast.get('fraztipo') == 'demando' and
            (query_ast.get('verbo') or {}).get('radiko', '').lower() == 'est'
        )
        features.append(has_def_pattern)

        # Question type (one-hot encoding)
        question_word = ''
        if query_ast.get('fraztipo') == 'demando':
            # Look for question word in query
            for node in [query_ast.get('subjekto'), query_ast.get('objekto')] + query_ast.get('aliaj', []):
                if node and isinstance(node, dict):
                    if node.get('tipo') == 'vorto':
                        root = node.get('radiko', '').lower()
                        if root in ['ki', 'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel']:
                            question_word = root
                            break

        # One-hot: [KIU/WHO, KIO/WHAT, KIE/WHERE, OTHER]
        q_type_onehot = [
            float(question_word in ['kiu']),      # WHO
            float(question_word in ['kio']),      # WHAT
            float(question_word in ['kie', 'kiam']),  # WHERE/WHEN
            float(question_word not in ['kiu', 'kio', 'kie', 'kiam']),  # OTHER
        ]
        features.extend(q_type_onehot)

        return torch.tensor(features, dtype=torch.float32)

    def forward(
        self,
        query_ast: Dict,
        doc_ast: Dict,
    ) -> torch.Tensor:
        """
        Score relevance of document to query.

        Args:
            query_ast: Parsed query AST
            doc_ast: Parsed document AST

        Returns:
            Relevance score in range [0, 1]
        """
        features = self.build_features(query_ast, doc_ast)
        score = self.scorer(features.unsqueeze(0))
        return score.squeeze(0)

    def score_batch(
        self,
        query_ast: Dict,
        doc_asts: List[Dict],
    ) -> torch.Tensor:
        """
        Score multiple documents against single query.

        Args:
            query_ast: Parsed query AST
            doc_asts: List of parsed document ASTs

        Returns:
            Tensor of relevance scores, shape (num_docs,)
        """
        features_batch = []
        for doc_ast in doc_asts:
            features = self.build_features(query_ast, doc_ast)
            features_batch.append(features)

        features_batch = torch.stack(features_batch)
        scores = self.scorer(features_batch)
        return scores.squeeze(1)

    def save(self, path: Path):
        """Save reranker model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'scorer_state_dict': self.scorer.state_dict(),
            'embed_dim': self.embed_dim,
            'feature_dim': self.feature_dim,
            'hidden_dims': self.scorer.hidden_dims,
        }, path)

        logger.info(f"Reranker saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        compositional_embedding: CompositionalEmbedding,
    ) -> 'ASTReranker':
        """
        Load reranker model.

        Args:
            path: Path to saved model
            compositional_embedding: Compositional embedding model

        Returns:
            Loaded ASTReranker
        """
        checkpoint = torch.load(path, map_location='cpu')

        reranker = cls(
            compositional_embedding=compositional_embedding,
            freeze_embedding=True,
            hidden_dims=checkpoint.get('hidden_dims', [256, 256, 128]),
        )

        reranker.scorer.load_state_dict(checkpoint['scorer_state_dict'])

        logger.info(f"Reranker loaded from {path}")
        return reranker


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
