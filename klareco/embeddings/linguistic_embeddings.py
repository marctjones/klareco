#!/usr/bin/env python3
"""
Linguistic embeddings for Klareco.

Captures semantic and linguistic similarity based on:
- ReVo dictionary semantic relations (synonyms, antonyms, hypernyms)
- Ekzercaro word similarity from curated pairs
- Fundamento core vocabulary

Examples:
- "hundo" similar to: kato, ĉevalo, besto (animals)
- "bela" similar to: belega, ĉarma, agrabla (beauty-related)
- NOT similar: "Parizo" (proper noun - no linguistic meaning)

Vocabulary: Curated content words with semantic meaning (~15K roots)
Training: Semantic similarity pairs + contrastive learning
Size: Stable, manually curated (grows slowly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Union
import json


class LinguisticEmbeddings(nn.Module):
    """
    Linguistic (semantic) embedding model.

    One embedding per content root with semantic meaning.
    Vocabulary excludes proper nouns and function words.

    Args:
        vocab_size: Number of unique content roots
        embedding_dim: Embedding dimension (default: 64)
        root_to_idx: Optional vocabulary mapping (root -> index)
        idx_to_root: Optional reverse mapping (index -> root)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        root_to_idx: Optional[Dict[str, int]] = None,
        idx_to_root: Optional[Dict[int, str]] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Vocabulary mappings
        self.root_to_idx = root_to_idx or {}
        self.idx_to_root = idx_to_root or {}

        # Single embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with larger variance to spread embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.5)

    def forward(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get embeddings for given indices.

        Args:
            indices: Tensor of root indices (shape: [batch] or [batch, seq])

        Returns:
            Embeddings tensor (shape: [..., embedding_dim])
        """
        return self.embeddings(indices)

    def get_normalized(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get L2-normalized embeddings.

        Args:
            indices: Tensor of root indices

        Returns:
            L2-normalized embeddings
        """
        embeddings = self.forward(indices)
        return F.normalize(embeddings, dim=-1)

    def similarity(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two root embeddings.

        Args:
            idx1: First root indices
            idx2: Second root indices

        Returns:
            Cosine similarity scores (range: [-1, 1])
        """
        emb1 = self.get_normalized(idx1)
        emb2 = self.get_normalized(idx2)
        return (emb1 * emb2).sum(dim=-1)

    def get_root_embedding(self, root: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a root word (string lookup).

        Args:
            root: Root word string

        Returns:
            Embedding tensor (shape: [embedding_dim]) or None if not in vocab
        """
        if root not in self.root_to_idx:
            return None

        idx = self.root_to_idx[root]
        with torch.no_grad():
            return self.forward(torch.tensor([idx]))[0]

    def has_root(self, root: str) -> bool:
        """Check if root is in vocabulary."""
        return root in self.root_to_idx

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path]) -> 'LinguisticEmbeddings':
        """
        Load model from checkpoint.

        Supports both new format and legacy RootEmbeddings checkpoints.

        Args:
            checkpoint_path: Path to saved checkpoint

        Returns:
            Loaded LinguisticEmbeddings model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
            vocab_size = config['vocab_size']
            embedding_dim = config['embedding_dim']
        else:
            # Legacy format (from train_root_embeddings.py)
            vocab_size = checkpoint['vocab_size']
            embedding_dim = checkpoint.get('embedding_dim', 64)

        # Extract vocabulary
        root_to_idx = checkpoint.get('root_to_idx', {})
        idx_to_root = checkpoint.get('idx_to_root', {})

        # Create model
        model = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            root_to_idx=root_to_idx,
            idx_to_root=idx_to_root
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'embeddings.weight' in state_dict:
                model.embeddings.weight.data = state_dict['embeddings.weight']
            else:
                model.load_state_dict(state_dict)
        else:
            # Direct weights
            model.embeddings.weight.data = checkpoint['embeddings.weight']

        return model

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        **extra_metadata
    ):
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            **extra_metadata: Additional metadata to save (epoch, loss, etc.)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim
            },
            'root_to_idx': self.root_to_idx,
            'idx_to_root': self.idx_to_root,
            **extra_metadata
        }

        torch.save(checkpoint, checkpoint_path)

    def get_embedding_statistics(self) -> dict:
        """
        Compute statistics about the embedding space (for debugging/analysis).

        Returns:
            Dictionary with mean similarity, std, min, max across random samples
        """
        with torch.no_grad():
            # Sample 200 random embeddings
            device = self.embeddings.weight.device
            sample_indices = torch.randint(0, self.vocab_size, (200,), device=device)
            sample_embs = self.get_normalized(sample_indices)

            # Compute pairwise cosine similarities
            sim_matrix = sample_embs @ sample_embs.T

            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            similarities = sim_matrix[mask]

            return {
                'mean_similarity': similarities.mean().item(),
                'std_similarity': similarities.std().item(),
                'min_similarity': similarities.min().item(),
                'max_similarity': similarities.max().item(),
                'num_samples': 200
            }
