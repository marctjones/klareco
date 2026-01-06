#!/usr/bin/env python3
"""
Dual parallel root embeddings for Klareco.

Implements two independent 64d embeddings per root:
1. Linguistic embedding: Trained on ReVo semantic relations + Ekzercaro co-occurrence
2. Topical embedding: Trained on corpus-level skip-gram context pairs

Total output: 128d (64d linguistic + 64d topical)

This addresses the retrieval accuracy problem where linguistic similarity alone
(e.g., hundo-kato-besto) doesn't capture topical/contextual similarity needed
for question answering.

Task #68: Core dual embeddings architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class DualRootEmbeddings(nn.Module):
    """
    Dual parallel root embeddings: linguistic + topical.

    Architecture:
    - Two independent nn.Embedding layers (64d each)
    - Supports three modes: 'linguistic', 'topical', 'combined'
    - Combined mode concatenates both embeddings (128d)
    - Backward compatible with single embedding systems

    Args:
        vocab_size: Number of unique roots in vocabulary
        embedding_dim: Dimension per embedding (default: 64)
        mode: Default embedding mode ('combined' recommended for retrieval)

    Example:
        >>> model = DualRootEmbeddings(vocab_size=5000, embedding_dim=64)
        >>> indices = torch.tensor([0, 1, 2])
        >>> emb = model(indices)  # Shape: (3, 128) in 'combined' mode
        >>> emb_ling = model(indices, mode='linguistic')  # Shape: (3, 64)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        mode: Literal['linguistic', 'topical', 'combined'] = 'combined'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.default_mode = mode

        # Two independent embedding layers
        self.linguistic_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.topical_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with larger variance to spread embeddings
        # Same initialization as RootEmbeddings for consistency
        nn.init.normal_(self.linguistic_embeddings.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.topical_embeddings.weight, mean=0.0, std=0.5)

    def forward(
        self,
        indices: torch.Tensor,
        mode: Optional[Literal['linguistic', 'topical', 'combined']] = None
    ) -> torch.Tensor:
        """
        Get embeddings for given indices.

        Args:
            indices: Tensor of root indices (shape: [batch] or [batch, seq])
            mode: Which embedding(s) to return (default: self.default_mode)
                - 'linguistic': Return only linguistic embeddings (64d)
                - 'topical': Return only topical embeddings (64d)
                - 'combined': Concatenate both embeddings (128d)

        Returns:
            Embeddings tensor (shape: [..., dim] where dim depends on mode)
        """
        if mode is None:
            mode = self.default_mode

        if mode == 'linguistic':
            return self.linguistic_embeddings(indices)
        elif mode == 'topical':
            return self.topical_embeddings(indices)
        elif mode == 'combined':
            ling = self.linguistic_embeddings(indices)
            topi = self.topical_embeddings(indices)
            return torch.cat([ling, topi], dim=-1)  # Concatenate on last dimension
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'linguistic', 'topical', or 'combined'")

    def get_normalized(
        self,
        indices: torch.Tensor,
        mode: Optional[Literal['linguistic', 'topical', 'combined']] = None
    ) -> torch.Tensor:
        """
        Get L2-normalized embeddings.

        Normalization is critical for cosine similarity computation in retrieval.

        Args:
            indices: Tensor of root indices
            mode: Which embedding(s) to return

        Returns:
            L2-normalized embeddings
        """
        embeddings = self.forward(indices, mode=mode)
        return F.normalize(embeddings, dim=-1)

    def similarity(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        mode: Optional[Literal['linguistic', 'topical', 'combined']] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two root embeddings.

        Args:
            idx1: First root indices
            idx2: Second root indices
            mode: Which embedding(s) to use for similarity

        Returns:
            Cosine similarity scores (range: [-1, 1])
        """
        emb1 = self.get_normalized(idx1, mode=mode)
        emb2 = self.get_normalized(idx2, mode=mode)
        return (emb1 * emb2).sum(dim=-1)

    def get_output_dim(self, mode: Optional[Literal['linguistic', 'topical', 'combined']] = None) -> int:
        """
        Get the output dimension for a given mode.

        Useful for downstream components that need to know embedding size.

        Args:
            mode: Which embedding mode

        Returns:
            Output dimension (64 for single, 128 for combined)
        """
        if mode is None:
            mode = self.default_mode

        if mode in ('linguistic', 'topical'):
            return self.embedding_dim
        else:  # combined
            return self.embedding_dim * 2

    def set_default_mode(self, mode: Literal['linguistic', 'topical', 'combined']):
        """
        Update the default embedding mode.

        Useful for switching between modes at inference time without
        changing all forward() calls.

        Args:
            mode: New default mode
        """
        if mode not in ('linguistic', 'topical', 'combined'):
            raise ValueError(f"Invalid mode: {mode}")
        self.default_mode = mode

    def freeze_linguistic(self):
        """
        Freeze linguistic embeddings (stop gradient updates).

        Useful for sequential training: train linguistic first, freeze,
        then train topical while keeping linguistic fixed.
        """
        self.linguistic_embeddings.weight.requires_grad = False

    def freeze_topical(self):
        """
        Freeze topical embeddings (stop gradient updates).

        Useful for sequential training: train topical first, freeze,
        then fine-tune linguistic.
        """
        self.topical_embeddings.weight.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreeze both embedding layers.

        Useful for joint fine-tuning after sequential training.
        """
        self.linguistic_embeddings.weight.requires_grad = True
        self.topical_embeddings.weight.requires_grad = True

    def load_linguistic_from_checkpoint(self, checkpoint_path: str):
        """
        Load pre-trained linguistic embeddings from a RootEmbeddings checkpoint.

        This allows migrating from the single-embedding baseline to dual embeddings
        by reusing the existing trained linguistic embeddings.

        Args:
            checkpoint_path: Path to checkpoint from train_root_embeddings.py
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract embeddings from RootEmbeddings checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # RootEmbeddings uses 'embeddings.weight', we use 'linguistic_embeddings.weight'
        if 'embeddings.weight' in state_dict:
            linguistic_weights = state_dict['embeddings.weight']

            # Verify dimensions match
            if linguistic_weights.shape != self.linguistic_embeddings.weight.shape:
                raise ValueError(
                    f"Dimension mismatch: checkpoint has {linguistic_weights.shape}, "
                    f"but model expects {self.linguistic_embeddings.weight.shape}"
                )

            # Load weights
            self.linguistic_embeddings.weight.data.copy_(linguistic_weights)
            print(f"Loaded linguistic embeddings from {checkpoint_path}")
        else:
            raise KeyError("Checkpoint does not contain 'embeddings.weight'")

    def get_embedding_statistics(self, mode: Optional[Literal['linguistic', 'topical', 'combined']] = None) -> dict:
        """
        Compute statistics about the embedding space (for debugging/analysis).

        Returns:
            Dictionary with mean similarity, std, min, max across random samples
        """
        if mode is None:
            mode = self.default_mode

        with torch.no_grad():
            # Sample 200 random embeddings
            device = self.linguistic_embeddings.weight.device
            sample_indices = torch.randint(0, self.vocab_size, (200,), device=device)
            sample_embs = self.get_normalized(sample_indices, mode=mode)

            # Compute pairwise cosine similarities
            sim_matrix = sample_embs @ sample_embs.T

            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            similarities = sim_matrix[mask]

            return {
                'mode': mode,
                'mean_similarity': similarities.mean().item(),
                'std_similarity': similarities.std().item(),
                'min_similarity': similarities.min().item(),
                'max_similarity': similarities.max().item(),
                'num_samples': 200
            }
