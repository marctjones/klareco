#!/usr/bin/env python3
"""
Hybrid embeddings that combine linguistic and topical embeddings intelligently.

Key insight: Different words have different types of semantic meaning:
- Content words (hundo, bela): BOTH linguistic + topical
- Proper nouns (Parizo, Napoleono): ONLY topical
- Function words (kaj, de, la): NEITHER (handled by AST)
- Rare/technical terms: ONLY topical (if not in linguistic vocab)

This class:
1. Loads linguistic and topical models separately
2. Looks up each root in both vocabularies
3. Returns available embeddings (may be 64d, 128d, or None)
4. Handles missing gracefully with configurable fallbacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Union, Literal, Tuple
import logging

from .linguistic_embeddings import LinguisticEmbeddings
from .topical_embeddings import TopicalEmbeddings

logger = logging.getLogger(__name__)


class HybridEmbeddings(nn.Module):
    """
    Hybrid embedding model combining linguistic and topical embeddings.

    Intelligently combines embeddings based on what's available:
    - Both available: concatenate [ling_64d, top_64d] = 128d
    - Only topical: [zeros_64d, top_64d] = 128d (or just 64d if pad=False)
    - Only linguistic: [ling_64d, zeros_64d] = 128d (or just 64d if pad=False)
    - Neither: None or zeros based on settings

    Args:
        linguistic_model: Pre-loaded LinguisticEmbeddings model
        topical_model: Pre-loaded TopicalEmbeddings model
        pad_missing: If True, pad missing embeddings with zeros to always return 128d
                     If False, return only available dimensions (64d or 128d)
        default_mode: Default combination mode
    """

    def __init__(
        self,
        linguistic_model: LinguisticEmbeddings,
        topical_model: TopicalEmbeddings,
        pad_missing: bool = True,
        default_mode: Literal['linguistic', 'topical', 'hybrid'] = 'hybrid'
    ):
        super().__init__()

        self.linguistic_model = linguistic_model
        self.topical_model = topical_model
        self.pad_missing = pad_missing
        self.default_mode = default_mode

        # Cache vocabulary sizes
        self.linguistic_vocab_size = linguistic_model.vocab_size
        self.topical_vocab_size = topical_model.vocab_size

        logger.info(f"HybridEmbeddings initialized:")
        logger.info(f"  Linguistic vocab: {self.linguistic_vocab_size:,} roots")
        logger.info(f"  Topical vocab: {self.topical_vocab_size:,} roots")
        logger.info(f"  Pad missing: {pad_missing}")
        logger.info(f"  Default mode: {default_mode}")

    @classmethod
    def from_checkpoints(
        cls,
        linguistic_checkpoint: Union[str, Path],
        topical_checkpoint: Union[str, Path],
        pad_missing: bool = True,
        default_mode: Literal['linguistic', 'topical', 'hybrid'] = 'hybrid'
    ) -> 'HybridEmbeddings':
        """
        Load hybrid model from separate linguistic and topical checkpoints.

        Args:
            linguistic_checkpoint: Path to linguistic model checkpoint
            topical_checkpoint: Path to topical model checkpoint
            pad_missing: Whether to pad missing embeddings
            default_mode: Default combination mode

        Returns:
            Loaded HybridEmbeddings model
        """
        logger.info(f"Loading linguistic model from {linguistic_checkpoint}")
        linguistic_model = LinguisticEmbeddings.from_checkpoint(linguistic_checkpoint)

        logger.info(f"Loading topical model from {topical_checkpoint}")
        topical_model = TopicalEmbeddings.from_checkpoint(topical_checkpoint)

        return cls(
            linguistic_model=linguistic_model,
            topical_model=topical_model,
            pad_missing=pad_missing,
            default_mode=default_mode
        )

    def get_root_embedding(
        self,
        root: str,
        mode: Optional[Literal['linguistic', 'topical', 'hybrid']] = None,
        return_mask: bool = False
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Dict[str, bool]]]:
        """
        Get embedding for a root word (string lookup).

        Args:
            root: Root word string
            mode: Which embedding(s) to use (default: self.default_mode)
            return_mask: If True, also return which embeddings were available

        Returns:
            Embedding tensor or None if not available in requested mode
            If return_mask=True: (embedding, {'linguistic': bool, 'topical': bool})
        """
        if mode is None:
            mode = self.default_mode

        # Check availability
        has_ling = self.linguistic_model.has_root(root)
        has_top = self.topical_model.has_root(root)

        mask = {'linguistic': has_ling, 'topical': has_top}

        # Get embeddings based on mode
        with torch.no_grad():
            if mode == 'linguistic':
                if not has_ling:
                    return (None, mask) if return_mask else None
                emb = self.linguistic_model.get_root_embedding(root)

            elif mode == 'topical':
                if not has_top:
                    return (None, mask) if return_mask else None
                emb = self.topical_model.get_root_embedding(root)

            else:  # hybrid
                # If neither embedding available, return None
                if not has_ling and not has_top:
                    return (None, mask) if return_mask else None

                embs = []

                # Add linguistic if available
                if has_ling:
                    embs.append(self.linguistic_model.get_root_embedding(root))
                elif self.pad_missing:
                    embs.append(torch.zeros(self.linguistic_model.embedding_dim))

                # Add topical if available
                if has_top:
                    embs.append(self.topical_model.get_root_embedding(root))
                elif self.pad_missing:
                    embs.append(torch.zeros(self.topical_model.embedding_dim))

                # Combine
                if not embs:
                    return (None, mask) if return_mask else None

                emb = torch.cat(embs)

        return (emb, mask) if return_mask else emb

    def get_batch_embeddings(
        self,
        roots: list[str],
        mode: Optional[Literal['linguistic', 'topical', 'hybrid']] = None
    ) -> torch.Tensor:
        """
        Get embeddings for a batch of root words.

        Args:
            roots: List of root word strings
            mode: Which embedding(s) to use (default: self.default_mode)

        Returns:
            Embeddings tensor (shape: [batch, dim])
            Missing roots get zero vectors if pad_missing=True, else raises error
        """
        if mode is None:
            mode = self.default_mode

        embeddings = []
        for root in roots:
            emb = self.get_root_embedding(root, mode=mode)
            if emb is None:
                if self.pad_missing:
                    # Use zero vector for missing roots
                    dim = self.get_output_dim(mode)
                    emb = torch.zeros(dim)
                else:
                    raise ValueError(f"Root '{root}' not in {mode} vocabulary and pad_missing=False")
            embeddings.append(emb)

        return torch.stack(embeddings)

    def get_output_dim(
        self,
        mode: Optional[Literal['linguistic', 'topical', 'hybrid']] = None
    ) -> int:
        """
        Get output dimension for a given mode.

        Args:
            mode: Which embedding mode

        Returns:
            Output dimension (64 for single mode, 128 for hybrid if pad_missing=True)
        """
        if mode is None:
            mode = self.default_mode

        if mode == 'linguistic':
            return self.linguistic_model.embedding_dim
        elif mode == 'topical':
            return self.topical_model.embedding_dim
        else:  # hybrid
            if self.pad_missing:
                return self.linguistic_model.embedding_dim + self.topical_model.embedding_dim
            else:
                # Variable size - can't determine without knowing specific root
                return 128  # Maximum possible

    def compute_similarity(
        self,
        root1: str,
        root2: str,
        mode: Optional[Literal['linguistic', 'topical', 'hybrid']] = None
    ) -> Optional[float]:
        """
        Compute cosine similarity between two roots.

        Args:
            root1: First root word
            root2: Second root word
            mode: Which embedding(s) to use

        Returns:
            Cosine similarity score or None if either root not available
        """
        emb1 = self.get_root_embedding(root1, mode=mode)
        emb2 = self.get_root_embedding(root2, mode=mode)

        if emb1 is None or emb2 is None:
            return None

        # Cosine similarity
        emb1_norm = F.normalize(emb1.unsqueeze(0), dim=-1)
        emb2_norm = F.normalize(emb2.unsqueeze(0), dim=-1)
        sim = (emb1_norm * emb2_norm).sum().item()

        return sim

    def get_vocabulary_info(self) -> dict:
        """
        Get information about vocabularies and their overlap.

        Returns:
            Dictionary with vocabulary statistics
        """
        ling_roots = set(self.linguistic_model.root_to_idx.keys())
        top_roots = set(self.topical_model.root_to_idx.keys())

        overlap = ling_roots & top_roots
        ling_only = ling_roots - top_roots
        top_only = top_roots - ling_roots

        return {
            'linguistic_vocab_size': len(ling_roots),
            'topical_vocab_size': len(top_roots),
            'overlap_size': len(overlap),
            'linguistic_only': len(ling_only),
            'topical_only': len(top_only),
            'overlap_percentage': len(overlap) / len(ling_roots) * 100 if ling_roots else 0
        }

    def analyze_root(self, root: str) -> dict:
        """
        Analyze what embeddings are available for a root.

        Args:
            root: Root word to analyze

        Returns:
            Dictionary with analysis results
        """
        emb, mask = self.get_root_embedding(root, mode='hybrid', return_mask=True)

        info = {
            'root': root,
            'has_linguistic': mask['linguistic'],
            'has_topical': mask['topical'],
            'embedding_available': emb is not None
        }

        if emb is not None:
            info['embedding_dim'] = emb.shape[0]

        # Classify root type
        if mask['linguistic'] and mask['topical']:
            info['type'] = 'content_word'  # Normal content word
        elif mask['topical'] and not mask['linguistic']:
            info['type'] = 'proper_noun_or_rare'  # Likely proper noun or rare term
        elif mask['linguistic'] and not mask['topical']:
            info['type'] = 'linguistic_only'  # In linguistic vocab but not seen in corpus
        else:
            info['type'] = 'unknown'  # Not in either vocab

        return info

    def freeze_linguistic(self):
        """Freeze linguistic embeddings (stop gradient updates)."""
        for param in self.linguistic_model.parameters():
            param.requires_grad = False
        logger.info("Froze linguistic embeddings")

    def freeze_topical(self):
        """Freeze topical embeddings (stop gradient updates)."""
        for param in self.topical_model.parameters():
            param.requires_grad = False
        logger.info("Froze topical embeddings")

    def unfreeze_all(self):
        """Unfreeze both embedding layers."""
        for param in self.linguistic_model.parameters():
            param.requires_grad = True
        for param in self.topical_model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all embeddings")
