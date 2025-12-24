#!/usr/bin/env python3
"""
Morpheme-aware embeddings for Esperanto.

Separates semantic and grammatical components:
- Semantic (128d): Roots and semantic affixes (learn meaning from context)
- Grammatical (16d): Aspectual suffixes and endings (encode linguistic properties)

Part of Issue #43 - Morpheme-aware training for M1 corpus.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import json
from pathlib import Path


class MorphemeAwareEmbedding(nn.Module):
    """
    Morpheme-aware embedding model that separates semantic and grammatical components.

    Architecture:
        Semantic components (learn MEANING):
        - Roots: hund, kat, bon, grand... (128d dense)
        - Semantic affixes: mal-, re-, -ej, -ul, -et... (128d dense)

        Grammatical components (encode PROPERTIES):
        - Aspectual suffixes: -ad, -ig, -iĝ... (16d sparse)
        - Grammatical prefixes: bo-, duon-, vic-... (16d sparse)
        - Endings: -o, -as, -is... (16d sparse)

    Parameters:
        root_vocab_size (int): Number of unique roots (default: 5000)
        semantic_dim (int): Dimension for semantic embeddings (default: 128)
        grammatical_dim (int): Dimension for grammatical embeddings (default: 16)

    Example:
        >>> embedder = MorphemeAwareEmbedding(root_vocab_size=5000)
        >>> morphemes = {
        ...     'root': 0,  # Index for 'hund'
        ...     'semantic_prefixes': [0],  # 'mal'
        ...     'semantic_suffixes': [1],  # 'ej'
        ...     'ending': 0  # 'o'
        ... }
        >>> result = embedder(morphemes)
        >>> result['semantic'].shape  # torch.Size([128])
        >>> result['grammatical'].shape  # torch.Size([48])
        >>> result['combined'].shape  # torch.Size([176])
    """

    def __init__(
        self,
        root_vocab_size: int = 5000,
        semantic_prefix_count: int = 8,
        semantic_suffix_count: int = 15,
        grammatical_prefix_count: int = 4,
        aspect_suffix_count: int = 13,
        ending_count: int = 17,
        semantic_dim: int = 128,
        grammatical_dim: int = 16,
        num_heads: int = 4
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.grammatical_dim = grammatical_dim

        # SEMANTIC EMBEDDINGS (learn from context)
        self.root_embed = nn.Embedding(
            root_vocab_size,
            semantic_dim,
            padding_idx=0  # Index 0 reserved for <PAD>
        )
        self.semantic_prefix_embed = nn.Embedding(
            semantic_prefix_count + 1,  # +1 for <NONE>
            semantic_dim,
            padding_idx=0
        )
        self.semantic_suffix_embed = nn.Embedding(
            semantic_suffix_count + 1,  # +1 for <NONE>
            semantic_dim,
            padding_idx=0
        )

        # GRAMMATICAL EMBEDDINGS (encode features)
        self.aspect_suffix_embed = nn.Embedding(
            aspect_suffix_count + 1,  # +1 for <NONE>
            grammatical_dim,
            padding_idx=0
        )
        self.grammatical_prefix_embed = nn.Embedding(
            grammatical_prefix_count + 1,  # +1 for <NONE>
            grammatical_dim,
            padding_idx=0
        )
        self.ending_embed = nn.Embedding(
            ending_count + 1,  # +1 for <NONE>
            grammatical_dim,
            padding_idx=0
        )

        # COMPOSITION (attention-based for semantic components)
        self.composer = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer normalization
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        self.grammatical_norm = nn.LayerNorm(grammatical_dim * 3)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize embeddings with Xavier uniform."""
        # Semantic embeddings
        nn.init.xavier_uniform_(self.root_embed.weight)
        nn.init.xavier_uniform_(self.semantic_prefix_embed.weight)
        nn.init.xavier_uniform_(self.semantic_suffix_embed.weight)

        # Grammatical embeddings (smaller initialization)
        nn.init.xavier_uniform_(self.aspect_suffix_embed.weight, gain=0.5)
        nn.init.xavier_uniform_(self.grammatical_prefix_embed.weight, gain=0.5)
        nn.init.xavier_uniform_(self.ending_embed.weight, gain=0.5)

    def forward(
        self,
        morphemes: Dict[str, Union[int, List[int], torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compose word embedding from morphemes.

        Args:
            morphemes: Dictionary with keys:
                - root: Root index (int or tensor)
                - semantic_prefixes: List of semantic prefix indices (optional)
                - semantic_suffixes: List of semantic suffix indices (optional)
                - grammatical_prefix: Grammatical prefix index (optional)
                - aspect_suffix: Aspectual suffix index (optional)
                - ending: Ending index

        Returns:
            Dictionary with:
                - semantic: 128d semantic embedding
                - grammatical: 48d grammatical feature vector
                - combined: 176d combined embedding
        """
        # Convert to tensors if needed
        root_idx = self._to_tensor(morphemes['root'])
        ending_idx = self._to_tensor(morphemes['ending'])

        # Compose semantic components
        semantic = self._compose_semantic(
            root_idx,
            morphemes.get('semantic_prefixes', []),
            morphemes.get('semantic_suffixes', [])
        )

        # Compose grammatical components
        grammatical = self._compose_grammatical(
            morphemes.get('grammatical_prefix', 0),
            morphemes.get('aspect_suffix', 0),
            ending_idx
        )

        # Combined embedding
        combined = torch.cat([semantic, grammatical], dim=-1)

        return {
            'semantic': semantic,
            'grammatical': grammatical,
            'combined': combined
        }

    def _compose_semantic(
        self,
        root_idx: torch.Tensor,
        prefix_indices: Union[List[int], torch.Tensor],
        suffix_indices: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compose semantic embedding using attention.

        Args:
            root_idx: Root index tensor
            prefix_indices: Semantic prefix indices
            suffix_indices: Semantic suffix indices

        Returns:
            Composed semantic embedding (128d)
        """
        batch_size = root_idx.shape[0] if root_idx.dim() > 0 else 1

        # Start with root embedding
        root_emb = self.root_embed(root_idx)
        if root_emb.dim() == 1:
            root_emb = root_emb.unsqueeze(0)  # [1, semantic_dim]

        components = [root_emb]

        # Add semantic prefixes
        if prefix_indices:
            prefix_idx = self._to_tensor(prefix_indices)
            if prefix_idx.dim() == 0:
                prefix_idx = prefix_idx.unsqueeze(0)
            prefix_emb = self.semantic_prefix_embed(prefix_idx)
            if prefix_emb.dim() == 1:
                prefix_emb = prefix_emb.unsqueeze(0)
            components.append(prefix_emb)

        # Add semantic suffixes
        if suffix_indices:
            suffix_idx = self._to_tensor(suffix_indices)
            if suffix_idx.dim() == 0:
                suffix_idx = suffix_idx.unsqueeze(0)
            suffix_emb = self.semantic_suffix_embed(suffix_idx)
            if suffix_emb.dim() == 1:
                suffix_emb = suffix_emb.unsqueeze(0)
            components.append(suffix_emb)

        # Compose with attention if multiple components
        if len(components) > 1:
            # Stack components: [batch, num_components, semantic_dim]
            component_stack = torch.cat(components, dim=0).unsqueeze(0)

            # Self-attention composition
            composed, _ = self.composer(
                component_stack,
                component_stack,
                component_stack
            )

            # Take mean of attended components
            semantic = composed.squeeze(0).mean(dim=0)
        else:
            semantic = components[0].squeeze(0)

        # Normalize
        semantic = self.semantic_norm(semantic)

        return semantic

    def _compose_grammatical(
        self,
        grammatical_prefix_idx: Union[int, torch.Tensor],
        aspect_suffix_idx: Union[int, torch.Tensor],
        ending_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose grammatical feature vector by concatenation.

        Args:
            grammatical_prefix_idx: Grammatical prefix index
            aspect_suffix_idx: Aspectual suffix index
            ending_idx: Ending index

        Returns:
            Grammatical feature vector (48d = 16d × 3)
        """
        # Convert to tensors
        gram_prefix_idx = self._to_tensor(grammatical_prefix_idx)
        aspect_idx = self._to_tensor(aspect_suffix_idx)

        # Get embeddings
        gram_prefix_emb = self.grammatical_prefix_embed(gram_prefix_idx)
        aspect_emb = self.aspect_suffix_embed(aspect_idx)
        ending_emb = self.ending_embed(ending_idx)

        # Ensure 1D tensors (squeeze batch dimension if needed)
        if gram_prefix_emb.dim() > 1:
            gram_prefix_emb = gram_prefix_emb.squeeze(0)
        if aspect_emb.dim() > 1:
            aspect_emb = aspect_emb.squeeze(0)
        if ending_emb.dim() > 1:
            ending_emb = ending_emb.squeeze(0)

        # Concatenate: [grammatical_dim * 3]
        grammatical = torch.cat([
            gram_prefix_emb,
            aspect_emb,
            ending_emb
        ], dim=-1)

        # Normalize
        grammatical = self.grammatical_norm(grammatical)

        return grammatical

    def _to_tensor(self, value: Union[int, List[int], torch.Tensor]) -> torch.Tensor:
        """Convert value to tensor."""
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, list):
            return torch.tensor(value, dtype=torch.long)
        else:
            return torch.tensor([value], dtype=torch.long)

    def get_output_dim(self) -> int:
        """Get combined output dimension."""
        return self.semantic_dim + self.grammatical_dim * 3

    def get_semantic_dim(self) -> int:
        """Get semantic embedding dimension."""
        return self.semantic_dim

    def get_grammatical_dim(self) -> int:
        """Get grammatical embedding dimension."""
        return self.grammatical_dim * 3


def load_affix_config(config_path: Path) -> Dict:
    """
    Load affix classification configuration.

    Args:
        config_path: Path to affix_classification.json

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return json.load(f)


def create_morpheme_vocabularies(
    affix_config: Dict
) -> Dict[str, Dict[str, int]]:
    """
    Create vocabulary mappings for morphemes.

    Args:
        affix_config: Affix classification configuration

    Returns:
        Dictionary with vocabulary mappings for each component
    """
    vocabularies = {}

    # Semantic prefixes
    semantic_prefixes = ['<NONE>'] + affix_config['semantic_prefixes']['affixes']
    vocabularies['semantic_prefixes'] = {
        prefix: idx for idx, prefix in enumerate(semantic_prefixes)
    }

    # Semantic suffixes
    semantic_suffixes = ['<NONE>'] + affix_config['semantic_suffixes']['affixes']
    vocabularies['semantic_suffixes'] = {
        suffix: idx for idx, suffix in enumerate(semantic_suffixes)
    }

    # Grammatical prefixes
    grammatical_prefixes = ['<NONE>'] + affix_config['grammatical_prefixes']['affixes']
    vocabularies['grammatical_prefixes'] = {
        prefix: idx for idx, prefix in enumerate(grammatical_prefixes)
    }

    # Grammatical suffixes (aspectual)
    grammatical_suffixes = ['<NONE>'] + affix_config['grammatical_suffixes']['affixes']
    vocabularies['grammatical_suffixes'] = {
        suffix: idx for idx, suffix in enumerate(grammatical_suffixes)
    }

    # Endings
    endings = ['<NONE>'] + affix_config['endings']['all_endings']
    vocabularies['endings'] = {
        ending: idx for idx, ending in enumerate(endings)
    }

    return vocabularies
