"""
Hybrid Word Encoder for Esperanto (v2.0)

Combines three sources of semantic information:
1. **Learned root embeddings** (128D) - from corpus co-occurrence
2. **Deterministic affix features** (12D) - from grammar rules (100% accurate)
3. **Lexicon features** (12D) - from hand-curated semantic annotations

Philosophy (from CLAUDE.md):
    "Make grammar, morphology, and linguistic structure 100% programmatic,
     focus learned capacity on reasoning"

Affixes are GRAMMAR → deterministic
Roots are LEXICAL → learned or looked up
Result: 152D word representation (128D + 24D features)

Usage:
    >>> encoder = HybridWordEncoder(root_embedder)
    >>> word_data = {
    ...     'text': 'pomisto',
    ...     'root': 'pom',
    ...     'affixes': ['ist'],
    ...     'pos': 'substantivo'
    ... }
    >>> emb = encoder.encode(word_data)
    >>> emb.shape
    torch.Size([152])  # 128D root + 12D affix + 12D lexicon
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from klareco.morphology.affix_semantics import (
    get_affix_features,
    ANIMACY_CLASSES,
    TYPE_CLASSES
)
from klareco.morphology.root_lexicon import (
    ROOT_LEXICON,
    get_root_features
)

logger = logging.getLogger(__name__)


class HybridWordEncoder(nn.Module):
    """
    Encode Esperanto words using learned + deterministic features.

    Architecture:
        word_data (dict)
           ├─→ root → Root Embedder → 128D (learned)
           ├─→ affixes → Affix Rules → 12D one-hot (deterministic)
           └─→ root → Lexicon Lookup → 12D one-hot (deterministic)

        Concatenate → 152D hybrid representation
    """

    def __init__(
        self,
        root_embedder,
        embed_dim: int = 128,
        use_lexicon: bool = True,
        use_affix_rules: bool = True
    ):
        """
        Args:
            root_embedder: Module that embeds roots (e.g., HybridRootEmbedder)
            embed_dim: Dimension of root embeddings (default: 128)
            use_lexicon: Whether to include lexicon features
            use_affix_rules: Whether to include affix rule features
        """
        super().__init__()

        self.root_embedder = root_embedder
        self.embed_dim = embed_dim
        self.use_lexicon = use_lexicon
        self.use_affix_rules = use_affix_rules

        # Calculate output dimension
        self.output_dim = embed_dim  # Root embedding
        if use_affix_rules:
            self.output_dim += len(ANIMACY_CLASSES) + len(TYPE_CLASSES)  # 4 + 16 = 20D
        if use_lexicon:
            self.output_dim += len(ANIMACY_CLASSES) + len(TYPE_CLASSES)  # 4 + 16 = 20D

        logger.info(f"HybridWordEncoder initialized:")
        logger.info(f"  Root embedding: {embed_dim}D")
        logger.info(f"  Affix features: {len(ANIMACY_CLASSES) + len(TYPE_CLASSES)}D (enabled: {use_affix_rules})")
        logger.info(f"  Lexicon features: {len(ANIMACY_CLASSES) + len(TYPE_CLASSES)}D (enabled: {use_lexicon})")
        logger.info(f"  Total output: {self.output_dim}D")

    def encode_animacy(self, animacy: str) -> torch.Tensor:
        """One-hot encode animacy feature."""
        idx = ANIMACY_CLASSES.index(animacy) if animacy in ANIMACY_CLASSES else ANIMACY_CLASSES.index('unknown')
        vec = torch.zeros(len(ANIMACY_CLASSES))
        vec[idx] = 1.0
        return vec

    def encode_type(self, type_str: str) -> torch.Tensor:
        """One-hot encode type feature."""
        idx = TYPE_CLASSES.index(type_str) if type_str in TYPE_CLASSES else TYPE_CLASSES.index('unknown')
        vec = torch.zeros(len(TYPE_CLASSES))
        vec[idx] = 1.0
        return vec

    def get_affix_features(self, word_data: Dict) -> torch.Tensor:
        """
        Get deterministic features from affixes (100% accurate, 0 parameters).

        Args:
            word_data: Dictionary with 'affixes' key

        Returns:
            Tensor of shape (20,): animacy (4D) + type (16D)
        """
        affixes = word_data.get('affixes', [])

        # Get affix semantics from deterministic rules
        affix_sem = get_affix_features(affixes)

        # One-hot encode
        animacy_vec = self.encode_animacy(affix_sem.get('animacy', 'unknown'))
        type_vec = self.encode_type(affix_sem.get('type', 'unknown'))

        return torch.cat([animacy_vec, type_vec])

    def get_lexicon_features(self, word_data: Dict) -> torch.Tensor:
        """
        Get features from hand-curated lexicon (if available).

        Args:
            word_data: Dictionary with 'root' key

        Returns:
            Tensor of shape (20,): animacy (4D) + type (16D)
        """
        root = word_data.get('root', '')

        # Lookup in lexicon
        lex_features = get_root_features(root)

        # One-hot encode
        animacy_vec = self.encode_animacy(lex_features.get('animacy', 'unknown'))
        type_vec = self.encode_type(lex_features.get('type', 'unknown'))

        return torch.cat([animacy_vec, type_vec])

    def encode(self, word_data: Dict) -> torch.Tensor:
        """
        Encode word using hybrid learned + deterministic features.

        Args:
            word_data: Dictionary with keys:
                - 'root': Root word (e.g., 'pom')
                - 'affixes': List of suffixes (e.g., ['ist'])
                - 'text': Full word (optional, for logging)

        Returns:
            Tensor of shape (output_dim,): hybrid representation
        """
        root = word_data.get('root', '')

        if not root:
            # Return zero vector if no root
            return torch.zeros(self.output_dim)

        # 1. Get learned root embedding (128D)
        with torch.no_grad():
            root_emb = self.root_embedder.get_embedding(root)
            if root_emb is None:
                logger.warning(f"Root '{root}' not in vocabulary, using zero vector")
                root_emb = torch.zeros(self.embed_dim)

        components = [root_emb]

        # 2. Get deterministic affix features (20D)
        if self.use_affix_rules:
            affix_features = self.get_affix_features(word_data)
            components.append(affix_features)

        # 3. Get lexicon features (20D)
        if self.use_lexicon:
            lex_features = self.get_lexicon_features(word_data)
            components.append(lex_features)

        # Concatenate all components
        return torch.cat(components)

    def encode_batch(self, word_data_list: List[Dict]) -> torch.Tensor:
        """
        Encode a batch of words.

        Args:
            word_data_list: List of word data dictionaries

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        embeddings = [self.encode(wd) for wd in word_data_list]
        return torch.stack(embeddings)

    def forward(self, word_data: Dict) -> torch.Tensor:
        """Forward pass (for nn.Module compatibility)."""
        return self.encode(word_data)


def explain_encoding(word_data: Dict, encoder: HybridWordEncoder) -> str:
    """
    Generate human-readable explanation of how word was encoded.

    Returns explanation string showing which features contributed.
    """
    root = word_data.get('root', '')
    affixes = word_data.get('affixes', [])
    text = word_data.get('text', root)

    lines = [f"\nEncoding: '{text}'"]
    lines.append("="*60)

    # Root contribution
    lines.append(f"\n1. ROOT '{root}' → {encoder.embed_dim}D learned embedding")
    # Check if root is in vocabulary
    emb = encoder.root_embedder.get_embedding(root)
    if emb is not None:
        lines.append(f"   ✓ Found in vocabulary")
    else:
        lines.append(f"   ⚠ Not in vocabulary (using zero vector)")

    # Affix contribution
    if encoder.use_affix_rules and affixes:
        affix_sem = get_affix_features(affixes)
        lines.append(f"\n2. AFFIXES {affixes} → 20D deterministic features")
        lines.append(f"   - Animacy: {affix_sem.get('animacy', 'unknown')}")
        lines.append(f"   - Type: {affix_sem.get('type', 'unknown')}")
        lines.append(f"   ✓ From grammar rules (100% accurate)")

    # Lexicon contribution
    if encoder.use_lexicon:
        lex_features = get_root_features(root)
        lines.append(f"\n3. LEXICON '{root}' → 20D hand-curated features")
        if root in ROOT_LEXICON:
            lines.append(f"   - Animacy: {lex_features.get('animacy', 'unknown')}")
            lines.append(f"   - Type: {lex_features.get('type', 'unknown')}")
            lines.append(f"   ✓ From hand-curated lexicon")
        else:
            lines.append(f"   - Not in lexicon (unknown)")

    lines.append(f"\n→ Total: {encoder.output_dim}D hybrid representation")
    lines.append("="*60)

    return "\n".join(lines)


if __name__ == '__main__':
    # Test the encoder
    from klareco.embeddings.hybrid import HybridRootEmbedder
    from pathlib import Path

    print("Loading root embedder...")
    production_path = Path("models/root_embeddings_phase1_fast/root_embeddings_best.pt")
    ast_path = Path("models/root_embeddings_fundamento_ast/root_embeddings_best.pt")

    if not production_path.exists() or not ast_path.exists():
        print(f"ERROR: Model files not found")
        print(f"  Production: {production_path} (exists: {production_path.exists()})")
        print(f"  AST: {ast_path} (exists: {ast_path.exists()})")
        print("\nRun training scripts first or skip model loading test")
        import sys
        sys.exit(1)

    root_embedder = HybridRootEmbedder(
        production_path=str(production_path),
        ast_path=str(ast_path)
    )

    print("\nCreating hybrid word encoder...")
    encoder = HybridWordEncoder(root_embedder)

    # Test cases
    test_words = [
        {'text': 'pomisto', 'root': 'pom', 'affixes': ['ist']},
        {'text': 'hundo', 'root': 'hund', 'affixes': []},
        {'text': 'manĝas', 'root': 'manĝ', 'affixes': []},
        {'text': 'tableto', 'root': 'tabl', 'affixes': ['et']},
    ]

    for word_data in test_words:
        emb = encoder.encode(word_data)
        print(f"\n{word_data['text']}: {emb.shape}")
        print(explain_encoding(word_data, encoder))
