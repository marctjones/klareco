#!/usr/bin/env python3
"""
Tests for morpheme-aware embeddings.

Part of Issue #43 - Morpheme-aware training for M1 corpus.
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from klareco.embeddings.morpheme_aware import (
    MorphemeAwareEmbedding,
    load_affix_config,
    create_morpheme_vocabularies
)


@pytest.fixture
def embedder():
    """Create a small morpheme-aware embedder for testing."""
    return MorphemeAwareEmbedding(
        root_vocab_size=100,
        semantic_prefix_count=8,
        semantic_suffix_count=15,
        grammatical_prefix_count=4,
        aspect_suffix_count=13,
        ending_count=17,
        semantic_dim=32,  # Smaller for testing
        grammatical_dim=8   # Smaller for testing
    )


def test_initialization(embedder):
    """Test that model initializes correctly."""
    assert embedder.semantic_dim == 32
    assert embedder.grammatical_dim == 8
    assert embedder.get_semantic_dim() == 32
    assert embedder.get_grammatical_dim() == 24  # 8 * 3
    assert embedder.get_output_dim() == 56  # 32 + 24


def test_forward_simple_root(embedder):
    """Test forward pass with just a root."""
    morphemes = {
        'root': 5,
        'ending': 1  # -o
    }

    result = embedder(morphemes)

    assert 'semantic' in result
    assert 'grammatical' in result
    assert 'combined' in result

    assert result['semantic'].shape == (32,)
    assert result['grammatical'].shape == (24,)
    assert result['combined'].shape == (56,)


def test_forward_with_semantic_prefix(embedder):
    """Test forward pass with semantic prefix."""
    morphemes = {
        'root': 5,
        'semantic_prefixes': [1],  # mal-
        'ending': 1
    }

    result = embedder(morphemes)

    assert result['semantic'].shape == (32,)
    assert result['grammatical'].shape == (24,)


def test_forward_with_semantic_suffix(embedder):
    """Test forward pass with semantic suffix."""
    morphemes = {
        'root': 5,
        'semantic_suffixes': [2],  # -ej
        'ending': 1
    }

    result = embedder(morphemes)

    assert result['semantic'].shape == (32,)


def test_forward_with_multiple_suffixes(embedder):
    """Test forward pass with multiple semantic suffixes."""
    morphemes = {
        'root': 5,
        'semantic_suffixes': [2, 3],  # -ej, -et
        'ending': 1
    }

    result = embedder(morphemes)

    assert result['semantic'].shape == (32,)


def test_forward_complex(embedder):
    """Test forward pass with all components."""
    morphemes = {
        'root': 5,
        'semantic_prefixes': [1],  # mal-
        'semantic_suffixes': [2, 3],  # -ej, -et
        'grammatical_prefix': 1,  # bo-
        'aspect_suffix': 2,  # -ig
        'ending': 2  # -oj
    }

    result = embedder(morphemes)

    assert result['semantic'].shape == (32,)
    assert result['grammatical'].shape == (24,)
    assert result['combined'].shape == (56,)


def test_same_root_similar_semantics(embedder):
    """Words with same root should have similar semantic embeddings."""
    # hundo (dog)
    morphemes1 = {
        'root': 10,
        'ending': 1  # -o
    }

    # hundoj (dogs)
    morphemes2 = {
        'root': 10,
        'ending': 2  # -oj
    }

    result1 = embedder(morphemes1)
    result2 = embedder(morphemes2)

    # Semantic should be very similar (same root)
    semantic_similarity = F.cosine_similarity(
        result1['semantic'].unsqueeze(0),
        result2['semantic'].unsqueeze(0)
    )

    assert semantic_similarity > 0.95, \
        f"Same root should have similar semantics: {semantic_similarity}"


def test_different_roots_different_semantics(embedder):
    """Words with different roots should have different semantic embeddings."""
    # hundo (dog)
    morphemes1 = {
        'root': 10,
        'ending': 1
    }

    # kato (cat)
    morphemes2 = {
        'root': 20,
        'ending': 1
    }

    result1 = embedder(morphemes1)
    result2 = embedder(morphemes2)

    # Semantic should be different (different roots)
    semantic_similarity = F.cosine_similarity(
        result1['semantic'].unsqueeze(0),
        result2['semantic'].unsqueeze(0)
    )

    # Should be less similar than same-root pairs
    assert semantic_similarity < 0.9, \
        f"Different roots should have different semantics: {semantic_similarity}"


def test_grammatical_change_only_affects_grammatical(embedder):
    """Grammatical changes should only affect grammatical component."""
    # hundo (nominative)
    morphemes1 = {
        'root': 10,
        'ending': 1  # -o
    }

    # hundon (accusative)
    morphemes2 = {
        'root': 10,
        'ending': 3  # -on
    }

    result1 = embedder(morphemes1)
    result2 = embedder(morphemes2)

    # Semantic should be nearly identical
    semantic_similarity = F.cosine_similarity(
        result1['semantic'].unsqueeze(0),
        result2['semantic'].unsqueeze(0)
    )
    assert semantic_similarity > 0.99, \
        "Grammar change shouldn't affect semantics"

    # Grammatical should be different
    grammatical_similarity = F.cosine_similarity(
        result1['grammatical'].unsqueeze(0),
        result2['grammatical'].unsqueeze(0)
    )
    assert grammatical_similarity < 0.9, \
        "Grammar change should affect grammatical component"


def test_semantic_suffix_changes_semantic(embedder):
    """Semantic suffix should change semantic embedding."""
    # hund (dog root)
    morphemes1 = {
        'root': 10,
        'ending': 1
    }

    # hundejo (kennel - dog place)
    morphemes2 = {
        'root': 10,
        'semantic_suffixes': [2],  # -ej
        'ending': 1
    }

    result1 = embedder(morphemes1)
    result2 = embedder(morphemes2)

    # Semantic should change (affix transforms meaning)
    semantic_similarity = F.cosine_similarity(
        result1['semantic'].unsqueeze(0),
        result2['semantic'].unsqueeze(0)
    )

    # Should be moderately similar (related but transformed)
    assert 0.5 < semantic_similarity < 0.9, \
        f"Semantic affix should transform meaning: {semantic_similarity}"


def test_batch_processing(embedder):
    """Test batch processing of multiple morphemes."""
    # Create batch with tensors
    batch_size = 4

    morphemes = {
        'root': torch.tensor([5, 10, 15, 20]),
        'ending': torch.tensor([1, 2, 1, 2])
    }

    result = embedder(morphemes)

    # Output should have batch dimension
    assert result['semantic'].shape == (32,)  # Still returns single embedding
    # Note: Current implementation doesn't handle batches
    # This test documents current behavior


def test_load_affix_config():
    """Test loading affix configuration."""
    config_path = Path(__file__).parent.parent / "config" / "affix_classification.json"

    if config_path.exists():
        config = load_affix_config(config_path)

        assert 'semantic_prefixes' in config
        assert 'semantic_suffixes' in config
        assert 'grammatical_prefixes' in config
        assert 'grammatical_suffixes' in config
        assert 'endings' in config

        # Check counts
        assert len(config['semantic_prefixes']['affixes']) == 8
        assert len(config['semantic_suffixes']['affixes']) == 15
        assert len(config['grammatical_prefixes']['affixes']) == 4
        assert len(config['grammatical_suffixes']['affixes']) == 13


def test_create_morpheme_vocabularies():
    """Test creating morpheme vocabularies from config."""
    config_path = Path(__file__).parent.parent / "config" / "affix_classification.json"

    if config_path.exists():
        config = load_affix_config(config_path)
        vocabs = create_morpheme_vocabularies(config)

        assert 'semantic_prefixes' in vocabs
        assert 'semantic_suffixes' in vocabs
        assert 'grammatical_prefixes' in vocabs
        assert 'grammatical_suffixes' in vocabs
        assert 'endings' in vocabs

        # Check <NONE> is index 0
        assert vocabs['semantic_prefixes']['<NONE>'] == 0
        assert vocabs['semantic_suffixes']['<NONE>'] == 0

        # Check mal is index 1 (first semantic prefix)
        assert vocabs['semantic_prefixes']['mal'] == 1

        # Check sizes
        assert len(vocabs['semantic_prefixes']) == 9  # 8 + <NONE>
        assert len(vocabs['semantic_suffixes']) == 16  # 15 + <NONE>


def test_embedding_norm(embedder):
    """Test that embeddings have reasonable norms."""
    morphemes = {
        'root': 5,
        'ending': 1
    }

    result = embedder(morphemes)

    # Check norms are reasonable (not too large or too small)
    semantic_norm = result['semantic'].norm().item()
    grammatical_norm = result['grammatical'].norm().item()
    combined_norm = result['combined'].norm().item()

    assert 0.1 < semantic_norm < 100, f"Semantic norm: {semantic_norm}"
    assert 0.1 < grammatical_norm < 100, f"Grammatical norm: {grammatical_norm}"
    assert 0.1 < combined_norm < 100, f"Combined norm: {combined_norm}"


def test_gradient_flow(embedder):
    """Test that gradients flow through the model."""
    morphemes = {
        'root': 5,
        'semantic_prefixes': [1],
        'semantic_suffixes': [2],
        'ending': 1
    }

    result = embedder(morphemes)

    # Create dummy loss
    loss = result['combined'].sum()
    loss.backward()

    # Check that embeddings have gradients
    assert embedder.root_embed.weight.grad is not None
    assert embedder.semantic_prefix_embed.weight.grad is not None
    assert embedder.semantic_suffix_embed.weight.grad is not None
    assert embedder.ending_embed.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
