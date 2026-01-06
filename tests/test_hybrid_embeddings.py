#!/usr/bin/env python3
"""
Tests for HybridEmbeddings combining linguistic and topical models.

Verifies:
- Independent vocabularies work correctly
- Smart combining based on availability
- Proper handling of missing embeddings
- Vocabulary overlap analysis
"""

import pytest
import torch
from klareco.embeddings.linguistic_embeddings import LinguisticEmbeddings
from klareco.embeddings.topical_embeddings import TopicalEmbeddings
from klareco.embeddings.hybrid_embeddings import HybridEmbeddings


@pytest.fixture
def linguistic_model():
    """Create small linguistic model for testing."""
    # Vocabulary: content words only (no proper nouns)
    roots = ['hund', 'kat', 'bel', 'grand', 'kur']
    root_to_idx = {root: idx for idx, root in enumerate(roots)}
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    return LinguisticEmbeddings(
        vocab_size=len(roots),
        embedding_dim=64,
        root_to_idx=root_to_idx,
        idx_to_root=idx_to_root
    )


@pytest.fixture
def topical_model():
    """Create small topical model for testing."""
    # Vocabulary: all corpus roots (including proper nouns)
    roots = ['hund', 'kat', 'bel', 'Parizo', 'Napoleono', 'algoritm']
    root_to_idx = {root: idx for idx, root in enumerate(roots)}
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    return TopicalEmbeddings(
        vocab_size=len(roots),
        embedding_dim=64,
        root_to_idx=root_to_idx,
        idx_to_root=idx_to_root
    )


@pytest.fixture
def hybrid_model(linguistic_model, topical_model):
    """Create hybrid model combining both."""
    return HybridEmbeddings(
        linguistic_model=linguistic_model,
        topical_model=topical_model,
        pad_missing=True
    )


def test_hybrid_initialization(hybrid_model):
    """Test hybrid model initializes correctly."""
    assert hybrid_model.linguistic_vocab_size == 5
    assert hybrid_model.topical_vocab_size == 6
    assert hybrid_model.pad_missing is True
    assert hybrid_model.default_mode == 'hybrid'


def test_content_word_both_embeddings(hybrid_model):
    """Content words should have both embeddings (128d)."""
    # 'hund' is in both vocabularies
    emb, mask = hybrid_model.get_root_embedding('hund', return_mask=True)

    assert emb is not None
    assert emb.shape == (128,)  # 64 + 64
    assert mask['linguistic'] is True
    assert mask['topical'] is True


def test_proper_noun_topical_only(hybrid_model):
    """Proper nouns should have only topical embedding."""
    # 'Parizo' is only in topical vocab
    emb, mask = hybrid_model.get_root_embedding('Parizo', return_mask=True)

    assert emb is not None
    assert emb.shape == (128,)  # 64 zeros + 64 topical
    assert mask['linguistic'] is False
    assert mask['topical'] is True

    # First 64 dimensions should be zeros (no linguistic embedding)
    assert torch.allclose(emb[:64], torch.zeros(64))
    # Last 64 dimensions should be non-zero (topical embedding)
    assert not torch.allclose(emb[64:], torch.zeros(64))


def test_linguistic_only_mode(hybrid_model):
    """Linguistic mode should return only linguistic embeddings."""
    # Content word - has linguistic
    emb = hybrid_model.get_root_embedding('hund', mode='linguistic')
    assert emb is not None
    assert emb.shape == (64,)

    # Proper noun - no linguistic
    emb = hybrid_model.get_root_embedding('Parizo', mode='linguistic')
    assert emb is None


def test_topical_only_mode(hybrid_model):
    """Topical mode should return only topical embeddings."""
    # Content word - has topical
    emb = hybrid_model.get_root_embedding('hund', mode='topical')
    assert emb is not None
    assert emb.shape == (64,)

    # Proper noun - has topical
    emb = hybrid_model.get_root_embedding('Parizo', mode='topical')
    assert emb is not None
    assert emb.shape == (64,)

    # Unknown word - no topical
    emb = hybrid_model.get_root_embedding('unknown', mode='topical')
    assert emb is None


def test_unknown_word_handling(hybrid_model):
    """Unknown words should return None or zeros based on pad_missing."""
    # Word not in either vocabulary
    emb = hybrid_model.get_root_embedding('foobar')
    assert emb is None  # Both missing = None even with pad_missing=True


def test_batch_embeddings(hybrid_model):
    """Test batch embedding lookup."""
    roots = ['hund', 'Parizo', 'kat']
    embs = hybrid_model.get_batch_embeddings(roots, mode='hybrid')

    assert embs.shape == (3, 128)

    # All should be non-zero (either real or padded)
    assert not torch.allclose(embs, torch.zeros(3, 128))


def test_vocabulary_overlap_analysis(hybrid_model):
    """Test vocabulary statistics."""
    info = hybrid_model.get_vocabulary_info()

    assert info['linguistic_vocab_size'] == 5
    assert info['topical_vocab_size'] == 6
    assert info['overlap_size'] == 3  # hund, kat, bel
    assert info['linguistic_only'] == 2  # grand, kur
    assert info['topical_only'] == 3  # Parizo, Napoleono, algoritm


def test_root_classification(hybrid_model):
    """Test automatic root type classification."""
    # Content word (both embeddings)
    info = hybrid_model.analyze_root('hund')
    assert info['type'] == 'content_word'
    assert info['has_linguistic'] is True
    assert info['has_topical'] is True

    # Proper noun (topical only)
    info = hybrid_model.analyze_root('Parizo')
    assert info['type'] == 'proper_noun_or_rare'
    assert info['has_linguistic'] is False
    assert info['has_topical'] is True

    # Linguistic only (rare case)
    info = hybrid_model.analyze_root('grand')
    assert info['type'] == 'linguistic_only'
    assert info['has_linguistic'] is True
    assert info['has_topical'] is False

    # Unknown
    info = hybrid_model.analyze_root('foobar')
    assert info['type'] == 'unknown'
    assert info['has_linguistic'] is False
    assert info['has_topical'] is False


def test_similarity_computation(hybrid_model):
    """Test similarity between roots."""
    # Both have both embeddings
    sim = hybrid_model.compute_similarity('hund', 'kat', mode='hybrid')
    assert sim is not None
    assert -1 <= sim <= 1

    # One has only topical (use topical mode)
    sim = hybrid_model.compute_similarity('Parizo', 'Napoleono', mode='topical')
    assert sim is not None

    # Can't compare in linguistic mode (Parizo not in linguistic vocab)
    sim = hybrid_model.compute_similarity('hund', 'Parizo', mode='linguistic')
    assert sim is None  # Parizo missing in linguistic


def test_output_dimensions(hybrid_model):
    """Test output dimension calculation."""
    assert hybrid_model.get_output_dim('linguistic') == 64
    assert hybrid_model.get_output_dim('topical') == 64
    assert hybrid_model.get_output_dim('hybrid') == 128


def test_freeze_operations(hybrid_model):
    """Test freezing/unfreezing embeddings."""
    # Initially trainable
    assert hybrid_model.linguistic_model.embeddings.weight.requires_grad is True
    assert hybrid_model.topical_model.embeddings.weight.requires_grad is True

    # Freeze linguistic
    hybrid_model.freeze_linguistic()
    assert hybrid_model.linguistic_model.embeddings.weight.requires_grad is False
    assert hybrid_model.topical_model.embeddings.weight.requires_grad is True

    # Unfreeze all
    hybrid_model.unfreeze_all()
    assert hybrid_model.linguistic_model.embeddings.weight.requires_grad is True
    assert hybrid_model.topical_model.embeddings.weight.requires_grad is True


def test_no_padding_mode():
    """Test hybrid model with pad_missing=False."""
    ling_roots = ['hund', 'kat']
    top_roots = ['hund', 'Parizo']

    ling_model = LinguisticEmbeddings(
        vocab_size=len(ling_roots),
        embedding_dim=64,
        root_to_idx={r: i for i, r in enumerate(ling_roots)},
        idx_to_root={i: r for i, r in enumerate(ling_roots)}
    )

    top_model = TopicalEmbeddings(
        vocab_size=len(top_roots),
        embedding_dim=64,
        root_to_idx={r: i for i, r in enumerate(top_roots)},
        idx_to_root={i: r for i, r in enumerate(top_roots)}
    )

    hybrid = HybridEmbeddings(ling_model, top_model, pad_missing=False)

    # Content word: both available = 128d
    emb = hybrid.get_root_embedding('hund')
    assert emb.shape == (128,)

    # Proper noun: only topical = 64d (no padding)
    emb = hybrid.get_root_embedding('Parizo')
    assert emb.shape == (64,)  # Only topical, no padding
