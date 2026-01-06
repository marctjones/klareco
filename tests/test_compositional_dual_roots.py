#!/usr/bin/env python3
"""
Tests for CompositionalEmbedding with dual root embeddings support.

Tests cover:
1. Backward compatibility (use_dual_roots=False)
2. Dual root embedding initialization
3. Mode switching (linguistic, topical, combined)
4. Loading linguistic embeddings from checkpoint
5. Freezing/unfreezing embeddings
6. Forward pass with dual roots
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from klareco.embeddings.compositional import CompositionalEmbedding


@pytest.fixture
def sample_vocabs():
    """Create sample vocabularies for testing."""
    root_vocab = {'<PAD>': 0, '<UNK>': 1, 'hund': 2, 'kat': 3, 'am': 4}
    prefix_vocab = {'<NONE>': 0, 're': 1, 'mal': 2}
    suffix_vocab = {'<NONE>': 0, 'ej': 1, 'et': 2}
    return root_vocab, prefix_vocab, suffix_vocab


class TestCompositionalEmbeddingBackwardCompatibility:
    """Test that existing single-embedding behavior is preserved."""

    def test_single_embedding_initialization(self, sample_vocabs):
        """Test traditional single embedding initialization."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=False
        )

        assert not model.use_dual_roots
        assert isinstance(model.root_embed, nn.Embedding)
        assert model.root_embed.weight.shape == (len(root_vocab), 128)
        assert model.root_dim == 128

    def test_single_embedding_forward(self, sample_vocabs):
        """Test forward pass with single embeddings."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=False
        )

        # Encode a word
        emb = model.encode_word(root='hund', ending='o')

        assert emb.shape == (128,)
        assert torch.is_tensor(emb)


class TestCompositionalEmbeddingDualRoots:
    """Test dual root embedding functionality."""

    def test_dual_embedding_initialization(self, sample_vocabs):
        """Test dual root embedding initialization."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True,
            root_embedding_mode='combined'
        )

        assert model.use_dual_roots
        assert model.root_embedding_mode == 'combined'
        # DualRootEmbeddings in combined mode: 128d (64d + 64d)
        assert model.root_dim == 128

    def test_dual_embedding_linguistic_mode(self, sample_vocabs):
        """Test dual embeddings in linguistic-only mode."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True,
            root_embedding_mode='linguistic'
        )

        assert model.root_dim == 64  # Linguistic only

    def test_dual_embedding_topical_mode(self, sample_vocabs):
        """Test dual embeddings in topical-only mode."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True,
            root_embedding_mode='topical'
        )

        assert model.root_dim == 64  # Topical only

    def test_set_root_embedding_mode(self, sample_vocabs):
        """Test changing root embedding mode at runtime."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True,
            root_embedding_mode='combined'
        )

        # Initially combined (128d)
        assert model.root_dim == 128

        # Switch to linguistic (64d)
        model.set_root_embedding_mode('linguistic')
        assert model.root_embedding_mode == 'linguistic'
        assert model.root_dim == 64

        # Switch to topical (64d)
        model.set_root_embedding_mode('topical')
        assert model.root_embedding_mode == 'topical'
        assert model.root_dim == 64

        # Switch back to combined (128d)
        model.set_root_embedding_mode('combined')
        assert model.root_embedding_mode == 'combined'
        assert model.root_dim == 128

    def test_set_mode_raises_without_dual_roots(self, sample_vocabs):
        """Test that setting mode raises error when use_dual_roots=False."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=False
        )

        with pytest.raises(RuntimeError, match="use_dual_roots=False"):
            model.set_root_embedding_mode('linguistic')

    def test_load_linguistic_embeddings(self, sample_vocabs):
        """Test loading linguistic embeddings from checkpoint."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        # Create a fake RootEmbeddings checkpoint
        vocab_size = len(root_vocab)
        embedding_dim = 64

        fake_checkpoint = {
            'model_state_dict': {
                'embeddings.weight': torch.randn(vocab_size, embedding_dim)
            },
            'epoch': 10,
            'loss': 0.5
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            torch.save(fake_checkpoint, checkpoint_path)

        try:
            model = CompositionalEmbedding(
                root_vocab=root_vocab,
                prefix_vocab=prefix_vocab,
                suffix_vocab=suffix_vocab,
                embed_dim=128,
                use_dual_roots=True,
                root_embedding_mode='combined'
            )

            # Save original topical embeddings
            original_topical = model.root_embed.topical_embeddings.weight.data.clone()

            # Load linguistic embeddings
            model.load_linguistic_embeddings(checkpoint_path)

            # Linguistic embeddings should match checkpoint
            assert torch.allclose(
                model.root_embed.linguistic_embeddings.weight.data,
                fake_checkpoint['model_state_dict']['embeddings.weight']
            )

            # Topical embeddings should be unchanged
            assert torch.allclose(model.root_embed.topical_embeddings.weight.data, original_topical)

        finally:
            Path(checkpoint_path).unlink()

    def test_load_linguistic_raises_without_dual_roots(self, sample_vocabs):
        """Test that loading linguistic embeddings raises error when use_dual_roots=False."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=False
        )

        with pytest.raises(RuntimeError, match="use_dual_roots=False"):
            model.load_linguistic_embeddings('/fake/path.pt')

    def test_freeze_linguistic_embeddings(self, sample_vocabs):
        """Test freezing linguistic embeddings."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True
        )

        # Initially both trainable
        assert model.root_embed.linguistic_embeddings.weight.requires_grad
        assert model.root_embed.topical_embeddings.weight.requires_grad

        # Freeze linguistic
        model.freeze_linguistic_embeddings()

        assert not model.root_embed.linguistic_embeddings.weight.requires_grad
        assert model.root_embed.topical_embeddings.weight.requires_grad

    def test_freeze_topical_embeddings(self, sample_vocabs):
        """Test freezing topical embeddings."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True
        )

        # Freeze topical
        model.freeze_topical_embeddings()

        assert model.root_embed.linguistic_embeddings.weight.requires_grad
        assert not model.root_embed.topical_embeddings.weight.requires_grad

    def test_unfreeze_all_embeddings(self, sample_vocabs):
        """Test unfreezing all embeddings."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True
        )

        # Freeze both
        model.freeze_linguistic_embeddings()
        model.freeze_topical_embeddings()

        assert not model.root_embed.linguistic_embeddings.weight.requires_grad
        assert not model.root_embed.topical_embeddings.weight.requires_grad

        # Unfreeze all
        model.unfreeze_all_embeddings()

        assert model.root_embed.linguistic_embeddings.weight.requires_grad
        assert model.root_embed.topical_embeddings.weight.requires_grad

    def test_freeze_raises_without_dual_roots(self, sample_vocabs):
        """Test that freezing methods raise error when use_dual_roots=False."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=False
        )

        with pytest.raises(RuntimeError, match="use_dual_roots=False"):
            model.freeze_linguistic_embeddings()

        with pytest.raises(RuntimeError, match="use_dual_roots=False"):
            model.freeze_topical_embeddings()

        with pytest.raises(RuntimeError, match="use_dual_roots=False"):
            model.unfreeze_all_embeddings()

    def test_forward_pass_with_dual_roots(self, sample_vocabs):
        """Test that forward pass works with dual roots."""
        root_vocab, prefix_vocab, suffix_vocab = sample_vocabs

        model = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=128,
            use_dual_roots=True,
            root_embedding_mode='combined'
        )

        # Encode a word
        emb = model.encode_word(root='hund', ending='o')

        # Should still produce 128d output (composition happens after root embedding)
        assert emb.shape == (128,)
        assert torch.is_tensor(emb)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
