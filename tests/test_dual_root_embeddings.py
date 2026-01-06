#!/usr/bin/env python3
"""
Tests for DualRootEmbeddings class.

Tests cover:
1. Basic initialization and forward pass
2. Mode selection (linguistic, topical, combined)
3. L2 normalization
4. Similarity computation
5. Freezing/unfreezing parameters
6. Loading from checkpoint
7. Output dimension queries
8. Embedding statistics
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from klareco.embeddings.dual_root_embeddings import DualRootEmbeddings


class TestDualRootEmbeddings:
    """Test suite for DualRootEmbeddings."""

    def test_initialization(self):
        """Test basic initialization."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        assert model.vocab_size == 100
        assert model.embedding_dim == 64
        assert model.default_mode == 'combined'

        # Check both embedding layers exist
        assert isinstance(model.linguistic_embeddings, nn.Embedding)
        assert isinstance(model.topical_embeddings, nn.Embedding)

        # Check dimensions
        assert model.linguistic_embeddings.weight.shape == (100, 64)
        assert model.topical_embeddings.weight.shape == (100, 64)

    def test_forward_linguistic_mode(self):
        """Test forward pass in linguistic mode."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        emb = model(indices, mode='linguistic')

        assert emb.shape == (3, 64)
        assert torch.is_tensor(emb)

    def test_forward_topical_mode(self):
        """Test forward pass in topical mode."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        emb = model(indices, mode='topical')

        assert emb.shape == (3, 64)
        assert torch.is_tensor(emb)

    def test_forward_combined_mode(self):
        """Test forward pass in combined mode."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        emb = model(indices, mode='combined')

        assert emb.shape == (3, 128)  # 64 + 64 = 128
        assert torch.is_tensor(emb)

        # Verify concatenation: first 64d should be linguistic, next 64d should be topical
        emb_ling = model(indices, mode='linguistic')
        emb_topi = model(indices, mode='topical')

        assert torch.allclose(emb[:, :64], emb_ling)
        assert torch.allclose(emb[:, 64:], emb_topi)

    def test_default_mode(self):
        """Test that default mode is used when mode is None."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64, mode='linguistic')
        indices = torch.tensor([0, 5, 10])

        emb = model(indices)  # No mode specified

        assert emb.shape == (3, 64)  # Should use default mode='linguistic'

    def test_set_default_mode(self):
        """Test changing default mode."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64, mode='linguistic')
        indices = torch.tensor([0, 5, 10])

        emb1 = model(indices)
        assert emb1.shape == (3, 64)

        model.set_default_mode('combined')
        emb2 = model(indices)
        assert emb2.shape == (3, 128)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        with pytest.raises(ValueError, match="Invalid mode"):
            model(indices, mode='invalid')

    def test_get_normalized(self):
        """Test L2 normalization."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        # Test each mode
        for mode in ['linguistic', 'topical', 'combined']:
            emb = model.get_normalized(indices, mode=mode)

            # Check L2 norm is approximately 1.0 for each embedding
            norms = torch.norm(emb, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_similarity(self):
        """Test cosine similarity computation."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        idx1 = torch.tensor([0, 1, 2])
        idx2 = torch.tensor([0, 1, 5])  # First two same, last different

        sim = model.similarity(idx1, idx2, mode='linguistic')

        assert sim.shape == (3,)
        # Same index should have similarity ~1.0
        assert sim[0] > 0.99
        assert sim[1] > 0.99
        # Different indices will have variable similarity

    def test_get_output_dim(self):
        """Test output dimension query."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        assert model.get_output_dim('linguistic') == 64
        assert model.get_output_dim('topical') == 64
        assert model.get_output_dim('combined') == 128
        assert model.get_output_dim() == 128  # Default mode

    def test_freeze_linguistic(self):
        """Test freezing linguistic embeddings."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        # Initially both should require grad
        assert model.linguistic_embeddings.weight.requires_grad
        assert model.topical_embeddings.weight.requires_grad

        model.freeze_linguistic()

        # Linguistic frozen, topical still trainable
        assert not model.linguistic_embeddings.weight.requires_grad
        assert model.topical_embeddings.weight.requires_grad

    def test_freeze_topical(self):
        """Test freezing topical embeddings."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        model.freeze_topical()

        # Topical frozen, linguistic still trainable
        assert model.linguistic_embeddings.weight.requires_grad
        assert not model.topical_embeddings.weight.requires_grad

    def test_unfreeze_all(self):
        """Test unfreezing all embeddings."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        model.freeze_linguistic()
        model.freeze_topical()

        # Both frozen
        assert not model.linguistic_embeddings.weight.requires_grad
        assert not model.topical_embeddings.weight.requires_grad

        model.unfreeze_all()

        # Both unfrozen
        assert model.linguistic_embeddings.weight.requires_grad
        assert model.topical_embeddings.weight.requires_grad

    def test_load_linguistic_from_checkpoint(self):
        """Test loading linguistic embeddings from RootEmbeddings checkpoint."""
        # Create a fake RootEmbeddings checkpoint
        vocab_size = 100
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
            model = DualRootEmbeddings(vocab_size=vocab_size, embedding_dim=embedding_dim)

            # Save original topical embeddings to verify they don't change
            original_topical = model.topical_embeddings.weight.data.clone()

            model.load_linguistic_from_checkpoint(checkpoint_path)

            # Linguistic embeddings should match checkpoint
            assert torch.allclose(
                model.linguistic_embeddings.weight.data,
                fake_checkpoint['model_state_dict']['embeddings.weight']
            )

            # Topical embeddings should be unchanged
            assert torch.allclose(model.topical_embeddings.weight.data, original_topical)

        finally:
            Path(checkpoint_path).unlink()

    def test_load_linguistic_dimension_mismatch(self):
        """Test that loading checkpoint with wrong dimensions raises error."""
        vocab_size = 100
        embedding_dim = 64

        # Create checkpoint with different dimension
        fake_checkpoint = {
            'model_state_dict': {
                'embeddings.weight': torch.randn(vocab_size, 32)  # Wrong dim
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            torch.save(fake_checkpoint, checkpoint_path)

        try:
            model = DualRootEmbeddings(vocab_size=vocab_size, embedding_dim=embedding_dim)

            with pytest.raises(ValueError, match="Dimension mismatch"):
                model.load_linguistic_from_checkpoint(checkpoint_path)

        finally:
            Path(checkpoint_path).unlink()

    def test_get_embedding_statistics(self):
        """Test embedding statistics computation."""
        model = DualRootEmbeddings(vocab_size=1000, embedding_dim=64)

        stats = model.get_embedding_statistics(mode='linguistic')

        assert 'mode' in stats
        assert 'mean_similarity' in stats
        assert 'std_similarity' in stats
        assert 'min_similarity' in stats
        assert 'max_similarity' in stats
        assert 'num_samples' in stats

        assert stats['mode'] == 'linguistic'
        # Allow small floating point error
        assert -1.01 <= stats['mean_similarity'] <= 1.01
        assert stats['std_similarity'] >= 0
        assert -1.01 <= stats['min_similarity'] <= 1.01
        assert -1.01 <= stats['max_similarity'] <= 1.01

    def test_batch_processing(self):
        """Test processing batches of different shapes."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)

        # 1D tensor (batch)
        indices_1d = torch.tensor([0, 5, 10])
        emb_1d = model(indices_1d, mode='combined')
        assert emb_1d.shape == (3, 128)

        # 2D tensor (batch x seq)
        indices_2d = torch.tensor([[0, 5, 10], [1, 2, 3]])
        emb_2d = model(indices_2d, mode='combined')
        assert emb_2d.shape == (2, 3, 128)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        # Combined mode
        emb = model(indices, mode='combined')
        loss = emb.sum()
        loss.backward()

        # Both embeddings should have gradients
        assert model.linguistic_embeddings.weight.grad is not None
        assert model.topical_embeddings.weight.grad is not None

        # Clear gradients
        model.zero_grad()

        # Linguistic mode only
        emb_ling = model(indices, mode='linguistic')
        loss_ling = emb_ling.sum()
        loss_ling.backward()

        # Only linguistic should have gradients
        assert model.linguistic_embeddings.weight.grad is not None
        assert model.topical_embeddings.weight.grad is None

    def test_linguistic_topical_independence(self):
        """Test that linguistic and topical embeddings are independent."""
        model = DualRootEmbeddings(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([0, 5, 10])

        emb_ling = model(indices, mode='linguistic')
        emb_topi = model(indices, mode='topical')

        # Embeddings should be different (initialized independently)
        assert not torch.allclose(emb_ling, emb_topi)

        # Modifying linguistic shouldn't affect topical
        with torch.no_grad():
            model.linguistic_embeddings.weight[0] += 1.0

        emb_ling_after = model(indices, mode='linguistic')
        emb_topi_after = model(indices, mode='topical')

        # Linguistic changed
        assert not torch.allclose(emb_ling, emb_ling_after)
        # Topical unchanged
        assert torch.allclose(emb_topi, emb_topi_after)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
