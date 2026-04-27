#!/usr/bin/env python3
"""
Index Integrity Tests.

Tests that the INDEX stage produced valid FAISS indexes with proper
embeddings and metadata.

Usage:
    pytest tests/test_index_integrity.py -v
"""

import json
import pytest
import numpy as np
from pathlib import Path


# Expected index locations
INDEX_PATHS = [
    Path('data/indexes/compositional'),
    Path('data/corpus_index_compositional'),
    Path('data/indexes/merged'),
]


class TestIndexExists:
    """Tests that index files exist."""

    @pytest.fixture
    def index_dir(self):
        """Find the index directory."""
        for path in INDEX_PATHS:
            if path.exists():
                return path
        return INDEX_PATHS[0]  # Default for error message

    def test_index_directory_exists(self, index_dir):
        """Index directory should exist (skip if data pipeline hasn't run)."""
        if not index_dir.exists():
            pytest.skip(f"Index not found — run index_kuzu.py first. Tried: {[str(p) for p in INDEX_PATHS]}")

    def test_embeddings_file_exists(self, index_dir):
        """Embeddings file should exist."""
        if not index_dir.exists():
            pytest.skip("Index directory not found")

        embeddings_file = index_dir / 'embeddings.npy'
        assert embeddings_file.exists(), \
            f"embeddings.npy not found in {index_dir}"

    def test_faiss_index_exists(self, index_dir):
        """FAISS index file should exist."""
        if not index_dir.exists():
            pytest.skip("Index directory not found")

        # Check for various FAISS file names
        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        assert len(faiss_files) > 0, \
            f"No FAISS index file found in {index_dir}"

    def test_metadata_exists(self, index_dir):
        """Metadata file should exist."""
        if not index_dir.exists():
            pytest.skip("Index directory not found")

        metadata_files = (
            list(index_dir.glob('*.jsonl')) +
            list(index_dir.glob('metadata.json')) +
            list(index_dir.glob('sentences.json'))
        )
        assert len(metadata_files) > 0, \
            f"No metadata file found in {index_dir}"


class TestEmbeddingsQuality:
    """Tests for embeddings file quality."""

    @pytest.fixture
    def index_dir(self):
        """Find the index directory."""
        for path in INDEX_PATHS:
            if path.exists():
                return path
        pytest.skip("No index directory found")

    @pytest.fixture
    def embeddings(self, index_dir):
        """Load embeddings array."""
        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")
        return np.load(emb_path)

    def test_embeddings_shape(self, embeddings):
        """Embeddings should be 2D array."""
        assert len(embeddings.shape) == 2, \
            f"Expected 2D array, got shape {embeddings.shape}"

    def test_embeddings_dimension(self, embeddings):
        """Embedding dimension should be reasonable (32-512)."""
        dim = embeddings.shape[1]
        assert 32 <= dim <= 512, \
            f"Unexpected embedding dimension: {dim}"

    def test_embeddings_count(self, embeddings):
        """Should have substantial number of embeddings."""
        count = embeddings.shape[0]
        # Should have at least 100K embeddings
        assert count >= 100_000, \
            f"Only {count:,} embeddings (expected 100K+)"

    def test_embeddings_dtype(self, embeddings):
        """Embeddings should be float32."""
        assert embeddings.dtype == np.float32, \
            f"Expected float32, got {embeddings.dtype}"

    def test_no_nan_values(self, embeddings):
        """Embeddings should not contain NaN."""
        nan_count = np.isnan(embeddings).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_no_inf_values(self, embeddings):
        """Embeddings should not contain Inf."""
        inf_count = np.isinf(embeddings).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values"

    def test_embeddings_not_zero(self, embeddings):
        """Embeddings should not be all zeros."""
        # Check first 1000 embeddings
        sample = embeddings[:1000]
        zero_rows = np.all(sample == 0, axis=1).sum()
        pct_zero = zero_rows / len(sample)
        assert pct_zero < 0.01, f"{pct_zero:.1%} of embeddings are all zeros"

    def test_embeddings_normalized(self, embeddings):
        """Embeddings should be approximately normalized (L2 norm ~1)."""
        # Sample first 1000
        sample = embeddings[:1000]
        norms = np.linalg.norm(sample, axis=1)

        # Allow some variation, but should be close to 1
        mean_norm = np.mean(norms)
        assert 0.5 < mean_norm < 2.0, \
            f"Mean norm {mean_norm:.2f} suggests unnormalized embeddings"

    def test_embeddings_diversity(self, embeddings):
        """Embeddings should be diverse (not collapsed)."""
        # Sample 100 random pairs
        n = min(1000, len(embeddings))
        sample = embeddings[:n]

        # Compute pairwise similarities for random pairs
        sims = []
        np.random.seed(42)
        for _ in range(100):
            i, j = np.random.choice(n, 2, replace=False)
            sim = np.dot(sample[i], sample[j]) / (
                np.linalg.norm(sample[i]) * np.linalg.norm(sample[j]) + 1e-8
            )
            sims.append(sim)

        mean_sim = np.mean(sims)
        # If mean similarity > 0.9, embeddings have collapsed
        assert mean_sim < 0.9, \
            f"Mean pairwise similarity {mean_sim:.2f} suggests embedding collapse"


class TestFAISSIndex:
    """Tests for FAISS index validity."""

    @pytest.fixture
    def index_dir(self):
        """Find the index directory."""
        for path in INDEX_PATHS:
            if path.exists():
                return path
        pytest.skip("No index directory found")

    def test_faiss_loadable(self, index_dir):
        """FAISS index should be loadable."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        if not faiss_files:
            pytest.skip("No FAISS index file found")

        faiss_file = faiss_files[0]
        try:
            index = faiss.read_index(str(faiss_file))
            assert index is not None
        except Exception as e:
            pytest.fail(f"Failed to load FAISS index: {e}")

    def test_faiss_dimensions_match(self, index_dir):
        """FAISS index dimensions should match embeddings."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        if not faiss_files:
            pytest.skip("No FAISS index file found")

        embeddings = np.load(emb_path)
        index = faiss.read_index(str(faiss_files[0]))

        assert index.d == embeddings.shape[1], \
            f"FAISS dim {index.d} != embeddings dim {embeddings.shape[1]}"

    def test_faiss_count_matches(self, index_dir):
        """FAISS index count should match embeddings."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        if not faiss_files:
            pytest.skip("No FAISS index file found")

        embeddings = np.load(emb_path)
        index = faiss.read_index(str(faiss_files[0]))

        assert index.ntotal == embeddings.shape[0], \
            f"FAISS count {index.ntotal} != embeddings count {embeddings.shape[0]}"

    def test_faiss_search_works(self, index_dir):
        """FAISS search should return results."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        if not faiss_files:
            pytest.skip("No FAISS index file found")

        embeddings = np.load(emb_path)
        index = faiss.read_index(str(faiss_files[0]))

        # Search with first embedding
        query = embeddings[0:1].astype(np.float32)
        D, I = index.search(query, k=5)

        assert I.shape == (1, 5), f"Unexpected search result shape: {I.shape}"
        assert I[0, 0] == 0, "First result should be the query itself"


class TestMetadataConsistency:
    """Tests for metadata consistency with embeddings."""

    @pytest.fixture
    def index_dir(self):
        """Find the index directory."""
        for path in INDEX_PATHS:
            if path.exists():
                return path
        pytest.skip("No index directory found")

    def test_metadata_count_matches(self, index_dir):
        """Metadata count should match embeddings."""
        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        embeddings = np.load(emb_path)
        emb_count = embeddings.shape[0]

        # Find metadata file
        metadata_files = list(index_dir.glob('*.jsonl'))
        if not metadata_files:
            # Try sentences.json or metadata.json
            for name in ['sentences.json', 'metadata.json']:
                if (index_dir / name).exists():
                    with open(index_dir / name) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            assert len(data) == emb_count, \
                                f"Metadata has {len(data)} entries, embeddings has {emb_count}"
                    return
            pytest.skip("No metadata file found")
            return

        # Count JSONL lines
        metadata_count = sum(1 for _ in open(metadata_files[0], encoding='utf-8'))
        assert metadata_count == emb_count, \
            f"Metadata has {metadata_count} entries, embeddings has {emb_count}"

    def test_metadata_has_text(self, index_dir):
        """Metadata entries should have text field."""
        metadata_files = list(index_dir.glob('*.jsonl'))
        if not metadata_files:
            pytest.skip("No JSONL metadata found")

        missing_text = 0
        total = 0

        with open(metadata_files[0], encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'text' not in entry:
                        missing_text += 1
                except:
                    pass

        assert missing_text == 0, f"{missing_text}/{total} entries missing text"


class TestIndexPerformance:
    """Performance-related tests for index."""

    @pytest.fixture
    def index_dir(self):
        """Find the index directory."""
        for path in INDEX_PATHS:
            if path.exists():
                return path
        pytest.skip("No index directory found")

    def test_search_latency(self, index_dir):
        """Search should complete in reasonable time."""
        import time
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        emb_path = index_dir / 'embeddings.npy'
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        faiss_files = list(index_dir.glob('*.bin')) + list(index_dir.glob('*.index'))
        if not faiss_files:
            pytest.skip("No FAISS index file found")

        embeddings = np.load(emb_path)
        index = faiss.read_index(str(faiss_files[0]))

        # Time 100 searches
        query = embeddings[0:1].astype(np.float32)
        start = time.time()
        for _ in range(100):
            index.search(query, k=10)
        elapsed = time.time() - start

        avg_ms = (elapsed / 100) * 1000
        # Each search should take < 100ms
        assert avg_ms < 100, f"Average search time {avg_ms:.1f}ms is too slow"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
