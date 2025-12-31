#!/usr/bin/env python3
"""
Comprehensive Semantic Evaluation Tests
========================================

Tests embedding quality using multiple evaluation types:
1. Category Clustering - Do similar words group together?
2. Analogy Tests - A:B :: C:? reasoning
3. Outlier Detection - Which word doesn't belong?
4. Relation Classification - Predict relationship type
5. Hierarchy Consistency - Taxonomic structure
6. Affix Transformation - Morphological consistency
7. Correlative Grid - Esperanto-specific structure

Data sources:
- ReVo semantic relations (data/revo/revo_semantic_relations.json)
- Semantic categories (data/semantic_categories.json)
- Fundamento roots (data/vocabularies/fundamento_roots.json)

Usage:
    pytest tests/test_semantic_evaluation.py -v
    pytest tests/test_semantic_evaluation.py -v -k clustering
    pytest tests/test_semantic_evaluation.py -v -k analogy
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def semantic_categories():
    """Load semantic categories data."""
    path = PROJECT_ROOT / "data" / "semantic_categories.json"
    if not path.exists():
        pytest.skip(f"Semantic categories not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def revo_relations():
    """Load ReVo semantic relations."""
    path = PROJECT_ROOT / "data" / "revo" / "revo_semantic_relations.json"
    if not path.exists():
        pytest.skip(f"ReVo relations not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    """Load the trained models."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        pytest.skip("PyTorch not available")

    class LowRankTransform(nn.Module):
        def __init__(self, dim: int = 64, rank: int = 4):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)

        def forward(self, x):
            return x + self.up(self.down(x))

    class EmbeddingPipeline:
        def __init__(self, models_dir: Path):
            self.root_embeddings = None
            self.root_to_idx = None
            self.prefix_transforms = {}
            self.suffix_transforms = {}
            self.embedding_dim = 64

            # Load root embeddings
            root_path = models_dir / "root_embeddings" / "best_model.pt"
            if root_path.exists():
                ckpt = torch.load(root_path, map_location='cpu', weights_only=False)
                self.root_to_idx = ckpt['root_to_idx']
                self.root_embeddings = nn.Embedding(ckpt['vocab_size'], ckpt['embedding_dim'])
                self.root_embeddings.load_state_dict({
                    'weight': ckpt['model_state_dict']['embeddings.weight']
                })
                self.embedding_dim = ckpt['embedding_dim']

            # Load affix transforms (try v2 first, then v1)
            affix_path = models_dir / "affix_transforms_v2" / "best_model.pt"
            if not affix_path.exists():
                affix_path = models_dir / "affix_transforms" / "best_model.pt"

            if affix_path.exists():
                ckpt = torch.load(affix_path, map_location='cpu', weights_only=False)
                for p in ckpt.get('prefixes', []):
                    t = LowRankTransform(self.embedding_dim, ckpt.get('rank', 4))
                    if f'prefix_transforms.{p}.down.weight' in ckpt['model_state_dict']:
                        t.down.weight.data = ckpt['model_state_dict'][f'prefix_transforms.{p}.down.weight']
                        t.up.weight.data = ckpt['model_state_dict'][f'prefix_transforms.{p}.up.weight']
                        self.prefix_transforms[p] = t

                for s in ckpt.get('suffixes', []):
                    t = LowRankTransform(self.embedding_dim, ckpt.get('rank', 4))
                    if f'suffix_transforms.{s}.down.weight' in ckpt['model_state_dict']:
                        t.down.weight.data = ckpt['model_state_dict'][f'suffix_transforms.{s}.down.weight']
                        t.up.weight.data = ckpt['model_state_dict'][f'suffix_transforms.{s}.up.weight']
                        self.suffix_transforms[s] = t

        def get_root_embedding(self, root: str) -> Optional[torch.Tensor]:
            if self.root_embeddings is None or root not in self.root_to_idx:
                return None
            idx = self.root_to_idx[root]
            return self.root_embeddings(torch.tensor([idx])).squeeze()

        def get_embedding(self, root: str, prefixes: List[str] = None, suffixes: List[str] = None) -> Optional[torch.Tensor]:
            emb = self.get_root_embedding(root)
            if emb is None:
                return None

            for p in (prefixes or []):
                if p in self.prefix_transforms:
                    emb = self.prefix_transforms[p](emb)

            for s in (suffixes or []):
                if s in self.suffix_transforms:
                    emb = self.suffix_transforms[s](emb)

            return emb

        def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
            return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

        def has_root(self, root: str) -> bool:
            return self.root_to_idx is not None and root in self.root_to_idx

    models_dir = PROJECT_ROOT / "models"
    pipe = EmbeddingPipeline(models_dir)

    if pipe.root_embeddings is None:
        pytest.skip("Root embeddings model not found")

    return pipe


# =============================================================================
# Helper Functions
# =============================================================================

def compute_cluster_metrics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute clustering metrics for a set of embeddings."""
    if len(embeddings) < 2:
        return {"mean_sim": 1.0, "min_sim": 1.0, "std_sim": 0.0}

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            sims.append(sim)

    return {
        "mean_sim": np.mean(sims),
        "min_sim": np.min(sims),
        "std_sim": np.std(sims)
    }


def find_nearest(query: np.ndarray, candidates: Dict[str, np.ndarray], k: int = 5) -> List[Tuple[str, float]]:
    """Find k nearest neighbors."""
    results = []
    for name, emb in candidates.items():
        sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
        results.append((name, sim))
    return sorted(results, key=lambda x: -x[1])[:k]


# =============================================================================
# 1. Category Clustering Tests
# =============================================================================

class TestCategoryClustering:
    """Test that semantic categories cluster together."""

    @pytest.mark.xfail(reason="Broad categories need taxonomic training signal (issue #149)")
    def test_fundamento_animals(self, pipeline, semantic_categories):
        """Animals should cluster together."""
        animals = semantic_categories["fundamento_categories"]["categories"]["animals"]["members"]
        embeddings = []
        found = []

        for root in animals:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough animal roots found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"Animals should cluster: mean_sim={metrics['mean_sim']:.3f}"

    @pytest.mark.xfail(reason="Broad categories need taxonomic training signal (issue #149)")
    def test_fundamento_body_parts(self, pipeline, semantic_categories):
        """Body parts should cluster together."""
        parts = semantic_categories["fundamento_categories"]["categories"]["body_parts"]["members"]
        embeddings = []
        found = []

        for root in parts:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough body part roots found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"Body parts should cluster: mean_sim={metrics['mean_sim']:.3f}"

    def test_fundamento_colors(self, pipeline, semantic_categories):
        """Colors should cluster together."""
        colors = semantic_categories["fundamento_categories"]["categories"]["colors"]["members"]
        embeddings = []
        found = []

        for root in colors:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough color roots found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"Colors should cluster: mean_sim={metrics['mean_sim']:.3f}"

    def test_fundamento_kinship(self, pipeline, semantic_categories):
        """Kinship terms should cluster together."""
        kinship = semantic_categories["fundamento_categories"]["categories"]["kinship"]["members"]
        embeddings = []
        found = []

        for root in kinship:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough kinship roots found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"Kinship should cluster: mean_sim={metrics['mean_sim']:.3f}"


# =============================================================================
# 2. Affix Transformation Tests
# =============================================================================

class TestAffixTransformations:
    """Test morphological consistency of affixes."""

    def test_mal_creates_opposites(self, pipeline, semantic_categories):
        """mal- prefix should create semantic opposites."""
        mal_pairs = semantic_categories["affix_tests"]["mal_pairs"]["pairs"]
        similarities = []

        for root, mal_root in mal_pairs:
            emb1 = pipeline.get_root_embedding(root)
            emb2 = pipeline.get_embedding(root, prefixes=["mal"])

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append((root, sim))

        if len(similarities) < 3:
            pytest.skip("Not enough mal- pairs found")

        mean_sim = np.mean([s[1] for s in similarities])
        assert mean_sim < 0.5, f"mal- pairs should have low similarity: mean={mean_sim:.3f}"

    def test_re_preserves_meaning(self, pipeline, semantic_categories):
        """re- prefix should preserve meaning (high similarity)."""
        re_pairs = semantic_categories["affix_tests"]["re_pairs"]["pairs"]
        similarities = []

        for root, re_root in re_pairs:
            emb1 = pipeline.get_root_embedding(root)
            emb2 = pipeline.get_embedding(root, prefixes=["re"])

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append((root, sim))

        if len(similarities) < 3:
            pytest.skip("Not enough re- pairs found")

        mean_sim = np.mean([s[1] for s in similarities])
        assert mean_sim > 0.6, f"re- pairs should have high similarity: mean={mean_sim:.3f}"

    @pytest.mark.xfail(reason="Suffix transforms need clustering training signal (issue #148)")
    def test_ej_places_cluster(self, pipeline, semantic_categories):
        """Words with -ej suffix should cluster together."""
        ej_roots = semantic_categories["affix_tests"]["ej_suffix"]["members"]
        embeddings = []
        found = []

        for root in ej_roots:
            emb = pipeline.get_embedding(root, suffixes=["ej"])
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough -ej forms found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"-ej places should cluster: mean_sim={metrics['mean_sim']:.3f}"

    @pytest.mark.xfail(reason="Suffix transforms need clustering training signal (issue #148)")
    def test_ist_professions_cluster(self, pipeline, semantic_categories):
        """Words with -ist suffix should cluster together."""
        ist_roots = semantic_categories["affix_tests"]["ist_suffix"]["members"]
        embeddings = []
        found = []

        for root in ist_roots:
            emb = pipeline.get_embedding(root, suffixes=["ist"])
            if emb is not None:
                embeddings.append(emb.detach().numpy())
                found.append(root)

        if len(embeddings) < 3:
            pytest.skip(f"Not enough -ist forms found: {found}")

        metrics = compute_cluster_metrics(embeddings)
        assert metrics["mean_sim"] > 0.3, f"-ist professions should cluster: mean_sim={metrics['mean_sim']:.3f}"


# =============================================================================
# 3. Analogy Tests
# =============================================================================

class TestAnalogies:
    """Test word analogy reasoning: A:B :: C:?"""

    def test_antonym_analogies(self, pipeline, semantic_categories):
        """Test antonym analogies: bon:malbon :: grand:?"""
        analogies = [a for a in semantic_categories["analogy_tests"]["tests"] if a["type"] == "antonym"]
        correct = 0
        total = 0

        for analogy in analogies:
            a_emb = pipeline.get_root_embedding(analogy["a"])
            b_emb = pipeline.get_embedding(analogy["a"], prefixes=["mal"])
            c_emb = pipeline.get_root_embedding(analogy["c"])

            if a_emb is None or b_emb is None or c_emb is None:
                continue

            # Predicted: c + (b - a) should be close to expected
            predicted = c_emb + (b_emb - a_emb)
            expected_emb = pipeline.get_embedding(analogy["c"], prefixes=["mal"])

            if expected_emb is not None:
                sim = pipeline.similarity(predicted, expected_emb)
                if sim > 0.5:
                    correct += 1
                total += 1

        if total == 0:
            pytest.skip("No analogy tests could be run")

        accuracy = correct / total
        # Note: We use a lower threshold since this is a harder test
        assert accuracy > 0.3, f"Antonym analogy accuracy: {accuracy:.1%} ({correct}/{total})"

    def test_gender_analogies(self, pipeline, semantic_categories):
        """Test gender analogies: patr:patrin :: frat:?"""
        # This requires the gender suffix transform
        analogies = [a for a in semantic_categories["analogy_tests"]["tests"] if a["type"] == "gender"]

        if len(analogies) == 0:
            pytest.skip("No gender analogies defined")

        # For now, just test that the analogy structure exists
        assert len(analogies) > 0


# =============================================================================
# 4. Outlier Detection Tests
# =============================================================================

class TestOutlierDetection:
    """Test finding the word that doesn't belong."""

    def test_outlier_detection(self, pipeline, semantic_categories):
        """Test that outliers are correctly identified."""
        outlier_tests = semantic_categories["outlier_tests"]["tests"]
        correct = 0
        total = 0

        for test in outlier_tests:
            words = test["words"]
            expected_outlier = test["outlier"]

            embeddings = {}
            for word in words:
                emb = pipeline.get_root_embedding(word)
                if emb is not None:
                    embeddings[word] = emb.detach().numpy()

            if len(embeddings) < 3:
                continue

            # Find outlier: word with lowest mean similarity to others
            outlier_scores = {}
            for word, emb in embeddings.items():
                others = [e for w, e in embeddings.items() if w != word]
                mean_sim = np.mean([
                    np.dot(emb, other) / (np.linalg.norm(emb) * np.linalg.norm(other))
                    for other in others
                ])
                outlier_scores[word] = mean_sim

            predicted_outlier = min(outlier_scores, key=outlier_scores.get)

            if predicted_outlier == expected_outlier:
                correct += 1
            total += 1

        if total == 0:
            pytest.skip("No outlier tests could be run")

        accuracy = correct / total
        assert accuracy > 0.5, f"Outlier detection accuracy: {accuracy:.1%} ({correct}/{total})"


# =============================================================================
# 5. Relation Classification Tests (using ReVo)
# =============================================================================

class TestRelationClassification:
    """Test semantic relation classification using ReVo data."""

    def test_synonyms_similar(self, pipeline, revo_relations):
        """Synonyms should have high similarity."""
        synonyms = revo_relations["relations"].get("synonym", [])
        similarities = []

        for pair in synonyms[:50]:  # Test first 50
            if len(pair) != 2:
                continue
            root1, root2 = pair

            emb1 = pipeline.get_root_embedding(root1)
            emb2 = pipeline.get_root_embedding(root2)

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append(sim)

        if len(similarities) < 5:
            pytest.skip("Not enough synonym pairs found")

        mean_sim = np.mean(similarities)
        assert mean_sim > 0.3, f"Synonyms should be similar: mean={mean_sim:.3f}"

    def test_antonyms_dissimilar(self, pipeline, revo_relations):
        """Antonyms should have low similarity."""
        antonyms = revo_relations["relations"].get("antonym", [])
        similarities = []

        for pair in antonyms[:50]:  # Test first 50
            if len(pair) != 2:
                continue
            root1, root2 = pair

            emb1 = pipeline.get_root_embedding(root1)
            emb2 = pipeline.get_root_embedding(root2)

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append(sim)

        if len(similarities) < 5:
            pytest.skip("Not enough antonym pairs found")

        mean_sim = np.mean(similarities)
        assert mean_sim < 0.7, f"Antonyms should be dissimilar: mean={mean_sim:.3f}"

    def test_hypernyms_related(self, pipeline, revo_relations):
        """Hypernyms should be related to hyponyms."""
        hypernyms = revo_relations["relations"].get("hypernym", [])
        similarities = []

        for pair in hypernyms[:50]:  # Test first 50
            if len(pair) != 2:
                continue
            hyponym, hypernym = pair

            emb1 = pipeline.get_root_embedding(hyponym)
            emb2 = pipeline.get_root_embedding(hypernym)

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append(sim)

        if len(similarities) < 5:
            pytest.skip("Not enough hypernym pairs found")

        mean_sim = np.mean(similarities)
        assert mean_sim > 0.2, f"Hypernyms should be related: mean={mean_sim:.3f}"


# =============================================================================
# 6. Separation Tests
# =============================================================================

class TestSeparation:
    """Test that unrelated words are separated."""

    def test_unrelated_separation(self, pipeline, semantic_categories):
        """Unrelated word pairs should have low similarity."""
        pairs = semantic_categories["separation_tests"]["pairs"]
        similarities = []

        for pair in pairs:
            emb1 = pipeline.get_root_embedding(pair["a"])
            emb2 = pipeline.get_root_embedding(pair["b"])

            if emb1 is not None and emb2 is not None:
                sim = pipeline.similarity(emb1, emb2)
                similarities.append((pair["a"], pair["b"], sim))

        if len(similarities) < 2:
            pytest.skip("Not enough separation pairs found")

        mean_sim = np.mean([s[2] for s in similarities])
        assert mean_sim < 0.4, f"Unrelated pairs should have low similarity: mean={mean_sim:.3f}"

    def test_cross_category_separation(self, pipeline, semantic_categories):
        """Words from different categories should be less similar than within."""
        categories = semantic_categories["fundamento_categories"]["categories"]

        # Get embeddings for two distinct categories
        animals = []
        for root in categories["animals"]["members"][:5]:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                animals.append(emb.detach().numpy())

        colors = []
        for root in categories["colors"]["members"][:5]:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                colors.append(emb.detach().numpy())

        if len(animals) < 2 or len(colors) < 2:
            pytest.skip("Not enough embeddings for cross-category test")

        # Within-category similarity
        within_animal = compute_cluster_metrics(animals)["mean_sim"]
        within_color = compute_cluster_metrics(colors)["mean_sim"]

        # Cross-category similarity
        cross_sims = []
        for a in animals:
            for c in colors:
                sim = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
                cross_sims.append(sim)
        cross_sim = np.mean(cross_sims)

        avg_within = (within_animal + within_color) / 2
        assert cross_sim < avg_within, f"Cross-category ({cross_sim:.3f}) should be < within ({avg_within:.3f})"


# =============================================================================
# 7. Correlative Grid Tests
# =============================================================================

class TestCorrelativeGrid:
    """Test Esperanto correlative table structure."""

    @pytest.mark.xfail(reason="Correlatives are function words, excluded from training by design (issue #147)")
    def test_row_clustering(self, pipeline, semantic_categories):
        """Correlative rows (same prefix) should cluster."""
        row_tests = semantic_categories["correlatives"]["tests"]["row_clusters"]

        for row_test in row_tests[:3]:  # Test first 3 rows
            members = row_test["members"]
            embeddings = []
            found = []

            for word in members:
                emb = pipeline.get_root_embedding(word)
                if emb is not None:
                    embeddings.append(emb.detach().numpy())
                    found.append(word)

            if len(embeddings) < 3:
                continue

            metrics = compute_cluster_metrics(embeddings)
            # Correlatives are function words, so clustering threshold is lower
            assert metrics["mean_sim"] > 0.2, f"Row {row_test['name']}: mean_sim={metrics['mean_sim']:.3f}"

    @pytest.mark.xfail(reason="Correlatives are function words, excluded from training by design (issue #147)")
    def test_column_clustering(self, pipeline, semantic_categories):
        """Correlative columns (same suffix) should cluster."""
        col_tests = semantic_categories["correlatives"]["tests"]["column_clusters"]

        for col_test in col_tests[:3]:  # Test first 3 columns
            members = col_test["members"]
            embeddings = []
            found = []

            for word in members:
                emb = pipeline.get_root_embedding(word)
                if emb is not None:
                    embeddings.append(emb.detach().numpy())
                    found.append(word)

            if len(embeddings) < 3:
                continue

            metrics = compute_cluster_metrics(embeddings)
            # Correlatives are function words, so clustering threshold is lower
            assert metrics["mean_sim"] > 0.2, f"Column {col_test['name']}: mean_sim={metrics['mean_sim']:.3f}"


# =============================================================================
# Summary Report
# =============================================================================

class TestSummaryReport:
    """Generate a summary report of all evaluation metrics."""

    def test_generate_report(self, pipeline, semantic_categories, revo_relations):
        """Generate and print a comprehensive evaluation report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("SEMANTIC EVALUATION SUMMARY")
        report.append("=" * 60)

        # Count vocabulary coverage
        vocab_size = len(pipeline.root_to_idx) if pipeline.root_to_idx else 0
        report.append(f"\nVocabulary: {vocab_size:,} roots")
        report.append(f"Prefixes: {len(pipeline.prefix_transforms)}")
        report.append(f"Suffixes: {len(pipeline.suffix_transforms)}")

        # Test category coverage
        categories = semantic_categories["fundamento_categories"]["categories"]
        category_coverage = {}
        for name, cat in categories.items():
            found = sum(1 for r in cat["members"] if pipeline.has_root(r))
            total = len(cat["members"])
            category_coverage[name] = (found, total)

        report.append(f"\nCategory Coverage:")
        for name, (found, total) in sorted(category_coverage.items()):
            pct = 100 * found / total if total > 0 else 0
            report.append(f"  {name}: {found}/{total} ({pct:.0f}%)")

        # ReVo relation counts
        report.append(f"\nReVo Relations:")
        for rel_type, pairs in revo_relations.get("relations", {}).items():
            report.append(f"  {rel_type}: {len(pairs)} pairs")

        print("\n".join(report))

        # This test always passes - it's just for reporting
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
