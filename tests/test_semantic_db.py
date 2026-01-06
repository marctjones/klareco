"""
Tests for Semantic Relation Database.

Tests loading and querying semantic relations from ReVo thesaurus.
"""

import pytest
from pathlib import Path
from klareco.rag.semantic_db import SemanticRelationDB


@pytest.fixture
def db():
    """Create semantic database instance."""
    # Use default path (data/raw/eo/dictionaries/revo/revo_semantic_relations.json)
    return SemanticRelationDB()


@pytest.fixture
def db_empty():
    """Create empty semantic database (for testing missing file)."""
    # Use non-existent path
    return SemanticRelationDB(revo_path=Path('/nonexistent/path/revo.json'))


class TestDatabaseLoading:
    """Test loading ReVo semantic relations."""

    def test_load_synonyms(self, db):
        """Test that synonyms are loaded."""
        stats = db.get_statistics()
        assert stats['synonym_roots'] > 0
        assert stats['total_synonym_pairs'] > 0

    def test_load_antonyms(self, db):
        """Test that antonyms are loaded."""
        stats = db.get_statistics()
        assert stats['antonym_roots'] > 0
        assert stats['total_antonym_pairs'] > 0

    def test_empty_database(self, db_empty):
        """Test graceful handling of missing file."""
        stats = db_empty.get_statistics()
        assert stats['synonym_roots'] == 0
        assert stats['antonym_roots'] == 0


class TestSynonymLookup:
    """Test synonym lookups."""

    def test_get_synonyms(self, db):
        """Test retrieving synonyms for a root."""
        # Note: Exact synonyms depend on ReVo data
        # Just check that method works and returns set
        syns = db.get_synonyms('hund')
        assert isinstance(syns, set)

    def test_synonym_symmetry(self, db):
        """Test that synonym relations are symmetric."""
        # If A is synonym of B, then B should be synonym of A
        for root, synonyms in list(db.synonyms.items())[:10]:  # Check first 10
            for syn in synonyms:
                assert root in db.synonyms.get(syn, set()), \
                    f"Symmetric relation broken: {root} → {syn} but not {syn} → {root}"

    def test_are_synonyms(self, db):
        """Test checking if two roots are synonyms."""
        # Get a known synonym pair
        if db.synonyms:
            root1 = list(db.synonyms.keys())[0]
            root2 = list(db.synonyms[root1])[0] if db.synonyms[root1] else None

            if root2:
                assert db.are_synonyms(root1, root2)
                assert db.are_synonyms(root2, root1)  # Symmetric

    def test_unknown_root_synonyms(self, db):
        """Test querying synonyms for unknown root."""
        syns = db.get_synonyms('xyzunknownroot')
        assert syns == set()


class TestAntonymLookup:
    """Test antonym lookups."""

    def test_get_antonyms(self, db):
        """Test retrieving antonyms for a root."""
        ants = db.get_antonyms('bon')
        assert isinstance(ants, set)

    def test_antonym_symmetry(self, db):
        """Test that antonym relations are symmetric."""
        # If A is antonym of B, then B should be antonym of A
        for root, antonyms in list(db.antonyms.items())[:10]:  # Check first 10
            for ant in antonyms:
                assert root in db.antonyms.get(ant, set()), \
                    f"Symmetric relation broken: {root} ↔ {ant}"

    def test_are_antonyms(self, db):
        """Test checking if two roots are antonyms."""
        # Get a known antonym pair
        if db.antonyms:
            root1 = list(db.antonyms.keys())[0]
            root2 = list(db.antonyms[root1])[0] if db.antonyms[root1] else None

            if root2:
                assert db.are_antonyms(root1, root2)
                assert db.are_antonyms(root2, root1)  # Symmetric

    def test_unknown_root_antonyms(self, db):
        """Test querying antonyms for unknown root."""
        ants = db.get_antonyms('xyzunknownroot')
        assert ants == set()


class TestHypernymLookup:
    """Test hypernym lookups (Phase 2 feature)."""

    def test_get_hypernyms(self, db):
        """Test retrieving hypernyms (more general terms)."""
        hypers = db.get_hypernyms('hund')
        assert isinstance(hypers, set)

    def test_get_hyponyms(self, db):
        """Test retrieving hyponyms (more specific terms)."""
        hypos = db.get_hyponyms('best')
        assert isinstance(hypos, set)


class TestExpansion:
    """Test expanding root sets with synonyms."""

    def test_expand_with_synonyms(self, db):
        """Test expanding a set of roots with their synonyms."""
        if not db.synonyms:
            pytest.skip("No synonyms in database")

        # Get a root with known synonyms
        root = list(db.synonyms.keys())[0]
        original = {root}

        expanded = db.expand_with_synonyms(original)

        # Should include original
        assert root in expanded

        # Should include synonyms (if any)
        expected_synonyms = db.get_synonyms(root)
        if expected_synonyms:
            assert len(expanded) > len(original)
            assert expected_synonyms.issubset(expanded)

    def test_expand_unknown_roots(self, db):
        """Test expanding unknown roots returns original set."""
        original = {'xyzunknown1', 'xyzunknown2'}
        expanded = db.expand_with_synonyms(original)

        assert expanded == original


class TestSemanticSimilarity:
    """Test semantic similarity calculation."""

    def test_self_similarity(self, db):
        """Test that a root is maximally similar to itself."""
        assert db.get_semantic_similarity('hund', 'hund') == 1.0

    def test_synonym_similarity(self, db):
        """Test that synonyms have similarity 1.0."""
        if db.synonyms:
            root1 = list(db.synonyms.keys())[0]
            root2 = list(db.synonyms[root1])[0] if db.synonyms[root1] else None

            if root2:
                assert db.get_semantic_similarity(root1, root2) == 1.0

    def test_antonym_similarity(self, db):
        """Test that antonyms have similarity -1.0."""
        if db.antonyms:
            root1 = list(db.antonyms.keys())[0]
            root2 = list(db.antonyms[root1])[0] if db.antonyms[root1] else None

            if root2:
                assert db.get_semantic_similarity(root1, root2) == -1.0

    def test_hypernym_similarity(self, db):
        """Test that hypernyms have similarity 0.5."""
        if db.hypernyms:
            root1 = list(db.hypernyms.keys())[0]
            root2 = list(db.hypernyms[root1])[0] if db.hypernyms[root1] else None

            if root2:
                sim = db.get_semantic_similarity(root1, root2)
                assert sim == 0.5

    def test_unrelated_similarity(self, db):
        """Test that unrelated roots have similarity 0.0."""
        sim = db.get_semantic_similarity('xyzunknown1', 'xyzunknown2')
        assert sim == 0.0


class TestStatistics:
    """Test database statistics."""

    def test_get_statistics(self, db):
        """Test getting database statistics."""
        stats = db.get_statistics()

        assert 'synonym_roots' in stats
        assert 'antonym_roots' in stats
        assert 'total_synonym_pairs' in stats
        assert 'total_antonym_pairs' in stats

        # All counts should be non-negative
        for key, value in stats.items():
            assert value >= 0

    def test_statistics_empty_db(self, db_empty):
        """Test statistics for empty database."""
        stats = db_empty.get_statistics()

        # All should be zero
        for key, value in stats.items():
            assert value == 0
