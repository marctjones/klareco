"""
Unit tests for two-stage hybrid retrieval.

Tests the keyword filtering + semantic reranking pipeline.
"""

import pytest
from pathlib import Path

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever


@pytest.fixture(scope="module")
def retriever():
    """Create retriever instance for tests."""
    index_dir = Path("data/corpus_index")
    model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

    if not index_dir.exists() or not model_path.exists():
        pytest.skip("Corpus index or model not available")

    return KlarecoRetriever(
        index_dir=str(index_dir),
        model_path=str(model_path),
        mode='tree_lstm',
        device='cpu'
    )


class TestKeywordExtraction:
    """Test keyword extraction from AST."""

    def test_extract_proper_name(self, retriever):
        """Should extract proper names as keywords."""
        ast = parse("Kiu estas Mitrandiro?")
        keywords = retriever._extract_keywords_from_ast(ast)

        assert 'mitrandiro' in keywords

    def test_extract_content_words(self, retriever):
        """Should extract content words (nouns, verbs, adjectives)."""
        ast = parse("La granda drakon flugis.")
        keywords = retriever._extract_keywords_from_ast(ast)

        assert 'grand' in keywords or 'granda' in keywords
        assert 'drak' in keywords or 'drakon' in keywords
        assert 'flug' in keywords or 'flugis' in keywords

    def test_skip_common_words(self, retriever):
        """Should skip very common words like 'estas', 'havas'."""
        ast = parse("Li estas feliĉa.")
        keywords = retriever._extract_keywords_from_ast(ast)

        assert 'est' not in keywords  # Too common
        assert 'feliĉ' in keywords or 'feliĉa' in keywords

    def test_skip_question_words(self, retriever):
        """Should skip question words."""
        ast = parse("Kiu estas tio?")
        keywords = retriever._extract_keywords_from_ast(ast)

        assert 'kiu' not in keywords
        assert 'kio' not in keywords


class TestStage1KeywordFiltering:
    """Test Stage 1: Keyword filtering."""

    def test_finds_keyword_matches(self, retriever):
        """Should find all sentences containing keywords."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        stage1 = result['stage1']

        # Should find multiple candidates
        assert stage1['total_candidates'] > 0
        assert 'mitrandiro' in stage1['keywords']

    def test_filters_by_keywords(self, retriever):
        """All stage1 candidates should contain at least one keyword."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        stage1 = result['stage1']
        keywords = stage1['keywords']
        candidates = stage1['candidates_shown']

        # Check each candidate contains at least one keyword
        for candidate in candidates:
            text = candidate.get('text', '').lower()
            has_keyword = any(kw in text for kw in keywords)
            assert has_keyword, f"Candidate missing keywords: {text[:100]}"


class TestStage2SemanticReranking:
    """Test Stage 2: Semantic reranking."""

    def test_reranks_by_relevance(self, retriever):
        """Should rerank candidates by semantic similarity."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        results = result['results']

        # Results should be sorted by score (descending)
        scores = [r.get('score', 0.0) for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_top_result_most_relevant(self, retriever):
        """Top result should be most semantically similar."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        results = result['results']

        # Top result should have highest score
        if len(results) > 1:
            assert results[0]['score'] >= results[1]['score']

    def test_all_results_contain_keywords(self, retriever):
        """All final results should contain keywords (stage1 filter worked)."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        keywords = result['stage1']['keywords']
        results = result['results']

        for r in results:
            text = r.get('text', '').lower()
            has_keyword = any(kw in text for kw in keywords)
            assert has_keyword, f"Final result missing keywords: {text[:100]}"


class TestHybridVsPureSemantic:
    """Compare hybrid retrieval to pure semantic."""

    def test_hybrid_better_precision(self, retriever):
        """Hybrid should have better precision than pure semantic."""
        query = "Kiu estas Mitrandiro?"
        ast = parse(query)

        # Pure semantic
        pure_results = retriever.retrieve_from_ast(ast, k=5, return_scores=True)

        # Hybrid
        hybrid_result = retriever.retrieve_hybrid(ast, k=5, return_stage1_info=True)
        hybrid_results = hybrid_result['results']

        # Count how many contain "Mitrandiro"
        pure_matches = sum(1 for r in pure_results if 'mitrandiro' in r.get('text', '').lower())
        hybrid_matches = sum(1 for r in hybrid_results if 'mitrandiro' in r.get('text', '').lower())

        # Hybrid should have more keyword matches in top-5
        assert hybrid_matches >= pure_matches, \
            f"Hybrid ({hybrid_matches}/5) should >= pure semantic ({pure_matches}/5)"


class TestStage1InfoReturn:
    """Test stage1 info return functionality."""

    def test_returns_stage1_info(self, retriever):
        """Should return stage1 info when requested."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        assert 'stage1' in result
        assert 'results' in result

    def test_stage1_info_structure(self, retriever):
        """Stage1 info should have required fields."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        stage1 = result['stage1']

        assert 'keywords' in stage1
        assert 'total_candidates' in stage1
        assert 'candidates_shown' in stage1
        assert 'candidates_reranked' in stage1

        assert isinstance(stage1['keywords'], list)
        assert isinstance(stage1['total_candidates'], int)
        assert isinstance(stage1['candidates_shown'], list)
        assert isinstance(stage1['candidates_reranked'], int)

    def test_no_stage1_info_when_not_requested(self, retriever):
        """Should return plain list when stage1 info not requested."""
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=False
        )

        # Should be a list, not dict
        assert isinstance(result, list)


class TestEdgeCases:
    """Test edge cases."""

    def test_no_keywords_extracted(self, retriever):
        """Should fallback to pure semantic when no keywords."""
        # Query with only common words
        ast = parse("Estas bona.")
        keywords = retriever._extract_keywords_from_ast(ast)

        # If no keywords, should still return results via fallback
        result = retriever.retrieve_hybrid(ast, k=5)
        assert len(result) > 0

    def test_no_keyword_matches(self, retriever):
        """Should fallback to pure semantic when no keyword matches."""
        # Use a made-up word that won't appear in corpus
        ast = parse("Kiu estas xyzabc123?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            return_stage1_info=True
        )

        # When no keywords found, falls back to pure semantic (returns list)
        if isinstance(result, dict):
            assert len(result['results']) > 0
        else:
            # Fallback to pure semantic returns plain list
            assert len(result) > 0


class TestPerformance:
    """Test performance characteristics."""

    def test_limits_candidates(self, retriever):
        """Should limit candidates for efficiency."""
        # Use a query that will definitely extract keywords
        ast = parse("Kiu estas Mitrandiro?")

        result = retriever.retrieve_hybrid(
            ast,
            k=5,
            keyword_candidates=50,  # Limit to 50
            return_stage1_info=True
        )

        # Should return dict with stage1 info for this query
        assert isinstance(result, dict)
        stage1 = result['stage1']

        # Should limit candidates even if more found
        assert stage1['candidates_reranked'] <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
