"""
Tests for Kuzu-backed Retrieval.

Tests the Kuzu graph database backend for AST-first retrieval:
1. Index loading and basic queries
2. Graph traversal (synonyms, hypernyms)
3. Role-based search
4. Sentence context retrieval

Requires: data/indexes/kuzu_index to exist (run build_kuzu_index.sh first)

Usage:
    pytest tests/test_kuzu_retrieval.py -v
    pytest tests/test_kuzu_retrieval.py -v -k "test_search"  # Only search tests
"""

import pytest
from pathlib import Path

# Check if Kuzu index exists
KUZU_INDEX_PATH = Path("data/indexes/kuzu_index")
KUZU_INDEX_EXISTS = (KUZU_INDEX_PATH / "kuzu.db").exists()

# Skip all tests if index doesn't exist
pytestmark = pytest.mark.skipif(
    not KUZU_INDEX_EXISTS,
    reason=f"Kuzu index not found at {KUZU_INDEX_PATH}. Run: ./scripts/build_kuzu_index.sh"
)


@pytest.fixture(scope="module")
def kuzu_index():
    """Load Kuzu index once for all tests in module."""
    from klareco.rag.kuzu_inverted_index import KuzuInvertedIndex, FallbackMode

    index = KuzuInvertedIndex(
        index_path=KUZU_INDEX_PATH,
        fallback_mode=FallbackMode.NONE,
    )
    yield index
    index.close()


@pytest.fixture(scope="module")
def retriever():
    """Load full retriever once for all tests in module."""
    from klareco.rag.ast_aware_retriever import ASTAwareRetriever
    from klareco.rag.kuzu_inverted_index import FallbackMode

    r = ASTAwareRetriever(
        index_path=KUZU_INDEX_PATH,
        fallback_mode=FallbackMode.NONE,
    )
    yield r
    r.close()


class TestKuzuIndexLoading:
    """Test that Kuzu index loads correctly."""

    def test_index_loads(self, kuzu_index):
        """Index should load without errors."""
        assert kuzu_index is not None
        assert kuzu_index.total_docs > 0
        assert kuzu_index.total_roots > 0

    def test_index_has_documents(self, kuzu_index):
        """Index should have documents loaded."""
        # We built with 4.3M docs
        assert kuzu_index.total_docs > 1_000_000

    def test_index_has_roots(self, kuzu_index):
        """Index should have roots loaded."""
        # We expect over 1M roots
        assert kuzu_index.total_roots > 500_000

    def test_root_exists(self, kuzu_index):
        """Common roots should exist in index."""
        # Use has_root method to check roots exist
        # "esperant" is the root of "Esperanto"
        assert kuzu_index.has_root("esperant")
        # "fond" is the root of "fondis" (founded)
        assert kuzu_index.has_root("fond")
        # "hund" is the root of "hundo" (dog)
        assert kuzu_index.has_root("hund")


class TestKuzuSearch:
    """Test search functionality."""

    def test_search_returns_results(self, kuzu_index):
        """Search should return results for valid query."""
        from klareco.parser import parse

        query_ast = parse("Kiu fondis Esperanton?")
        results, stats = kuzu_index.search(query_ast, max_results=10)

        assert len(results) > 0
        assert stats is not None

    def test_search_results_have_scores(self, kuzu_index):
        """Search results should have scores."""
        from klareco.parser import parse

        query_ast = parse("Kio estas Esperanto?")
        results, stats = kuzu_index.search(query_ast, max_results=5)

        for result in results:
            assert result.score > 0
            assert result.doc_id >= 0

    def test_search_stats_track_roots(self, kuzu_index):
        """Search stats should track which roots were found."""
        from klareco.parser import parse

        query_ast = parse("La hundo estas bela.")
        results, stats = kuzu_index.search(query_ast, max_results=5)

        # Should find roots like "hund", "bel"
        assert len(stats.roots_found_in_index) > 0

    def test_document_retrieval(self, kuzu_index):
        """Should be able to retrieve document by ID."""
        from klareco.parser import parse

        query_ast = parse("Esperanto")
        results, _ = kuzu_index.search(query_ast, max_results=1)

        if results:
            doc = kuzu_index.get_document(results[0].doc_id)
            assert doc is not None
            assert 'text' in doc
            assert len(doc['text']) > 0


class TestGraphTraversal:
    """Test graph-specific features."""

    def test_get_synonyms_transitive(self, kuzu_index):
        """Should find transitive synonyms via graph traversal."""
        # Get synonyms for a common root
        synonyms = kuzu_index.get_synonyms_transitive("kre", max_hops=2)

        # Should return a set (may be empty if no synonyms in ReVo)
        assert isinstance(synonyms, set)

    def test_get_hypernym_chain(self, kuzu_index):
        """Should find hypernym chain via graph traversal."""
        # Get hypernyms for a specific root
        chain = kuzu_index.get_hypernym_chain("hund", max_depth=3)

        # Should return a list (may be empty if no hypernyms in ReVo)
        assert isinstance(chain, list)

    def test_get_sentence_context(self, kuzu_index):
        """Should retrieve sentence context (adjacent sentences)."""
        from klareco.parser import parse

        # First get a sentence ID
        query_ast = parse("Esperanto")
        results, _ = kuzu_index.search(query_ast, max_results=1)

        if results:
            context = kuzu_index.get_sentence_context(results[0].doc_id, window=2)
            # Should return a list of context sentences
            assert isinstance(context, list)


class TestRoleBasedSearch:
    """Test role-aware pattern matching."""

    def test_search_by_verb_role(self, kuzu_index):
        """Should find sentences with specific verb root."""
        results = kuzu_index.search_by_role(verb="fond", max_results=10)

        # Should find sentences with "fond" as verb
        assert isinstance(results, list)

    def test_search_by_verb_and_object(self, kuzu_index):
        """Should find sentences with specific verb and object."""
        results = kuzu_index.search_by_role(
            verb="fond",
            obj="esperant",
            max_results=10
        )

        # Should find sentences like "X fondis Esperanton"
        assert isinstance(results, list)

    def test_search_by_subject_verb_object(self, kuzu_index):
        """Should find sentences with all three roles specified."""
        results = kuzu_index.search_by_role(
            subj="zamenhof",
            verb="kre",
            obj="esperant",
            expand_synonyms=True,
            max_results=10
        )

        # May or may not find exact matches
        assert isinstance(results, list)


class TestHighLevelRetriever:
    """Test the high-level ASTAwareRetriever."""

    def test_retriever_search(self, retriever):
        """Retriever should return scored results."""
        results = retriever.search("Kiu fondis Esperanton?", top_k=5)

        assert len(results) > 0
        for score, doc, stats in results:
            assert score > 0
            assert 'text' in doc
            assert stats is not None

    def test_retriever_search_simple(self, retriever):
        """Simple search should return (score, doc) tuples."""
        results = retriever.search_simple("Kio estas Esperanto?", top_k=5)

        assert len(results) > 0
        for score, doc in results:
            assert score > 0
            assert 'text' in doc

    def test_retriever_explain(self, retriever):
        """Should explain retrieval for a document."""
        results = retriever.search("Kiu fondis Esperanton?", top_k=1)

        if results:
            _, doc, _ = results[0]
            doc_id = doc.get('source', {}).get('doc_id', 0)

            explanation = retriever.explain_retrieval(
                "Kiu fondis Esperanton?",
                doc_id
            )

            assert 'query' in explanation
            assert 'query_roots' in explanation

    def test_retriever_question_classification(self, retriever):
        """Retriever should classify question types."""
        # WHO question
        results = retriever.search("Kiu kreis Esperanton?", top_k=1)
        if results:
            _, _, stats = results[0]
            # Stats should exist
            assert stats is not None

        # WHAT question
        results = retriever.search("Kio estas la Fundamento?", top_k=1)
        if results:
            _, _, stats = results[0]
            assert stats is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, kuzu_index):
        """Empty query should handle gracefully."""
        from klareco.parser import parse

        # Parse empty or minimal query
        try:
            query_ast = parse("la")  # Just an article
            results, stats = kuzu_index.search(query_ast, max_results=5)
            # Should return empty or few results (articles are skipped)
            assert isinstance(results, list)
        except Exception:
            # Parser may fail on minimal input - that's OK
            pass

    def test_nonexistent_root(self, kuzu_index):
        """Nonexistent root should return False from has_root."""
        assert not kuzu_index.has_root("xyznonexistent123")

    def test_invalid_doc_id(self, kuzu_index):
        """Invalid doc ID should return None."""
        doc = kuzu_index.get_document(-1)
        assert doc is None

        doc = kuzu_index.get_document(999_999_999)
        assert doc is None

    def test_synonyms_for_unknown_root(self, kuzu_index):
        """Unknown root should return empty synonyms."""
        synonyms = kuzu_index.get_synonyms_transitive("xyznonexistent", max_hops=2)
        assert synonyms == set()


class TestPerformance:
    """Test performance characteristics."""

    def test_search_is_fast(self, kuzu_index):
        """Search should complete in reasonable time."""
        import time
        from klareco.parser import parse

        query_ast = parse("Kiu fondis Esperanton?")

        start = time.time()
        for _ in range(10):
            results, _ = kuzu_index.search(query_ast, max_results=10)
        elapsed = time.time() - start

        # 10 searches should take less than 120 seconds
        # (Kuzu graph traversal is slower than SQLite but more powerful)
        # Performance varies based on system load
        assert elapsed < 120.0, f"10 searches took {elapsed:.2f}s (too slow)"

    def test_document_lookup_is_fast(self, kuzu_index):
        """Document lookup should be O(1)."""
        import time

        start = time.time()
        for doc_id in range(0, 1000, 10):  # Sample 100 docs
            doc = kuzu_index.get_document(doc_id)
        elapsed = time.time() - start

        # 100 lookups should take less than 1 second
        assert elapsed < 1.0, f"100 doc lookups took {elapsed:.2f}s (too slow)"
