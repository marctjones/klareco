"""
Retrieval Performance Test Suite.

Tests actual retrieval quality using diagnostic questions from the benchmark dataset.
These tests verify that the retrieval system finds relevant documents for known queries.

Run with: pytest tests/test_retrieval_performance.py -v
Run slow tests: pytest tests/test_retrieval_performance.py -v -m slow

Requires: data/indexes/slot_hybrid index to be built.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Test data location
DIAGNOSTIC_QUESTIONS_PATH = Path("data/benchmarks/datasets/diagnostic_retriever_questions.jsonl")
INDEX_PATH = Path("data/indexes/slot_hybrid")


def load_diagnostic_questions() -> List[Dict[str, Any]]:
    """Load diagnostic questions from benchmark dataset."""
    if not DIAGNOSTIC_QUESTIONS_PATH.exists():
        pytest.skip(f"Diagnostic questions not found at {DIAGNOSTIC_QUESTIONS_PATH}")

    questions = []
    with open(DIAGNOSTIC_QUESTIONS_PATH) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def check_answer_in_results(results: List[Tuple[float, Dict]], acceptable_answers: List[str]) -> Tuple[bool, int]:
    """
    Check if any acceptable answer appears in the results.

    Returns:
        (found, rank) - whether answer was found and at what rank (1-indexed, 0 if not found)
    """
    for rank, (score, doc) in enumerate(results, 1):
        text = doc.get('text', '').lower()
        for answer in acceptable_answers:
            if answer.lower() in text:
                return True, rank
    return False, 0


def compute_mrr(ranks: List[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    reciprocals = [1.0 / r if r > 0 else 0.0 for r in ranks]
    return sum(reciprocals) / len(ranks)


@pytest.fixture(scope="module")
def ast_aware_retriever():
    """Load ASTAwareRetriever once for all tests in this module."""
    if not INDEX_PATH.exists():
        pytest.skip(f"Index not found at {INDEX_PATH}")

    from klareco.rag.ast_aware_retriever import ASTAwareRetriever

    try:
        retriever = ASTAwareRetriever(INDEX_PATH, use_prefilter=True)
        return retriever
    except Exception as e:
        pytest.skip(f"Failed to load retriever: {e}")


@pytest.fixture(scope="module")
def diagnostic_questions():
    """Load diagnostic questions once for all tests."""
    return load_diagnostic_questions()


class TestRetrieverAvailability:
    """Verify retrievers can be loaded."""

    def test_ast_aware_retriever_loads(self, ast_aware_retriever):
        """ASTAwareRetriever should load successfully."""
        assert ast_aware_retriever is not None

    def test_retriever_has_documents(self, ast_aware_retriever):
        """Retriever should have indexed documents."""
        # Check document count via doc_offsets (byte offsets for document lookup)
        assert len(ast_aware_retriever.doc_offsets) > 0

    def test_retriever_has_prefilter(self, ast_aware_retriever):
        """Retriever should have HNSW prefilter loaded."""
        assert ast_aware_retriever.prefilter_retriever is not None


class TestDiagnosticQuestions:
    """Test retrieval on diagnostic questions."""

    def test_diagnostic_questions_exist(self, diagnostic_questions):
        """Diagnostic questions should be loaded."""
        assert len(diagnostic_questions) > 0

    def test_diagnostic_question_structure(self, diagnostic_questions):
        """Each question should have required fields."""
        for q in diagnostic_questions:
            assert 'id' in q
            assert 'question' in q
            assert 'acceptable_answers' in q
            assert isinstance(q['acceptable_answers'], list)


class TestASTAwareRetrieval:
    """Test ASTAwareRetriever on diagnostic questions."""

    @pytest.mark.parametrize("question_id,expected_pass", [
        ("diag_ast_01", True),   # Fundamento question
        ("diag_ast_02", True),   # Suffix -uj- question
        pytest.param("diag_ast_03", False, marks=pytest.mark.xfail(reason="Known issue: direction expression not matched")),
    ])
    def test_ast_aware_questions(self, ast_aware_retriever, diagnostic_questions, question_id, expected_pass):
        """Test ASTAware-targeted diagnostic questions."""
        # Find the question
        question = next((q for q in diagnostic_questions if q['id'] == question_id), None)
        if question is None:
            pytest.skip(f"Question {question_id} not found")

        # Run retrieval
        results = ast_aware_retriever.search(question['question'], top_k=10)

        # Check if answer found
        found, rank = check_answer_in_results(results, question['acceptable_answers'])

        # Report
        if not found:
            # Get actual results for debugging
            top_texts = [doc.get('text', '')[:80] for _, doc in results[:3]]
            pytest.fail(
                f"Question: {question['question']}\n"
                f"Expected answers: {question['acceptable_answers']}\n"
                f"Top 3 results: {top_texts}"
            )

        # Verify found in top 10
        assert found, f"Answer not found for: {question['question']}"
        assert rank <= 10, f"Answer found but at rank {rank} > 10"

    @pytest.mark.parametrize("question_id", [
        "diag_hnsw_01",  # Hundo question
        "diag_hnsw_02",  # Birdo/arbo question
        "diag_hnsw_03",  # Parizo question
    ])
    def test_hnsw_questions(self, ast_aware_retriever, diagnostic_questions, question_id):
        """Test HNSW-targeted diagnostic questions (simple embedding similarity)."""
        question = next((q for q in diagnostic_questions if q['id'] == question_id), None)
        if question is None:
            pytest.skip(f"Question {question_id} not found")

        results = ast_aware_retriever.search(question['question'], top_k=10)
        found, rank = check_answer_in_results(results, question['acceptable_answers'])

        assert found, f"Answer not found for: {question['question']}"
        assert rank <= 10

    @pytest.mark.parametrize("question_id,expected_pass", [
        pytest.param("diag_faiss_01", False, marks=pytest.mark.xfail(reason="Known issue: pastro sentence not in corpus")),
        ("diag_faiss_02", True),   # Urbo/miliono question
        pytest.param("diag_faiss_03", False, marks=pytest.mark.xfail(reason="Known issue: cigarejo sentence not in corpus")),
    ])
    def test_faiss_questions(self, ast_aware_retriever, diagnostic_questions, question_id, expected_pass):
        """Test FAISS-targeted diagnostic questions (slot matching)."""
        question = next((q for q in diagnostic_questions if q['id'] == question_id), None)
        if question is None:
            pytest.skip(f"Question {question_id} not found")

        results = ast_aware_retriever.search(question['question'], top_k=10)
        found, rank = check_answer_in_results(results, question['acceptable_answers'])

        assert found, f"Answer not found for: {question['question']}"
        assert rank <= 10

    @pytest.mark.parametrize("question_id", [
        "diag_hybrid_01",  # Plumujo question
        "diag_hybrid_02",  # Frukto-arbo question
        "diag_hybrid_03",  # Bulonja Deklaracio question
    ])
    def test_hybrid_questions(self, ast_aware_retriever, diagnostic_questions, question_id):
        """Test Hybrid-targeted diagnostic questions (combined strategies)."""
        question = next((q for q in diagnostic_questions if q['id'] == question_id), None)
        if question is None:
            pytest.skip(f"Question {question_id} not found")

        results = ast_aware_retriever.search(question['question'], top_k=10)
        found, rank = check_answer_in_results(results, question['acceptable_answers'])

        assert found, f"Answer not found for: {question['question']}"
        assert rank <= 10


class TestRetrievalMetrics:
    """Test overall retrieval metrics."""

    @pytest.mark.slow
    def test_overall_top5_recall(self, ast_aware_retriever, diagnostic_questions):
        """Test that at least 50% of questions have answers in top-5."""
        found_count = 0
        ranks = []

        for question in diagnostic_questions:
            results = ast_aware_retriever.search(question['question'], top_k=5)
            found, rank = check_answer_in_results(results, question['acceptable_answers'])

            if found:
                found_count += 1
                ranks.append(rank)
            else:
                ranks.append(0)

        recall_at_5 = found_count / len(diagnostic_questions)

        # Minimum 50% recall at top-5
        assert recall_at_5 >= 0.5, f"Top-5 recall {recall_at_5:.1%} < 50%"

    @pytest.mark.slow
    def test_overall_top10_recall(self, ast_aware_retriever, diagnostic_questions):
        """Test that at least 70% of questions have answers in top-10."""
        found_count = 0
        ranks = []

        for question in diagnostic_questions:
            results = ast_aware_retriever.search(question['question'], top_k=10)
            found, rank = check_answer_in_results(results, question['acceptable_answers'])

            if found:
                found_count += 1
                ranks.append(rank)
            else:
                ranks.append(0)

        recall_at_10 = found_count / len(diagnostic_questions)

        # Minimum 70% recall at top-10
        assert recall_at_10 >= 0.7, f"Top-10 recall {recall_at_10:.1%} < 70%"

    @pytest.mark.slow
    def test_mrr_above_threshold(self, ast_aware_retriever, diagnostic_questions):
        """Test that MRR is above 0.3."""
        ranks = []

        for question in diagnostic_questions:
            results = ast_aware_retriever.search(question['question'], top_k=10)
            found, rank = check_answer_in_results(results, question['acceptable_answers'])
            ranks.append(rank)

        mrr = compute_mrr(ranks)

        # Minimum MRR of 0.3
        assert mrr >= 0.3, f"MRR {mrr:.3f} < 0.3"


class TestDefinitionQuestions:
    """Test definition questions specifically (the pattern we fixed)."""

    def test_kio_estas_hundo(self, ast_aware_retriever):
        """Test basic definition: 'Kio estas hundo?'"""
        results = ast_aware_retriever.search("Kio estas hundo?", top_k=10)

        # Should find documents about dogs
        found = any('hundo' in doc.get('text', '').lower() for _, doc in results[:5])
        assert found, "No documents about 'hundo' in top 5"

    def test_kio_estas_esperanto(self, ast_aware_retriever):
        """Test important definition: 'Kio estas Esperanto?'"""
        results = ast_aware_retriever.search("Kio estas Esperanto?", top_k=10)

        # Should find documents about Esperanto
        found = any('esperant' in doc.get('text', '').lower() for _, doc in results[:5])
        assert found, "No documents about 'Esperanto' in top 5"

    def test_kio_estas_plumujo(self, ast_aware_retriever):
        """Test suffix definition: 'Kio estas plumujo?'"""
        results = ast_aware_retriever.search("Kio estas plumujo?", top_k=10)

        # Should find documents about plumujo or plumoj
        acceptable = ['plumujo', 'plum', 'skatolo']
        found = any(
            any(ans in doc.get('text', '').lower() for ans in acceptable)
            for _, doc in results[:5]
        )
        assert found, "No documents about 'plumujo' or 'plumoj' in top 5"


class TestWhoQuestions:
    """Test WHO questions."""

    def test_kiu_fondis_esperanton(self, ast_aware_retriever):
        """Test classic WHO question: 'Kiu fondis Esperanton?'"""
        results = ast_aware_retriever.search("Kiu fondis Esperanton?", top_k=10)

        # Should find documents mentioning Zamenhof
        acceptable = ['zamenhof', 'fondis', 'kreis']
        found = any(
            any(ans in doc.get('text', '').lower() for ans in acceptable)
            for _, doc in results[:5]
        )
        assert found, "No documents about Esperanto's founder in top 5"


class TestHowQuestions:
    """Test HOW questions."""

    def test_kiel_oni_formas(self, ast_aware_retriever):
        """Test grammar HOW question."""
        results = ast_aware_retriever.search("Kiel oni formas la pasintecon?", top_k=10)

        # Should find documents about past tense formation
        acceptable = ['-is', 'pasint', 'tempo', 'verbo']
        found = any(
            any(ans in doc.get('text', '').lower() for ans in acceptable)
            for _, doc in results[:10]
        )
        assert found, "No documents about past tense formation in top 10"


class TestRetrievalExplanation:
    """Test that retrieval provides explanations."""

    def test_results_have_text(self, ast_aware_retriever):
        """Results should have text field."""
        results = ast_aware_retriever.search("Kio estas hundo?", top_k=5)

        for score, doc in results:
            assert 'text' in doc, "Result missing 'text' field"
            assert isinstance(doc['text'], str), "'text' should be string"

    def test_results_have_scores(self, ast_aware_retriever):
        """Results should have valid scores."""
        results = ast_aware_retriever.search("Kio estas hundo?", top_k=5)

        for score, doc in results:
            assert isinstance(score, (int, float)), "Score should be numeric"
            assert score >= 0, "Score should be non-negative"

    def test_results_are_sorted(self, ast_aware_retriever):
        """Results should be sorted by score (descending)."""
        results = ast_aware_retriever.search("Kio estas hundo?", top_k=5)

        scores = [score for score, _ in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


class TestRetrieverComparison:
    """Compare all active retrievers on diagnostic questions."""

    @pytest.fixture(scope="class")
    def hnsw_retriever(self):
        """Load HNSWSlotRetriever."""
        if not INDEX_PATH.exists():
            pytest.skip(f"Index not found at {INDEX_PATH}")

        from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
        from klareco.embeddings.hybrid import HybridEmbedder

        try:
            embedder = HybridEmbedder.from_index(INDEX_PATH)
            retriever = HNSWSlotRetriever(INDEX_PATH, embedder)
            return retriever
        except Exception as e:
            pytest.skip(f"Failed to load HNSWSlotRetriever: {e}")

    @pytest.fixture(scope="class")
    def faiss_retriever(self):
        """Load FAISSSlotRetriever."""
        if not INDEX_PATH.exists():
            pytest.skip(f"Index not found at {INDEX_PATH}")

        from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
        from klareco.embeddings.hybrid import HybridEmbedder

        try:
            embedder = HybridEmbedder.from_index(INDEX_PATH)
            retriever = FAISSSlotRetriever(INDEX_PATH, embedder)
            return retriever
        except Exception as e:
            pytest.skip(f"Failed to load FAISSSlotRetriever: {e}")

    @pytest.fixture(scope="class")
    def hybrid_faiss_retriever(self):
        """Load HybridFAISSMmapRetriever."""
        if not INDEX_PATH.exists():
            pytest.skip(f"Index not found at {INDEX_PATH}")

        from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
        from klareco.embeddings.hybrid import HybridEmbedder

        try:
            embedder = HybridEmbedder.from_index(INDEX_PATH)
            retriever = HybridFAISSMmapRetriever(INDEX_PATH, embedder)
            return retriever
        except Exception as e:
            pytest.skip(f"Failed to load HybridFAISSMmapRetriever: {e}")

    @pytest.mark.slow
    def test_hnsw_basic_retrieval(self, hnsw_retriever):
        """HNSWSlotRetriever should retrieve documents."""
        results = hnsw_retriever.search("Kio estas hundo?", top_k=5)
        assert len(results) > 0
        # Verify results have text
        for score, doc in results:
            assert 'text' in doc

    @pytest.mark.slow
    def test_faiss_basic_retrieval(self, faiss_retriever):
        """FAISSSlotRetriever should retrieve documents."""
        results = faiss_retriever.search("Kio estas hundo?", top_k=5)
        assert len(results) > 0
        for score, doc in results:
            assert 'text' in doc

    @pytest.mark.slow
    def test_hybrid_faiss_basic_retrieval(self, hybrid_faiss_retriever):
        """HybridFAISSMmapRetriever should retrieve documents."""
        results = hybrid_faiss_retriever.search("Kio estas hundo?", top_k=5)
        assert len(results) > 0
        for score, doc in results:
            assert 'text' in doc


# Benchmark report generation
@pytest.mark.slow
def test_generate_benchmark_report(ast_aware_retriever, diagnostic_questions, tmp_path):
    """Generate a benchmark report for all diagnostic questions."""
    results_data = []

    for question in diagnostic_questions:
        results = ast_aware_retriever.search(question['question'], top_k=10)
        found, rank = check_answer_in_results(results, question['acceptable_answers'])

        results_data.append({
            'id': question['id'],
            'question': question['question'],
            'found': found,
            'rank': rank,
            'top_result': results[0][1].get('text', '')[:100] if results else 'N/A',
        })

    # Calculate metrics
    top1_recall = sum(1 for r in results_data if r['rank'] == 1) / len(results_data)
    top5_recall = sum(1 for r in results_data if 0 < r['rank'] <= 5) / len(results_data)
    top10_recall = sum(1 for r in results_data if r['rank'] > 0) / len(results_data)
    mrr = compute_mrr([r['rank'] for r in results_data])

    # Write report
    report_path = tmp_path / "benchmark_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RETRIEVAL BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUMMARY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-1 Recall:  {top1_recall:.1%}\n")
        f.write(f"Top-5 Recall:  {top5_recall:.1%}\n")
        f.write(f"Top-10 Recall: {top10_recall:.1%}\n")
        f.write(f"MRR:           {mrr:.3f}\n\n")

        f.write("DETAILED RESULTS\n")
        f.write("-" * 40 + "\n")
        for r in results_data:
            status = "✓" if r['found'] else "✗"
            f.write(f"{status} [{r['id']}] Rank: {r['rank'] or 'N/A'}\n")
            f.write(f"   Q: {r['question']}\n")
            f.write(f"   Top: {r['top_result'][:60]}...\n\n")

    print(f"\nBenchmark report written to: {report_path}")
    print(f"Top-1: {top1_recall:.1%}, Top-5: {top5_recall:.1%}, Top-10: {top10_recall:.1%}, MRR: {mrr:.3f}")

    # Assertions for CI
    assert top5_recall >= 0.5, f"Top-5 recall {top5_recall:.1%} below 50% threshold"
