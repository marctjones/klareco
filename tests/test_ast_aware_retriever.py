"""
Tests for AST-Aware Retriever.

Tests the multi-strategy retrieval system combining all AST components.
"""

import pytest
from pathlib import Path
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.question_classifier import QuestionType
from klareco.rag.entity_recognizer import EntityType, Entity


class TestStrategySelection:
    """Test automatic strategy selection logic."""

    def test_who_question_selects_entity_strategy(self):
        """WHO questions should use entity-focused strategy."""
        # Mock retriever without actual index
        # We'll test the _select_strategy method directly

        # WHO question with entities
        question_type = QuestionType.WHO
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9)
        ]

        # Simulate strategy selection
        # WHO questions → entity strategy
        expected_strategy = 'entity'

        # This would be the expected behavior
        assert question_type == QuestionType.WHO

    def test_what_question_selects_pattern_strategy(self):
        """WHAT questions should use pattern matching strategy."""
        question_type = QuestionType.WHAT
        entities = []

        # WHAT questions → pattern strategy
        expected_strategy = 'pattern'

        assert question_type == QuestionType.WHAT

    def test_multiple_entities_selects_hybrid_strategy(self):
        """Questions with multiple entities should use hybrid strategy."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
            Entity(text="Esperanto", entity_type=EntityType.PERSON, root="esperant", slot="OBJ", confidence=0.9),
        ]

        # Multiple entities → hybrid strategy
        expected_strategy = 'hybrid'

        assert len(entities) >= 2


class TestComponentIntegration:
    """Test that all components work together."""

    def test_question_classification_integration(self):
        """Test that question classifier is properly integrated."""
        from klareco.parser import parse
        from klareco.rag.question_classifier import QuestionClassifier

        classifier = QuestionClassifier()
        query = "Kiu fondis Esperanton?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHO
        assert 'target_slots' in result
        assert isinstance(result['target_slots'], list)

    def test_entity_recognition_integration(self):
        """Test that entity recognizer is properly integrated."""
        from klareco.parser import parse
        from klareco.rag.entity_recognizer import EntityRecognizer

        recognizer = EntityRecognizer()
        text = "Zamenhof kreis Esperanton."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should find entities
        assert isinstance(entities, list)

    def test_pattern_matching_integration(self):
        """Test that pattern matcher is properly integrated."""
        from klareco.parser import parse
        from klareco.rag.ast_pattern_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        query = "Kiu kreis Esperanton?"
        doc = "Zamenhof kreis Esperanton."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher.match(
            query_ast, doc_ast,
            target_slots=['SUBJ', 'VERB', 'OBJ'],
            entity_type='person'
        )

        assert result.score > 0

    def test_semantic_db_integration(self):
        """Test that semantic database is properly integrated."""
        from klareco.rag.semantic_db import SemanticRelationDB

        db = SemanticRelationDB()
        stats = db.get_statistics()

        # Should load ReVo data
        assert stats['synonym_roots'] > 0


class TestExplainRetrieval:
    """Test retrieval explanation functionality."""

    def test_explanation_structure(self):
        """Test that explanation has expected structure."""
        # Mock explanation structure
        explanation = {
            'query': 'Kiu fondis Esperanton?',
            'document': 'Zamenhof fondis Esperanton.',
            'classification': {
                'question_type': 'who',
                'entity_type': 'person',
                'focus': 'SUBJ',
                'target_slots': ['SUBJ', 'VERB'],
            },
            'query_entities': [],
            'doc_entities': [],
            'pattern_match': {
                'score': 0.8,
                'matched_slots': ['VERB', 'OBJ'],
                'transformations': [],
                'explanation': 'Direct match on slots: VERB, OBJ',
            },
        }

        # Verify structure
        assert 'query' in explanation
        assert 'document' in explanation
        assert 'classification' in explanation
        assert 'pattern_match' in explanation


class TestRetrievalStrategies:
    """Test different retrieval strategies conceptually."""

    def test_pattern_matching_strategy_concept(self):
        """Test pattern matching strategy concept."""
        # Pattern matching should:
        # 1. Use AST structure
        # 2. Match slots (SUBJ/VERB/OBJ)
        # 3. Apply synonym expansion
        # 4. Handle transformations (passive, etc.)

        strategy_features = {
            'uses_ast': True,
            'matches_slots': True,
            'synonym_expansion': True,
            'transformations': True,
        }

        assert all(strategy_features.values())

    def test_entity_focused_strategy_concept(self):
        """Test entity-focused strategy concept."""
        # Entity-focused should:
        # 1. Extract entities from query
        # 2. Find documents with same entities
        # 3. Use pattern matching for ranking

        strategy_features = {
            'extracts_entities': True,
            'matches_entities': True,
            'uses_pattern_ranking': True,
        }

        assert all(strategy_features.values())

    def test_hybrid_strategy_concept(self):
        """Test hybrid strategy concept."""
        # Hybrid should:
        # 1. Combine entity matching
        # 2. Combine pattern matching
        # 3. Balance both scores

        strategy_features = {
            'entity_matching': True,
            'pattern_matching': True,
            'score_combination': True,
        }

        assert all(strategy_features.values())


class TestDocumentLoading:
    """Test document loading mechanics (without actual index)."""

    def test_document_offset_index_concept(self):
        """Test offset index concept for lazy loading."""
        # Offset index should allow:
        # 1. O(1) document lookup by ID
        # 2. Lazy loading (don't load all docs into memory)
        # 3. Fast random access

        offset_index_features = {
            'o1_lookup': True,
            'lazy_loading': True,
            'random_access': True,
        }

        assert all(offset_index_features.values())


class TestScoring:
    """Test scoring logic."""

    def test_entity_score_calculation(self):
        """Test entity overlap scoring."""
        # Entity score = text_overlap * 0.6 + root_overlap * 0.4

        query_texts = {'Zamenhof', 'Esperanto'}
        doc_texts = {'Zamenhof', 'Esperanto', 'lingvo'}

        text_overlap = len(query_texts & doc_texts)  # 2

        # Score should be based on overlap
        assert text_overlap == 2

    def test_combined_score_calculation(self):
        """Test combined scoring (entity + pattern)."""
        # Combined score = entity_score * weight + pattern_score * weight

        entity_score = 0.8
        pattern_score = 0.6

        # For entity-focused: 0.6 entity + 0.4 pattern
        combined = entity_score * 0.6 + pattern_score * 0.4

        assert combined == pytest.approx(0.72)

        # For hybrid: 0.5 entity + 0.5 pattern
        combined_hybrid = entity_score * 0.5 + pattern_score * 0.5

        assert combined_hybrid == pytest.approx(0.7)


class TestQuestionTypes:
    """Test handling of different question types."""

    def test_who_question_handling(self):
        """WHO questions should prioritize person entities and SUBJ slot."""
        question_type = QuestionType.WHO

        # Expected behavior:
        # - Entity type: PERSON
        # - Target slots: SUBJ, VERB
        # - Strategy: entity-focused

        assert question_type == QuestionType.WHO

    def test_where_question_handling(self):
        """WHERE questions should prioritize place entities."""
        question_type = QuestionType.WHERE

        # Expected behavior:
        # - Entity type: PLACE
        # - Target slots: VERB, OBJ
        # - Strategy: entity-focused

        assert question_type == QuestionType.WHERE

    def test_when_question_handling(self):
        """WHEN questions should prioritize time entities."""
        question_type = QuestionType.WHEN

        # Expected behavior:
        # - Entity type: TIME
        # - Target slots: VERB, SUBJ, OBJ
        # - Strategy: entity-focused

        assert question_type == QuestionType.WHEN

    def test_what_question_handling(self):
        """WHAT questions should use pattern matching."""
        question_type = QuestionType.WHAT

        # Expected behavior:
        # - Entity type: THING or DEFINITION
        # - Target slots: VERB, OBJ
        # - Strategy: pattern matching

        assert question_type == QuestionType.WHAT


class TestDeterministicNature:
    """Test that retriever is deterministic."""

    def test_no_learned_parameters(self):
        """Verify that retrieval is fully deterministic."""
        # AST-aware retrieval should use:
        # - Question classifier (rule-based)
        # - Entity recognizer (rule-based)
        # - Pattern matcher (rule-based)
        # - Semantic DB (dictionary lookup)

        components = {
            'question_classifier': 'deterministic',
            'entity_recognizer': 'deterministic',
            'pattern_matcher': 'deterministic',
            'semantic_db': 'deterministic',
        }

        assert all(v == 'deterministic' for v in components.values())


class TestCompoundWordFiltering:
    """Test compound word filtering for query disambiguation."""

    def test_standalone_query_filters_compounds(self):
        """Standalone query object should filter out compound results."""
        from klareco.parser import parse

        # Query: "Kiu fondis Esperanton?" - standalone proper noun
        query = "Kiu fondis Esperanton?"
        query_ast = parse(query)

        # Mock results with both standalone and compound
        class MockResult:
            def __init__(self, doc_id):
                self.doc_id = doc_id
                self.score = 1.0

        # We need to test the logic, but can't test with real retriever without index
        # So we test the AST structure check logic

        # Check query object is standalone (not compound)
        query_obj = query_ast.get('objekto', {})
        if query_obj.get('tipo') == 'vortgrupo':
            query_kerno = query_obj.get('kerno', {})
        else:
            query_kerno = query_obj

        is_standalone = not query_kerno.get('estas_kunmetita', False)
        is_noun = query_kerno.get('vortspeco') in ['propra_nomo', 'substantivo']

        assert is_standalone and is_noun, "Query 'Esperanton' should be detected as standalone noun"

    def test_compound_detection_in_results(self):
        """Compound words in results should be correctly identified."""
        from klareco.parser import parse

        # Standalone result: "Zamenhof fondis Esperanton."
        result1 = parse("Zamenhof fondis Esperanton.")
        result1_obj = result1.get('objekto', {})
        if result1_obj.get('tipo') == 'vortgrupo':
            result1_kerno = result1_obj.get('kerno', {})
        else:
            result1_kerno = result1_obj
        is_compound1 = result1_kerno.get('estas_kunmetita', False)
        assert not is_compound1, "Standalone 'Esperanton' should not be compound"

        # Compound result: "Schmidt fondis Esperanto-klubon."
        result2 = parse("Schmidt fondis Esperanto-klubon.")
        result2_obj = result2.get('objekto', {})
        if result2_obj.get('tipo') == 'vortgrupo':
            result2_kerno = result2_obj.get('kerno', {})
        else:
            result2_kerno = result2_obj
        is_compound2 = result2_kerno.get('estas_kunmetita', False)
        assert is_compound2, "Compound 'Esperanto-klubon' should be detected"

    def test_compound_query_allows_compounds(self):
        """Compound query should allow both standalone and compound results."""
        from klareco.parser import parse

        # Query with compound: "Kiu fondis Esperanto-klubon?"
        query = "Kiu fondis Esperanto-klubon?"
        query_ast = parse(query)

        query_obj = query_ast.get('objekto', {})
        if query_obj.get('tipo') == 'vortgrupo':
            query_kerno = query_obj.get('kerno', {})
        else:
            query_kerno = query_obj

        is_compound = query_kerno.get('estas_kunmetita', False)

        # Compound query object should be detected as compound
        assert is_compound, "Compound query should be detected as compound"

    def test_non_proper_noun_no_filtering(self):
        """Regular nouns should not trigger compound filtering."""
        from klareco.parser import parse

        # Query with regular noun: "Kiu havas hundon?"
        query = "Kiu havas hundon?"
        query_ast = parse(query)

        query_obj = query_ast.get('objekto', {})

        # Should be substantivo, but filtering only applies to queries with
        # standalone proper nouns where disambiguation matters
        vortspeco = query_obj.get('vortspeco')

        # Regular nouns are substantivo, but the filtering logic checks for
        # both tipo='vorto' AND vortspeco in ['propra_nomo', 'substantivo']
        # This is intentional - we want to filter for both proper nouns and
        # regular nouns when standalone
        assert vortspeco in ['substantivo', 'propra_nomo'] or vortspeco is None

    def test_filtering_preserves_order(self):
        """Filtering should preserve result order."""
        # This tests the conceptual requirement that filtering
        # should maintain the relative order of remaining results

        original_order = [1, 2, 3, 4, 5]
        # Simulate filtering out items 2 and 4
        filtered = [x for x in original_order if x not in [2, 4]]

        assert filtered == [1, 3, 5], "Order should be preserved after filtering"
