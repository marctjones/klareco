"""
Tests for AST Pattern Matcher.

Tests deterministic pattern matching using AST structure.
"""

import pytest
from klareco.parser import parse
from klareco.rag.ast_pattern_matcher import ASTPatternMatcher


@pytest.fixture
def matcher():
    """Create pattern matcher with simple synonym database."""
    synonym_db = {
        'kre': {'establ', 'fond'},  # create, establish, found
        'establ': {'kre', 'fond'},
        'fond': {'kre', 'establ'},
        'hund': {'kant'},  # dog, canine
        'kant': {'hund'},
    }
    return ASTPatternMatcher(synonym_db=synonym_db)


@pytest.fixture
def matcher_no_synonyms():
    """Create pattern matcher without synonyms."""
    return ASTPatternMatcher()


class TestDirectSlotMatching:
    """Test direct slot-to-slot matching."""

    def test_exact_match_all_slots(self, matcher_no_synonyms):
        """Test exact match on all slots."""
        query = "La hundo vidas la katon."
        doc = "La hundo vidas la katon."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['SUBJ', 'VERB', 'OBJ'],
            entity_type='thing'
        )

        assert result.score > 0
        assert 'SUBJ' in result.matched_slots
        assert 'VERB' in result.matched_slots
        assert 'OBJ' in result.matched_slots

    def test_partial_match_verb_obj(self, matcher_no_synonyms):
        """Test partial match on VERB and OBJ."""
        query = "Kion la hundo vidas?"
        doc = "La kato vidas la hundon."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'OBJ'],
            entity_type='thing'
        )

        # Should match VERB (vidas)
        assert result.score > 0
        assert 'VERB' in result.matched_slots

    def test_no_match(self, matcher_no_synonyms):
        """Test no match when sentences are completely different."""
        query = "La hundo kuras."
        doc = "La kato dormas."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['SUBJ', 'VERB', 'OBJ'],
            entity_type='thing'
        )

        # Different verbs and subjects
        assert result.score == 0.0
        assert len(result.matched_slots) == 0


class TestSynonymMatching:
    """Test synonym expansion in matching."""

    def test_synonym_verb_match(self, matcher):
        """Test synonym match on verbs."""
        query = "Kiu kreis Esperanton?"
        doc = "Zamenhof fondis Esperanton."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'OBJ'],
            entity_type='person'
        )

        # Should match VERB (kre ≈ fond) and OBJ (Esperanto)
        assert result.score > 0
        assert 'VERB' in result.matched_slots
        assert 'OBJ' in result.matched_slots
        assert "Synonym match" in result.explanation

    def test_synonym_subject_match(self, matcher):
        """Test synonym match on subjects."""
        query = "La hundo kuras."
        doc = "La kanto kuras."  # Using synonym 'kanto' for 'hundo'

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher.match(
            query_ast, doc_ast,
            target_slots=['SUBJ', 'VERB'],
            entity_type='thing'
        )

        # Should match SUBJ (hund ≈ kant) and VERB (kuras)
        assert result.score > 0
        assert 'SUBJ' in result.matched_slots


class TestPassiveTransformation:
    """Test passive voice transformation matching."""

    def test_active_to_passive(self, matcher_no_synonyms):
        """Test matching active question to passive answer."""
        query = "Kiu fondis Esperanton?"  # Active: who founded
        doc = "Esperanto estis fondita de Zamenhof."  # Passive: was founded by

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'SUBJ'],
            entity_type='person'
        )

        # Should detect passive transformation
        assert result.score > 0
        assert 'passive' in result.transformations
        assert 'VERB' in result.matched_slots

    def test_passive_detection(self, matcher_no_synonyms):
        """Test passive voice detection."""
        active = "La hundo vidas la katon."
        passive = "La kato estas vidata de la hundo."

        active_ast = parse(active)
        passive_ast = parse(passive)

        # Check that passive construction is detected
        assert matcher_no_synonyms._has_passive_construction(passive_ast)

        # Check that active is not detected as passive
        assert not matcher_no_synonyms._has_passive_construction(active_ast)


class TestAppositiveMatching:
    """Test appositive and fragment matching for definitions."""

    def test_definition_question_match(self, matcher_no_synonyms):
        """Test matching 'Kio estas X?' to definitions."""
        query = "Kio estas Esperanto?"
        doc = "Esperanto estas internacia planlingvo."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'OBJ'],
            entity_type='definition'
        )

        # Should match because Esperanto appears in subject
        assert result.score > 0


class TestRootExtraction:
    """Test root extraction from AST nodes."""

    def test_extract_roots_from_word(self, matcher_no_synonyms):
        """Test extracting roots from simple word."""
        ast = parse("La hundo kuras.")
        subj = ast.get('subjekto')

        roots = matcher_no_synonyms._extract_roots(subj)

        # Should extract 'hund' (not 'la' - that's an article)
        assert 'hund' in roots
        assert 'la' not in roots  # Article filtered out

    def test_extract_roots_from_word_group(self, matcher_no_synonyms):
        """Test extracting roots from word group."""
        ast = parse("La granda hundo kuras.")
        subj = ast.get('subjekto')

        roots = matcher_no_synonyms._extract_roots(subj)

        # Should extract both 'hund' and 'grand'
        assert 'hund' in roots
        assert 'grand' in roots

    def test_filter_question_words(self, matcher_no_synonyms):
        """Test that question words are filtered out."""
        ast = parse("Kiu hundo kuras?")
        subj = ast.get('subjekto')

        roots = matcher_no_synonyms._extract_roots(subj)

        # Should NOT extract 'kiu' (question word)
        assert 'kiu' not in roots

    def test_filter_function_words(self, matcher_no_synonyms):
        """Test that function words are filtered."""
        ast = parse("Mi kaj vi.")
        subj = ast.get('subjekto')

        roots = matcher_no_synonyms._extract_roots(subj)

        # Should not extract pronouns
        assert 'mi' not in roots


class TestSynonymExpansion:
    """Test synonym expansion logic."""

    def test_expand_with_synonyms(self, matcher):
        """Test expanding roots with synonyms."""
        roots = {'kre'}
        expanded = matcher._expand_synonyms(roots)

        # Should include original and synonyms
        assert 'kre' in expanded
        assert 'establ' in expanded
        assert 'fond' in expanded

    def test_expand_without_synonyms(self, matcher):
        """Test expanding roots with no known synonyms."""
        roots = {'unknown'}
        expanded = matcher._expand_synonyms(roots)

        # Should only include original
        assert expanded == {'unknown'}


class TestRealBenchmarkQuestions:
    """Test on real benchmark questions."""

    def test_zamenhof_question(self, matcher):
        """Test: Kiu fondis Esperanton?"""
        query = "Kiu fondis Esperanton?"
        doc = "Esperanto estis kreita de D-ro Zamenhof."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'OBJ'],
            entity_type='person'
        )

        # Should match:
        # - VERB: fondis ≈ kreita (synonym + passive)
        # - OBJ: Esperanto
        assert result.score > 0
        assert 'VERB' in result.matched_slots or 'OBJ' in result.matched_slots

    def test_esperanto_definition(self, matcher_no_synonyms):
        """Test: Kio estas Esperanto?"""
        query = "Kio estas Esperanto?"
        doc = "Esperanto estas internacia lingvo."

        query_ast = parse(query)
        doc_ast = parse(doc)

        result = matcher_no_synonyms.match(
            query_ast, doc_ast,
            target_slots=['VERB', 'OBJ'],
            entity_type='definition'
        )

        # Should match as definition question
        assert result.score > 0
