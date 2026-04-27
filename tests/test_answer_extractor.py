#!/usr/bin/env python3
"""
Tests for AST-based answer extraction.

Tests answer extraction for all question types:
- WHO (kiu)
- WHAT (kio)
- WHERE (kie)
- WHEN (kiam)
- HOW_MANY (kiom)
- WHY (kial)
- HOW (kiel)
- WHICH (kia)
- WHOSE (kies)
"""

import pytest
from klareco.parser import parse
from klareco.rag.answer_extractor import ASTAnswerExtractor


class TestQuestionTypeDetection:
    """Test question type detection from correlatives."""

    def test_detect_who(self):
        """Test WHO (kiu) detection."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiu fondis Esperanton?")
        q_type = extractor._detect_question_type(query_ast)
        assert q_type == 'WHO'

    def test_detect_what(self):
        """Test WHAT (kio) detection."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kio estas Esperanto?")
        q_type = extractor._detect_question_type(query_ast)
        assert q_type == 'WHAT'

    def test_detect_where(self):
        """Test WHERE (kie) detection."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kie naskiĝis Zamenhof?")
        q_type = extractor._detect_question_type(query_ast)
        assert q_type == 'WHERE'

    def test_detect_when(self):
        """Test WHEN (kiam) detection."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiam estis fondita Esperanto?")
        q_type = extractor._detect_question_type(query_ast)
        assert q_type == 'WHEN'

    def test_detect_how_many(self):
        """Test HOW_MANY (kiom) detection."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiom da homoj parolas Esperanton?")
        q_type = extractor._detect_question_type(query_ast)
        assert q_type == 'HOW_MANY'

    def test_non_question(self):
        """Test that non-questions return None."""
        extractor = ASTAnswerExtractor()
        ast = parse("Zamenhof fondis Esperanton.")
        q_type = extractor._detect_question_type(ast)
        assert q_type is None


class TestWHOExtraction:
    """Test WHO (kiu) answer extraction."""

    def test_extract_person_subject(self):
        """Test extracting person from subject position."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiu fondis Esperanton?")
        doc_ast = parse("Zamenhof fondis Esperanton en 1887.")
        doc_text = "Zamenhof fondis Esperanton en 1887."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert answer['text'] == 'Zamenhof'
        assert answer['method'] in ('ast_pattern_match', 'ast_ranked_match')
        assert answer['confidence'] > 0.7

    def test_extract_person_with_suffix(self):
        """Test extracting person with -ul suffix."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiu kreis la lingvon?")
        doc_ast = parse("La verkulo kreis la lingvon.")
        doc_text = "La verkulo kreis la lingvon."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'verkulo' in answer['text']

    def test_no_verb_match(self):
        """Test that extraction fails when verbs don't match."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiu fondis Esperanton?")
        doc_ast = parse("Zamenhof parolis pri lingvo.")
        doc_text = "Zamenhof parolis pri lingvo."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        # Should not extract since verbs don't match (fondis vs parolis)
        assert answer is None


class TestWHATExtraction:
    """Test WHAT (kio) answer extraction."""

    def test_extract_definition(self):
        """Test extracting definition after 'estas'."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kio estas Esperanto?")
        doc_ast = parse("Esperanto estas planlingvo.")
        doc_text = "Esperanto estas planlingvo."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'planlingvo' in answer['text']
        assert answer['explanation']  # has some explanation

    def test_extract_object(self):
        """Test extracting object when query has 'kio' in object position."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kion Zamenhof kreis?")
        doc_ast = parse("Zamenhof kreis Esperanton.")
        doc_text = "Zamenhof kreis Esperanton."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'Esperanto' in answer['text']


class TestWHEREExtraction:
    """Test WHERE (kie) answer extraction."""

    def test_extract_location_with_en(self):
        """Test extracting location with 'en' preposition."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kie naskiĝis Zamenhof?")
        doc_ast = parse("Zamenhof naskiĝis en Bjalistoko.")
        doc_text = "Zamenhof naskiĝis en Bjalistoko."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'Bjalistoko' in answer['text']
        assert answer['confidence'] > 0.6

    def test_extract_location_with_ej_suffix(self):
        """Test extracting location with -ej suffix."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kie li loĝas?")
        doc_ast = parse("Li loĝas en domo.")  # 'domo' not 'domejo', but test -ej
        doc_text = "Li loĝas en lernejo."
        doc_ast = parse(doc_text)

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        # Should find either the prepositional phrase or -ej suffix
        assert answer is not None


class TestWHENExtraction:
    """Test WHEN (kiam) answer extraction."""

    def test_extract_year(self):
        """Test extracting year."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiam estis fondita Esperanto?")
        doc_ast = parse("Esperanto estis fondita en 1887.")
        doc_text = "Esperanto estis fondita en 1887."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert '1887' in answer['text']
        assert answer['confidence'] > 0.9

    def test_extract_time_adverb(self):
        """Test extracting time adverb."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiam li venos?")
        doc_ast = parse("Li venos morgaŭ.")
        doc_text = "Li venos morgaŭ."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'morgaŭ' in answer['text']


class TestHOWMANYExtraction:
    """Test HOW_MANY (kiom) answer extraction."""

    def test_extract_number(self):
        """Test extracting numeric quantity."""
        extractor = ASTAnswerExtractor()
        query_ast = parse("Kiom da homoj parolas Esperanton?")
        # Simplified document for testing
        doc_ast = parse("Du milionoj da homoj parolas Esperanton.")
        doc_text = "Du milionoj da homoj parolas Esperanton."

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        # Should extract either "du" or "milionoj"
        assert answer is not None
        assert answer['confidence'] > 0.8


class TestHelperMethods:
    """Test helper methods."""

    def test_is_person_with_capital(self):
        """Test person detection for proper nouns."""
        extractor = ASTAnswerExtractor()
        ast = parse("Zamenhof kreis lingvon.")
        subjekto = ast['subjekto']

        is_person = extractor._is_person(subjekto)
        assert is_person is True

    def test_is_person_with_ul_suffix(self):
        """Test person detection for -ul suffix."""
        extractor = ASTAnswerExtractor()
        ast = parse("La verkulo skribis libron.")
        subjekto = ast['subjekto']

        is_person = extractor._is_person(subjekto)
        assert is_person is True

    def test_looks_like_time_year(self):
        """Test time detection for year."""
        extractor = ASTAnswerExtractor()
        assert extractor._looks_like_time('1887') is True
        assert extractor._looks_like_time('2024') is True
        assert extractor._looks_like_time('999') is False  # Too short
        assert extractor._looks_like_time('3000') is False  # Too far

    def test_looks_like_time_month(self):
        """Test time detection for month."""
        extractor = ASTAnswerExtractor()
        assert extractor._looks_like_time('januaro') is True
        assert extractor._looks_like_time('decembro') is True

    def test_is_number_word(self):
        """Test number word detection."""
        extractor = ASTAnswerExtractor()
        assert extractor._is_number_word('du') is True
        assert extractor._is_number_word('dek') is True
        assert extractor._is_number_word('mil') is True
        assert extractor._is_number_word('multe') is True
        assert extractor._is_number_word('hundo') is False


class TestIntegrationTests:
    """Integration tests with realistic queries."""

    def test_full_pipeline_who(self):
        """Test full extraction pipeline for WHO question."""
        extractor = ASTAnswerExtractor()

        # Query: Who founded Esperanto?
        query_ast = parse("Kiu fondis Esperanton?")

        # Document: Zamenhof founded Esperanto in 1887.
        doc_text = "Zamenhof fondis Esperanton en 1887."
        doc_ast = parse(doc_text)

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'Zamenhof' in answer['text']
        assert answer['method'] in ('ast_pattern_match', 'ast_ranked_match')
        assert 0.7 <= answer['confidence'] <= 1.0
        assert 'explanation' in answer
        assert 'ast' in answer

    def test_full_pipeline_what(self):
        """Test full extraction pipeline for WHAT question."""
        extractor = ASTAnswerExtractor()

        query_ast = parse("Kio estas Esperanto?")
        doc_text = "Esperanto estas internacia planlingvo."
        doc_ast = parse(doc_text)

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        # Should extract either "internacia" or "planlingvo" or both
        assert answer['confidence'] > 0.7

    def test_full_pipeline_where(self):
        """Test full extraction pipeline for WHERE question."""
        extractor = ASTAnswerExtractor()

        query_ast = parse("Kie naskiĝis Zamenhof?")
        doc_text = "Zamenhof naskiĝis en Bjalistoko."
        doc_ast = parse(doc_text)

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is not None
        assert 'Bjalistoko' in answer['text']
        assert answer['confidence'] > 0.6

    def test_no_answer_found(self):
        """Test that None is returned when no answer found."""
        extractor = ASTAnswerExtractor()

        query_ast = parse("Kiu fondis Esperanton?")
        # Irrelevant document
        doc_text = "La hundo dormas."
        doc_ast = parse(doc_text)

        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        assert answer is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
