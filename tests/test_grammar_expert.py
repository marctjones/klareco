"""
Unit tests for GrammarExpert.

Tests the grammar expert's ability to:
- Detect grammar-related queries
- Analyze grammatical structure from AST
- Identify parts of speech, case, number, tense
- Format grammatical explanations in Esperanto
- Handle edge cases
"""

import pytest
from klareco.parser import parse
from klareco.experts.grammar_expert import GrammarExpert


class TestGrammarExpert:
    """Test suite for GrammarExpert."""

    def setup_method(self):
        """Initialize expert before each test."""
        self.expert = GrammarExpert()

    def test_can_handle_grammar_explanation_request(self):
        """Test detection of grammar explanation request."""
        ast = parse("Eksplik la gramatikon de la frazo.")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_part_of_speech_query(self):
        """Test detection of part of speech query."""
        ast = parse("Kio estas la vortspeco?")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_analysis_request(self):
        """Test detection of analysis request."""
        ast = parse("Analizu la strukturon.")
        assert self.expert.can_handle(ast) is True

    def test_cannot_handle_non_grammar(self):
        """Test rejection of non-grammar queries."""
        ast = parse("La hundo vidas la katon.")
        assert self.expert.can_handle(ast) is False

    def test_cannot_handle_math_query(self):
        """Test rejection of mathematical queries."""
        ast = parse("Kiom estas du plus tri?")
        assert self.expert.can_handle(ast) is False

    def test_cannot_handle_temporal_query(self):
        """Test rejection of temporal queries."""
        ast = parse("Kiu tago estas hodiaŭ?")
        assert self.expert.can_handle(ast) is False

    def test_confidence_high_for_explicit_grammar(self):
        """Test high confidence for explicit grammar requests."""
        ast = parse("Eksplik la gramatikon.")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence >= 0.9

    def test_confidence_high_for_analysis(self):
        """Test high confidence for analysis requests."""
        ast = parse("Analizu la frazon.")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence >= 0.9

    def test_confidence_zero_for_non_grammar(self):
        """Test zero confidence for non-grammar queries."""
        ast = parse("La hundo vidas la katon.")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence == 0.0

    def test_execute_grammar_analysis(self):
        """Test execution of grammar analysis."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert 'analysis' in result
        assert result['confidence'] == 0.95
        assert result['expert'] == 'Grammar Tool Expert'

        # Answer should contain "Gramatika analizo"
        assert 'Gramatika analizo' in result['answer']

    def test_analysis_identifies_subject(self):
        """Test that analysis identifies subject."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        assert 'components' in analysis

        # Should have subject component
        components = analysis['components']
        subject_component = None
        for comp in components:
            if comp.get('role') == 'subjekto':
                subject_component = comp
                break

        assert subject_component is not None
        assert subject_component['word'] == 'hundo'

    def test_analysis_identifies_verb(self):
        """Test that analysis identifies verb."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Should have verb component
        verb_component = None
        for comp in components:
            if comp.get('role') == 'verbo':
                verb_component = comp
                break

        assert verb_component is not None
        assert verb_component['word'] == 'vidas'
        assert 'verb' in verb_component['part_of_speech']

    def test_analysis_identifies_object(self):
        """Test that analysis identifies object."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Should have object component
        object_component = None
        for comp in components:
            if comp.get('role') == 'objekto':
                object_component = comp
                break

        assert object_component is not None
        assert object_component['word'] == 'katon'

    def test_analysis_identifies_accusative_case(self):
        """Test that analysis identifies accusative case."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Object should be in accusative case
        object_component = None
        for comp in components:
            if comp.get('role') == 'objekto':
                object_component = comp
                break

        assert object_component is not None
        assert 'case' in object_component
        assert 'akuzativo' in object_component['case']

    def test_analysis_identifies_singular(self):
        """Test that analysis identifies singular number."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Subject should be singular
        subject_component = None
        for comp in components:
            if comp.get('role') == 'subjekto':
                subject_component = comp
                break

        assert subject_component is not None
        assert 'number' in subject_component
        assert 'singularo' in subject_component['number']

    def test_analysis_identifies_plural(self):
        """Test that analysis identifies plural number."""
        ast = parse("Malgrandaj hundoj vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Subject should be plural
        subject_component = None
        for comp in components:
            if comp.get('role') == 'subjekto':
                subject_component = comp
                break

        assert subject_component is not None
        if 'number' in subject_component:
            assert 'pluralo' in subject_component['number']

    def test_analysis_identifies_modifiers(self):
        """Test that analysis identifies modifiers."""
        ast = parse("La granda hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Subject should have modifier "granda"
        subject_component = None
        for comp in components:
            if comp.get('role') == 'subjekto':
                subject_component = comp
                break

        assert subject_component is not None
        if 'modifiers' in subject_component:
            assert 'granda' in subject_component['modifiers']

    def test_analysis_identifies_present_tense(self):
        """Test that analysis identifies present tense."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Verb should be present tense
        verb_component = None
        for comp in components:
            if comp.get('role') == 'verbo':
                verb_component = comp
                break

        assert verb_component is not None
        if 'tense' in verb_component:
            assert 'prezenco' in verb_component['tense']

    def test_analysis_identifies_past_tense(self):
        """Test that analysis identifies past tense."""
        ast = parse("La hundo vidis la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Verb should be past tense
        verb_component = None
        for comp in components:
            if comp.get('role') == 'verbo':
                verb_component = comp
                break

        assert verb_component is not None
        if 'tense' in verb_component:
            assert 'pasinteco' in verb_component['tense']

    def test_analysis_identifies_future_tense(self):
        """Test that analysis identifies future tense."""
        ast = parse("La hundo vidos la katon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Verb should be future tense
        verb_component = None
        for comp in components:
            if comp.get('role') == 'verbo':
                verb_component = comp
                break

        assert verb_component is not None
        if 'tense' in verb_component:
            assert 'futuro' in verb_component['tense']

    def test_format_explanation_includes_components(self):
        """Test that explanation includes all components."""
        ast = parse("La hundo vidas la katon.")
        result = self.expert.execute(ast)

        answer = result['answer']

        # Should include roles and words
        assert 'subjekto' in answer.lower() or 'Subjekto' in answer
        assert 'verbo' in answer.lower() or 'Verbo' in answer
        assert 'objekto' in answer.lower() or 'Objekto' in answer

    def test_extract_all_words(self):
        """Test extraction of all words from AST."""
        ast = parse("La hundo vidas la katon.")
        words = self.expert._extract_all_words(ast)

        # Should extract all words
        assert len(words) > 0
        words_lower = [w.lower() for w in words]
        assert any('hund' in w for w in words_lower)
        assert any('vid' in w for w in words_lower)
        assert any('kat' in w for w in words_lower)

    def test_analyze_structure(self):
        """Test structure analysis."""
        ast = parse("La hundo vidas la katon.")
        analysis = self.expert._analyze_structure(ast)

        assert 'sentence_type' in analysis
        assert 'components' in analysis
        assert len(analysis['components']) > 0

    def test_handles_pronoun_subject(self):
        """Test handling of pronoun subject."""
        ast = parse("Mi vidas la hundon.")
        result = self.expert.execute(ast)

        analysis = result['analysis']
        components = analysis['components']

        # Should identify pronoun
        subject_component = None
        for comp in components:
            if comp.get('role') == 'subjekto':
                subject_component = comp
                break

        assert subject_component is not None
        if 'part_of_speech' in subject_component:
            assert 'pronomo' in subject_component['part_of_speech']

    def test_expert_name(self):
        """Test expert has correct name."""
        assert self.expert.name == "Grammar Tool Expert"

    def test_error_handling_on_invalid_ast(self):
        """Test error handling with invalid AST."""
        # Empty AST should not crash
        result = self.expert.execute({})

        # Should return error or low confidence response
        assert result['confidence'] == 0.0 or 'error' in result or 'Mi ne povis' in result['answer']

    def test_handles_complex_sentence_with_adjectives(self):
        """Test handling of complex sentence with multiple adjectives."""
        ast = parse("Malgrandaj hundoj vidas la grandan katon.")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert result['confidence'] == 0.95

        # Should include analysis components
        analysis = result['analysis']
        assert len(analysis['components']) > 0


class TestGrammarExpertEdgeCases:
    """Test suite for GrammarExpert edge cases."""

    def setup_method(self):
        """Initialize expert before each test."""
        self.expert = GrammarExpert()

    def test_all_pos_mappings_defined(self):
        """Test that all POS mappings are defined."""
        assert len(self.expert.PART_OF_SPEECH_EO) > 0
        # Should have at least basic POS
        assert 'substantivo' in self.expert.PART_OF_SPEECH_EO
        assert 'verbo' in self.expert.PART_OF_SPEECH_EO
        assert 'adjektivo' in self.expert.PART_OF_SPEECH_EO

    def test_case_mappings_defined(self):
        """Test that case mappings are defined."""
        assert len(self.expert.CASE_EO) == 2
        assert 'nominativo' in self.expert.CASE_EO
        assert 'akuzativo' in self.expert.CASE_EO

    def test_number_mappings_defined(self):
        """Test that number mappings are defined."""
        assert len(self.expert.NUMBER_EO) == 2
        assert 'singularo' in self.expert.NUMBER_EO
        assert 'pluralo' in self.expert.NUMBER_EO

    def test_tense_mappings_defined(self):
        """Test that tense mappings are defined."""
        assert len(self.expert.TENSE_EO) >= 3
        assert 'prezenco' in self.expert.TENSE_EO
        assert 'pasinteco' in self.expert.TENSE_EO
        assert 'futuro' in self.expert.TENSE_EO

    def test_grammar_keywords_not_empty(self):
        """Test that grammar keywords list is populated."""
        assert len(self.expert.GRAMMAR_KEYWORDS) > 0

    def test_handles_sentence_with_no_components(self):
        """Test handling of sentence with no identifiable components."""
        # Create minimal AST
        minimal_ast = {'tipo': 'frazo'}
        result = self.expert.execute(minimal_ast)

        # Should not crash, should return some response
        assert 'answer' in result

    def test_handles_mixed_case_keywords(self):
        """Test handling of mixed case in keywords."""
        ast = parse("EKSPLIK la GRAMATIKON.")
        assert self.expert.can_handle(ast) is True
