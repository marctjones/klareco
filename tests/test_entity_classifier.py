"""
Tests for Entity Type Classification.

Tests deterministic classification of entities into semantic types:
LANGUAGE, PERSON, ORGANIZATION, PLACE, EVENT, UNKNOWN.
"""

import pytest
from klareco.entity_classifier import EntityClassifier, EntityType
from klareco.parser import parse


class TestLanguageClassification:
    """Test classification of language entities."""

    def test_standalone_language(self):
        """Standalone language name should be LANGUAGE."""
        classifier = EntityClassifier()

        # Esperanto
        ast = parse("Mi parolas Esperanton.")
        obj = ast.get('objekto', {}).get('kerno', {})
        assert classifier.classify(obj) == EntityType.LANGUAGE

    def test_language_roots(self):
        """All language roots should be detected."""
        classifier = EntityClassifier()

        language_roots = ['esperant', 'angl', 'franc', 'german', 'rus']
        for root in language_roots:
            ast_node = {'radiko': root, 'tipo': 'vorto', 'vortspeco': 'substantivo'}
            assert classifier.classify(ast_node) == EntityType.LANGUAGE


class TestOrganizationClassification:
    """Test classification of organization entities."""

    def test_compound_with_organization_marker(self):
        """Compound with organization marker should be ORGANIZATION."""
        classifier = EntityClassifier()

        # Esperanto-klubo
        ast = parse("Schmidt fondis Esperanto-klubon.")
        obj = ast.get('objekto', {})
        if obj.get('tipo') == 'vortgrupo':
            obj = obj.get('kerno', {})

        result = classifier.classify(obj)
        assert result == EntityType.ORGANIZATION, f"Expected ORGANIZATION, got {result}"

    def test_standalone_organization_marker(self):
        """Standalone organization marker should be ORGANIZATION."""
        classifier = EntityClassifier()

        org_markers = ['klub', 'societ', 'asocia', 'ligo']
        for marker in org_markers:
            ast_node = {'radiko': marker, 'tipo': 'vorto', 'vortspeco': 'substantivo'}
            assert classifier.classify(ast_node) == EntityType.ORGANIZATION

    def test_language_compound_without_marker_is_organization(self):
        """Language compound without specific marker → ORGANIZATION."""
        classifier = EntityClassifier()

        # Esperanto-movado (Esperanto movement)
        ast_node = {
            'radiko': 'mov',
            'tipo': 'vorto',
            'estas_kunmetita': True,
            'kunmetajhoj': [
                {'radiko': 'esperant', 'tipo': 'vorto'}
            ]
        }

        result = classifier.classify(ast_node)
        # "mov" is an organization marker
        assert result == EntityType.ORGANIZATION


class TestPlaceClassification:
    """Test classification of place entities."""

    def test_compound_with_place_marker(self):
        """Compound with place marker should be PLACE."""
        classifier = EntityClassifier()

        place_markers = ['urb', 'land', 'regi', 'ŝtat']
        for marker in place_markers:
            ast_node = {
                'radiko': marker,
                'tipo': 'vorto',
                'vortspeco': 'substantivo',
                'estas_kunmetita': False
            }
            assert classifier.classify(ast_node) == EntityType.PLACE


class TestEventClassification:
    """Test classification of event entities."""

    def test_event_markers(self):
        """Event markers should be classified as EVENT."""
        classifier = EntityClassifier()

        event_markers = ['kongres', 'konferenc', 'simozi', 'fest']
        for marker in event_markers:
            ast_node = {'radiko': marker, 'tipo': 'vorto', 'vortspeco': 'substantivo'}
            assert classifier.classify(ast_node) == EntityType.EVENT

    def test_compound_event(self):
        """Compound with event marker should be EVENT."""
        classifier = EntityClassifier()

        # Universala Kongreso
        ast_node = {
            'radiko': 'kongres',
            'tipo': 'vorto',
            'vortspeco': 'substantivo',
            'estas_kunmetita': True,
            'kunmetajhoj': [
                {'radiko': 'univers', 'tipo': 'vorto'}
            ]
        }
        assert classifier.classify(ast_node) == EntityType.EVENT


class TestPersonClassification:
    """Test classification of person entities."""

    def test_known_person(self):
        """Known person should be classified as PERSON."""
        classifier = EntityClassifier()

        # Zamenhof
        ast_node = {
            'radiko': 'zamenhof',
            'tipo': 'vorto',
            'vortspeco': 'propra_nomo'
        }
        assert classifier.classify(ast_node) == EntityType.PERSON

    def test_unknown_proper_noun_is_unknown(self):
        """Unknown proper noun should be UNKNOWN (needs context)."""
        classifier = EntityClassifier()

        # Random proper noun
        ast_node = {
            'radiko': 'smith',
            'tipo': 'vorto',
            'vortspeco': 'propra_nomo'
        }
        assert classifier.classify(ast_node) == EntityType.UNKNOWN


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_ast(self):
        """Empty AST should return UNKNOWN."""
        classifier = EntityClassifier()
        assert classifier.classify({}) == EntityType.UNKNOWN
        assert classifier.classify(None) == EntityType.UNKNOWN

    def test_vortgrupo_extraction(self):
        """Should extract kerno from vortgrupo."""
        classifier = EntityClassifier()

        ast = parse("Mi parolas Esperanton.")
        obj = ast.get('objekto', {})  # This is a vortgrupo

        # Should extract kerno and classify
        result = classifier.classify(obj)
        assert result == EntityType.LANGUAGE

    def test_compound_priority(self):
        """Compound markers should take priority."""
        classifier = EntityClassifier()

        # Language + organization marker → ORGANIZATION
        ast_node = {
            'radiko': 'klub',
            'tipo': 'vorto',
            'estas_kunmetita': True,
            'kunmetajhoj': [
                {'radiko': 'esperant', 'tipo': 'vorto'}
            ]
        }
        assert classifier.classify(ast_node) == EntityType.ORGANIZATION


class TestSimplifiedClassification:
    """Test simplified text-based classification."""

    def test_classify_from_text_language(self):
        """classify_from_text should detect languages."""
        classifier = EntityClassifier()
        assert classifier.classify_from_text('esperant') == EntityType.LANGUAGE
        assert classifier.classify_from_text('angl') == EntityType.LANGUAGE

    def test_classify_from_text_organization(self):
        """classify_from_text should detect organization markers."""
        classifier = EntityClassifier()
        assert classifier.classify_from_text('klub') == EntityType.ORGANIZATION
        assert classifier.classify_from_text('asocia') == EntityType.ORGANIZATION

    def test_classify_from_text_proper_noun_unknown(self):
        """Unknown proper noun should be UNKNOWN."""
        classifier = EntityClassifier()
        result = classifier.classify_from_text('smith', is_proper_noun=True)
        assert result == EntityType.UNKNOWN


class TestStatistics:
    """Test statistics reporting."""

    def test_get_statistics(self):
        """Should return vocabulary statistics."""
        classifier = EntityClassifier()
        stats = classifier.get_statistics()

        assert stats['languages'] > 0
        assert stats['organization_markers'] > 0
        assert stats['place_markers'] > 0
        assert stats['event_markers'] > 0
        assert stats['known_people'] >= 0


class TestRealWorldExamples:
    """Test with real-world Esperanto examples."""

    def test_esperanto_language_vs_organization(self):
        """Should distinguish Esperanto (language) from Esperanto-klubo (org)."""
        classifier = EntityClassifier()

        # Language
        ast1 = parse("Zamenhof kreis Esperanton.")
        obj1 = ast1.get('objekto', {})
        if obj1.get('tipo') == 'vortgrupo':
            obj1 = obj1.get('kerno', {})
        assert classifier.classify(obj1) == EntityType.LANGUAGE

        # Organization
        ast2 = parse("Schmidt fondis Esperanto-klubon.")
        obj2 = ast2.get('objekto', {})
        if obj2.get('tipo') == 'vortgrupo':
            obj2 = obj2.get('kerno', {})
        assert classifier.classify(obj2) == EntityType.ORGANIZATION

    def test_uea_organization(self):
        """UEA (Universala Esperanto-Asocio) should be ORGANIZATION."""
        classifier = EntityClassifier()

        ast_node = {
            'radiko': 'asocia',
            'tipo': 'vorto',
            'estas_kunmetita': True,
            'kunmetajhoj': [
                {'radiko': 'esperant', 'tipo': 'vorto'},
                {'radiko': 'univers', 'tipo': 'vorto'}
            ]
        }
        assert classifier.classify(ast_node) == EntityType.ORGANIZATION
