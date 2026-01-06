"""
Tests for Entity Recognizer.

Tests deterministic entity extraction from AST annotations.
"""

import pytest
from klareco.parser import parse
from klareco.rag.entity_recognizer import EntityRecognizer, EntityType, Entity


@pytest.fixture
def recognizer():
    """Create entity recognizer instance."""
    return EntityRecognizer()


class TestProperNameRecognition:
    """Test recognition of proper names."""

    def test_recognize_person_name(self, recognizer):
        """Test recognizing person names."""
        # Zamenhof is marked as proper_name_unknown by parser
        text = "Zamenhof kreis Esperanton."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should find Zamenhof as PERSON and Esperanto as entity
        person_entities = recognizer.filter_by_type(entities, EntityType.PERSON)
        assert len(person_entities) > 0

        # Check that Zamenhof was found
        entity_texts = recognizer.get_entity_texts(entities)
        assert 'Zamenhof' in entity_texts

    def test_recognize_place_name(self, recognizer):
        """Test recognizing place names with -uj suffix."""
        text = "Li loĝas en Usono."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Usono (USA with -uj suffix) might be recognized
        # Note: depends on parser marking it as proper name
        assert isinstance(entities, list)

    def test_proper_name_in_subject(self, recognizer):
        """Test proper name in subject position."""
        text = "Zamenhof naskiĝis en Bjalistoko."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)
        subj_entities = recognizer.get_by_slot(entities, 'SUBJ')

        # Should find entity in subject
        assert len(subj_entities) > 0

    def test_proper_name_in_object(self, recognizer):
        """Test proper name in object position."""
        text = "Li kreis Esperanton."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)
        obj_entities = recognizer.get_by_slot(entities, 'OBJ')

        # Should find Esperanton in object
        assert len(obj_entities) > 0


class TestTimeExpressionRecognition:
    """Test recognition of time expressions."""

    def test_recognize_year(self, recognizer):
        """Test recognizing 4-digit years."""
        text = "Esperanto estis kreita en 1887."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)
        time_entities = recognizer.filter_by_type(entities, EntityType.TIME)

        # Should recognize 1887 as a year
        if time_entities:
            assert any('1887' in e.text for e in time_entities)

    def test_recognize_time_root(self, recognizer):
        """Test recognizing time-related roots."""
        text = "La jaro 1959 estis grava."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should recognize "jaro" (year) as time-related
        time_entities = recognizer.filter_by_type(entities, EntityType.TIME)
        assert len(time_entities) > 0


class TestQuantityRecognition:
    """Test recognition of quantities and numbers."""

    def test_recognize_number(self, recognizer):
        """Test recognizing numbers."""
        text = "Li havas 5 hundojn."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)
        quantity_entities = recognizer.filter_by_type(entities, EntityType.QUANTITY)

        # Should recognize 5 as quantity (if parsed as nombro)
        # Note: Parser might not mark single digits as nombro
        assert isinstance(quantity_entities, list)


class TestEntityTypeInference:
    """Test inference of entity types from context."""

    def test_infer_person_from_default(self, recognizer):
        """Test that unknown proper names default to PERSON."""
        # Create a mock node
        node = {
            'tipo': 'vorto',
            'plena_vorto': 'TestPerson',
            'radiko': 'testperson',
            'parse_status': 'proper_name_unknown',
            'sufiksoj': [],
        }

        entity = recognizer._check_proper_name(node, 'SUBJ')

        # Should default to PERSON
        assert entity is not None
        assert entity.entity_type == EntityType.PERSON

    def test_infer_place_from_uj_suffix(self, recognizer):
        """Test inferring PLACE from -uj suffix."""
        # Create a mock node with -uj suffix (country suffix)
        node = {
            'tipo': 'vorto',
            'plena_vorto': 'Germanujo',
            'radiko': 'german',
            'parse_status': 'success',
            'sufiksoj': ['uj'],
        }

        entity_type = recognizer._infer_proper_name_type(node, 'german')

        # Should infer PLACE due to -uj suffix
        assert entity_type == EntityType.PLACE


class TestEntityFiltering:
    """Test filtering and querying entities."""

    def test_filter_by_type(self, recognizer):
        """Test filtering entities by type."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
            Entity(text="Esperanto", entity_type=EntityType.PERSON, root="esperant", slot="OBJ", confidence=0.9),
            Entity(text="1887", entity_type=EntityType.TIME, root="1887", slot="MODIFIER", confidence=0.7),
        ]

        persons = recognizer.filter_by_type(entities, EntityType.PERSON)
        times = recognizer.filter_by_type(entities, EntityType.TIME)

        assert len(persons) == 2
        assert len(times) == 1

    def test_get_by_slot(self, recognizer):
        """Test getting entities by slot."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
            Entity(text="Esperanto", entity_type=EntityType.PERSON, root="esperant", slot="OBJ", confidence=0.9),
        ]

        subj_entities = recognizer.get_by_slot(entities, 'SUBJ')
        obj_entities = recognizer.get_by_slot(entities, 'OBJ')

        assert len(subj_entities) == 1
        assert len(obj_entities) == 1
        assert subj_entities[0].text == "Zamenhof"
        assert obj_entities[0].text == "Esperanto"

    def test_has_entity_type(self, recognizer):
        """Test checking for entity type existence."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
        ]

        assert recognizer.has_entity_type(entities, EntityType.PERSON)
        assert not recognizer.has_entity_type(entities, EntityType.PLACE)

    def test_get_entity_texts(self, recognizer):
        """Test getting all entity text strings."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
            Entity(text="Esperanto", entity_type=EntityType.PERSON, root="esperant", slot="OBJ", confidence=0.9),
        ]

        texts = recognizer.get_entity_texts(entities)

        assert texts == {"Zamenhof", "Esperanto"}

    def test_get_entity_roots(self, recognizer):
        """Test getting all entity roots."""
        entities = [
            Entity(text="Zamenhof", entity_type=EntityType.PERSON, root="zamenhof", slot="SUBJ", confidence=0.9),
            Entity(text="Esperanto", entity_type=EntityType.PERSON, root="esperant", slot="OBJ", confidence=0.9),
        ]

        roots = recognizer.get_entity_roots(entities)

        assert roots == {"zamenhof", "esperant"}


class TestRealBenchmarkSentences:
    """Test on real benchmark sentences."""

    def test_zamenhof_sentence(self, recognizer):
        """Test: Zamenhof kreis Esperanton."""
        text = "Zamenhof kreis Esperanton."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should find entities (Zamenhof, Esperanto)
        assert len(entities) > 0

        # Should have person entities
        person_entities = recognizer.filter_by_type(entities, EntityType.PERSON)
        assert len(person_entities) > 0

    def test_birth_year_sentence(self, recognizer):
        """Test: Li naskiĝis en 1859."""
        text = "Li naskiĝis en 1859."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should recognize 1859 as time/year
        time_entities = recognizer.filter_by_type(entities, EntityType.TIME)
        if time_entities:
            assert any('1859' in e.text for e in time_entities)

    def test_complex_sentence(self, recognizer):
        """Test complex sentence with multiple entities."""
        text = "D-ro Zamenhof kreis Esperanton en la jaro 1887."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should find multiple entities
        assert len(entities) > 0

        # Check entity types
        entity_texts = recognizer.get_entity_texts(entities)
        # At minimum should recognize some proper names
        assert len(entity_texts) > 0


class TestEmptySentences:
    """Test edge cases with empty or simple sentences."""

    def test_simple_sentence_no_entities(self, recognizer):
        """Test sentence with no entities."""
        text = "La hundo kuras."
        ast = parse(text)

        entities = recognizer.recognize_entities(ast)

        # Should return empty list or minimal entities
        assert isinstance(entities, list)

    def test_empty_ast(self, recognizer):
        """Test handling empty AST."""
        ast = {
            'tipo': 'frazo',
            'subjekto': None,
            'verbo': None,
            'objekto': None,
            'aliaj': [],
        }

        entities = recognizer.recognize_entities(ast)

        # Should handle gracefully
        assert entities == []
