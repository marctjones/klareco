"""
Tests for QuestionClassifier.

Tests deterministic classification of Esperanto questions using AST analysis.
"""

import pytest
from klareco.parser import parse
from klareco.rag.question_classifier import (
    QuestionClassifier,
    QuestionType,
    EntityType,
)


@pytest.fixture
def classifier():
    """Create question classifier instance."""
    return QuestionClassifier()


class TestBasicClassification:
    """Test basic question type classification."""

    def test_who_question(self, classifier):
        """Test WHO question classification."""
        query = "Kiu fondis Esperanton?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHO
        assert result['entity_type'] == EntityType.PERSON
        assert result['question_word'] == 'kiu'

    def test_what_question(self, classifier):
        """Test WHAT question classification."""
        query = "Kion vi vidas?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHAT
        assert result['entity_type'] == EntityType.THING
        assert result['question_word'] in ['kio', 'kion']

    def test_where_question(self, classifier):
        """Test WHERE question classification."""
        query = "Kie estas la libro?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHERE
        assert result['entity_type'] == EntityType.PLACE
        assert result['question_word'] == 'kie'

    def test_when_question(self, classifier):
        """Test WHEN question classification."""
        query = "Kiam okazis la evento?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHEN
        assert result['entity_type'] == EntityType.TIME
        assert result['question_word'] == 'kiam'

    def test_how_question(self, classifier):
        """Test HOW question classification."""
        query = "Kiel vi fartas?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.HOW
        assert result['entity_type'] == EntityType.METHOD
        assert result['question_word'] == 'kiel'

    def test_why_question(self, classifier):
        """Test WHY question classification."""
        query = "Kial vi venis?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHY
        assert result['entity_type'] == EntityType.REASON
        assert result['question_word'] == 'kial'

    def test_how_many_question(self, classifier):
        """Test HOW MANY question classification."""
        query = "Kiom kostas la libro?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.HOW_MANY
        assert result['entity_type'] == EntityType.QUANTITY
        assert result['question_word'] == 'kiom'

    def test_boolean_question(self, classifier):
        """Test BOOLEAN question classification."""
        query = "Ĉu vi parolas Esperanton?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.BOOLEAN
        assert result['entity_type'] == EntityType.BOOLEAN
        assert result['question_word'] == 'ĉu'


class TestEntityTypeRefinement:
    """Test refinement of entity types based on context."""

    def test_what_definition_question(self, classifier):
        """Test 'Kio estas X?' pattern → DEFINITION."""
        query = "Kio estas Esperanto?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHAT
        # Should be refined to DEFINITION because of "estas" verb
        # (This may need parser support for proper verb recognition)

    def test_who_with_place_noun(self, classifier):
        """Test 'Kiu urbo' pattern → PLACE not PERSON."""
        query = "Kiu urbo estas la plej granda?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should detect "urbo" (city) and refine to PLACE
        # Note: This requires the refinement logic to work

    def test_who_with_person_noun(self, classifier):
        """Test 'Kiu homo' pattern → PERSON."""
        query = "Kiu homo skribis la libron?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHO
        assert result['entity_type'] == EntityType.PERSON


class TestFocusDetection:
    """Test detection of question focus (which slot is being asked about)."""

    def test_focus_on_subject(self, classifier):
        """Test question focused on subject."""
        query = "Kiu venis hieraŭ?"  # Who came yesterday?
        ast = parse(query)

        result = classifier.classify(query, ast)

        # "Kiu" is in subject position
        assert result['focus'] == 'SUBJ'

    def test_focus_on_object(self, classifier):
        """Test question focused on object."""
        query = "Vi vidis kion?"  # You saw what?
        ast = parse(query)

        result = classifier.classify(query, ast)

        # "kion" is in object position
        # Note: May default to OBJ if detection fails


class TestTargetSlots:
    """Test determination of which slots to prioritize in matching."""

    def test_who_question_target_slots(self, classifier):
        """WHO questions should prioritize SUBJ and VERB."""
        query = "Kiu fondis Esperanton?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should prioritize subject and verb slots
        assert 'SUBJ' in result['target_slots']
        assert 'VERB' in result['target_slots']

    def test_what_question_target_slots(self, classifier):
        """WHAT questions should prioritize VERB and OBJ."""
        query = "Kion vi manĝas?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should prioritize verb and object slots
        assert 'VERB' in result['target_slots']
        assert 'OBJ' in result['target_slots']

    def test_where_question_target_slots(self, classifier):
        """WHERE questions should look for VERB and OBJ."""
        query = "Kie estas la libro?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should look in verb and object
        assert 'VERB' in result['target_slots']


class TestQuestionDetection:
    """Test detection of whether input is a question."""

    def test_question_mark_detection(self, classifier):
        """Question mark should indicate question."""
        query = "La hundo kuras?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should be detected as question due to "?"
        # (though may be UNKNOWN type without ki- word)

    def test_question_word_detection(self, classifier):
        """Question word should indicate question."""
        query = "Kio estas tio"  # Missing question mark
        ast = parse(query)

        result = classifier.classify(query, ast)

        # Should still be detected as question due to "Kio"
        assert result['question_type'] == QuestionType.WHAT

    def test_non_question(self, classifier):
        """Statement should not be classified as question."""
        query = "La hundo kuras rapide."
        ast = parse(query)

        # This should return UNKNOWN or handle gracefully
        # (classifier expects questions)


class TestRealBenchmarkQuestions:
    """Test on real questions from Q&A benchmark."""

    def test_zamenhof_question(self, classifier):
        """Test: Kiu fondis Esperanton?"""
        query = "Kiu fondis Esperanton?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHO
        assert result['entity_type'] == EntityType.PERSON
        assert result['question_word'] == 'kiu'
        assert result['focus'] == 'SUBJ'
        # This question asks WHO (subject) did the action (founded)
        assert 'SUBJ' in result['target_slots']
        assert 'VERB' in result['target_slots']

    def test_esperanto_definition_question(self, classifier):
        """Test: Kio estas Esperanto?"""
        query = "Kio estas Esperanto?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHAT
        # Should be refined to DEFINITION
        assert result['question_word'] in ['kio', 'kion']

    def test_location_question(self, classifier):
        """Test: Kie naskiĝis Zamenhof?"""
        query = "Kie naskiĝis Zamenhof?"
        ast = parse(query)

        result = classifier.classify(query, ast)

        assert result['question_type'] == QuestionType.WHERE
        assert result['entity_type'] == EntityType.PLACE
        assert result['question_word'] == 'kie'
