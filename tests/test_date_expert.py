"""
Unit tests for DateExpert.

Tests the date/time expert's ability to:
- Detect temporal queries
- Extract date/time information from AST
- Format dates and times in Esperanto
- Handle various temporal query types
- Handle edge cases
"""

import pytest
from datetime import datetime, timedelta
from klareco.parser import parse
from klareco.experts.date_expert import DateExpert


class TestDateExpert:
    """Test suite for DateExpert."""

    def setup_method(self):
        """Initialize expert before each test."""
        self.expert = DateExpert()

    def test_can_handle_current_date_query(self):
        """Test detection of current date query."""
        ast = parse("Kiu tago estas hodiaŭ?")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_current_time_query(self):
        """Test detection of current time query."""
        ast = parse("Kioma horo estas?")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_current_day_query(self):
        """Test detection of day of week query."""
        ast = parse("Kiu tago de la semajno estas hodiaŭ?")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_yesterday_query(self):
        """Test detection of yesterday query."""
        ast = parse("Kiu dato estis hieraŭ?")
        assert self.expert.can_handle(ast) is True

    def test_can_handle_tomorrow_query(self):
        """Test detection of tomorrow query."""
        ast = parse("Kiu dato estos morgaŭ?")
        assert self.expert.can_handle(ast) is True

    def test_cannot_handle_non_temporal(self):
        """Test rejection of non-temporal queries."""
        ast = parse("La hundo vidas la katon.")
        assert self.expert.can_handle(ast) is False

    def test_cannot_handle_math_query(self):
        """Test rejection of mathematical queries."""
        ast = parse("Kiom estas du plus tri?")
        assert self.expert.can_handle(ast) is False

    def test_confidence_high_for_clear_temporal(self):
        """Test high confidence for clear temporal queries."""
        ast = parse("Kiu tago estas hodiaŭ?")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence >= 0.9

    def test_confidence_high_for_time_query(self):
        """Test high confidence for time queries."""
        ast = parse("Kioma horo estas?")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence >= 0.9

    def test_confidence_zero_for_non_temporal(self):
        """Test zero confidence for non-temporal queries."""
        ast = parse("La hundo vidas la katon.")
        confidence = self.expert.estimate_confidence(ast)
        assert confidence == 0.0

    def test_execute_current_date(self):
        """Test execution of current date query."""
        ast = parse("Kiu tago estas hodiaŭ?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert 'timestamp' in result
        assert result['confidence'] == 0.95
        assert result['expert'] == 'Date/Time Tool Expert'
        assert result['query_type'] == 'current_date'

        # Answer should contain "Hodiaŭ estas"
        assert 'Hodiaŭ estas' in result['answer']

    def test_execute_current_time(self):
        """Test execution of current time query."""
        ast = parse("Kioma horo estas?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert result['confidence'] == 0.95
        assert result['query_type'] == 'current_time'

        # Answer should contain "Estas la XX:XX"
        assert 'Estas la' in result['answer']
        assert ':' in result['answer']

    def test_execute_current_day_of_week(self):
        """Test execution of day of week query."""
        ast = parse("Kiu tago de la semajno estas hodiaŭ?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert result['confidence'] == 0.95
        assert result['query_type'] == 'current_day'

        # Answer should contain "Hodiaŭ estas" and a day name
        assert 'Hodiaŭ estas' in result['answer']

        # Should contain one of the Esperanto day names
        day_names = ['lundo', 'mardo', 'merkredo', 'ĵaŭdo', 'vendredo', 'sabato', 'dimanĉo']
        assert any(day in result['answer'] for day in day_names)

    def test_execute_yesterday(self):
        """Test execution of yesterday query."""
        ast = parse("Kiu dato estis hieraŭ?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert result['confidence'] == 0.95
        assert result['query_type'] == 'yesterday'

    def test_execute_tomorrow(self):
        """Test execution of tomorrow query."""
        ast = parse("Kiu dato estos morgaŭ?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert result['confidence'] == 0.95
        assert result['query_type'] == 'tomorrow'

    def test_format_current_date_structure(self):
        """Test date format structure."""
        now = datetime.now()
        formatted = self.expert._format_current_date(now)

        # Should contain "Hodiaŭ estas la", day number, "de", month name, year
        assert 'Hodiaŭ estas la' in formatted
        assert str(now.year) in formatted
        assert 'de' in formatted

        # Should contain month name
        month_names = list(self.expert.MONTHS_EO.values())
        assert any(month in formatted for month in month_names)

    def test_format_current_time_structure(self):
        """Test time format structure."""
        now = datetime.now()
        formatted = self.expert._format_current_time(now)

        # Should contain "Estas la" and HH:MM format
        assert 'Estas la' in formatted
        assert ':' in formatted

        # Should have proper time format (XX:XX)
        time_part = formatted.split('la ')[-1]
        assert len(time_part.split(':')) == 2

    def test_format_current_day_structure(self):
        """Test day of week format structure."""
        now = datetime.now()
        formatted = self.expert._format_current_day(now)

        # Should contain "Hodiaŭ estas" and day name
        assert 'Hodiaŭ estas' in formatted

        # Should contain the correct day name
        expected_day = self.expert.DAYS_EO[now.weekday()]
        assert expected_day in formatted

    def test_determine_query_type_time(self):
        """Test query type determination for time."""
        ast = parse("Kioma horo estas?")
        query_type = self.expert._determine_query_type(ast)
        assert query_type == 'current_time'

    def test_determine_query_type_date(self):
        """Test query type determination for date."""
        ast = parse("Kiu dato estas hodiaŭ?")
        query_type = self.expert._determine_query_type(ast)
        assert query_type == 'current_date'

    def test_determine_query_type_day(self):
        """Test query type determination for day of week."""
        ast = parse("Kiu tago de la semajno?")
        query_type = self.expert._determine_query_type(ast)
        assert query_type == 'current_day'

    def test_extract_temporal_keywords(self):
        """Test extraction of temporal keywords from AST."""
        ast = parse("Kiu tago estas hodiaŭ?")
        words = self.expert._extract_all_words(ast)

        # Should extract all words including temporal keywords
        assert len(words) > 0
        words_lower = [w.lower() for w in words]
        assert any('hodiaŭ' in w or 'hodiaux' in w or 'hodi' in w for w in words_lower)

    def test_handles_simple_temporal_query(self):
        """Test handling of simple temporal query."""
        ast = parse("hodiaŭ")
        # Should handle even single-word temporal query
        assert self.expert.can_handle(ast) is True

    def test_answer_includes_timestamp(self):
        """Test that answer includes ISO timestamp."""
        ast = parse("Kiu tago estas hodiaŭ?")
        result = self.expert.execute(ast)

        assert 'timestamp' in result
        # Should be valid ISO format
        timestamp = result['timestamp']
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp)

    def test_execute_includes_explanation(self):
        """Test that execute includes explanation field."""
        ast = parse("Kiu tago estas hodiaŭ?")
        result = self.expert.execute(ast)

        assert 'explanation' in result
        assert len(result['explanation']) > 0

    def test_handles_multiple_temporal_keywords(self):
        """Test handling query with multiple temporal keywords."""
        ast = parse("Kiu estas la dato kaj horo hodiaŭ?")
        assert self.expert.can_handle(ast) is True

        confidence = self.expert.estimate_confidence(ast)
        # Should have high confidence due to multiple temporal words
        assert confidence >= 0.8

    def test_confidence_medium_for_single_keyword(self):
        """Test medium confidence for single temporal keyword."""
        ast = parse("Kio estas la dato?")
        confidence = self.expert.estimate_confidence(ast)
        # Should have medium confidence (single temporal word)
        assert 0.5 <= confidence < 0.9

    def test_expert_name(self):
        """Test expert has correct name."""
        assert self.expert.name == "Date/Time Tool Expert"

    def test_error_handling_on_invalid_ast(self):
        """Test error handling with invalid AST."""
        # Empty AST should not crash
        result = self.expert.execute({})

        # Should return error response
        assert result['confidence'] == 0.0 or 'error' in result


class TestDateExpertEdgeCases:
    """Test suite for DateExpert edge cases."""

    def setup_method(self):
        """Initialize expert before each test."""
        self.expert = DateExpert()

    def test_handles_malformed_temporal_query(self):
        """Test handling of malformed temporal query."""
        ast = parse("tago horo dato")
        # Should still recognize temporal keywords
        assert self.expert.can_handle(ast) is True

    def test_handles_mixed_case_keywords(self):
        """Test handling of mixed case in keywords."""
        ast = parse("HODIAŬ estas kio?")
        assert self.expert.can_handle(ast) is True

    def test_all_day_names_defined(self):
        """Test that all 7 days are defined."""
        assert len(self.expert.DAYS_EO) == 7
        for i in range(7):
            assert i in self.expert.DAYS_EO

    def test_all_month_names_defined(self):
        """Test that all 12 months are defined."""
        assert len(self.expert.MONTHS_EO) == 12
        for i in range(1, 13):
            assert i in self.expert.MONTHS_EO

    def test_temporal_keywords_not_empty(self):
        """Test that temporal keywords list is populated."""
        assert len(self.expert.TEMPORAL_KEYWORDS) > 0

    def test_question_words_not_empty(self):
        """Test that question words list is populated."""
        assert len(self.expert.QUESTION_WORDS) > 0
