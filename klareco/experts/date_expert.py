"""
Date/Time Tool Expert - Handles temporal queries.

This expert can:
- Answer "what day/time is it" questions
- Parse dates and times from queries
- Perform date arithmetic (add/subtract days, months, etc.)
- Convert between timezones

Examples:
- "Kiu tago estas hodiaŭ?" → "Hodiaŭ estas mardo, la 11-a de novembro, 2025"
- "Kioma horo estas?" → "Estas la 22:45"
- "Kiu dato estas post kvin tagoj?" → "La 16-a de novembro, 2025"
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base import Expert


class DateExpert(Expert):
    """
    Expert for handling date and time queries.

    Uses Python's datetime library for precise temporal calculations.
    """

    # Esperanto day names
    DAYS_EO = {
        0: 'lundo',
        1: 'mardo',
        2: 'merkredo',
        3: 'ĵaŭdo',
        4: 'vendredo',
        5: 'sabato',
        6: 'dimanĉo'
    }

    # Esperanto month names
    MONTHS_EO = {
        1: 'januaro',
        2: 'februaro',
        3: 'marto',
        4: 'aprilo',
        5: 'majo',
        6: 'junio',
        7: 'julio',
        8: 'aŭgusto',
        9: 'septembro',
        10: 'oktobro',
        11: 'novembro',
        12: 'decembro'
    }

    # Temporal keywords in Esperanto
    TEMPORAL_KEYWORDS = [
        'hodiaŭ',  # today
        'hieraŭ',  # yesterday
        'morgaŭ',  # tomorrow
        'tago',    # day
        'dato',    # date
        'horo',    # hour/time
        'tempo',   # time
        'jaro',    # year
        'monato',  # month
        'semajno', # week
        'minuto',  # minute
        'sekundo', # second
    ]

    # Question words for temporal queries
    QUESTION_WORDS = [
        'kiu',     # which
        'kiom',    # how much/many
        'kiam',    # when
    ]

    def __init__(self):
        """Initialize Date/Time Expert."""
        super().__init__("Date/Time Tool Expert")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this is a temporal query.

        Looks for:
        - Temporal keywords (today, time, date, etc.)
        - Question words + temporal nouns

        Args:
            ast: Parsed query AST

        Returns:
            True if this appears to be a date/time query
        """
        if not ast or ast.get('tipo') != 'frazo':
            return False

        # Extract all words from AST
        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Check for temporal keywords
        has_temporal = any(
            keyword in word
            for word in words_lower
            for keyword in self.TEMPORAL_KEYWORDS
        )

        # Check for question + temporal combination
        has_question = any(
            qword in word
            for word in words_lower
            for qword in self.QUESTION_WORDS
        )

        return has_temporal or (has_question and any('tag' in w or 'hor' in w or 'dat' in w for w in words_lower))

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        High confidence if:
        - Direct temporal query ("kiu tago", "kioma horo")
        - Contains "hodiaŭ", "hieraŭ", "morgaŭ"

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]
        text = ' '.join(words_lower)

        # Very high confidence patterns
        # Check for specific word combinations (order-independent)
        # For time queries: kioma/kiu + horo
        if ('kioma' in text or 'kiu' in text) and 'horo' in text:
            return 0.98

        # For day queries: kiu + tago + semajno (day of week)
        if 'kiu' in text and 'tago' in text and 'semajno' in text:
            return 0.98

        # For very direct temporal queries with action words
        if ('hodiaŭ' in text or 'hieraŭ' in text or 'morgaŭ' in text) and 'estas' in text:
            return 0.98
        if 'kiam' in text and 'estas' in text:
            return 0.98

        # High confidence: temporal keywords present
        temporal_count = sum(
            1 for word in words_lower
            if any(kw in word for kw in self.TEMPORAL_KEYWORDS)
        )

        if temporal_count >= 2:
            return 0.90

        if temporal_count == 1:
            return 0.75

        return 0.5

    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute temporal query.

        Args:
            ast: Parsed query AST

        Returns:
            Response with date/time information
        """
        # Validate AST
        if not ast or ast.get('tipo') != 'frazo':
            return {
                'answer': 'Mi ne povas procezi malplenan aŭ malvalidan demandon.',
                'confidence': 0.0,
                'expert': self.name,
                'error': 'Invalid or empty AST'
            }

        try:
            # Get current date/time
            now = datetime.now()

            # Determine query type
            query_type = self._determine_query_type(ast)

            # Execute appropriate handler
            if query_type == 'current_date':
                answer = self._format_current_date(now)
                explanation = "Raportas la nunan daton"

            elif query_type == 'current_time':
                answer = self._format_current_time(now)
                explanation = "Raportas la nunan tempon"

            elif query_type == 'current_day':
                answer = self._format_current_day(now)
                explanation = "Raportas la nunan tagon de la semajno"

            elif query_type == 'yesterday':
                yesterday = now - timedelta(days=1)
                answer = self._format_full_date(yesterday)
                explanation = "Raportas hieraŭan daton"

            elif query_type == 'tomorrow':
                tomorrow = now + timedelta(days=1)
                answer = self._format_full_date(tomorrow)
                explanation = "Raportas morgaŭan daton"

            else:
                # Default: provide full date and time
                answer = self._format_full_datetime(now)
                explanation = "Raportas la nunan daton kaj tempon"

            return {
                'answer': answer,
                'timestamp': now.isoformat(),
                'query_type': query_type,
                'confidence': 0.95,
                'expert': self.name,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'answer': f"Eraro dum trakto de tempa demando: {str(e)}",
                'confidence': 0.0,
                'expert': self.name,
                'error': str(e)
            }

    def _extract_all_words(self, ast: Dict[str, Any]) -> List[str]:
        """Extract all words from AST recursively."""
        words = []

        if isinstance(ast, dict):
            if ast.get('tipo') == 'vorto':
                word = ast.get('plena_vorto', '') or ast.get('radiko', '')
                if word:
                    words.append(word)

            # Recursively extract from all fields
            for value in ast.values():
                if isinstance(value, (dict, list)):
                    words.extend(self._extract_all_words(value))

        elif isinstance(ast, list):
            for item in ast:
                words.extend(self._extract_all_words(item))

        return words

    def _determine_query_type(self, ast: Dict[str, Any]) -> str:
        """
        Determine the type of temporal query.

        Returns:
            Query type string: 'current_date', 'current_time', 'current_day', etc.
        """
        words = self._extract_all_words(ast)
        text = ' '.join(words).lower()

        # Check for day of week queries (kiu tago + semajno)
        if ('kiu' in text or 'kio' in text) and 'tago' in text and 'semajno' in text:
            return 'current_day'

        # Check for time queries (kioma/kiu horo)
        if ('kioma' in text or 'kiu' in text) and 'horo' in text:
            return 'current_time'

        # Check for specific patterns
        if 'kiu dato' in text:
            return 'current_date'

        if 'hieraŭ' in text:
            return 'yesterday'

        if 'morgaŭ' in text:
            return 'tomorrow'

        # Check for time-related words
        if any(word in text for word in ['horo', 'tempo', 'minuto']):
            return 'current_time'

        # Default to date
        return 'current_date'

    def _format_current_date(self, dt: datetime) -> str:
        """Format current date in Esperanto."""
        day = dt.day
        month = self.MONTHS_EO[dt.month]
        year = dt.year

        return f"Hodiaŭ estas la {day}-a de {month}, {year}"

    def _format_current_time(self, dt: datetime) -> str:
        """Format current time in Esperanto."""
        hour = dt.hour
        minute = dt.minute

        return f"Estas la {hour:02d}:{minute:02d}"

    def _format_current_day(self, dt: datetime) -> str:
        """Format current day of week in Esperanto."""
        day_name = self.DAYS_EO[dt.weekday()]

        return f"Hodiaŭ estas {day_name}"

    def _format_full_date(self, dt: datetime) -> str:
        """Format full date with day of week in Esperanto."""
        day_name = self.DAYS_EO[dt.weekday()]
        day = dt.day
        month = self.MONTHS_EO[dt.month]
        year = dt.year

        return f"{day_name.capitalize()}, la {day}-a de {month}, {year}"

    def _format_full_datetime(self, dt: datetime) -> str:
        """Format full date and time in Esperanto."""
        date_str = self._format_full_date(dt)
        hour = dt.hour
        minute = dt.minute

        return f"{date_str}, je la {hour:02d}:{minute:02d}"


# Export
__all__ = ['DateExpert']
