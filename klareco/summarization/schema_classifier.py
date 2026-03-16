"""
Schema Classifier - Deterministic Query Type Detection

VERSION: v2.1
COMPATIBLE WITH: Pure Esperanto queries, v2.1 database schema
STAGE: Summarization - Phase 0

Description:
    Classifies Esperanto queries into schema types using deterministic
    pattern matching. Supports biographical, definitional, and event schemas.

Schema Types:
    - biographical: Person-focused (birth, death, achievements, roles)
    - definitional: Concept-focused (category, properties, function)
    - event: Time-focused (what happened, when, where, participants)

Usage:
    from klareco.summarization import SchemaClassifier

    classifier = SchemaClassifier()
    result = classifier.classify("Kiu estis Zamenhof?")
    # Returns: {'schema': 'biographical', 'confidence': 0.95, 'indicators': [...]}

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Result of schema classification."""
    schema: str  # 'biographical', 'definitional', 'event'
    confidence: float  # 0.0-1.0
    indicators: List[str]  # What patterns triggered this classification
    subject: Optional[str] = None  # Extracted subject (if detected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': self.schema,
            'confidence': self.confidence,
            'indicators': self.indicators,
            'subject': self.subject
        }


class SchemaClassifier:
    """
    Deterministic schema classifier using pattern matching.

    Uses linguistic patterns in Esperanto queries to detect:
    - Biographical: Person-focused summaries
    - Definitional: Concept/category summaries
    - Event: Time-based event summaries
    """

    def __init__(self):
        """Initialize pattern matchers."""
        # Biographical patterns
        self.biographical_patterns = [
            # Direct person queries
            (r'\bkiu\s+(estis|estas|fondis|kreis)\b', 0.90, "who_was_did"),
            (r'\brakontu\s+pri\s+[A-ZĈĜĤĴŜŬ]', 0.95, "tell_about_person"),
            (r'\bbiografi[ao]\b', 0.95, "biography_keyword"),
            (r'\b(naskiĝ|mort|viv)[oi]s?\b', 0.85, "life_events"),
            (r'\b(patr|matr|frat|fil)[oi]n?\b', 0.70, "family_relations"),
            (r'\bprofesi[ao]\b', 0.75, "profession"),
            (r'\b(kreist|fondint|elpensint|fondinto)[oi]s?\b', 0.80, "creator_founder"),
            # Person name + verb (e.g., "Zamenhof kreis")
            (r'\b[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+\s+(kreis|fondis|estis|vivis|mortis|naskiĝis)\b', 0.90, "person_action"),
            # Questions about person's motivations/influences
            (r'\b(kial|kiuj)\s+[A-ZĈĜĤĴŜŬ]', 0.85, "person_why_who"),
            (r'\binspiris\s+[A-ZĈĜĤĴŜŬ]', 0.85, "influenced_person"),
            # Name detection (capitalized words not at sentence start)
            (r'(?<!^)\b[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+\b', 0.65, "proper_name"),
        ]

        # Definitional patterns
        self.definitional_patterns = [
            # Direct definition queries
            (r'\bkio\s+(estas|estis)\b', 0.90, "what_is"),
            (r'\bdifinu\b', 0.95, "define_command"),
            (r'\bpriskribu\b', 0.85, "describe_command"),
            (r'\bklarig[ui]\b', 0.85, "explain_command"),
            (r'\bsignif[oi]s?\b', 0.80, "meaning"),
            (r'\bkarakteri[sz]o\b', 0.75, "characteristic"),
            (r'\btip[oi]n?\b', 0.70, "type"),
            (r'\bspecon?\b', 0.70, "species_type"),
            (r'\bkategorio\b', 0.75, "category"),
            # Common defined entities (languages, systems, concepts)
            (r'\b(lingvo|sistemo|ideo|koncepto|teorio)\b', 0.65, "abstract_concept"),
            # Special case: "Esperanto" is a concept/language, not a person
            (r'\bEsperanto[jn]?\b', 0.90, "esperanto_language"),
            # Questions about properties/usage (not biographical)
            (r'\b(kiuj|kiom)\s+(da\s+)?homoj\s+parolas\b', 0.85, "usage_question"),
        ]

        # Event patterns
        self.event_patterns = [
            # Direct event queries
            (r'\bkio\s+(okazis|okazas)\b', 0.95, "what_happened"),
            (r'\brakontu\s+pri\s+(la\s+)?[a-zĉĝĥĵŝŭ]+\s+(kongres|konfer|milit|revoluci)', 0.90, "tell_about_event"),
            (r'\bkiam\s+(okazis|estis)\b', 0.85, "when_happened"),
            (r'\bkie\s+(okazis|estis)\b', 0.80, "where_happened"),
            # Temporal expressions
            (r'\ben\s+\d{3,4}\b', 0.85, "year_reference"),
            (r'\b(en|dum)\s+(la\s+)?(jar[oi]|monat[oi]|tag[oi])\b', 0.75, "temporal_reference"),
            (r'\b(antaŭ|post|inter)\s+\d', 0.70, "relative_time"),
            # Event nouns
            (r'\b(event|okazaĵ|fest|kongres|konfer|milit|revoluci|manifest|fund)[oi]n?\b', 0.80, "event_noun"),
            (r'\b(ĉe|en)\s+[A-ZĈĜĤĴŜŬ]', 0.65, "location_reference"),
        ]

        # Question word weights (when ambiguous)
        self.question_weights = {
            'kiu': 'biographical',   # who (person)
            'kio': 'definitional',   # what (thing/concept)
            'kiam': 'event',         # when (time)
            'kie': 'event',          # where (place)
            'kial': 'definitional',  # why (reason/explanation)
            'kiel': 'definitional',  # how (manner/method)
        }

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify query into schema type.

        Args:
            query: Esperanto query string

        Returns:
            ClassificationResult with schema, confidence, and indicators
        """
        query_lower = query.lower()

        # Score each schema type
        bio_score, bio_indicators = self._score_patterns(query, self.biographical_patterns)
        def_score, def_indicators = self._score_patterns(query, self.definitional_patterns)
        event_score, event_indicators = self._score_patterns(query, self.event_patterns)

        # Boost scores based on question word
        question_word = self._extract_question_word(query_lower)
        if question_word in self.question_weights:
            boost = 0.15
            if self.question_weights[question_word] == 'biographical':
                bio_score += boost
            elif self.question_weights[question_word] == 'definitional':
                def_score += boost
            elif self.question_weights[question_word] == 'event':
                event_score += boost

        # Determine winner
        scores = {
            'biographical': (bio_score, bio_indicators),
            'definitional': (def_score, def_indicators),
            'event': (event_score, event_indicators)
        }

        schema, (confidence, indicators) = max(scores.items(), key=lambda x: x[1][0])

        # Extract subject if possible
        subject = self._extract_subject(query)

        # Fallback to definitional if no clear winner
        if confidence < 0.40:
            schema = 'definitional'
            confidence = 0.50
            indicators = ['fallback_definitional']

        # Cap confidence at 0.95 for deterministic classifier
        confidence = min(confidence, 0.95)

        return ClassificationResult(
            schema=schema,
            confidence=confidence,
            indicators=indicators,
            subject=subject
        )

    def _score_patterns(self, query: str, patterns: List[tuple]) -> tuple:
        """
        Score query against pattern list.

        Args:
            query: Query string
            patterns: List of (regex, weight, indicator_name) tuples

        Returns:
            (total_score, indicator_names)
        """
        query_lower = query.lower()
        score = 0.0
        indicators = []

        for pattern, weight, indicator in patterns:
            # Use original query for patterns with capital letters, lowercase for others
            search_string = query if re.search(r'[A-Z]', pattern) else query_lower
            if re.search(pattern, search_string):
                score += weight
                indicators.append(indicator)

        return score, indicators

    def _extract_question_word(self, query_lower: str) -> Optional[str]:
        """Extract question word (ki-vorto) from query."""
        question_words = ['kiu', 'kio', 'kiam', 'kie', 'kial', 'kiel', 'kiom', 'kies']
        for word in question_words:
            if re.search(r'\b' + word + r'\b', query_lower):
                return word
        return None

    def _extract_subject(self, query: str) -> Optional[str]:
        """
        Extract subject from query (person name, concept, event).

        Simple heuristic: look for capitalized words or words after "pri".
        """
        # Look for "pri [SUBJECT]"
        match = re.search(r'\bpri\s+(?:la\s+)?([A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+(?:\s+[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+)?)', query)
        if match:
            return match.group(1)

        # Look for capitalized word NOT at start of sentence (likely proper name)
        # Skip question words at start
        query_words = query.split()
        for i, word in enumerate(query_words):
            if i == 0:
                continue  # Skip first word
            match = re.match(r'^([A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+)$', word)
            if match:
                return match.group(1)

        # Look for common subjects after "estas"
        match = re.search(r'\bestas\s+([a-zĉĝĥĵŝŭ]+(?:\s+[a-zĉĝĥĵŝŭ]+)?)\b', query.lower())
        if match:
            return match.group(1)

        return None

    def classify_batch(self, queries: List[str]) -> List[ClassificationResult]:
        """Classify multiple queries."""
        return [self.classify(query) for query in queries]

    def explain(self, query: str) -> str:
        """
        Explain classification decision (for debugging).

        Args:
            query: Query to explain

        Returns:
            Human-readable explanation string
        """
        result = self.classify(query)

        explanation = f"Query: {query}\n"
        explanation += f"Schema: {result.schema}\n"
        explanation += f"Confidence: {result.confidence:.2f}\n"
        explanation += f"Subject: {result.subject or 'None detected'}\n"
        explanation += f"Indicators:\n"
        for indicator in result.indicators:
            explanation += f"  - {indicator}\n"

        return explanation
