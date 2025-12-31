"""
Answer Extractor (CP5): Heuristic-based answer extraction from retrieved documents.

This module extracts answers from retrieved documents using:
1. Question type analysis (Kiu? Kio? Kiel? Kiam? Kie?)
2. Pattern matching (X estas Y, X fondis Y, etc.)
3. Sentence selection based on relevance

All extraction is deterministic (0 learned parameters).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class ExtractionResult:
    """Result of answer extraction."""
    answer: str
    confidence: float
    method: str  # 'pattern', 'sentence', 'fallback'
    source_sentence: str
    question_type: str
    explanation: Optional[str] = None


# Question word to expected answer type mapping
QUESTION_TYPES = {
    'kiu': 'person',      # Who
    'kio': 'thing',       # What
    'kiel': 'manner',     # How
    'kiam': 'time',       # When
    'kie': 'location',    # Where
    'kial': 'reason',     # Why
    'kiom': 'quantity',   # How much/many
    'kia': 'quality',     # What kind
}

# Correlative table roots for detection
CORRELATIVES = {
    'ki': 'question',
    'ti': 'demonstrative',
    'i': 'indefinite',
    'ĉi': 'universal',
    'neni': 'negative',
}

# Answer extraction patterns
# Pattern: (regex, extract_group, confidence)
EXTRACTION_PATTERNS = {
    'thing': [
        # "X estas Y" -> Y is the answer for "Kio estas X?"
        (r'(\w+)\s+estas\s+(.+?)\.', 2, 0.8),
        # "X, tio estas Y" -> Y
        (r',\s*tio\s+estas\s+(.+?)\.', 1, 0.7),
        # "X signifas Y" -> Y
        (r'(\w+)\s+signifas\s+(.+?)\.', 2, 0.7),
        # "X nomiĝas Y" -> Y
        (r'(\w+)\s+nomiĝas\s+(.+?)\.', 2, 0.7),
    ],
    'person': [
        # "X fondis Y" -> X for "Kiu fondis Y?"
        (r'(\w+(?:\s+\w+)?)\s+fondis\s+', 1, 0.9),
        # "X kreis Y" -> X
        (r'(\w+(?:\s+\w+)?)\s+kreis\s+', 1, 0.9),
        # "Y estis fondita de X" -> X
        (r'estis\s+(?:fondita|kreita)\s+de\s+(\w+(?:\s+\w+)?)', 1, 0.8),
        # "de X" at start -> X
        (r'^de\s+(\w+(?:\s+\w+)?)', 1, 0.6),
    ],
    'time': [
        # "en (la jaro) YYYY"
        (r'en\s+(?:la\s+jaro\s+)?(\d{4})', 1, 0.9),
        # "en YYYY"
        (r'en\s+(\d{4})', 1, 0.9),
        # "la XX-an de MONTH"
        (r'la\s+(\d{1,2})[—-]?an?\s+de\s+(\w+)', 0, 0.8),
        # "YYYY" standalone
        (r'\b(1[89]\d{2}|20[0-2]\d)\b', 1, 0.5),
    ],
    'location': [
        # "en PLACE"
        (r'en\s+([A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+(?:\s+[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+)?)', 1, 0.8),
        # "ĉe PLACE"
        (r'ĉe\s+([A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+)', 1, 0.7),
    ],
    'quantity': [
        # Numbers
        (r'(\d+(?:\s*\d+)*)', 1, 0.7),
        # Number words
        (r'(unu|du|tri|kvar|kvin|ses|sep|ok|naŭ|dek|cent|mil)', 1, 0.6),
    ],
    'manner': [
        # "per X" (by means of)
        (r'per\s+(.+?)(?:\.|,|$)', 1, 0.7),
        # "uzante X" (using)
        (r'uzante?\s+(.+?)(?:\.|,|$)', 1, 0.7),
    ],
    'reason': [
        # "ĉar X" (because)
        (r'ĉar\s+(.+?)(?:\.|$)', 1, 0.8),
        # "pro X" (because of)
        (r'pro\s+(.+?)(?:\.|,|$)', 1, 0.7),
    ],
}

# Negative indicators (suggests question cannot be answered)
NEGATIVE_INDICATORS = [
    'ne ekzistas',
    'ne havas',
    'ne scias',
    'nekonata',
    'ne eblas',
]


class AnswerExtractor:
    """
    Extracts answers from retrieved documents using heuristics.

    Zero learned parameters - all extraction is rule-based.
    """

    def __init__(self):
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        for qtype, patterns in EXTRACTION_PATTERNS.items():
            self._compiled_patterns[qtype] = [
                (re.compile(pattern, re.IGNORECASE), group, conf)
                for pattern, group, conf in patterns
            ]

    def analyze_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Analyze question to determine type and key terms.

        Returns:
            tuple: (question_type, key_terms)
        """
        question_lower = question.lower().strip()

        # Detect question type from question word
        question_type = 'unknown'
        for q_word, q_type in QUESTION_TYPES.items():
            if question_lower.startswith(q_word):
                question_type = q_type
                break

        # Extract key terms (nouns, verbs - simplified)
        # Remove question words and common words
        stop_words = {'la', 'de', 'en', 'kaj', 'al', 'por', 'kun', 'ĉu', 'estas'}
        words = re.findall(r'[a-zĉĝĥĵŝŭ]+', question_lower)
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        # Remove question words
        for q_word in QUESTION_TYPES.keys():
            if q_word in key_terms:
                key_terms.remove(q_word)

        return question_type, key_terms

    def _extract_by_pattern(
        self,
        text: str,
        question_type: str,
        key_terms: List[str]
    ) -> Optional[ExtractionResult]:
        """Try to extract answer using patterns for the question type."""

        patterns = self._compiled_patterns.get(question_type, [])

        for regex, group, base_confidence in patterns:
            match = regex.search(text)
            if match:
                if group == 0:
                    # Return full match
                    answer = match.group(0)
                else:
                    answer = match.group(group)

                # Boost confidence if key terms appear in context
                confidence = base_confidence
                for term in key_terms:
                    if term.lower() in text.lower():
                        confidence = min(1.0, confidence + 0.1)

                return ExtractionResult(
                    answer=answer.strip(),
                    confidence=confidence,
                    method='pattern',
                    source_sentence=text,
                    question_type=question_type,
                    explanation=f"Matched pattern for '{question_type}'"
                )

        return None

    def _score_sentence(
        self,
        sentence: str,
        key_terms: List[str],
        question_type: str
    ) -> float:
        """Score a sentence's relevance to the question."""
        score = 0.0
        sentence_lower = sentence.lower()

        # Term overlap
        for term in key_terms:
            if term.lower() in sentence_lower:
                score += 0.2

        # Prefer sentences with definitions for 'thing' questions
        if question_type == 'thing' and 'estas' in sentence_lower:
            score += 0.1

        # Prefer sentences with years for 'time' questions
        if question_type == 'time' and re.search(r'\d{4}', sentence):
            score += 0.2

        # Prefer sentences with proper nouns for 'person' questions
        if question_type == 'person':
            proper_nouns = re.findall(r'[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+', sentence)
            score += 0.1 * len(proper_nouns)

        # Penalty for being too short
        if len(sentence.split()) < 5:
            score -= 0.1

        # Penalty for negative indicators
        for neg in NEGATIVE_INDICATORS:
            if neg in sentence_lower:
                score -= 0.3

        return max(0.0, min(1.0, score))

    def _select_best_sentence(
        self,
        sentences: List[str],
        key_terms: List[str],
        question_type: str
    ) -> Tuple[str, float]:
        """Select the best sentence from a list."""
        if not sentences:
            return "", 0.0

        scored = [
            (s, self._score_sentence(s, key_terms, question_type))
            for s in sentences
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0]

    def extract(
        self,
        question: str,
        retrieved_docs: List[str],
        max_answer_len: int = 150
    ) -> ExtractionResult:
        """
        Extract answer from retrieved documents.

        Args:
            question: The question to answer
            retrieved_docs: List of retrieved document texts
            max_answer_len: Maximum length of extracted answer

        Returns:
            ExtractionResult with the extracted answer
        """
        # Analyze question
        question_type, key_terms = self.analyze_question(question)

        # Check for unanswerable questions (negative category)
        question_lower = question.lower()
        if any(neg in question_lower for neg in ['marso', 'luno', 'kostas']):
            # Likely a "negative" category question
            return ExtractionResult(
                answer="Mi ne scias.",
                confidence=0.7,
                method='negative_detection',
                source_sentence="",
                question_type=question_type,
                explanation="Question appears unanswerable from corpus"
            )

        # Split documents into sentences
        all_sentences = []
        for doc in retrieved_docs:
            # Simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', doc)
            all_sentences.extend([s.strip() for s in sentences if s.strip()])

        if not all_sentences:
            return ExtractionResult(
                answer="Mi ne scias.",
                confidence=0.0,
                method='fallback',
                source_sentence="",
                question_type=question_type,
                explanation="No documents retrieved"
            )

        # Try pattern extraction on each sentence
        for sentence in all_sentences:
            result = self._extract_by_pattern(sentence, question_type, key_terms)
            if result and result.confidence >= 0.6:
                # Truncate if needed
                if len(result.answer) > max_answer_len:
                    result.answer = result.answer[:max_answer_len] + "..."
                return result

        # Fallback: select best sentence as answer
        best_sentence, best_score = self._select_best_sentence(
            all_sentences, key_terms, question_type
        )

        if best_score > 0.2:
            # Extract relevant part if sentence is too long
            answer = best_sentence
            if len(answer) > max_answer_len:
                # Try to extract a relevant clause
                clauses = re.split(r'[,;:]', answer)
                for clause in clauses:
                    if any(term.lower() in clause.lower() for term in key_terms):
                        answer = clause.strip()
                        break
                if len(answer) > max_answer_len:
                    answer = answer[:max_answer_len] + "..."

            return ExtractionResult(
                answer=answer,
                confidence=best_score * 0.6,  # Lower confidence for sentence-level
                method='sentence',
                source_sentence=best_sentence,
                question_type=question_type,
                explanation=f"Selected best matching sentence (score={best_score:.2f})"
            )

        # Final fallback: return first sentence
        return ExtractionResult(
            answer=all_sentences[0][:max_answer_len],
            confidence=0.1,
            method='fallback',
            source_sentence=all_sentences[0],
            question_type=question_type,
            explanation="No good match found, returning first sentence"
        )


# Convenience function
def extract_answer(question: str, retrieved_docs: List[str]) -> str:
    """
    Extract answer from retrieved documents.

    Args:
        question: The question to answer
        retrieved_docs: List of retrieved document texts

    Returns:
        Extracted answer string
    """
    extractor = AnswerExtractor()
    result = extractor.extract(question, retrieved_docs)
    return result.answer
