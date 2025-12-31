"""
Deterministic Reranker (CP4): AST-based document reranking for Q&A.

This module reranks retrieved documents using only deterministic AST features:
1. Question-Answer Type Match (Kiu→person, Kio→thing, etc.)
2. Grammar Agreement (number, tense)
3. Semantic Role Match (subject/object alignment)
4. Entity Overlap (matching roots weighted by rarity)

All reranking is deterministic (0 learned parameters).
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RerankResult:
    """Result of reranking a document."""
    text: str
    original_score: float
    rerank_score: float
    combined_score: float
    features: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


# Question word to expected answer type and semantic role
QUESTION_TYPE_MAP = {
    'kiu': {'type': 'person', 'expected_role': 'subjekto', 'weight': 1.0},
    'kio': {'type': 'thing', 'expected_role': 'subjekto', 'weight': 1.0},
    'kiel': {'type': 'manner', 'expected_role': 'aliaj', 'weight': 0.8},
    'kiam': {'type': 'time', 'expected_role': 'aliaj', 'weight': 0.9},
    'kie': {'type': 'location', 'expected_role': 'aliaj', 'weight': 0.9},
    'kial': {'type': 'reason', 'expected_role': 'aliaj', 'weight': 0.7},
    'kiom': {'type': 'quantity', 'expected_role': 'aliaj', 'weight': 0.8},
    'kia': {'type': 'quality', 'expected_role': 'priskriboj', 'weight': 0.7},
    'kiun': {'type': 'person', 'expected_role': 'objekto', 'weight': 1.0},
    'kion': {'type': 'thing', 'expected_role': 'objekto', 'weight': 1.0},
}

# Expected content patterns for each answer type
ANSWER_TYPE_PATTERNS = {
    'person': [
        (r'[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+(?:\s+[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+)?', 0.3),  # Proper name
        (r'\b(homo|persono|viro|virino|ulo|ano)\b', 0.2),  # Person words
    ],
    'thing': [
        (r'estas\s+\w+', 0.2),  # Definition pattern
        (r'\b(afero|objekto|ilo|aĵo)\b', 0.1),  # Thing words
    ],
    'time': [
        (r'\b(1[89]\d{2}|20[0-2]\d)\b', 0.4),  # Years
        (r'\b(en|dum|post|antaŭ)\s+(la\s+)?(jar|monat|tag)', 0.3),  # Time phrases
        (r'\b(januaro|februaro|marto|aprilo|majo|junio|julio|aŭgusto|septembro|oktobro|novembro|decembro)\b', 0.2),
    ],
    'location': [
        (r'\b(en|ĉe|sur|sub)\s+[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+', 0.3),  # Location prepositions
        (r'\b(urbo|lando|loko|regiono|mondo)\b', 0.1),  # Location words
    ],
    'manner': [
        (r'\b(per|uzante|kun|laŭ)\s+\w+', 0.2),  # Manner phrases
        (r'\w+e\b', 0.1),  # Adverbs (ending in -e)
    ],
    'reason': [
        (r'\b(ĉar|pro|tial|sekve)\b', 0.3),  # Reason words
        (r'\b(kaŭzo|kialo|motivo)\b', 0.2),
    ],
    'quantity': [
        (r'\b\d+\b', 0.3),  # Numbers
        (r'\b(unu|du|tri|kvar|kvin|ses|sep|ok|naŭ|dek|cent|mil)\b', 0.2),
        (r'\b(multe|malmulte|kelkaj|ĉiuj)\b', 0.1),
    ],
    'quality': [
        (r'\w+a\b', 0.1),  # Adjectives (ending in -a)
    ],
}

# Common Esperanto roots for IDF weighting (more common = lower weight)
COMMON_ROOTS = {
    'est': 0.1, 'hav': 0.2, 'far': 0.2, 'ir': 0.3, 'ven': 0.3,
    'vid': 0.3, 'dir': 0.3, 'sci': 0.4, 'vol': 0.4, 'pov': 0.4,
    'dez': 0.5, 'am': 0.5, 'don': 0.5, 'pren': 0.5, 'met': 0.5,
}


class DeterministicReranker:
    """
    Reranks documents using AST-based features.

    Zero learned parameters - all features are deterministic.
    """

    def __init__(
        self,
        type_match_weight: float = 0.3,
        grammar_weight: float = 0.2,
        role_weight: float = 0.2,
        entity_weight: float = 0.3,
    ):
        """
        Initialize reranker with feature weights.

        Args:
            type_match_weight: Weight for question-answer type matching
            grammar_weight: Weight for grammar agreement features
            role_weight: Weight for semantic role matching
            entity_weight: Weight for entity/root overlap
        """
        self.type_match_weight = type_match_weight
        self.grammar_weight = grammar_weight
        self.role_weight = role_weight
        self.entity_weight = entity_weight

        # Compile regex patterns
        self._compiled_patterns = {}
        for answer_type, patterns in ANSWER_TYPE_PATTERNS.items():
            self._compiled_patterns[answer_type] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze question to extract type and key information.

        Returns dict with:
            - question_type: Expected answer type (person, thing, time, etc.)
            - expected_role: Expected AST role (subjekto, objekto, aliaj)
            - key_roots: Key content roots from the question
            - number: singular/plural
            - tense: past/present/future
        """
        question_lower = question.lower().strip()

        # Detect question type
        question_type = 'unknown'
        expected_role = 'aliaj'
        type_weight = 0.5

        for q_word, info in QUESTION_TYPE_MAP.items():
            if question_lower.startswith(q_word):
                question_type = info['type']
                expected_role = info['expected_role']
                type_weight = info['weight']
                break

        # Extract key roots (simple approach: words > 3 chars, not function words)
        function_words = {'la', 'de', 'en', 'kaj', 'al', 'por', 'kun', 'ĉu', 'estas',
                          'kiu', 'kio', 'kiel', 'kiam', 'kie', 'kial', 'kiom', 'kia'}
        words = re.findall(r'[a-zĉĝĥĵŝŭ]+', question_lower)
        key_roots = []
        for word in words:
            if word not in function_words and len(word) > 3:
                # Strip common endings to get root
                root = self._extract_root(word)
                if root and len(root) > 2:
                    key_roots.append(root)

        # Detect number (plural markers)
        number = 'singular'
        if re.search(r'\b\w+oj\b', question_lower):
            number = 'plural'

        # Detect tense from question verbs
        tense = 'present'
        if re.search(r'\b\w+is\b', question_lower):
            tense = 'past'
        elif re.search(r'\b\w+os\b', question_lower):
            tense = 'future'

        return {
            'question_type': question_type,
            'expected_role': expected_role,
            'type_weight': type_weight,
            'key_roots': key_roots,
            'number': number,
            'tense': tense,
        }

    def _extract_root(self, word: str) -> str:
        """Extract root from Esperanto word by stripping endings."""
        for ending in ['ojn', 'ajn', 'oj', 'aj', 'on', 'an', 'on', 'en',
                       'as', 'is', 'os', 'us', 'o', 'a', 'e', 'i', 'u', 'n', 'j']:
            if word.endswith(ending) and len(word) > len(ending) + 2:
                return word[:-len(ending)]
        return word

    def _score_type_match(self, doc: str, question_analysis: Dict) -> float:
        """Score how well document matches expected answer type."""
        answer_type = question_analysis['question_type']
        if answer_type == 'unknown':
            return 0.0

        patterns = self._compiled_patterns.get(answer_type, [])
        score = 0.0

        for regex, weight in patterns:
            matches = regex.findall(doc)
            if matches:
                score += weight * min(len(matches), 3)  # Cap at 3 matches

        return min(1.0, score)

    def _score_grammar_agreement(self, doc: str, question_analysis: Dict) -> float:
        """Score grammar agreement between question and document."""
        score = 0.0

        # Number agreement
        doc_lower = doc.lower()
        if question_analysis['number'] == 'plural':
            if re.search(r'\b\w+oj\b', doc_lower):
                score += 0.5
        else:
            # Singular - penalize if doc is mostly plural
            plural_count = len(re.findall(r'\b\w+oj\b', doc_lower))
            singular_count = len(re.findall(r'\b\w+o\b', doc_lower))
            if singular_count >= plural_count:
                score += 0.3

        # Tense agreement (present questions can match any tense)
        q_tense = question_analysis['tense']
        if q_tense == 'past':
            if re.search(r'\b\w+is\b', doc_lower):
                score += 0.5
        elif q_tense == 'future':
            if re.search(r'\b\w+os\b', doc_lower):
                score += 0.5
        else:  # present
            score += 0.3  # Present can match anything

        return min(1.0, score)

    def _score_entity_overlap(self, doc: str, question_analysis: Dict) -> float:
        """Score entity/root overlap with IDF-like weighting."""
        key_roots = question_analysis['key_roots']
        if not key_roots:
            return 0.0

        doc_lower = doc.lower()
        doc_words = re.findall(r'[a-zĉĝĥĵŝŭ]+', doc_lower)
        doc_roots = set(self._extract_root(w) for w in doc_words)

        score = 0.0
        matched = 0

        for root in key_roots:
            if root in doc_roots or any(root in dr for dr in doc_roots):
                matched += 1
                # Weight by rarity (inverse common frequency)
                weight = 1.0 - COMMON_ROOTS.get(root, 0.0)
                score += weight

        if matched == 0:
            return 0.0

        # Normalize by number of key roots
        return min(1.0, score / len(key_roots))

    def _score_role_match(self, doc: str, question_analysis: Dict) -> float:
        """Score based on expected semantic role presence."""
        expected_role = question_analysis['expected_role']
        answer_type = question_analysis['question_type']

        score = 0.0
        doc_lower = doc.lower()

        # Heuristic checks for role presence
        if expected_role == 'subjekto':
            # Check for sentence-initial subject patterns
            if answer_type == 'person':
                if re.match(r'^[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+', doc):
                    score += 0.5
            if re.search(r'^(la\s+)?\w+o\s+(est|far|dir|vid)', doc_lower):
                score += 0.3

        elif expected_role == 'objekto':
            # Check for accusative patterns
            if re.search(r'\w+n\b', doc_lower):
                score += 0.4

        elif expected_role == 'aliaj':
            # Check for prepositional phrases
            if re.search(r'\b(en|sur|sub|ĉe|per|kun|por|al)\s+', doc_lower):
                score += 0.3

        return min(1.0, score)

    def score_document(
        self,
        doc: str,
        question_analysis: Dict,
        original_score: float = 0.0
    ) -> RerankResult:
        """
        Score a single document against the question.

        Args:
            doc: Document text
            question_analysis: Result from analyze_question()
            original_score: Original retrieval score (0-1)

        Returns:
            RerankResult with scores and explanation
        """
        # Compute feature scores
        type_score = self._score_type_match(doc, question_analysis)
        grammar_score = self._score_grammar_agreement(doc, question_analysis)
        entity_score = self._score_entity_overlap(doc, question_analysis)
        role_score = self._score_role_match(doc, question_analysis)

        # Weighted combination
        rerank_score = (
            self.type_match_weight * type_score +
            self.grammar_weight * grammar_score +
            self.entity_weight * entity_score +
            self.role_weight * role_score
        )

        # Combine with original score (60% rerank, 40% original)
        combined_score = 0.6 * rerank_score + 0.4 * original_score

        features = {
            'type_match': type_score,
            'grammar': grammar_score,
            'entity_overlap': entity_score,
            'role_match': role_score,
        }

        explanation = (
            f"type={type_score:.2f}, grammar={grammar_score:.2f}, "
            f"entity={entity_score:.2f}, role={role_score:.2f}"
        )

        return RerankResult(
            text=doc,
            original_score=original_score,
            rerank_score=rerank_score,
            combined_score=combined_score,
            features=features,
            explanation=explanation,
        )

    def rerank(
        self,
        question: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on question-document matching.

        Args:
            question: The question being asked
            documents: List of retrieved documents
            original_scores: Optional list of original retrieval scores
            top_k: Return only top-k results (None = all)

        Returns:
            List of RerankResult sorted by combined_score descending
        """
        if not documents:
            return []

        # Default scores if not provided
        if original_scores is None:
            original_scores = [1.0 - i/len(documents) for i in range(len(documents))]

        # Analyze question once
        question_analysis = self.analyze_question(question)

        # Score each document
        results = []
        for doc, orig_score in zip(documents, original_scores):
            result = self.score_document(doc, question_analysis, orig_score)
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]

        return results


# Convenience function
def rerank_documents(
    question: str,
    documents: List[str],
    original_scores: Optional[List[float]] = None,
    top_k: Optional[int] = None
) -> List[RerankResult]:
    """
    Rerank documents for a question using deterministic features.

    Args:
        question: The question being asked
        documents: List of retrieved documents
        original_scores: Optional list of original retrieval scores
        top_k: Return only top-k results

    Returns:
        List of RerankResult sorted by score
    """
    reranker = DeterministicReranker()
    return reranker.rerank(question, documents, original_scores, top_k)
