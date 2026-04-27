#!/usr/bin/env python3
"""
Fact Importance Scorer - Rank Facts by Relevance to Query

Scores fact importance using question-aware heuristics. Deterministic scoring
with explainable breakdown.

Design Philosophy:
- Mostly deterministic (weighted heuristics)
- Explainable (score breakdown for each component)
- Question-aware (different scoring for WHAT/WHO/WHERE/WHEN)
- Entity-centric (query entity gets priority)

Scoring Components (weights):
- Question Relevance: 0.4 (most important!)
- Definitional Priority: 0.3
- Entity Centrality: 0.2
- Semantic Completeness: 0.1
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum

from klareco.rag.unified_extractor import Fact, RelationType


class QuestionType(Enum):
    """Question types for question-aware scoring."""
    WHAT = "kio"
    WHO = "kiu"
    WHERE = "kie"
    WHEN = "kiam"
    HOW = "kiel"
    WHY = "kial"
    OTHER = "alia"


@dataclass
class ScoreBreakdown:
    """Explainable score breakdown."""
    question_relevance: float
    definitional_priority: float
    entity_centrality: float
    semantic_completeness: float
    embedding_similarity: float
    final_score: float

    def __str__(self):
        return (f"Score={self.final_score:.2f} "
                f"[Q:{self.question_relevance:.2f}, "
                f"D:{self.definitional_priority:.2f}, "
                f"E:{self.entity_centrality:.2f}, "
                f"C:{self.semantic_completeness:.2f}, "
                f"Emb:{self.embedding_similarity:.2f}]")


class FactImportanceScorer:
    """Score fact importance with explainable breakdown."""

    # Scoring weights (sum to 1.0)
    # Phase 2: Added embedding_similarity (0.1), reduced other weights proportionally
    WEIGHTS = {
        'question_relevance': 0.35,  # Was 0.4
        'definitional': 0.30,         # Unchanged (most stable)
        'centrality': 0.20,           # Unchanged
        'completeness': 0.05,         # Was 0.1
        'embedding_similarity': 0.10  # NEW: Phase 2
    }

    # Source quality weights (Issue #683 - Quick Win +10% precision)
    SOURCE_WEIGHTS = {
        'wikipedia': 1.0,      # Highest quality
        'fundamento': 0.9,     # Official Esperanto foundation
        'gutenberg': 0.7,      # Literary texts, variable quality
        'revo': 0.85,          # Dictionary definitions
        'database': 0.5,       # Generic corpus
        'unknown': 0.3         # Fallback for unspecified sources
    }

    def __init__(self, use_embeddings: bool = True,
                 embedding_path: Optional[str] = None):
        """
        Initialize importance scorer.

        Args:
            use_embeddings: Whether to use embedding similarity (Phase 2)
            embedding_path: Path to root embeddings checkpoint (optional)
        """
        self.use_embeddings = use_embeddings
        self.embeddings = None

        if use_embeddings:
            # Import and load embeddings lazily
            try:
                from klareco.rag.ast_semantic_ranker import load_embeddings
                from pathlib import Path

                if embedding_path:
                    self.embeddings = load_embeddings(Path(embedding_path))
                else:
                    # Use default path
                    default_path = Path('models/root_embeddings/best_model.pt')
                    if default_path.exists():
                        self.embeddings = load_embeddings(default_path)
            except Exception as e:
                # Embeddings optional - fall back to deterministic scoring
                self.embeddings = None

    def score(self, fact: Fact, question_type: QuestionType,
              query_entity: Optional[str] = None,
              query_roots: Optional[List[str]] = None,
              source_metadata: Optional[Dict] = None) -> ScoreBreakdown:
        """
        Score fact importance with explainable breakdown.

        Args:
            fact: Fact to score
            question_type: Type of question (WHAT, WHO, etc.)
            query_entity: Entity being queried about
            query_roots: Query roots for embedding similarity (Phase 2)
            source_metadata: Source document metadata (position, source, etc.)

        Returns:
            ScoreBreakdown with component scores and final score
        """
        # Compute component scores
        q_score = self._score_question_relevance(fact, question_type, query_entity)
        d_score = self._score_fact_quality(fact, source_metadata or {}, question_type)
        e_score = self._score_entity_centrality(fact, query_entity)
        c_score = self._score_completeness(fact)
        emb_score = self._score_embedding_similarity(fact, query_roots or [])

        # Weighted combination
        final = (
            q_score * self.WEIGHTS['question_relevance'] +
            d_score * self.WEIGHTS['definitional'] +
            e_score * self.WEIGHTS['centrality'] +
            c_score * self.WEIGHTS['completeness'] +
            emb_score * self.WEIGHTS['embedding_similarity']
        )

        return ScoreBreakdown(
            question_relevance=q_score,
            definitional_priority=d_score,
            entity_centrality=e_score,
            semantic_completeness=c_score,
            embedding_similarity=emb_score,
            final_score=final
        )

    def _score_question_relevance(self, fact: Fact, question_type: QuestionType,
                                   query_entity: Optional[str]) -> float:
        """
        Score how well fact answers the question type.

        WHAT questions → prioritize IS-A facts about query entity
        WHO questions → prioritize agent/person facts
        WHERE questions → prioritize LOCATED-AT facts
        WHEN questions → prioritize facts with temporal modifiers

        Improvements (2026-03-29):
        - Proper noun-aware matching (capitalized words = proper nouns)
        - Exact entity matching to avoid "fundament" matching "fundamentoj"
        - Stronger IS-A prioritization for WHAT questions

        Phase 1 Ranking Improvements (2026-03-29):
        - Boost IS-A + WHAT combination (+0.2) to fix "Kio estas hundo?" ranking #22 → #1
        - Boost agent + WHO combination (+0.15) to fix "Kiu fondis?" ranking #16 → top 3
        - Penalize generic facts (0.1) for better discrimination
        """
        score = 0.0

        if question_type == QuestionType.WHAT:
            # "What is X?" → IS-A facts about X are perfect
            if fact.relation == RelationType.IS_A:
                if query_entity and fact.entity:
                    # Check for exact match (proper noun aware)
                    if self._entity_matches(query_entity, fact, exact=True):
                        score = 1.0  # Perfect match!
                    elif self._entity_matches(query_entity, fact, exact=False):
                        score = 0.8  # Related match (substring)
                    else:
                        score = 0.5  # IS-A fact, but not about query entity
                else:
                    score = 0.5  # IS-A fact, but no entity info

            # Other facts about query entity - significantly lower than IS-A
            elif query_entity and fact.entity:
                if self._entity_matches(query_entity, fact, exact=True):
                    score = 0.5  # Relevant fact, but not definitional (was 0.7)
                elif self._entity_matches(query_entity, fact, exact=False):
                    score = 0.3  # Related fact (was 0.4)
                else:
                    score = 0.1  # Not about query entity

            # Related facts (mentions query entity)
            elif query_entity and query_entity.lower() in str(fact).lower():
                score = 0.2

            else:
                score = 0.1  # Generic fact

        elif question_type == QuestionType.WHO:
            # "Who created X?" → prioritize CREATED-BY, FOUNDED, etc.
            if fact.relation in [RelationType.CREATED_BY, RelationType.FOUNDED]:
                if query_entity and fact.entity:
                    if self._entity_matches(query_entity, fact, exact=True):
                        score = 1.0  # Perfect match
                    elif self._entity_matches(query_entity, fact, exact=False):
                        score = 0.8  # Related match
                    else:
                        score = 0.6
                else:
                    score = 0.6

            # Facts with agent argument
            elif 'aganto' in fact.arguments:
                score = 0.8

            else:
                score = 0.1  # Generic fact (was 0.2)

        elif question_type == QuestionType.WHERE:
            # "Where is X?" → prioritize LOCATED-AT, BORN
            if fact.relation in [RelationType.LOCATED_AT, RelationType.BORN]:
                if query_entity and fact.entity:
                    if self._entity_matches(query_entity, fact, exact=True):
                        score = 1.0
                    elif self._entity_matches(query_entity, fact, exact=False):
                        score = 0.8
                    else:
                        score = 0.7
                else:
                    score = 0.7

            # Has location modifier
            elif 'loko' in fact.modifiers or 'loko' in fact.arguments:
                score = 0.8

            else:
                score = 0.1  # Generic fact (was 0.2)

        elif question_type == QuestionType.WHEN:
            # "When was X created?" → prioritize facts with time modifiers
            if 'tempo' in fact.modifiers:
                if query_entity and fact.entity:
                    if self._entity_matches(query_entity, fact, exact=True):
                        score = 1.0
                    elif self._entity_matches(query_entity, fact, exact=False):
                        score = 0.8
                    else:
                        score = 0.7
                else:
                    score = 0.8

            # CREATED-BY, PUBLISHED, BORN often have time info
            elif fact.relation in [RelationType.CREATED_BY, RelationType.PUBLISHED,
                                   RelationType.BORN]:
                score = 0.6

            else:
                score = 0.1  # Generic fact (was 0.2)

        else:
            # Generic scoring for other question types
            if query_entity and fact.entity:
                if self._entity_matches(query_entity, fact, exact=True):
                    score = 0.7
                elif self._entity_matches(query_entity, fact, exact=False):
                    score = 0.4
                else:
                    score = 0.3
            else:
                score = 0.3

        # Phase 1 Ranking Improvements: Apply targeted boosts
        # These boosts address ranking failures where correct answers are retrieved but ranked too low

        # Boost 1: IS-A + WHAT combination (fix "Kio estas hundo?" ranking #22 → #1)
        # For WHAT questions, we want IS-A facts where query entity is either:
        # 1. The entity (subject): "hund IS-A besto" → answers "What is a dog?"
        # 2. The type (object): "mi IS-A hund" → less relevant but still about dogs
        if question_type == QuestionType.WHAT and fact.relation == RelationType.IS_A:
            if query_entity:
                # Check if query entity matches fact entity (primary case)
                if self._entity_matches(query_entity, fact, exact=True):
                    score = min(1.0, score + 0.2)  # Definitional boost
                # Also check if query entity is mentioned in type argument (secondary case)
                elif 'tipo' in fact.arguments:
                    type_arg = str(fact.arguments['tipo']).lower()
                    query_lower = query_entity.lower()
                    if query_lower in type_arg or type_arg in query_lower:
                        score = min(1.0, score + 0.1)  # Smaller boost for reverse direction

        # Boost 2: Agent + WHO combination (fix "Kiu fondis Esperanton?" ranking #16 → top 3)
        if question_type == QuestionType.WHO:
            if 'aganto' in fact.arguments:
                # Check if the fact is about the query entity
                if query_entity and self._entity_matches(query_entity, fact, exact=True):
                    score = min(1.0, score + 0.15)  # Exact entity match boost

        return score

    def _entity_matches(self, query_entity: str, fact, exact: bool = True) -> bool:
        """
        Check if query entity matches fact entity with proper noun awareness.

        Args:
            query_entity: Entity from query (e.g., "fundament")
            fact: Fact object with entity and proper noun annotations
            exact: If True, require exact root match. If False, allow substring.

        Returns:
            True if entities match according to criteria

        Examples:
            - Query "fundament", Fact(entity="fundament", is_proper=True, cap="Fundamento")
              → exact=True returns True (proper noun root match)
            - Query "fundament", Fact(entity="fundamentoj", is_proper=False)
              → exact=True returns False (different root)
            - Query "fundament", Fact(entity="fundamentoj", is_proper=False)
              → exact=False returns True (substring match)
        """
        query_lower = query_entity.lower()
        fact_entity = fact.entity if hasattr(fact, 'entity') else str(fact)
        fact_lower = fact_entity.lower()

        # Use AST proper noun annotation if available
        is_proper_noun = getattr(fact, 'entity_is_proper_noun', False)
        cap_form = getattr(fact, 'entity_capitalized_form', None)

        # Exact mode: check for proper noun or exact root match
        if exact:
            # Use explicit proper noun annotation from AST
            if is_proper_noun:
                # Proper noun: match if query is prefix + Esperanto ending
                # "fundament" matches "Fundamento" but not "fundamentoj"
                if cap_form:
                    cap_lower = cap_form.lower()
                    return (cap_lower.startswith(query_lower) and
                            (len(cap_lower) == len(query_lower) or
                             cap_lower[len(query_lower)] in 'oaej'))
                else:
                    # Fallback to entity field
                    return (fact_lower.startswith(query_lower) and
                            (len(fact_lower) == len(query_lower) or
                             fact_lower[len(query_lower)] in 'oaej'))
            else:
                # Common noun: require exact root match (strip Esperanto endings)
                # "hund" matches "hundo", "hundoj", "hundon" but not "hundego"
                fact_root = self._strip_esperanto_endings(fact_lower)
                return query_lower == fact_root

        # Substring mode: allow any substring match
        else:
            return query_lower in fact_lower

    def _strip_esperanto_endings(self, word: str) -> str:
        """
        Strip Esperanto grammatical endings to get root.

        Args:
            word: Esperanto word (lowercase)

        Returns:
            Root form

        Examples:
            - "hundo" → "hund"
            - "hundoj" → "hund"
            - "hundon" → "hund"
            - "bela" → "bel"
            - "fundamentoj" → "fundament"
        """
        # Strip case endings (-n, -jn)
        if word.endswith('jn'):
            word = word[:-2]
        elif word.endswith('n'):
            word = word[:-1]

        # Strip plural (-j)
        if word.endswith('j'):
            word = word[:-1]

        # Strip part-of-speech endings (-o, -a, -e, -i, -as, -is, -os, -us, -u)
        if word.endswith('o') or word.endswith('a') or word.endswith('e'):
            word = word[:-1]
        elif word.endswith('as') or word.endswith('is') or word.endswith('os') or word.endswith('us'):
            word = word[:-2]
        elif word.endswith('i') or word.endswith('u'):
            word = word[:-1]

        return word

    def _score_fact_quality(self, fact: Fact, source_metadata: Dict, question_type: QuestionType) -> float:
        """
        Score fact quality based on question type.

        RENAMED from _score_definitional() - now question-type-aware.

        Phase 2 Fix: Remove WHAT-bias. Previously only IS-A got 0.9, causing
        44% penalty on WHO/WHERE/WHEN facts. Now each question type defines
        what constitutes a "high quality" fact.

        Phase 1 improvements:
        - Sentence complexity (simpler = more direct)
        - Clause depth (main clause = more central)
        - Context awareness (anaphora, continuation, etymology)

        Issue #683: Apply source quality weighting for +10% precision.
        """
        score = 0.0

        # 1. QUESTION-TYPE-AWARE BASE SCORE
        if question_type == QuestionType.WHAT:
            if fact.relation == RelationType.IS_A:
                score = 0.9
            else:
                score = 0.5

        elif question_type == QuestionType.WHO:
            # WHO: agent/creator facts are high quality
            if fact.relation in [RelationType.CREATED_BY, RelationType.FOUNDED]:
                score = 0.9  # Was 0.5!
            elif 'aganto' in fact.arguments:
                score = 0.9  # Was 0.5!
            else:
                score = 0.5

        elif question_type == QuestionType.WHERE:
            # WHERE: location facts are high quality
            if fact.relation in [RelationType.LOCATED_AT, RelationType.BORN]:
                score = 0.9  # Was 0.5!
            elif 'loko' in fact.modifiers or 'loko' in fact.arguments:
                score = 0.9  # Was 0.5!
            else:
                score = 0.5

        elif question_type == QuestionType.WHEN:
            # WHEN: temporal facts are high quality
            if 'tempo' in fact.modifiers:
                score = 0.9  # Was 0.5!
            elif fact.relation in [RelationType.CREATED_BY, RelationType.PUBLISHED,
                                 RelationType.BORN, RelationType.DIED]:
                score = 0.7
            else:
                score = 0.5

        elif question_type == QuestionType.WHY:
            # WHY: causal facts are high quality
            if 'purpose' in fact.modifiers or 'reason' in fact.modifiers:
                score = 0.9
            elif 'maniero' in fact.modifiers:
                score = 0.7
            else:
                score = 0.5
        else:
            # OTHER/HOW
            if fact.relation == RelationType.IS_A:
                score = 0.7
            else:
                score = 0.5

        # 2. Sentence complexity (simpler = more direct information)
        if hasattr(fact, 'source_text'):
            word_count = len(fact.source_text.split())
            if word_count <= 10:
                complexity_mult = 1.0   # Short & direct
            elif word_count <= 20:
                complexity_mult = 0.9   # Medium
            else:
                complexity_mult = 0.7   # Complex (often narrative)
            score *= complexity_mult

        # 3. Clause depth (main clause = more central)
        # Get from fact AST if available
        clause_depth = self._get_clause_depth(fact)
        if clause_depth == 0:
            depth_mult = 1.0    # Main clause
        elif clause_depth == 1:
            depth_mult = 0.7    # Relative clause
        else:
            depth_mult = 0.4    # Deeply nested
        score *= depth_mult

        # 4. Entity role (subject = primary)
        entity_role = self._get_entity_role(fact)
        if entity_role == 'SUBJECT':
            role_mult = 1.0     # Primary role
        elif entity_role == 'OBJECT':
            role_mult = 0.8     # Secondary role
        elif entity_role == 'MODIFIER':
            role_mult = 0.5     # Tertiary role
        else:
            role_mult = 0.7     # Unknown/mentioned

        score *= role_mult

        # 5. First sentence in document (often definitional)
        sentence_pos = source_metadata.get('sentence_position', -1)
        if sentence_pos == 0:
            score = min(1.0, score + 0.2)
        elif sentence_pos == 1:
            score = min(1.0, score + 0.1)

        # 6. Quick Win #683: Apply source quality weighting
        source = self._detect_source(source_metadata)
        source_weight = self.SOURCE_WEIGHTS.get(source, self.SOURCE_WEIGHTS['unknown'])
        score *= source_weight

        # 7. CONTEXT BOOST (Phase 2 - essentially free, question-type-aware!)
        context_boost = self._calculate_context_boost(fact, source_metadata, question_type)
        score = min(1.0, score + context_boost)

        return score

    def _detect_source(self, source_metadata: Dict) -> str:
        """
        Detect source from metadata (Issue #683).

        Returns: Source name (wikipedia, fundamento, gutenberg, revo, database, unknown)
        """
        # Check explicit source field
        if 'source' in source_metadata:
            source_str = source_metadata['source'].lower()
            for known_source in self.SOURCE_WEIGHTS.keys():
                if known_source in source_str:
                    return known_source

        # Check metadata dict for source indicators
        metadata_str = str(source_metadata).lower()
        if 'wikipedia' in metadata_str:
            return 'wikipedia'
        elif 'fundamento' in metadata_str:
            return 'fundamento'
        elif 'gutenberg' in metadata_str:
            return 'gutenberg'
        elif 'revo' in metadata_str:
            return 'revo'
        elif source_metadata:
            return 'database'
        else:
            return 'unknown'

    def _score_entity_centrality(self, fact: Fact, query_entity: Optional[str]) -> float:
        """
        Score how central the query entity is to this fact.

        Entity as subject (highest centrality) > object > modifier
        """
        if not query_entity:
            return 0.5  # Neutral if no query entity

        query_lower = query_entity.lower()

        # Entity is the main subject of fact
        if fact.entity and fact.entity.lower() == query_lower:
            return 1.0

        # Entity appears in arguments (object position)
        for arg_val in fact.arguments.values():
            if isinstance(arg_val, str) and query_lower in arg_val.lower():
                return 0.7

        # Entity appears in modifiers
        for mod_val in fact.modifiers.values():
            if isinstance(mod_val, str) and query_lower in mod_val.lower():
                return 0.4

        # Entity doesn't appear
        return 0.0

    def _score_completeness(self, fact: Fact) -> float:
        """
        Score semantic completeness of the fact.

        Facts with more information (arguments + modifiers) are more complete.
        """
        score = 0.3  # Base score for having entity + relation

        # Has required arguments
        if fact.arguments:
            score += 0.4

        # Has enriching modifiers
        if fact.modifiers:
            score += 0.3

        return min(score, 1.0)  # Cap at 1.0

    def _score_embedding_similarity(self, fact: Fact, query_roots: List[str]) -> float:
        """
        Score semantic similarity using root embeddings (Phase 2).

        Computes cosine similarity between query roots and fact entity/argument roots.
        Provides learned signal beyond deterministic matching.

        Args:
            fact: Fact to score
            query_roots: Roots from query (e.g., ["kiu", "fond", "esperant"])

        Returns:
            Similarity score [0, 1]
        """
        if not self.use_embeddings or not self.embeddings or not query_roots:
            return 0.5  # Neutral score if embeddings unavailable

        try:
            import torch

            # Extract roots from fact
            fact_roots = []
            if fact.entity:
                fact_roots.append(fact.entity.lower())
            for arg_val in fact.arguments.values():
                if isinstance(arg_val, str):
                    fact_roots.append(arg_val.lower())

            if not fact_roots:
                return 0.5  # No roots to compare

            # Get embeddings for query and fact roots
            query_vecs = []
            for root in query_roots:
                if root.lower() in self.embeddings:
                    query_vecs.append(self.embeddings[root.lower()])

            fact_vecs = []
            for root in fact_roots:
                if root in self.embeddings:
                    fact_vecs.append(self.embeddings[root])

            if not query_vecs or not fact_vecs:
                return 0.5  # Missing embeddings

            # Compute average embeddings
            query_avg = torch.stack(query_vecs).mean(dim=0)
            fact_avg = torch.stack(fact_vecs).mean(dim=0)

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                query_avg.unsqueeze(0), fact_avg.unsqueeze(0)
            ).item()

            # Normalize from [-1, 1] to [0, 1]
            normalized = (cos_sim + 1.0) / 2.0

            return normalized

        except Exception as e:
            # Fallback if embedding computation fails
            return 0.5

    def _get_clause_depth(self, fact: Fact) -> int:
        """
        Get clause depth from fact source_ast.

        Returns:
            0 = main clause (most central)
            1 = relative clause
            2+ = deeply nested
        """
        # Check if fact has source_ast attribute
        if not hasattr(fact, 'source_ast') or not fact.source_ast:
            return 0  # Assume main clause if no AST

        # Simple heuristic: count nesting level
        # In practice, you'd traverse the AST to find clause depth
        # For now, use a simple approximation based on sentence structure

        source_text = getattr(fact, 'source_text', '')
        if not source_text:
            return 0

        # Count relative pronouns and subordinating conjunctions
        subordinators = ['kiu', 'kio', 'kie', 'kiam', 'kiom', 'kies',
                        'ke', 'ĉu', 'se', 'kvankam', 'dum', 'ĉar']

        depth = 0
        text_lower = source_text.lower()
        for sub in subordinators:
            if f' {sub} ' in f' {text_lower} ':
                depth += 1

        return min(depth, 2)  # Cap at 2

    def _get_entity_role(self, fact: Fact) -> str:
        """
        Determine entity's grammatical role in the fact.

        Returns:
            'SUBJECT', 'OBJECT', 'MODIFIER', or 'MENTIONED'
        """
        # Check if fact has source_ast with grammatical role info
        if not hasattr(fact, 'source_ast') or not fact.source_ast:
            return 'MENTIONED'  # Unknown role

        ast = fact.source_ast

        # Check if entity is in subject position
        subjekto = ast.get('subjekto', {})
        if subjekto:
            if subjekto.get('tipo') == 'vortgrupo':
                kerno = subjekto.get('kerno', {})
                if kerno.get('radiko') == fact.entity:
                    return 'SUBJECT'
            elif subjekto.get('radiko') == fact.entity:
                return 'SUBJECT'

        # Check if entity is in object position
        objekto = ast.get('objekto', {})
        if objekto:
            if objekto.get('tipo') == 'vortgrupo':
                kerno = objekto.get('kerno', {})
                if kerno.get('radiko') == fact.entity:
                    return 'OBJECT'
            elif objekto.get('radiko') == fact.entity:
                return 'OBJECT'

        # Check if entity is in modifier position
        aliaj = ast.get('aliaj', [])
        for alia in aliaj:
            if isinstance(alia, dict) and alia.get('radiko') == fact.entity:
                return 'MODIFIER'

        return 'MENTIONED'

    def _calculate_context_boost(self, fact: Fact, source_metadata: Dict, question_type: QuestionType) -> float:
        """
        Calculate context-based boost with question-type awareness (Phase 2).

        Context is fetched from Kuzu SEKVA_FRAZOTEKSTO relationships (essentially free!).

        Phase 2 Fix: Add question-specific context patterns for WHO/WHERE/WHEN.

        Returns:
            boost: 0.0 to 0.3
        """
        boost = 0.0

        prev_text = source_metadata.get('prev_text', '')
        next_text = source_metadata.get('next_text', '')

        if not prev_text and not next_text:
            return 0.0

        source_text = getattr(fact, 'source_text', '')
        if not source_text:
            return 0.0

        # UNIVERSAL BOOSTS (all question types)

        # 1. ANAPHORA RESOLUTION (+0.2)
        if prev_text and self._has_pronouns(source_text):
            if self._resolves_anaphora(source_text, prev_text, fact.entity):
                boost += 0.2

        # 2. TOPIC COHERENCE (+0.1)
        if fact.entity and (prev_text or next_text):
            neighbor_text = (prev_text + ' ' + next_text).lower()
            if fact.entity.lower() in neighbor_text:
                boost += 0.1

        # QUESTION-TYPE-SPECIFIC BOOSTS

        if question_type == QuestionType.WHAT:
            # Definitional continuation (+0.15)
            if next_text and self._is_definitional(source_text):
                if self._continues_definition(source_text, next_text):
                    boost += 0.15

            # Etymology/origin (+0.15)
            if next_text:
                etymology_markers = ['nomo venas', 'devenas de', 'nomita laŭ',
                                   'la nomo', 'venas el', 'nomiĝas laŭ']
                if any(marker in next_text.lower() for marker in etymology_markers):
                    boost += 0.15

        elif question_type == QuestionType.WHO:
            # Biographical continuation (+0.15)
            if next_text:
                bio_markers = ['li estis', 'ŝi estis', 'li naskiĝis', 'ŝi naskiĝis',
                             'li mortis', 'ŝi mortis', 'lia', 'ŝia']
                if any(marker in next_text.lower() for marker in bio_markers):
                    boost += 0.15

            # Role/occupation context (+0.15)
            if prev_text or next_text:
                role_markers = ['doktor', 'profesor', 'verkist', 'lingvist',
                              'kreint', 'fondint', 'aŭtor']
                neighbor = (prev_text + ' ' + next_text).lower()
                if any(marker in neighbor for marker in role_markers):
                    boost += 0.15

        elif question_type == QuestionType.WHERE:
            # Geographic context (+0.15)
            if next_text or prev_text:
                geo_markers = ['en ', 'ĉe ', 'apud ', 'proksim', 'urb', 'land',
                             'region', 'kontinent']
                neighbor = (prev_text + ' ' + next_text).lower()
                if any(marker in neighbor for marker in geo_markers):
                    boost += 0.15

            # Location relationship (+0.15)
            if next_text:
                loc_relations = ['troviĝas', 'situas', 'loĝas', 'estas en']
                if any(marker in next_text.lower() for marker in loc_relations):
                    boost += 0.15

        elif question_type == QuestionType.WHEN:
            # Temporal sequence (+0.15)
            if next_text or prev_text:
                time_markers = ['en ', 'dum ', 'antaŭ ', 'post ', 'jar', 'jarcent',
                              'epok', 'period']
                neighbor = (prev_text + ' ' + next_text).lower()
                if any(marker in neighbor for marker in time_markers):
                    boost += 0.15

            # Date/year context (+0.15)
            if next_text or prev_text:
                import re
                neighbor = prev_text + ' ' + next_text
                if re.search(r'\b\d{4}\b', neighbor):  # Year patterns
                    boost += 0.15

        return min(0.3, boost)  # Cap at 30% boost

    def _has_pronouns(self, text: str) -> bool:
        """Check if text contains Esperanto pronouns."""
        pronouns = ['li', 'ŝi', 'ĝi', 'ili', 'tio', 'tiu', 'si']
        text_lower = ' ' + text.lower() + ' '
        return any(f' {p} ' in text_lower for p in pronouns)

    def _resolves_anaphora(self, current: str, previous: str, entity: Optional[str]) -> bool:
        """Check if previous sentence provides pronoun referent."""
        if not entity:
            return False

        # Heuristic: previous mentions entity + current has pronoun
        return entity.lower() in previous.lower() and self._has_pronouns(current)

    def _is_definitional(self, text: str) -> bool:
        """Check if sentence is definitional (copula pattern)."""
        text_lower = text.lower()
        return ' estas ' in text_lower or ' estis ' in text_lower

    def _continues_definition(self, current: str, next_text: str) -> bool:
        """Check if next sentence continues the definition."""
        next_lower = next_text.lower()

        # Look for continuation markers + pronoun reference
        has_pronoun = self._has_pronouns(next_text)
        continuation_markers = ['ankaŭ', 'kaj', 'plu', 'krome', 'tio estas',
                               'ĝi havas', 'ili estas', 'ĝi konsistas']
        has_continuation = any(marker in next_lower for marker in continuation_markers)

        return has_pronoun and has_continuation


def classify_question_type(query: str) -> QuestionType:
    """
    Classify question type from query text.

    Simple rule-based classification using question words.
    """
    query_lower = query.lower()

    if query_lower.startswith('kio') or 'what' in query_lower:
        return QuestionType.WHAT
    elif query_lower.startswith('kiu') or 'who' in query_lower:
        return QuestionType.WHO
    elif query_lower.startswith('kie') or 'where' in query_lower:
        return QuestionType.WHERE
    elif query_lower.startswith('kiam') or 'when' in query_lower:
        return QuestionType.WHEN
    elif query_lower.startswith('kiel') or 'how' in query_lower:
        return QuestionType.HOW
    elif query_lower.startswith('kial') or 'why' in query_lower:
        return QuestionType.WHY
    else:
        return QuestionType.OTHER

# Backward-compatible alias
ImportanceScorer = FactImportanceScorer
