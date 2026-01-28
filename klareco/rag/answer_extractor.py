#!/usr/bin/env python3
"""
AST-Based Answer Extraction (Deterministic)

Extracts precise answers from retrieved documents by matching AST patterns.

This is the PRIMARY answer extraction method for Klareco's RAG system.
It uses deterministic pattern matching on Abstract Syntax Trees to extract
grammatically and semantically correct answers.

Architecture:
1. Parse question AST to determine question type (WHO, WHAT, WHERE, WHEN, HOW_MANY)
2. Parse document AST
3. Match patterns based on question type
4. Extract answer as complete vortgrupo (not just root)
5. Return structured answer with confidence and explanation

Question Type Detection:
- WHO (kiu): Extract person/agent (subject or object with animate semantics)
- WHAT (kio): Extract thing/concept (subject, object, or predicate)
- WHERE (kie): Extract location (aliaj with location semantics)
- WHEN (kiam): Extract time (aliaj with temporal semantics)
- HOW_MANY (kiom): Extract quantity (numeric modifier)
- WHICH (kiu + noun): Extract specific instance from category
- WHY (kial): Extract reason/cause (aliaj with causal semantics)
- HOW (kiel): Extract manner (aliaj with manner semantics)

Example:
    Query: "Kiu fondis Esperanton?"
    Document: "Zamenhof fondis Esperanton en 1887."

    Question type: WHO (kiu)
    Match pattern: subject of "fond"
    Extract: "Zamenhof" (complete vortgrupo)

    Answer: {
        'text': 'Zamenhof',
        'confidence': 0.95,
        'method': 'ast_pattern_match',
        'explanation': 'Subject of verb "fond" matching query pattern',
        'ast': {...}
    }
"""

from typing import Dict, Optional, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class ASTAnswerExtractor:
    """
    Deterministic answer extraction using AST pattern matching.

    This is the first-tier extraction method in the cascading fallback system.
    """

    # Question type mapping based on correlative suffix
    QUESTION_TYPES = {
        'u': 'WHO',      # kiu (who/which person)
        'o': 'WHAT',     # kio (what thing)
        'e': 'WHERE',    # kie (where location)
        'am': 'WHEN',    # kiam (when time)
        'om': 'HOW_MANY',# kiom (how many/much)
        'al': 'WHY',     # kial (why reason)
        'el': 'HOW',     # kiel (how manner)
        'a': 'WHICH',    # kia (which kind)
        'es': 'WHOSE',   # kies (whose possession)
    }

    # Ordinals to skip when extracting WHAT answers (predicate extraction)
    ORDINALS = {
        'unua', 'dua', 'tria', 'kvara', 'kvina', 'sesa', 'sepa', 'oka', 'naŭa', 'deka',
        'unue', 'due', 'trie', 'kvare', 'kvine', 'sese', 'sepe', 'oke', 'naŭe', 'deke',
        # Cardinal numbers (for HOW_MANY validation)
        'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
    }

    # Pronouns to reject for WHO questions
    PRONOUNS = {
        'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili',
        'si', 'oni', 'mem',
    }

    # Manual verb synonyms (high-priority pairs not in ReVo)
    # Format: root -> set of synonymous roots
    MANUAL_VERB_SYNONYMS = {
        'fond': {'kre', 'establ', 'komenc'},       # found ≈ create ≈ establish ≈ begin
        'kre': {'fond', 'far', 'produk'},          # create ≈ found ≈ make ≈ produce
        'establ': {'fond', 'kre', 'starigt'},      # establish ≈ found ≈ create ≈ start
        'komenc': {'fond', 'start', 'ekig'},       # begin ≈ found ≈ start ≈ initiate
        'naski': {'nask', 'genat'},                # born ≈ birth ≈ beget
        'mort': {'perdiĝ', 'forpas'},              # die ≈ perish ≈ pass away
        'viv': {'ekzist', 'log', 'rest'},          # live ≈ exist ≈ reside ≈ stay
        'far': {'kre', 'produk', 'fabrik'},        # make ≈ create ≈ produce ≈ manufacture
        'skrib': {'redakt', 'kompoz', 'ver'},      # write ≈ edit ≈ compose ≈ author
        'dir': {'parol', 'ekster', 'ekspr'},       # say ≈ speak ≈ utter ≈ express
        'pens': {'opini', 'kred', 'konsider'},     # think ≈ opine ≈ believe ≈ consider
        'vid': {'rimark', 'observ', 'pert'},       # see ≈ notice ≈ observe ≈ perceive
    }

    def __init__(self, revo_path: Optional[str] = None):
        """
        Initialize answer extractor with verb synonym support.

        Args:
            revo_path: Optional path to ReVo semantic relations JSON
                      (defaults to data/raw/eo/dictionaries/revo/revo_semantic_relations.json)
        """
        # Load ReVo synonym relations
        self.verb_synonyms = self._load_verb_synonyms(revo_path)
        logger.info(f"Loaded {len(self.verb_synonyms)} verb roots with synonym relations")

    def _load_verb_synonyms(self, revo_path: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Load verb synonyms from ReVo + manual additions.

        Combines:
        1. ReVo synonym relations (1,027 pairs from dictionary)
        2. Manual high-priority verb synonyms (12 pairs)

        Args:
            revo_path: Path to ReVo semantic relations JSON

        Returns:
            Dict mapping root -> set of synonym roots
        """
        import json
        from pathlib import Path

        synonyms = {}

        # Start with manual synonyms
        for root, syns in self.MANUAL_VERB_SYNONYMS.items():
            synonyms[root] = set(syns)

        # Load ReVo synonyms
        if revo_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            revo_path = project_root / "data/raw/eo/dictionaries/revo/revo_semantic_relations.json"
        else:
            revo_path = Path(revo_path)

        if revo_path.exists():
            try:
                with open(revo_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract synonym relations
                for relation in data.get('relations', {}).get('synonym', []):
                    source = relation['source']
                    target = relation['target']

                    # Add bidirectional relation
                    if source not in synonyms:
                        synonyms[source] = set()
                    synonyms[source].add(target)

                    if target not in synonyms:
                        synonyms[target] = set()
                    synonyms[target].add(source)

                logger.debug(f"Loaded {len(data['relations']['synonym'])} synonym pairs from ReVo")
            except Exception as e:
                logger.warning(f"Failed to load ReVo synonyms: {e}")
        else:
            logger.warning(f"ReVo semantic relations not found at {revo_path}")

        return synonyms

    def _are_verbs_similar(self, verb1: str, verb2: str) -> bool:
        """
        Check if two verb roots are semantically similar.

        Uses:
        1. Exact match
        2. 4-character prefix match (handles inflections: fond/fondi)
        3. ReVo + manual synonym relations

        Args:
            verb1: First verb root
            verb2: Second verb root

        Returns:
            True if verbs are similar
        """
        if not verb1 or not verb2:
            return False

        # Exact match
        if verb1 == verb2:
            return True

        # 4-char prefix match (existing heuristic)
        if len(verb1) >= 4 and len(verb2) >= 4:
            if verb1[:4] == verb2[:4]:
                return True

        # Synonym relation check
        # Check both 4-char prefix (for inflections) and full root
        for v1 in [verb1, verb1[:4] if len(verb1) >= 4 else verb1]:
            if v1 in self.verb_synonyms:
                syns = self.verb_synonyms[v1]
                for v2 in [verb2, verb2[:4] if len(verb2) >= 4 else verb2]:
                    if v2 in syns:
                        return True

        return False

    def extract_answer(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str,
        use_subclause_scoring: bool = True,
    ) -> Optional[Dict]:
        """
        Extract answer from document AST based on query pattern.

        Args:
            query_ast: Parsed query AST
            doc_ast: Parsed document AST
            doc_text: Original document text
            use_subclause_scoring: If True, decompose complex sentences into subclauses
                                  and extract from best-matching subclause (default: True)

        Returns:
            {
                'text': str,           # Answer text
                'confidence': float,   # [0-1] confidence score
                'method': str,         # 'ast_pattern_match' or 'subclause_match'
                'explanation': str,    # Why this was extracted
                'ast': Dict,          # Full AST of answer
                'span': Tuple[int, int]  # Character offsets in doc_text (if available)
            }
            or None if no answer found
        """
        # Detect question type
        question_type = self._detect_question_type(query_ast)
        if not question_type:
            logger.debug("Could not detect question type")
            return None

        logger.debug(f"Question type: {question_type}")

        # Check if sentence is complex (should try subclause decomposition)
        is_complex = self._is_complex_sentence(doc_ast)

        if use_subclause_scoring and is_complex:
            logger.debug("Complex sentence detected, using subclause scoring")
            answer = self._extract_from_best_subclause(
                query_ast, doc_ast, doc_text, question_type
            )
            # If subclause extraction succeeded, return it
            if answer:
                return answer

            # Otherwise fall back to whole-sentence extraction
            logger.debug("Subclause extraction failed, falling back to whole sentence")

        # Extract answer based on question type (whole sentence)
        answer = None
        if question_type == 'WHO':
            answer = self._extract_who(query_ast, doc_ast, doc_text)
        elif question_type == 'WHAT':
            answer = self._extract_what(query_ast, doc_ast, doc_text)
        elif question_type == 'WHERE':
            answer = self._extract_where(query_ast, doc_ast, doc_text)
        elif question_type == 'WHEN':
            answer = self._extract_when(query_ast, doc_ast, doc_text)
        elif question_type == 'HOW_MANY':
            answer = self._extract_how_many(query_ast, doc_ast, doc_text)
        elif question_type == 'WHY':
            answer = self._extract_why(query_ast, doc_ast, doc_text)
        elif question_type == 'HOW':
            answer = self._extract_how(query_ast, doc_ast, doc_text)
        elif question_type == 'WHICH':
            answer = self._extract_which(query_ast, doc_ast, doc_text)
        elif question_type == 'WHOSE':
            answer = self._extract_whose(query_ast, doc_ast, doc_text)
        else:
            logger.warning(f"Unsupported question type: {question_type}")
            return None

        # Validate answer before returning
        if answer:
            answer_text = answer.get('text', '')
            answer_ast = answer.get('ast')

            if not self._validate_answer(question_type, answer_text, answer_ast):
                logger.debug(f"Answer validation failed for '{answer_text}'")
                return None

        return answer

    def _is_complex_sentence(self, doc_ast: Dict) -> bool:
        """
        Check if sentence is complex (has multiple clauses).

        Heuristic: Count clause boundaries in aliaj.
        Complex if has 1+ clause boundaries (coordination, subordination, etc.).

        Args:
            doc_ast: Document AST

        Returns:
            True if complex sentence
        """
        aliaj = doc_ast.get('aliaj', [])
        clause_boundary_count = sum(1 for word in aliaj if self._is_clause_boundary(word))

        return clause_boundary_count >= 1

    def _extract_from_best_subclause(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str,
        question_type: str,
    ) -> Optional[Dict]:
        """
        Extract answer from best-matching subclause.

        Strategy:
        1. Decompose document into subclauses
        2. Score each subclause against query
        3. Extract from top-scoring subclause
        4. Add subclause scoring info to result

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text
            question_type: Question type (WHO, WHAT, etc.)

        Returns:
            Answer dict or None
        """
        # Extract subclauses
        subclauses = self._extract_subclauses(doc_ast)

        if len(subclauses) <= 1:
            logger.debug("No subclauses found (simple sentence)")
            return None

        # Score each subclause
        scored_subclauses = []
        for i, subclause in enumerate(subclauses):
            score = self._score_subclause(query_ast, subclause)
            scored_subclauses.append({
                'index': i,
                'score': score,
                'subclause': subclause,
                'type': subclause.get('subclause_type', 'unknown'),
            })

        # Sort by score
        scored_subclauses.sort(key=lambda x: x['score'], reverse=True)

        logger.debug(f"Subclause scores: {[(s['index'], s['type'], s['score']) for s in scored_subclauses[:3]]}")

        # Try extraction from top-scoring subclauses
        for ranked_subclause in scored_subclauses:
            if ranked_subclause['score'] == 0:
                break  # No point trying subclauses with zero score

            subclause = ranked_subclause['subclause']

            # Call appropriate extraction method on subclause
            answer = None
            if question_type == 'WHO':
                answer = self._extract_who(query_ast, subclause, doc_text)
            elif question_type == 'WHAT':
                answer = self._extract_what(query_ast, subclause, doc_text)
            elif question_type == 'WHERE':
                answer = self._extract_where(query_ast, subclause, doc_text)
            elif question_type == 'WHEN':
                answer = self._extract_when(query_ast, subclause, doc_text)
            elif question_type == 'HOW_MANY':
                answer = self._extract_how_many(query_ast, subclause, doc_text)

            if answer:
                # Add subclause info to answer
                answer['method'] = 'subclause_match'
                answer['explanation'] = (
                    f"{answer['explanation']} "
                    f"(from {ranked_subclause['type']} subclause, score: {ranked_subclause['score']:.1f})"
                )
                logger.debug(f"Extracted from subclause #{ranked_subclause['index']}: {answer['text']}")
                return answer

        logger.debug("No valid extraction from any subclause")
        return None

    def _detect_question_type(self, query_ast: Dict) -> Optional[str]:
        """
        Detect question type from query AST.

        Looks for correlative (kiu, kio, kie, etc.) in subject, object, or aliaj.

        Args:
            query_ast: Parsed query AST

        Returns:
            Question type string (WHO, WHAT, WHERE, etc.) or None
        """
        # Check if it's marked as a question
        if query_ast.get('fraztipo') != 'demando':
            return None

        # Check subject for correlative
        subjekto = query_ast.get('subjekto')
        if subjekto:
            q_type = self._check_correlative(subjekto)
            if q_type:
                return q_type

        # Check object for correlative (e.g., "Kion X kreis?")
        objekto = query_ast.get('objekto')
        if objekto:
            q_type = self._check_correlative(objekto)
            if q_type:
                return q_type

        # Check aliaj (question words can appear in modifiers)
        for modifier in query_ast.get('aliaj', []):
            q_type = self._check_correlative(modifier)
            if q_type:
                return q_type

        return None

    def _check_correlative(self, node: Dict) -> Optional[str]:
        """
        Check if node contains correlative and return question type.

        Args:
            node: AST node (vortgrupo or vorto)

        Returns:
            Question type or None
        """
        # Handle vortgrupo - check kerno
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._check_correlative(kerno)

        # Handle vorto - check if it's a correlative
        if node.get('tipo') == 'vorto':
            if node.get('vortspeco') == 'korelativo':
                # Get correlative suffix (u, o, e, am, om, etc.)
                suffix = node.get('korelativo_sufikso', '')
                return self.QUESTION_TYPES.get(suffix)

        return None

    def _extract_who(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHO answer (person/agent) with multi-candidate ranking.

        Strategy:
        1. Collect ALL person candidates (subject, proper nouns, -ul/-ist words)
        2. Score each: pattern_score + proximity_score + validation_score
        3. Return highest-scoring candidate

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        query_verb = self._get_verb_root(query_ast)
        doc_verb = self._get_verb_root(doc_ast)

        # Check if verbs match (using synonym support)
        if not query_verb or not doc_verb:
            # No verb to match, fall back to collecting any person candidates
            verb_match = False
        else:
            verb_match = self._are_verbs_similar(query_verb, doc_verb)

        # Collect all person candidates
        candidates = []

        # Candidate 1: Subject (if verb matches and looks like person)
        subjekto = doc_ast.get('subjekto')
        if subjekto:
            answer_text = self._vortgrupo_to_text(subjekto)
            if answer_text and self._is_person(subjekto):
                candidates.append({
                    'ast': subjekto,
                    'text': answer_text,
                    'pattern_score': 0.9 if verb_match else 0.5,  # High score if verb matches
                    'source': 'subject',
                })

        # Candidate 2: Check for passive voice agent ("de X")
        # Look for "de" + person in aliaj
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio' and modifier.get('radiko') == 'de':
                    # Check next item
                    if i + 1 < len(aliaj):
                        agent = aliaj[i + 1]
                        agent_text = self._vortgrupo_to_text(agent)
                        if agent_text and self._is_person(agent):
                            # Check if this is a passive voice construction
                            # In Esperanto passive: "Esperanto estis fondita de Zamenhof"
                            # Parser puts participle "fondita" as priskribo of subject
                            is_passive = self._is_passive_voice(doc_ast)

                            candidates.append({
                                'ast': agent,
                                'text': agent_text,
                                'pattern_score': 0.95 if is_passive else 0.7,  # Higher if passive
                                'source': 'passive_agent',
                            })

        # Candidate 3: Other proper nouns in aliaj (not after "de")
        used_positions = set()  # Track positions already added as passive agents
        for i, modifier in enumerate(aliaj):
            if i in used_positions:
                continue

            # Check if previous word was "de" (already handled)
            if i > 0 and aliaj[i-1].get('radiko') == 'de':
                used_positions.add(i)
                continue

            modifier_text = self._vortgrupo_to_text(modifier)
            if modifier_text and self._is_person(modifier):
                candidates.append({
                    'ast': modifier,
                    'text': modifier_text,
                    'pattern_score': 0.6,  # Lower score - not grammatical role
                    'source': 'proper_noun',
                })

        # Candidate 4: Object (if subject doesn't look like person)
        objekto = doc_ast.get('objekto')
        if objekto:
            objekto_text = self._vortgrupo_to_text(objekto)
            if objekto_text and self._is_person(objekto):
                candidates.append({
                    'ast': objekto,
                    'text': objekto_text,
                    'pattern_score': 0.7 if verb_match else 0.4,
                    'source': 'object',
                })

        if not candidates:
            return None

        # Score each candidate
        for candidate in candidates:
            # Proximity score: how close to query terms?
            candidate['proximity_score'] = self._score_candidate_proximity(
                candidate['ast'], query_ast, doc_ast
            )

            # Validation score: does it pass type validation?
            is_valid = self._validate_answer('WHO', candidate['text'], candidate['ast'])
            candidate['validation_score'] = 1.0 if is_valid else 0.0

            # Total score (weighted combination)
            candidate['total_score'] = (
                candidate['pattern_score'] * 0.4 +
                candidate['proximity_score'] * 0.4 +
                candidate['validation_score'] * 0.2
            )

        # Return best candidate
        best = max(candidates, key=lambda c: c['total_score'])

        # Log candidates for debugging
        if len(candidates) > 1:
            logger.debug(f"WHO candidates ranked:")
            for i, c in enumerate(sorted(candidates, key=lambda x: x['total_score'], reverse=True)):
                logger.debug(f"  {i+1}. '{c['text']}' (score={c['total_score']:.3f}, "
                           f"pattern={c['pattern_score']:.2f}, "
                           f"proximity={c['proximity_score']:.2f}, "
                           f"valid={c['validation_score']:.0f}, "
                           f"source={c['source']})")

        return {
            'text': best['text'],
            'confidence': best['total_score'],
            'method': 'ast_ranked_match',
            'explanation': f"{best['source'].replace('_', ' ').title()} (pattern={best['pattern_score']:.2f}, proximity={best['proximity_score']:.2f})",
            'ast': best['ast'],
        }

    def _extract_what(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHAT answer (thing/concept) with multi-candidate ranking.

        Strategy:
        1. Collect ALL thing/concept candidates (predicates, objects, subjects)
        2. Score each by pattern + proximity + validation
        3. Return highest-scoring candidate

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        query_verb = self._get_verb_root(query_ast)
        doc_verb = self._get_verb_root(doc_ast)

        candidates = []

        # Check for "estas" questions (definitions)
        if query_verb == 'est' and doc_verb == 'est':
            # Candidate 1: Predicates after "estas" in aliaj
            aliaj = doc_ast.get('aliaj', [])
            for modifier in aliaj:
                if modifier.get('tipo') == 'vorto':
                    vortspeco = modifier.get('vortspeco')
                    radiko = modifier.get('radiko', '').lower()

                    # Skip ordinals (unua, dua, etc.)
                    if radiko in self.ORDINALS:
                        continue

                    # Substantives (high priority)
                    if vortspeco == 'substantivo':
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            candidates.append({
                                'ast': modifier,
                                'text': answer_text,
                                'pattern_score': 0.9,  # High - substantive predicate
                                'source': 'predicate_noun',
                            })

                    # Adjectives (lower priority)
                    elif vortspeco == 'adjektivo':
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            candidates.append({
                                'ast': modifier,
                                'text': answer_text,
                                'pattern_score': 0.75,  # Medium - adjective predicate
                                'source': 'predicate_adj',
                            })

            # Candidate 2: Object (fallback for "estas")
            objekto = doc_ast.get('objekto')
            if objekto:
                answer_text = self._vortgrupo_to_text(objekto)
                if answer_text:
                    candidates.append({
                        'ast': objekto,
                        'text': answer_text,
                        'pattern_score': 0.7,  # Lower - less typical
                        'source': 'object_estas',
                    })

        # Check if verbs match (non-estas questions, using synonym support)
        verb_match = False
        if query_verb and doc_verb:
            verb_match = self._are_verbs_similar(query_verb, doc_verb)

        if verb_match:
            # Candidate 3: Object (if query has "kio" as object)
            query_obj = query_ast.get('objekto')
            if query_obj and self._is_correlative(query_obj, 'kio'):
                objekto = doc_ast.get('objekto')
                if objekto:
                    answer_text = self._vortgrupo_to_text(objekto)
                    if answer_text:
                        candidates.append({
                            'ast': objekto,
                            'text': answer_text,
                            'pattern_score': 0.9,  # High - object matches pattern
                            'source': 'object',
                        })

            # Candidate 4: Subject (if not already added)
            subjekto = doc_ast.get('subjekto')
            if subjekto:
                answer_text = self._vortgrupo_to_text(subjekto)
                if answer_text and not any(c['text'] == answer_text for c in candidates):
                    candidates.append({
                        'ast': subjekto,
                        'text': answer_text,
                        'pattern_score': 0.8,  # Medium-high
                        'source': 'subject',
                    })

        if not candidates:
            return None

        # Score each candidate
        for candidate in candidates:
            candidate['proximity_score'] = self._score_candidate_proximity(
                candidate['ast'], query_ast, doc_ast
            )

            is_valid = self._validate_answer('WHAT', candidate['text'], candidate['ast'])
            candidate['validation_score'] = 1.0 if is_valid else 0.0

            candidate['total_score'] = (
                candidate['pattern_score'] * 0.4 +
                candidate['proximity_score'] * 0.4 +
                candidate['validation_score'] * 0.2
            )

        # Return best candidate
        best = max(candidates, key=lambda c: c['total_score'])

        # Log for debugging
        if len(candidates) > 1:
            logger.debug(f"WHAT candidates ranked:")
            for i, c in enumerate(sorted(candidates, key=lambda x: x['total_score'], reverse=True)):
                logger.debug(f"  {i+1}. '{c['text']}' (score={c['total_score']:.3f}, "
                           f"pattern={c['pattern_score']:.2f}, "
                           f"proximity={c['proximity_score']:.2f}, "
                           f"source={c['source']})")

        return {
            'text': best['text'],
            'confidence': best['total_score'],
            'method': 'ast_ranked_match',
            'explanation': f"{best['source'].replace('_', ' ').title()} (pattern={best['pattern_score']:.2f}, proximity={best['proximity_score']:.2f})",
            'ast': best['ast'],
        }

    def _extract_where(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHERE answer (location) with multi-candidate ranking.

        Strategy:
        1. Collect ALL location candidates (prepositional phrases, -ej words, place names)
        2. Score each by pattern + proximity + validation
        3. Return highest-scoring candidate

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Location prepositions
        LOCATION_PREPS = {'en', 'sur', 'apud', 'ĉe', 'antaŭ', 'post', 'sub',
                          'super', 'inter', 'ekster', 'ĉirkaŭ', 'trans'}

        candidates = []

        # Candidate 1: Prepositional phrases with location prepositions
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in LOCATION_PREPS:
                        # Look ahead for object (skip function words)
                        j = i + 1
                        while j < len(aliaj):
                            next_item = aliaj[j]

                            # Skip function words and punctuation
                            next_radiko = next_item.get('radiko', '')
                            if next_radiko in {'la', ',', '.', 'kaj', 'sed'}:
                                j += 1
                                continue

                            # Found potential location object
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text and self._is_place(next_item):
                                candidates.append({
                                    'ast': next_item,
                                    'text': answer_text,
                                    'pattern_score': 0.95,  # High - prepositional phrase
                                    'source': f'prep_{radiko}',
                                })
                            break

        # Candidate 2: Words with -ej suffix (place for)
        for key in ['subjekto', 'objekto']:
            node = doc_ast.get(key)
            if node and 'ej' in self._get_suffixes(node):
                answer_text = self._vortgrupo_to_text(node)
                if answer_text:
                    candidates.append({
                        'ast': node,
                        'text': answer_text,
                        'pattern_score': 0.85,  # Medium - suffix indicator
                        'source': 'suffix_ej',
                    })

        # Candidate 3: Place names in subject/object
        for key in ['subjekto', 'objekto']:
            node = doc_ast.get(key)
            if node:
                answer_text = self._vortgrupo_to_text(node)
                if answer_text and self._is_place(node):
                    # Check if not already added
                    if not any(c['text'] == answer_text for c in candidates):
                        candidates.append({
                            'ast': node,
                            'text': answer_text,
                            'pattern_score': 0.7,  # Lower - no preposition
                            'source': key,
                        })

        if not candidates:
            return None

        # Score each candidate
        for candidate in candidates:
            candidate['proximity_score'] = self._score_candidate_proximity(
                candidate['ast'], query_ast, doc_ast
            )

            is_valid = self._validate_answer('WHERE', candidate['text'], candidate['ast'])
            candidate['validation_score'] = 1.0 if is_valid else 0.0

            candidate['total_score'] = (
                candidate['pattern_score'] * 0.4 +
                candidate['proximity_score'] * 0.4 +
                candidate['validation_score'] * 0.2
            )

        # Return best candidate
        best = max(candidates, key=lambda c: c['total_score'])

        # Log for debugging
        if len(candidates) > 1:
            logger.debug(f"WHERE candidates ranked:")
            for i, c in enumerate(sorted(candidates, key=lambda x: x['total_score'], reverse=True)):
                logger.debug(f"  {i+1}. '{c['text']}' (score={c['total_score']:.3f}, "
                           f"pattern={c['pattern_score']:.2f}, "
                           f"proximity={c['proximity_score']:.2f}, "
                           f"source={c['source']})")

        return {
            'text': best['text'],
            'confidence': best['total_score'],
            'method': 'ast_ranked_match',
            'explanation': f"{best['source'].replace('_', ' ').title()} (pattern={best['pattern_score']:.2f}, proximity={best['proximity_score']:.2f})",
            'ast': best['ast'],
        }

    def _extract_when(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHEN answer (time).

        Strategy:
        1. Look for time prepositions (en, dum, post, antaŭ)
        2. Look for year/date patterns (1887, januaro, etc.)
        3. Look for time adverbs (hieraŭ, hodiaŭ, morgaŭ)

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Time prepositions
        TIME_PREPS = {'en', 'dum', 'post', 'antaŭ', 'ekde', 'ĝis'}

        # Check aliaj for time modifiers
        # Preposition and object are consecutive items in aliaj
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                # Check for time preposition
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in TIME_PREPS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            # Check if it looks like time
                            if answer_text and self._looks_like_time(answer_text):
                                return {
                                    'text': answer_text,
                                    'confidence': 0.95,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Time expression after "{radiko}"',
                                    'ast': next_item,
                                }

                # Time adverbs (hieraŭ, hodiaŭ, etc.)
                # Note: Parser may classify these as 'partiklo' or 'adverbo'
                vortspeco = modifier.get('vortspeco')
                if vortspeco in ['adverbo', 'partiklo']:
                    radiko = modifier.get('radiko', '')
                    if radiko in {'hieraŭ', 'hodiaŭ', 'morgaŭ', 'nun', 'tiam'}:
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            return {
                                'text': answer_text,
                                'confidence': 0.9,
                                'method': 'ast_pattern_match',
                                'explanation': 'Time adverb',
                                'ast': modifier,
                            }

        return None

    def _extract_how_many(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract HOW_MANY answer (quantity).

        Strategy:
        1. Look for numbers in document
        2. Look for quantity words (multe, malmulte, etc.)
        3. Extract numeric modifiers

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Check priskriboj (modifiers) for numbers
        for key in ['subjekto', 'objekto']:
            node = doc_ast.get(key)
            if node and node.get('tipo') == 'vortgrupo':
                for priskribo in node.get('priskriboj', []):
                    if priskribo.get('tipo') == 'vorto':
                        radiko = priskribo.get('radiko', '')
                        # Check if it's a number
                        if radiko.isdigit() or self._is_number_word(radiko):
                            answer_text = self._vortgrupo_to_text(priskribo)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.95,
                                    'method': 'ast_pattern_match',
                                    'explanation': 'Numeric modifier',
                                    'ast': priskribo,
                                }

        # Check aliaj for standalone numbers
        for modifier in doc_ast.get('aliaj', []):
            if modifier.get('tipo') == 'vorto':
                radiko = modifier.get('radiko', '')
                if radiko.isdigit() or self._is_number_word(radiko):
                    answer_text = self._vortgrupo_to_text(modifier)
                    if answer_text:
                        return {
                            'text': answer_text,
                            'confidence': 0.9,
                            'method': 'ast_pattern_match',
                            'explanation': 'Number in sentence',
                            'ast': modifier,
                        }

        return None

    def _extract_why(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHY answer (reason/cause).

        Strategy:
        1. Look for causal prepositions (pro, ĉar)
        2. Look for purpose constructions (por + infinitive)
        3. Extract clause after causal marker

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Causal markers
        CAUSAL_MARKERS = {'pro', 'ĉar', 'por', 'tial'}

        # Check aliaj for causal phrases
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in CAUSAL_MARKERS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.85,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Reason/cause after "{radiko}"',
                                    'ast': next_item,
                                }

        return None

    def _extract_how(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract HOW answer (manner).

        Strategy:
        1. Look for manner adverbs (ending in -e)
        2. Look for manner prepositions (per, kun)
        3. Extract adverbial phrases

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Manner prepositions
        MANNER_PREPS = {'per', 'kun', 'sen', 'laŭ'}

        # Check aliaj for manner modifiers
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                # Prepositional phrases
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in MANNER_PREPS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.85,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Manner expression with "{radiko}"',
                                    'ast': next_item,
                                }

                # Adverbs (ending in -e)
                if modifier.get('vortspeco') == 'adverbo':
                    answer_text = self._vortgrupo_to_text(modifier)
                    if answer_text:
                        return {
                            'text': answer_text,
                            'confidence': 0.8,
                            'method': 'ast_pattern_match',
                            'explanation': 'Manner adverb',
                            'ast': modifier,
                        }

        return None

    def _extract_which(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHICH answer (specific instance from category).

        Similar to WHO/WHAT but expects a specific selection.

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # For now, treat like WHO extraction
        return self._extract_who(query_ast, doc_ast, doc_text)

    def _extract_whose(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHOSE answer (possession).

        Strategy:
        1. Look for possessive constructions (de + possessor)
        2. Look for possessive adjectives (mia, via, lia, etc.)

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Check for "de" prepositional phrases (possession)
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio':
                    if modifier.get('radiko') == 'de':
                        # Get next item (possessor)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.9,
                                    'method': 'ast_pattern_match',
                                    'explanation': 'Possessor after "de"',
                                    'ast': next_item,
                                }

        return None

    # -------------------------------------------------------------------------
    # Answer Validation
    # -------------------------------------------------------------------------

    def _validate_answer(
        self,
        question_type: str,
        answer_text: str,
        answer_ast: Optional[Dict] = None
    ) -> bool:
        """
        Validate extracted answer matches expected answer type.

        This is a deterministic sanity check to reject clearly wrong extractions.

        Args:
            question_type: Question type (WHO, WHAT, WHERE, etc.)
            answer_text: Extracted answer text
            answer_ast: Optional AST of answer

        Returns:
            True if answer is valid, False if clearly wrong
        """
        answer_lower = answer_text.lower()

        # WHO questions should not return pronouns
        if question_type == 'WHO':
            # Check if answer is a pronoun
            if answer_lower in self.PRONOUNS:
                logger.debug(f"Rejecting pronoun '{answer_text}' for WHO question")
                return False

            # Check for generic words that aren't person names
            generic_non_persons = {'komparo', 'grupo', 'aro', 'afero', 'io'}
            if answer_lower in generic_non_persons:
                logger.debug(f"Rejecting generic non-person '{answer_text}' for WHO question")
                return False

        # HOW_MANY questions should contain numbers
        elif question_type == 'HOW_MANY':
            # Check if answer contains digits or number words
            has_digit = any(c.isdigit() for c in answer_text)
            is_number_word = self._is_number_word(answer_lower)

            if not (has_digit or is_number_word):
                logger.debug(f"Rejecting non-numeric '{answer_text}' for HOW_MANY question")
                return False

            # Reject index-style answers (all caps, single word)
            if answer_text.isupper() and len(answer_text.split()) == 1:
                logger.debug(f"Rejecting index entry '{answer_text}' for HOW_MANY question")
                return False

        # WHEN questions should look like time
        elif question_type == 'WHEN':
            if not self._looks_like_time(answer_text):
                logger.debug(f"Rejecting non-time '{answer_text}' for WHEN question")
                return False

        # WHAT questions should not be pronouns or ordinals
        elif question_type == 'WHAT':
            # Already handled ordinals in extraction, but double-check
            radiko = answer_ast.get('radiko', '').lower() if answer_ast else ''
            if radiko in self.ORDINALS:
                logger.debug(f"Rejecting ordinal '{answer_text}' for WHAT question")
                return False

            # Reject pronouns
            if answer_lower in self.PRONOUNS:
                logger.debug(f"Rejecting pronoun '{answer_text}' for WHAT question")
                return False

        return True

    # -------------------------------------------------------------------------
    # Subclause Decomposition (for complex sentences)
    # -------------------------------------------------------------------------

    def _is_clause_boundary(self, word: Dict) -> bool:
        """
        Check if word marks a clause boundary.

        Clause boundaries are indicated by:
        - Participles (fondita, kreita → participial clause)
        - Relative pronouns (kiu, kio, kia, kie, kiam, etc.)
        - Coordinating conjunctions (kaj, sed, aŭ)
        - Subordinating conjunctions (ke, ĉar, se, kvankam)

        Args:
            word: AST word node

        Returns:
            True if word marks clause boundary
        """
        if word.get('tipo') != 'vorto':
            return False

        # Participles (indicate participial clauses)
        if word.get('participo_tempo'):
            return True

        # Relative/interrogative correlatives
        radiko = word.get('radiko', '').lower()
        if radiko in {'kiu', 'kio', 'kia', 'kie', 'kiam', 'kiel', 'kial', 'kiom', 'kies'}:
            return True

        # Coordinating conjunctions
        if word.get('vortspeco') == 'konjunkcio':
            if radiko in {'kaj', 'sed', 'aŭ', 'nek'}:
                return True

        # Subordinating conjunctions/particles
        if word.get('vortspeco') in ['konjunkcio', 'partiklo']:
            if radiko in {'ke', 'ĉar', 'se', 'kvankam', 'dum', 'post', 'antaŭ'}:
                return True

        return False

    def _extract_subclauses(self, doc_ast: Dict) -> List[Dict]:
        """
        Extract subclauses from complex sentence using AST structure.

        Strategy:
        1. Main clause (subjekto-verbo-objekto) always included
        2. Scan aliaj for clause boundaries (participles, conjunctions, relative pronouns)
        3. Group consecutive words between boundaries into subclauses

        Args:
            doc_ast: Document AST (frazo)

        Returns:
            List of subclause dicts with structure similar to full AST
        """
        subclauses = []

        # Main clause (always included)
        main_clause = {
            'tipo': 'subclause',
            'subclause_type': 'main',
            'subjekto': doc_ast.get('subjekto'),
            'verbo': doc_ast.get('verbo'),
            'objekto': doc_ast.get('objekto'),
            'aliaj': [],  # Will add non-clause-boundary modifiers
        }

        # Collect aliaj that are NOT clause boundaries (belong to main clause)
        aliaj = doc_ast.get('aliaj', [])
        current_subclause_words = []

        for word in aliaj:
            # Check if this starts a new subclause
            if self._is_clause_boundary(word):
                # Save current subclause if it has content
                if current_subclause_words:
                    subclause = self._make_subclause(current_subclause_words)
                    subclauses.append(subclause)
                    current_subclause_words = []

                # Start new subclause with boundary word
                current_subclause_words.append(word)
            else:
                # Add to current subclause (or main clause if empty)
                if current_subclause_words:
                    current_subclause_words.append(word)
                else:
                    # Belongs to main clause
                    main_clause['aliaj'].append(word)

        # Add final subclause
        if current_subclause_words:
            subclause = self._make_subclause(current_subclause_words)
            subclauses.append(subclause)

        # Prepend main clause
        subclauses.insert(0, main_clause)

        return subclauses

    def _make_subclause(self, words: List[Dict]) -> Dict:
        """
        Create subclause dict from list of words.

        Attempts to identify subject/verb/object structure within the subclause.

        Args:
            words: List of word AST nodes

        Returns:
            Subclause dict
        """
        subclause = {
            'tipo': 'subclause',
            'subclause_type': 'subordinate',
            'subjekto': None,
            'verbo': None,
            'objekto': None,
            'aliaj': [],
        }

        # Try to find verb in subclause
        for word in words:
            if word.get('tipo') == 'vorto':
                vortspeco = word.get('vortspeco')

                # Identify verb
                if vortspeco == 'verbo' and not subclause['verbo']:
                    subclause['verbo'] = word

                # Identify substantives (potential subject/object)
                elif vortspeco == 'substantivo':
                    # If no subject yet, assume this is subject
                    if not subclause['subjekto']:
                        subclause['subjekto'] = word
                    # Otherwise assume object
                    elif not subclause['objekto']:
                        subclause['objekto'] = word
                    else:
                        subclause['aliaj'].append(word)

                # Everything else goes in aliaj
                else:
                    subclause['aliaj'].append(word)

        return subclause

    def _score_subclause(self, query_ast: Dict, subclause: Dict) -> float:
        """
        Score subclause relevance to query.

        Uses same method as sentence retrieval:
        - Extract roots from both
        - Count matches
        - Weight by role (verb > subject > object)

        Args:
            query_ast: Query AST
            subclause: Subclause dict

        Returns:
            Relevance score (higher is better)
        """
        score = 0.0

        # Verb match (highest weight, with synonym support)
        query_verb = self._get_verb_root(query_ast)
        subclause_verb = self._get_verb_root(subclause)

        if query_verb and subclause_verb:
            if self._are_verbs_similar(query_verb, subclause_verb):
                # Full similarity (exact, prefix, or synonym)
                score += 5.0 if query_verb == subclause_verb else 4.0

        # Root matches (subject/object)
        query_roots = self._extract_roots(query_ast)
        subclause_roots = self._extract_roots(subclause)

        matches = set(query_roots) & set(subclause_roots)
        score += len(matches) * 2.0

        return score

    def _extract_roots(self, ast: Dict) -> List[str]:
        """
        Extract all content roots from AST.

        Args:
            ast: AST dict (frazo or subclause)

        Returns:
            List of root strings
        """
        roots = []

        # Extract from subject
        if ast.get('subjekto'):
            roots.extend(self._extract_roots_from_node(ast['subjekto']))

        # Extract from verb
        if ast.get('verbo'):
            roots.extend(self._extract_roots_from_node(ast['verbo']))

        # Extract from object
        if ast.get('objekto'):
            roots.extend(self._extract_roots_from_node(ast['objekto']))

        # Extract from aliaj
        for modifier in ast.get('aliaj', []):
            roots.extend(self._extract_roots_from_node(modifier))

        return roots

    def _extract_roots_from_node(self, node: Dict) -> List[str]:
        """Extract roots from AST node (vorto or vortgrupo)."""
        roots = []

        if node.get('tipo') == 'vorto':
            radiko = node.get('radiko', '').lower()
            if radiko:
                roots.append(radiko)

        elif node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                roots.extend(self._extract_roots_from_node(kerno))

            for priskribo in node.get('priskriboj', []):
                roots.extend(self._extract_roots_from_node(priskribo))

        return roots

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_verb_root(self, ast: Dict) -> Optional[str]:
        """Extract verb root from AST."""
        verbo = ast.get('verbo')
        if verbo and verbo.get('tipo') == 'vorto':
            return verbo.get('radiko')
        return None

    def _is_passive_voice(self, ast: Dict) -> bool:
        """
        Check if sentence uses passive voice construction.

        In Esperanto passive voice, the participle appears as a priskribo (modifier)
        of the subject: "Esperanto estis fondita de Zamenhof"

        AST structure:
        - verbo: "estis" (to be)
        - subjekto.priskriboj: contains passive participle "fondita"
          - participo_voĉo: "pasiva"
          - participo_tempo: "pasinteco" (past participle)

        Args:
            ast: AST dict (frazo or subclause)

        Returns:
            True if passive voice construction detected
        """
        # Check if verb is "esti" (to be)
        verbo = ast.get('verbo')
        if not verbo or verbo.get('tipo') != 'vorto':
            return False

        verb_root = verbo.get('radiko', '')
        if verb_root != 'est':
            return False

        # Check if subject has passive participle modifier
        subjekto = ast.get('subjekto')
        if not subjekto or subjekto.get('tipo') != 'vortgrupo':
            return False

        # Look for passive participle in priskriboj
        for priskribo in subjekto.get('priskriboj', []):
            if priskribo.get('tipo') == 'vorto':
                # Check for passive participle markers
                if priskribo.get('participo_voĉo') == 'pasiva':
                    return True
                # Also check suffix 'it' (passive participle suffix)
                if 'it' in priskribo.get('sufiksoj', []):
                    return True

        return False

    def _vortgrupo_to_text(self, node: Dict) -> Optional[str]:
        """
        Convert vortgrupo AST node to text.

        Reconstructs the original text representation of a word group.

        Args:
            node: AST node (vortgrupo or vorto)

        Returns:
            Text string or None
        """
        if node.get('tipo') == 'vorto':
            return node.get('plena_vorto')

        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                # For now, just return the core word
                # TODO: Include priskriboj (modifiers)
                return self._vortgrupo_to_text(kerno)

        return None

    def _is_person(self, node: Dict) -> bool:
        """
        Check if node represents a person (enhanced validation).

        Heuristics:
        - Has -ul suffix (person characterized by)
        - Has -ist suffix (professional)
        - Has -in suffix (feminine)
        - Is a proper noun (starts with capital) BUT NOT:
          - Compound words ending in -o (things like "Esperanto-versio")
          - Place-indicating suffixes (-ej = place)
          - Common place names

        Args:
            node: AST node

        Returns:
            True if likely a person
        """
        suffixes = self._get_suffixes(node)

        # Strong person indicators (suffixes)
        if 'ul' in suffixes or 'ist' in suffixes or 'in' in suffixes:
            return True

        text = self._vortgrupo_to_text(node)
        if not text:
            return False

        # Reject compound words ending in -o (things, not people)
        # "Esperanto-versio", "radio-stacio", etc.
        if '-' in text and text.endswith('o'):
            return False

        # Reject place-indicating suffixes
        place_suffixes = {'ej'}  # -ejo = place for
        if any(suf in suffixes for suf in place_suffixes):
            return False

        # Reject common place names (cities, countries)
        # This is a small gazetteer - can be expanded
        place_names = {
            # Cities
            'Barcelono', 'Varsovio', 'Parizo', 'Berlino', 'Londono', 'Romo',
            'Moskvo', 'Pekino', 'Tokio', 'Nov-Jorko', 'Bjalistoko', 'Suwałki',
            # Countries
            'Pollando', 'Francio', 'Germanio', 'Anglio', 'Italio', 'Rusio',
            'Ĉinio', 'Japanio', 'Usono', 'Hispanio', 'Britio',
            # Regions
            'Eŭropo', 'Azio', 'Afriko', 'Ameriko',
        }
        if text in place_names:
            return False

        # Check if proper noun (after exclusions)
        if text[0].isupper():
            return True

        # Check if correlative (kiu)
        if node.get('tipo') == 'vorto':
            if node.get('korelativo_sufikso') == 'u':
                return True

        return False

    def _is_place(self, node: Dict) -> bool:
        """
        Check if node represents a place/location.

        Heuristics:
        - Has -ej suffix (place for)
        - Is in place name gazetteer
        - Is a proper noun with location indicators

        Args:
            node: AST node

        Returns:
            True if likely a place
        """
        suffixes = self._get_suffixes(node)

        # Strong place indicator (-ejo)
        if 'ej' in suffixes:
            return True

        text = self._vortgrupo_to_text(node)
        if not text:
            return False

        # Check place name gazetteer
        place_names = {
            # Cities
            'Barcelono', 'Varsovio', 'Parizo', 'Berlino', 'Londono', 'Romo',
            'Moskvo', 'Pekino', 'Tokio', 'Nov-Jorko', 'Bjalistoko', 'Suwałki',
            # Countries
            'Pollando', 'Francio', 'Germanio', 'Anglio', 'Italio', 'Rusio',
            'Ĉinio', 'Japanio', 'Usono', 'Hispanio', 'Britio',
            # Regions
            'Eŭropo', 'Azio', 'Afriko', 'Ameriko',
        }
        if text in place_names:
            return True

        # Check for location-related words
        location_roots = {'urb', 'vilaĝ', 'land', 'region', 'loko', 'teren'}
        if node.get('tipo') == 'vorto':
            radiko = node.get('radiko', '').lower()
            if any(radiko.startswith(loc) for loc in location_roots):
                return True

        return False

    def _get_suffixes(self, node: Dict) -> List[str]:
        """Extract list of suffixes from node."""
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._get_suffixes(kerno)

        if node.get('tipo') == 'vorto':
            return node.get('sufiksoj', [])

        return []

    def _is_correlative(self, node: Dict, radiko: str) -> bool:
        """Check if node is a specific correlative (e.g., 'kio')."""
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._is_correlative(kerno, radiko)

        if node.get('tipo') == 'vorto':
            return (node.get('vortspeco') == 'korelativo' and
                    node.get('radiko') == radiko)

        return False

    def _looks_like_time(self, text: str) -> bool:
        """
        Heuristic check if text looks like a time expression.

        Args:
            text: Text string

        Returns:
            True if looks like time
        """
        # Check for year (4 digits)
        if text.isdigit() and len(text) == 4:
            year = int(text)
            if 1000 <= year <= 2100:
                return True

        # Check for month names
        months = {'januaro', 'februaro', 'marto', 'aprilo', 'majo', 'junio',
                  'julio', 'aŭgusto', 'septembro', 'oktobro', 'novembro', 'decembro'}
        if text.lower() in months:
            return True

        # Check for time words
        time_words = {'jaro', 'monato', 'semajno', 'tago', 'horo', 'minuto'}
        for word in time_words:
            if word in text.lower():
                return True

        return False

    def _is_number_word(self, radiko: str) -> bool:
        """
        Check if root is a number word.

        Args:
            radiko: Root string

        Returns:
            True if number word
        """
        number_words = {
            'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
            'cent', 'mil', 'milion', 'miliard',
            'multe', 'malmulte', 'kelke', 'sufiĉe'
        }
        return radiko.lower() in number_words

    # -------------------------------------------------------------------------
    # Position Tracking and Proximity Scoring (for multi-candidate ranking)
    # -------------------------------------------------------------------------

    def _get_word_position(self, target_node: Dict, doc_ast: Dict) -> Optional[int]:
        """
        Get the position index of a word/node in the document AST.

        Args:
            target_node: AST node to find
            doc_ast: Document AST

        Returns:
            Position index (0-based) or None if not found
        """
        position = 0

        # Check subjekto
        if doc_ast.get('subjekto'):
            pos = self._find_node_position(target_node, doc_ast['subjekto'], position)
            if pos is not None:
                return pos
            position += self._count_words(doc_ast['subjekto'])

        # Check verbo
        if doc_ast.get('verbo'):
            if self._nodes_equal(target_node, doc_ast['verbo']):
                return position
            position += 1

        # Check objekto
        if doc_ast.get('objekto'):
            pos = self._find_node_position(target_node, doc_ast['objekto'], position)
            if pos is not None:
                return pos
            position += self._count_words(doc_ast['objekto'])

        # Check aliaj
        for modifier in doc_ast.get('aliaj', []):
            pos = self._find_node_position(target_node, modifier, position)
            if pos is not None:
                return pos
            position += self._count_words(modifier)

        return None

    def _find_node_position(self, target_node: Dict, search_node: Dict, start_pos: int) -> Optional[int]:
        """
        Recursively find target node within search node.

        Args:
            target_node: Node to find
            search_node: Node to search within
            start_pos: Starting position offset

        Returns:
            Position or None
        """
        if self._nodes_equal(target_node, search_node):
            return start_pos

        # If search_node is vortgrupo, check within it
        if search_node.get('tipo') == 'vortgrupo':
            pos = start_pos

            # Check priskriboj (modifiers)
            for priskribo in search_node.get('priskriboj', []):
                result = self._find_node_position(target_node, priskribo, pos)
                if result is not None:
                    return result
                pos += self._count_words(priskribo)

            # Check kerno
            if search_node.get('kerno'):
                result = self._find_node_position(target_node, search_node['kerno'], pos)
                if result is not None:
                    return result

        return None

    def _nodes_equal(self, node1: Dict, node2: Dict) -> bool:
        """
        Check if two AST nodes represent the same word.

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if same word
        """
        if node1.get('tipo') != node2.get('tipo'):
            return False

        if node1.get('tipo') == 'vorto':
            # Compare by full word text
            return node1.get('plena_vorto') == node2.get('plena_vorto')

        return False

    def _count_words(self, node: Dict) -> int:
        """
        Count number of words in AST node.

        Args:
            node: AST node

        Returns:
            Word count
        """
        if not node:
            return 0

        if node.get('tipo') == 'vorto':
            return 1

        if node.get('tipo') == 'vortgrupo':
            count = 0
            for priskribo in node.get('priskriboj', []):
                count += self._count_words(priskribo)
            if node.get('kerno'):
                count += self._count_words(node['kerno'])
            return count

        return 0

    def _find_root_positions(self, root: str, doc_ast: Dict) -> List[int]:
        """
        Find all positions where a root appears in document.

        Args:
            root: Root string to find
            doc_ast: Document AST

        Returns:
            List of position indices
        """
        positions = []
        position = 0

        # Check subjekto
        if doc_ast.get('subjekto'):
            positions.extend(self._find_root_in_node(root, doc_ast['subjekto'], position))
            position += self._count_words(doc_ast['subjekto'])

        # Check verbo
        if doc_ast.get('verbo'):
            verbo = doc_ast['verbo']
            if verbo.get('tipo') == 'vorto' and verbo.get('radiko', '').lower() == root:
                positions.append(position)
            position += 1

        # Check objekto
        if doc_ast.get('objekto'):
            positions.extend(self._find_root_in_node(root, doc_ast['objekto'], position))
            position += self._count_words(doc_ast['objekto'])

        # Check aliaj
        for modifier in doc_ast.get('aliaj', []):
            positions.extend(self._find_root_in_node(root, modifier, position))
            position += self._count_words(modifier)

        return positions

    def _find_root_in_node(self, root: str, node: Dict, start_pos: int) -> List[int]:
        """
        Find root in AST node recursively.

        Args:
            root: Root to find
            node: Node to search
            start_pos: Starting position

        Returns:
            List of positions
        """
        positions = []

        if node.get('tipo') == 'vorto':
            if node.get('radiko', '').lower() == root:
                positions.append(start_pos)

        elif node.get('tipo') == 'vortgrupo':
            pos = start_pos

            # Check priskriboj
            for priskribo in node.get('priskriboj', []):
                positions.extend(self._find_root_in_node(root, priskribo, pos))
                pos += self._count_words(priskribo)

            # Check kerno
            if node.get('kerno'):
                positions.extend(self._find_root_in_node(root, node['kerno'], pos))

        return positions

    def _score_candidate_proximity(
        self,
        candidate_ast: Dict,
        query_ast: Dict,
        doc_ast: Dict
    ) -> float:
        """
        Score candidate by proximity to query terms in document.

        Strategy:
        - Find candidate position in document
        - Find positions of all query roots
        - Measure average distance to query roots
        - Return: 1.0 / (1 + avg_distance)

        Args:
            candidate_ast: Candidate answer node
            query_ast: Query AST
            doc_ast: Document AST

        Returns:
            Proximity score (0.0-1.0, higher is better)
        """
        candidate_position = self._get_word_position(candidate_ast, doc_ast)
        if candidate_position is None:
            return 0.5  # Couldn't find position, use neutral score

        # Extract query roots (excluding question words)
        query_roots = []
        for root in self._extract_roots(query_ast):
            # Skip correlatives (kiu, kio, etc.)
            if root not in {'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel', 'kiom', 'kies'}:
                query_roots.append(root)

        if not query_roots:
            return 0.5  # No content roots in query

        # Find distances to each query root
        distances = []
        for root in query_roots:
            root_positions = self._find_root_positions(root, doc_ast)
            if root_positions:
                # Use minimum distance to this root
                min_dist = min(abs(candidate_position - pos) for pos in root_positions)
                distances.append(min_dist)

        if not distances:
            return 0.3  # Query roots not found in document

        # Average distance
        avg_distance = sum(distances) / len(distances)

        # Convert to score: closer = higher score
        # Distance 0 → score 1.0
        # Distance 5 → score 0.167
        # Distance 10 → score 0.091
        proximity_score = 1.0 / (1 + avg_distance)

        return proximity_score
