"""
AST Pattern Matcher for flexible structural question-answer matching.

Matches question ASTs against corpus sentence ASTs with support for:
- Slot-based matching (SUBJ/VERB/OBJ)
- Structural transformations (passive, appositive, fragments)
- Synonym expansion using semantic relations
- Focus-aware matching based on question type

This is a DETERMINISTIC matcher - no learned parameters, just rule-based
pattern matching using AST structure.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of pattern matching."""
    score: float
    matched_slots: Set[str]
    transformations: List[str]  # Applied transformations (e.g., "passive", "appositive")
    explanation: str


class ASTPatternMatcher:
    """
    Deterministic AST pattern matcher.

    Matches question ASTs against sentence ASTs using structural patterns
    and semantic relations.
    """

    def __init__(
        self,
        synonym_db: Optional[Dict[str, Set[str]]] = None,
        antonym_db: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Initialize pattern matcher.

        Args:
            synonym_db: Root synonym dictionary (root → set of synonym roots)
            antonym_db: Root antonym dictionary (root → set of antonym roots)
        """
        self.synonym_db = synonym_db or {}
        self.antonym_db = antonym_db or {}

    def match(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        target_slots: List[str],
        entity_type: str,
    ) -> MatchResult:
        """
        Match question AST against document AST.

        Args:
            query_ast: Parsed question AST
            doc_ast: Parsed document sentence AST
            target_slots: Priority slots from question classifier
            entity_type: Expected entity type from question classifier

        Returns:
            MatchResult with score and explanation
        """
        score = 0.0
        matched_slots = set()
        transformations = []
        explanations = []

        # Strategy 1: Direct slot matching
        direct_score, direct_slots = self._match_direct_slots(
            query_ast, doc_ast, target_slots
        )
        score += direct_score
        matched_slots.update(direct_slots)
        if direct_slots:
            explanations.append(f"Direct match on slots: {', '.join(direct_slots)}")

        # Strategy 2: Synonym expansion
        if self.synonym_db:
            synonym_score, synonym_slots = self._match_with_synonyms(
                query_ast, doc_ast, target_slots
            )
            score += synonym_score
            matched_slots.update(synonym_slots)
            if synonym_slots:
                explanations.append(f"Synonym match on slots: {', '.join(synonym_slots)}")

        # Strategy 3: Passive transformation
        passive_score, passive_slots = self._match_passive_transform(
            query_ast, doc_ast
        )
        if passive_score > 0:
            score += passive_score
            matched_slots.update(passive_slots)
            transformations.append("passive")
            explanations.append("Passive voice transformation detected")

        # Strategy 4: Appositive/fragment matching
        appositive_score, appositive_slots = self._match_appositive(
            query_ast, doc_ast, entity_type
        )
        if appositive_score > 0:
            score += appositive_score
            matched_slots.update(appositive_slots)
            transformations.append("appositive")
            explanations.append("Appositive/fragment match")

        explanation = "; ".join(explanations) if explanations else "No match"

        return MatchResult(
            score=score,
            matched_slots=matched_slots,
            transformations=transformations,
            explanation=explanation,
        )

    def _match_direct_slots(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        target_slots: List[str],
    ) -> Tuple[float, Set[str]]:
        """
        Match slots directly (exact root matching).

        Returns:
            (score, matched_slots)
        """
        score = 0.0
        matched_slots = set()

        # Map English slot names to Esperanto AST field names
        slot_mapping = {
            'SUBJ': 'subjekto',
            'VERB': 'verbo',
            'OBJ': 'objekto',
        }

        # Weight priority slots higher
        slot_weights = {
            slot: 1.0 if slot in target_slots else 0.5
            for slot in ['SUBJ', 'VERB', 'OBJ']
        }

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            ast_field = slot_mapping[slot]
            query_node = query_ast.get(ast_field)
            doc_node = doc_ast.get(ast_field)

            if query_node and doc_node:
                # Extract roots from both nodes
                query_roots = self._extract_roots(query_node)
                doc_roots = self._extract_roots(doc_node)

                # Check for overlap
                overlap = query_roots & doc_roots

                if overlap:
                    # Score based on Jaccard similarity
                    union = query_roots | doc_roots
                    jaccard = len(overlap) / len(union) if union else 0.0
                    score += slot_weights[slot] * jaccard
                    matched_slots.add(slot)

        return score, matched_slots

    def _match_with_synonyms(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        target_slots: List[str],
    ) -> Tuple[float, Set[str]]:
        """
        Match slots with synonym expansion.

        Returns:
            (score, matched_slots)
        """
        score = 0.0
        matched_slots = set()

        # Map English slot names to Esperanto AST field names
        slot_mapping = {
            'SUBJ': 'subjekto',
            'VERB': 'verbo',
            'OBJ': 'objekto',
        }

        # Weight priority slots higher
        slot_weights = {
            slot: 0.8 if slot in target_slots else 0.4  # Lower than direct match
            for slot in ['SUBJ', 'VERB', 'OBJ']
        }

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            ast_field = slot_mapping[slot]
            query_node = query_ast.get(ast_field)
            doc_node = doc_ast.get(ast_field)

            if query_node and doc_node:
                # Extract roots
                query_roots = self._extract_roots(query_node)
                doc_roots = self._extract_roots(doc_node)

                # Expand with synonyms
                query_expanded = self._expand_synonyms(query_roots)
                doc_expanded = self._expand_synonyms(doc_roots)

                # Check for overlap in expanded sets
                overlap = query_expanded & doc_expanded

                if overlap:
                    # Score based on Jaccard similarity (discounted for synonym match)
                    union = query_expanded | doc_expanded
                    jaccard = len(overlap) / len(union) if union else 0.0
                    score += slot_weights[slot] * jaccard
                    matched_slots.add(slot)

        return score, matched_slots

    def _match_passive_transform(
        self,
        query_ast: Dict,
        doc_ast: Dict,
    ) -> Tuple[float, Set[str]]:
        """
        Match with passive voice transformation.

        Detects if question is active and document is passive (or vice versa),
        and matches SUBJ ↔ OBJ accordingly.

        Returns:
            (score, matched_slots)
        """
        query_verb = query_ast.get('verbo')
        doc_verb = doc_ast.get('verbo')

        if not query_verb or not doc_verb:
            return 0.0, set()

        # Check if one is passive and one is active
        # Use full passive construction check (esti + passive participle)
        query_passive = self._has_passive_construction(query_ast)
        doc_passive = self._has_passive_construction(doc_ast)

        if query_passive == doc_passive:
            # Same voice, not a transformation
            return 0.0, set()

        score = 0.0
        matched_slots = set()

        # Extract verb roots - for passive, get root from participle
        if query_passive:
            query_verb_roots = self._get_passive_participle_roots(query_ast)
        else:
            query_verb_roots = self._extract_roots(query_verb)

        if doc_passive:
            doc_verb_roots = self._get_passive_participle_roots(doc_ast)
        else:
            doc_verb_roots = self._extract_roots(doc_verb)

        # Expand with synonyms if available
        query_verb_expanded = self._expand_synonyms(query_verb_roots)
        doc_verb_expanded = self._expand_synonyms(doc_verb_roots)

        # Match verb roots (even if one is participle, one is active)
        # Try both direct match and synonym match
        if query_verb_roots & doc_verb_roots or query_verb_expanded & doc_verb_expanded:
            score += 0.5
            matched_slots.add('VERB')

            # In passive transformation: SUBJ ↔ OBJ swap
            if query_passive and not doc_passive:
                # Query passive, doc active: query.SUBJ ↔ doc.OBJ
                query_subj = query_ast.get('subjekto')
                doc_obj = doc_ast.get('objekto')

                if query_subj and doc_obj:
                    query_roots = self._extract_roots(query_subj)
                    doc_roots = self._extract_roots(doc_obj)

                    if query_roots & doc_roots:
                        score += 0.5
                        matched_slots.add('SUBJ')
                        matched_slots.add('OBJ')

            elif not query_passive and doc_passive:
                # Query active, doc passive: query.OBJ ↔ doc.SUBJ
                query_obj = query_ast.get('objekto')
                doc_subj = doc_ast.get('subjekto')

                if query_obj and doc_subj:
                    query_roots = self._extract_roots(query_obj)
                    doc_roots = self._extract_roots(doc_subj)

                    if query_roots & doc_roots:
                        score += 0.5
                        matched_slots.add('SUBJ')
                        matched_slots.add('OBJ')

        return score, matched_slots

    def _match_appositive(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        entity_type: str,
    ) -> Tuple[float, Set[str]]:
        """
        Match appositive constructions and sentence fragments.

        Example:
        Q: "Kiu estas Zamenhof?"
        A: "Zamenhof, la kreinto de Esperanto, naskiĝis en 1859."

        Returns:
            (score, matched_slots)
        """
        # For DEFINITION questions ("Kio estas X?"), look for:
        # - X in subject or object position
        # - Definition in appositive or predicate

        if entity_type != 'definition':
            return 0.0, set()

        score = 0.0
        matched_slots = set()

        # Extract the entity being defined from query
        query_obj = query_ast.get('objekto')
        if not query_obj:
            return 0.0, set()

        query_entity_roots = self._extract_roots(query_obj)

        # Check if entity appears in document SUBJ or OBJ
        doc_subj = doc_ast.get('subjekto')
        doc_obj = doc_ast.get('objekto')

        entity_found = False

        if doc_subj:
            doc_subj_roots = self._extract_roots(doc_subj)
            if query_entity_roots & doc_subj_roots:
                entity_found = True
                matched_slots.add('SUBJ')
                score += 0.5

        if doc_obj:
            doc_obj_roots = self._extract_roots(doc_obj)
            if query_entity_roots & doc_obj_roots:
                entity_found = True
                matched_slots.add('OBJ')
                score += 0.5

        # If entity found, this is a potential definition
        if entity_found:
            score += 0.5  # Bonus for finding the entity being defined

        return score, matched_slots

    def _extract_roots(self, node: Dict) -> Set[str]:
        """
        Extract all content word roots from an AST node.

        Recursively traverses vortgrupo structure and collects roots.
        Filters out function words (pronouns, articles, etc.).
        """
        roots = set()

        if node.get('tipo') == 'vorto':
            # Extract root from word
            root = node.get('radiko', '').lower()
            vortspeco = node.get('vortspeco', '')

            # Filter out function words
            if root and vortspeco not in ['pronomo', 'artikolo', 'prepozicio', 'konjunkcio']:
                # Skip ki- question words
                if not root.startswith('ki'):
                    roots.add(root)

        elif node.get('tipo') == 'vortgrupo':
            # Recursively extract from word group
            if node.get('kerno'):
                roots.update(self._extract_roots(node['kerno']))

            if node.get('priskriboj'):
                for modifier in node['priskriboj']:
                    roots.update(self._extract_roots(modifier))

        return roots

    def _expand_synonyms(self, roots: Set[str]) -> Set[str]:
        """Expand a set of roots with their synonyms."""
        expanded = set(roots)

        for root in roots:
            if root in self.synonym_db:
                expanded.update(self.synonym_db[root])

        return expanded

    def _is_passive(self, verb_node: Dict) -> bool:
        """
        Check if a verb is in passive voice.

        Note: Esperanto passive voice is expressed with "esti + passive participle"
        (e.g., "estas vidata"), which the parser treats as:
        - VERB: "estas"
        - Subject modifier: "vidata" (passive participle)

        So we need to check if the verb is "esti" AND there's a passive participle
        in the sentence structure.
        """
        if verb_node.get('tipo') != 'vorto':
            return False

        # Check for passive participle suffix "-at-", "-it-"
        sufiksoj = verb_node.get('sufiksoj', [])

        # Passive if has "it" or "at" suffix (passive participles)
        # or if marked as passive participle
        if 'it' in sufiksoj or 'at' in sufiksoj:
            return True

        if verb_node.get('participo_voĉo') == 'pasiva':
            return True

        return False

    def _has_passive_construction(self, ast: Dict) -> bool:
        """
        Check if sentence has passive construction (esti + passive participle).

        Looks for passive participle in subject modifiers (where parser puts them).
        """
        # Check if verb is "esti"
        verb = ast.get('verbo')
        if not verb or verb.get('radiko') != 'est':
            return False

        # Check for passive participle in subject modifiers
        subj = ast.get('subjekto')
        if not subj or subj.get('tipo') != 'vortgrupo':
            return False

        modifiers = subj.get('priskriboj', [])
        for mod in modifiers:
            if mod.get('tipo') == 'vorto':
                if mod.get('participo_voĉo') == 'pasiva':
                    return True
                sufiksoj = mod.get('sufiksoj', [])
                if 'at' in sufiksoj or 'it' in sufiksoj:
                    return True

        return False

    def _get_passive_participle_roots(self, ast: Dict) -> Set[str]:
        """
        Extract verb roots from passive participle in passive construction.

        In "Esperanto estis fondita", extracts "fond" from the participle "fondita".
        """
        roots = set()

        subj = ast.get('subjekto')
        if not subj or subj.get('tipo') != 'vortgrupo':
            return roots

        # Check modifiers for passive participle
        modifiers = subj.get('priskriboj', [])
        for mod in modifiers:
            if mod.get('tipo') == 'vorto':
                # Check if this is a passive participle
                if mod.get('participo_voĉo') == 'pasiva':
                    root = mod.get('radiko', '').lower()
                    if root:
                        roots.add(root)
                else:
                    # Also check for -it/-at suffix
                    sufiksoj = mod.get('sufiksoj', [])
                    if 'at' in sufiksoj or 'it' in sufiksoj:
                        root = mod.get('radiko', '').lower()
                        if root:
                            roots.add(root)

        return roots
