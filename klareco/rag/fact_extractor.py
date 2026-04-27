#!/usr/bin/env python3
"""
Fact Extractor - DEPRECATED

Use klareco.rag.unified_extractor.UnifiedASTExtractor instead.
Fact and RelationType are re-exported here for backwards compatibility.
"""

import warnings
warnings.warn(
    "klareco.rag.fact_extractor is deprecated. "
    "Use klareco.rag.unified_extractor.UnifiedASTExtractor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from klareco.rag.unified_extractor import Fact, RelationType

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RelationType(Enum):
    """Semantic relation types mapped from verb roots."""
    IS_A = "IS-A"              # estas → IS-A (category membership)
    HAS = "HAS"                # havas → HAS (property/possession)
    CREATED_BY = "CREATED-BY"  # kreis → CREATED-BY (creation)
    LOCATED_AT = "LOCATED-AT"  # loĝas, troviĝas → LOCATED-AT
    BORN = "BORN"              # naskiĝis → BORN
    DIED = "DIED"              # mortis → DIED
    PUBLISHED = "PUBLISHED"    # publikigis → PUBLISHED
    USED_BY = "USED-BY"        # uzas → USED-BY
    FOUNDED = "FOUNDED"        # fondis → FOUNDED
    ACTION = "ACTION"          # Generic action (fallback)


# Verb root → relation type mapping
VERB_TO_RELATION = {
    'est': RelationType.IS_A,
    'hav': RelationType.HAS,
    'kre': RelationType.CREATED_BY,
    'lok': RelationType.LOCATED_AT,
    'trov': RelationType.LOCATED_AT,
    'nask': RelationType.BORN,
    'mort': RelationType.DIED,
    'publik': RelationType.PUBLISHED,
    'uz': RelationType.USED_BY,
    'fond': RelationType.FOUNDED,
}


@dataclass
class Fact:
    """Structured semantic fact extracted from AST."""
    entity: str                          # Main entity (usually from subjekto/objekto)
    relation: RelationType               # Semantic relation type
    arguments: Dict[str, Any] = field(default_factory=dict)   # Required arguments
    modifiers: Dict[str, Any] = field(default_factory=dict)   # Optional modifiers
    source_sentence: Optional[str] = None     # Original sentence text
    source_ast: Optional[Dict] = None         # Original AST
    confidence: float = 1.0                   # Extraction confidence

    # Linguistic annotations (for importance scoring)
    entity_is_proper_noun: bool = False       # Is entity a proper noun?
    entity_capitalized_form: Optional[str] = None  # Capitalized form (e.g., "Fundamento")

    # Citation tracking (Issue #674)
    citation_id: Optional[int] = None         # Citation number [1], [2], etc.
    sentence_id: Optional[str] = None         # Database sentence ID
    doc_title: Optional[str] = None           # Article/document title
    doc_metadata: Optional[Dict] = None       # Full document metadata

    def __str__(self):
        args_str = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        mods_str = ", ".join(f"{k}={v}" for k, v in self.modifiers.items()) if self.modifiers else "none"
        return f"Fact({self.entity}, {self.relation.value}, args=[{args_str}], mods=[{mods_str}])"


class FactExtractor:
    """Extract semantic facts from ASTs using deterministic rules."""

    def __init__(self):
        self.verb_to_relation = VERB_TO_RELATION

    def extract(self, ast: Dict, source_sentence: Optional[str] = None) -> List[Fact]:
        """
        Extract all facts from an AST.

        Extracts from:
        1. Main verb clause (existing)
        2. Participial noun phrases ("kreinto de X", "fondinto de Y")
        3. Nested/subordinate clauses (TODO)

        Args:
            ast: Parsed AST (from klareco.parser.parse)
            source_sentence: Original sentence text (optional)

        Returns:
            List of extracted Fact objects
        """
        facts = []

        if not ast or not isinstance(ast, dict):
            return facts

        if ast.get('tipo') == 'frazo':
            # 1. Extract from main verb clause
            fact = self._extract_from_frazo(ast, source_sentence)
            if fact:
                facts.append(fact)

            # 2. Extract from participial noun phrases
            participial_facts = self._extract_from_participial_nouns(ast, source_sentence)
            facts.extend(participial_facts)

            # 3. Extract from nested/subordinate clauses
            nested_facts = self._extract_from_nested_clauses(ast, source_sentence)
            facts.extend(nested_facts)

        return facts

    def _extract_from_frazo(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract fact from frazo (sentence) AST node."""
        # Get verb to determine relation type
        verbo = frazo.get('verbo')
        if not verbo:
            return None

        verb_root = self._get_root(verbo)
        if not verb_root:
            return None

        # Map verb → relation type
        relation = self.verb_to_relation.get(verb_root, RelationType.ACTION)

        # Extract entity and arguments based on relation type
        if relation == RelationType.IS_A:
            return self._extract_is_a(frazo, source_sentence)
        elif relation == RelationType.CREATED_BY:
            return self._extract_created_by(frazo, source_sentence)
        elif relation == RelationType.HAS:
            return self._extract_has(frazo, source_sentence)
        elif relation == RelationType.LOCATED_AT:
            return self._extract_located_at(frazo, source_sentence)
        elif relation == RelationType.BORN:
            return self._extract_born(frazo, source_sentence)
        elif relation == RelationType.PUBLISHED:
            return self._extract_published(frazo, source_sentence)
        else:
            # Generic action extraction
            return self._extract_action(frazo, source_sentence, relation)

    def _extract_is_a(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract IS-A fact: 'X estas Y' → X IS-A Y"""
        subjekto = frazo.get('subjekto')
        objekto = frazo.get('objekto')

        if not subjekto:
            return None

        # Get entity with proper noun information
        entity_info = self._get_entity_info(subjekto)
        if not entity_info:
            return None

        entity, is_proper, cap_form = entity_info

        # For copula "estas", the predicate nominative can be in objekto OR aliaj
        category = None
        if objekto:
            category = self._get_entity_name(objekto)
        else:
            # Check aliaj for nominative substantivo (predicate nominative)
            aliaj = frazo.get('aliaj', [])
            for alia in aliaj:
                if isinstance(alia, dict):
                    if (alia.get('vortspeco') == 'substantivo' and
                        alia.get('kazo') == 'nominativo'):
                        category = self._get_entity_name(alia)
                        break

        if not entity or not category:
            return None

        # Extract modifiers from aliaj (adjectives, adverbs, etc.)
        # But exclude the category we just found
        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        return Fact(
            entity=entity,
            relation=RelationType.IS_A,
            arguments={'tipo': category},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0,
            entity_is_proper_noun=is_proper,
            entity_capitalized_form=cap_form
        )

    def _extract_created_by(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract CREATED-BY fact: 'X kreis Y' → Y CREATED-BY X"""
        subjekto = frazo.get('subjekto')
        objekto = frazo.get('objekto')

        if not subjekto or not objekto:
            return None

        agent = self._get_entity_name(subjekto)
        entity = self._get_entity_name(objekto)

        if not agent or not entity:
            return None

        # Extract temporal modifiers
        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        return Fact(
            entity=entity,
            relation=RelationType.CREATED_BY,
            arguments={'aganto': agent},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0
        )

    def _extract_has(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract HAS fact: 'X havas Y' → X HAS Y"""
        subjekto = frazo.get('subjekto')
        objekto = frazo.get('objekto')

        if not subjekto or not objekto:
            return None

        entity = self._get_entity_name(subjekto)
        property_val = self._get_entity_name(objekto)

        if not entity or not property_val:
            return None

        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        return Fact(
            entity=entity,
            relation=RelationType.HAS,
            arguments={'eco': property_val},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0
        )

    def _extract_located_at(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract LOCATED-AT fact: 'X loĝas en Y' → X LOCATED-AT Y"""
        subjekto = frazo.get('subjekto')

        if not subjekto:
            return None

        entity = self._get_entity_name(subjekto)

        # Location is often in aliaj with preposition 'en', 'ĉe', etc.
        location = None
        aliaj = frazo.get('aliaj', [])
        for alia in aliaj:
            if isinstance(alia, dict):
                prep = alia.get('prepozicio')
                if prep in ['en', 'ĉe', 'de']:
                    location = self._get_entity_name(alia)
                    break

        if not entity:
            return None

        modifiers = self._extract_modifiers(aliaj)

        return Fact(
            entity=entity,
            relation=RelationType.LOCATED_AT,
            arguments={'loko': location} if location else {},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=0.8 if location else 0.5
        )

    def _extract_born(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract BORN fact: 'X naskiĝis en Y' → X BORN (location=Y, time=...)"""
        subjekto = frazo.get('subjekto')

        if not subjekto:
            return None

        entity = self._get_entity_name(subjekto)

        # Extract birth location and time from aliaj
        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        return Fact(
            entity=entity,
            relation=RelationType.BORN,
            arguments={},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0
        )

    def _extract_published(self, frazo: Dict, source_sentence: Optional[str]) -> Optional[Fact]:
        """Extract PUBLISHED fact: 'X publikigis Y' → Y PUBLISHED (agent=X, time=...)"""
        subjekto = frazo.get('subjekto')
        objekto = frazo.get('objekto')

        if not objekto:
            return None

        entity = self._get_entity_name(objekto)
        agent = self._get_entity_name(subjekto) if subjekto else None

        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        arguments = {}
        if agent:
            arguments['aganto'] = agent

        return Fact(
            entity=entity,
            relation=RelationType.PUBLISHED,
            arguments=arguments,
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0
        )

    def _extract_action(self, frazo: Dict, source_sentence: Optional[str],
                       relation: RelationType) -> Optional[Fact]:
        """Extract generic action fact."""
        subjekto = frazo.get('subjekto')
        objekto = frazo.get('objekto')

        if not subjekto:
            return None

        entity = self._get_entity_name(subjekto)

        arguments = {}
        if objekto:
            obj_name = self._get_entity_name(objekto)
            if obj_name:
                arguments['objekto'] = obj_name

        modifiers = self._extract_modifiers(frazo.get('aliaj', []))

        return Fact(
            entity=entity,
            relation=relation,
            arguments=arguments,
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=0.7  # Lower confidence for generic actions
        )

    def _extract_from_participial_nouns(self, frazo: Dict, source_sentence: Optional[str]) -> List[Fact]:
        """
        Extract facts from participial noun phrases.

        Patterns:
        - "La kreinto de Esperanto" → CREATED-BY fact
        - "Zamenhof, la kreinto-iniciatinto de Esperanto" → CREATED-BY fact
        - "La fondinto de la asocio" → FOUNDED fact

        Participial nouns are marked with sufiksoj containing "int" (past),
        "ant" (present), or "ont" (future).
        """
        facts = []

        # Check subjekto, objekto, and aliaj for participial nouns
        nodes_to_check = []

        subjekto = frazo.get('subjekto')
        if subjekto:
            nodes_to_check.append(subjekto)

        objekto = frazo.get('objekto')
        if objekto:
            nodes_to_check.append(objekto)

        aliaj = frazo.get('aliaj', [])
        nodes_to_check.extend(aliaj)

        # Process each node
        for node in nodes_to_check:
            if not isinstance(node, dict):
                continue

            # Get the core word (handle vortgrupo)
            core_node = node
            if node.get('tipo') == 'vortgrupo':
                core_node = node.get('kerno', {})

            # Check if it's a participial noun
            if not self._is_participial_noun(core_node):
                continue

            # Extract the verb root
            verb_root = core_node.get('radiko', '')

            # Check if it's a compound word with participial component
            kunmetajhoj = core_node.get('kunmetajhoj', [])
            if kunmetajhoj:
                # Check each component for participial markers
                for component in kunmetajhoj:
                    if self._is_participial_noun(component):
                        verb_root = component.get('radiko', '')
                        break

            if not verb_root:
                continue

            # Map verb root to relation type
            relation = self.verb_to_relation.get(verb_root.lower(), RelationType.ACTION)

            # Find the object of the action (after "de" preposition)
            # Pattern: "kreinto de Esperanto" → entity = "esperant"
            entity = self._find_prepositional_object(aliaj, 'de')

            # Find the agent (often proper noun in apposition or subjekto)
            agent = None

            # Check for proper nouns in aliaj (appositive pattern)
            # Collect all consecutive capitalized words (e.g., "Ludoviko Lazaro Zamenhof")
            proper_nouns = []
            for alia in aliaj:
                if isinstance(alia, dict):
                    plena_vorto = alia.get('plena_vorto', '')
                    vortspeco = alia.get('vortspeco', '')

                    # Skip the entity we already found (Esperanto)
                    if entity and alia.get('radiko', '').lower() == entity.lower():
                        continue

                    # Collect capitalized words (proper nouns)
                    if plena_vorto and plena_vorto[0].isupper():
                        proper_nouns.append(alia.get('radiko', '').lower())
                    elif proper_nouns:
                        # Stop collecting when we hit a non-capitalized word
                        # (unless it's a preposition/conjunction)
                        if vortspeco not in ['prepozicio', 'konjunkcio']:
                            break

            # Take the last proper noun as the agent (e.g., "Zamenhof" from "Ludoviko Lazaro Zamenhof")
            if proper_nouns:
                agent = proper_nouns[-1]

            # If no entity found, might be reversed pattern
            # "Zamenhof estas la kreinto" → entity from subjekto
            if not entity:
                if subjekto and self._is_participial_noun(core_node):
                    # Pattern: predicate nominative with participial noun
                    # Use subjekto as agent
                    continue  # Skip this pattern for now

            if not entity:
                continue

            # Build fact based on relation type
            arguments = {}
            if relation in [RelationType.CREATED_BY, RelationType.FOUNDED, RelationType.PUBLISHED]:
                if agent:
                    arguments['aganto'] = agent

            modifiers = self._extract_modifiers(aliaj)

            fact = Fact(
                entity=entity,
                relation=relation,
                arguments=arguments,
                modifiers=modifiers,
                source_sentence=source_sentence,
                source_ast=frazo,
                confidence=0.9  # High confidence for participial patterns
            )
            facts.append(fact)

        return facts

    def _is_participial_noun(self, node: Dict) -> bool:
        """Check if node is a participial noun (-into, -anto, -onto)."""
        if not isinstance(node, dict):
            return False

        sufiksoj = node.get('sufiksoj', [])
        if not sufiksoj:
            return False

        # Check for participial suffixes
        # -int- (past active), -ant- (present active), -ont- (future active)
        participial_suffixes = ['int', 'ant', 'ont', 'it', 'at', 'ot']

        for suffix in sufiksoj:
            if suffix in participial_suffixes:
                return True

        return False

    def _find_prepositional_object(self, aliaj: List[Dict], preposition: str) -> Optional[str]:
        """
        Find the object of a prepositional phrase in aliaj.

        Pattern: "de Esperanto" → "esperant"
        """
        prep_found = False

        for alia in aliaj:
            if not isinstance(alia, dict):
                continue

            # Check if it's the preposition we're looking for
            if alia.get('vortspeco') == 'prepozicio' and alia.get('radiko') == preposition:
                prep_found = True
                continue

            # If we found the prep, next substantivo or proper noun is the object
            if prep_found and alia.get('vortspeco') in ('substantivo', 'propra_nomo'):
                return alia.get('radiko', '')

        return None

    def _extract_from_nested_clauses(self, frazo: Dict, source_sentence: Optional[str]) -> List[Fact]:
        """
        Extract facts from nested/subordinate clauses.

        Patterns:
        - "...sub kiu la kuracisto publikigis..." → PUBLISHED fact
        - "...kiun Zamenhof kreis..." → CREATED-BY fact

        Nested clauses are marked by correlatives (kiu, kio, kie, kiam).
        Parser may place them in objekto or aliaj.
        """
        facts = []

        # Check rilata_subfrazo nodes in subjekto.priskriboj and objekto.priskriboj.
        # These are relative clauses now properly parsed by the deterministic handler.
        for slot_name in ('subjekto', 'objekto'):
            slot = frazo.get(slot_name)
            if not slot or not isinstance(slot, dict):
                continue
            antecedent = slot.get('kerno', {})
            antecedent_root = self._get_root(antecedent) if antecedent else None
            if not antecedent_root:
                continue
            for prisk in slot.get('priskriboj', []):
                if not isinstance(prisk, dict) or prisk.get('tipo') != 'rilata_subfrazo':
                    continue
                rf = self._extract_from_rilata_subfrazo(prisk, antecedent_root, source_sentence)
                if rf:
                    facts.append(rf)

        # Check if objekto is a correlative (relative clause pattern)
        objekto = frazo.get('objekto')
        if objekto and isinstance(objekto, dict):
            if objekto.get('tipo') == 'vortgrupo':
                kerno = objekto.get('kerno', {})
            else:
                kerno = objekto

            if kerno.get('vortspeco') == 'korelativo':
                # Pattern: "Subjekto [kiun Agent Verb]" - relative clause with correlative in objekto
                # The relative clause verb may be in aliaj when parser's main verbo is null.
                aliaj = frazo.get('aliaj', [])
                subjekto = frazo.get('subjekto')
                verbo = frazo.get('verbo')

                # Find verb: prefer main verbo, else scan aliaj for content verb
                verb_node = verbo
                if verb_node is None:
                    for alia in aliaj:
                        if isinstance(alia, dict) and alia.get('vortspeco') == 'verbo':
                            root = alia.get('radiko', '').lower()
                            if root != 'est':
                                verb_node = alia
                                break

                if verb_node:
                    verb_root = verb_node.get('radiko', '').lower()
                    relation = self.verb_to_relation.get(verb_root, RelationType.ACTION)

                    # Entity is the subject of the main clause
                    entity = None
                    if subjekto:
                        entity = self._get_entity_name(subjekto)

                    # Agent is first proper noun or substantivo in aliaj
                    agent = None
                    for alia in aliaj:
                        if isinstance(alia, dict):
                            plena = alia.get('plena_vorto', '')
                            if plena and plena[0].isupper():
                                agent = alia.get('radiko', '').lower()
                                break
                            elif alia.get('vortspeco') == 'substantivo' and not agent:
                                agent = alia.get('radiko', '').lower()
                                break

                    if entity and relation in [RelationType.CREATED_BY, RelationType.PUBLISHED, RelationType.FOUNDED]:
                        arguments = {}
                        if agent:
                            arguments['aganto'] = agent

                        modifiers = self._extract_modifiers(aliaj)

                        fact = Fact(
                            entity=entity,
                            relation=relation,
                            arguments=arguments,
                            modifiers=modifiers,
                            source_sentence=source_sentence,
                            source_ast=frazo,
                            confidence=0.8
                        )
                        facts.append(fact)

        # Also check aliaj for correlatives (other pattern)
        aliaj = frazo.get('aliaj', [])

        # Scan for relative pronouns/correlatives
        i = 0
        while i < len(aliaj):
            alia = aliaj[i]

            if not isinstance(alia, dict):
                i += 1
                continue

            # Check if it's a relative correlative
            if alia.get('vortspeco') == 'korelativo':
                # Found a correlative, look for verb in following elements
                verb_idx = self._find_verb_after_position(aliaj, i + 1)

                if verb_idx is not None:
                    # Extract subsequence from correlative to verb (and a bit after)
                    # Build a mini-clause structure
                    clause_elements = aliaj[i:min(verb_idx + 10, len(aliaj))]

                    # Try to construct a fact from this subsequence
                    nested_fact = self._extract_from_clause_subsequence(
                        clause_elements, verb_idx - i, source_sentence
                    )

                    if nested_fact:
                        facts.append(nested_fact)

                    # Skip past this clause
                    i = verb_idx + 5
                    continue

            i += 1

        return facts

    def _extract_from_rilata_subfrazo(
        self, rilata: Dict, antecedent_root: str, source_sentence: Optional[str]
    ) -> Optional[Fact]:
        """
        Extract a fact from a rilata_subfrazo node (relative clause).

        The antecedent noun (what the relative clause modifies) becomes the
        entity.  The subject of the relative clause becomes the agent.
        Only high-value semantic relations are extracted.
        """
        verbo = rilata.get('verbo')
        if not verbo:
            return None
        verb_root = self._get_root(verbo)
        if not verb_root:
            return None

        relation = self.verb_to_relation.get(verb_root, RelationType.ACTION)
        if relation not in {
            RelationType.CREATED_BY, RelationType.FOUNDED,
            RelationType.PUBLISHED, RelationType.BORN,
            RelationType.DIED, RelationType.LOCATED_AT,
        }:
            return None

        # Agent is the subject of the relative clause (if it is a content word)
        agent = None
        rilata_subj = rilata.get('subjekto')
        if rilata_subj and isinstance(rilata_subj, dict):
            kerno = rilata_subj.get('kerno', {})
            vs = kerno.get('vortspeco', '')
            radiko = kerno.get('radiko', '')
            if vs in ('propra_nomo', 'substantivo', 'pronomo') and radiko and radiko.lower() != 'kiu':
                agent = radiko.lower()

        if not agent:
            return None

        return Fact(
            entity=antecedent_root,
            relation=relation,
            arguments={'aganto': agent},
            modifiers=self._extract_modifiers(rilata.get('aliaj', [])),
            source_sentence=source_sentence,
            source_ast=rilata,
            confidence=0.8,
        )

    def _find_verb_after_position(self, aliaj: List[Dict], start_pos: int) -> Optional[int]:
        """Find the next verb in aliaj starting from start_pos."""
        for i in range(start_pos, min(start_pos + 15, len(aliaj))):
            if isinstance(aliaj[i], dict):
                if aliaj[i].get('vortspeco') == 'verbo':
                    return i
        return None

    def _extract_from_clause_subsequence(
        self, elements: List[Dict], verb_offset: int, source_sentence: Optional[str]
    ) -> Optional[Fact]:
        """
        Extract fact from a subsequence of clause elements.

        Elements is a slice from aliaj containing a relative clause.
        verb_offset is the index of the verb within this slice.
        """
        if verb_offset >= len(elements):
            return None

        verb = elements[verb_offset]
        verb_root = verb.get('radiko', '').lower()

        # Map verb to relation type
        relation = self.verb_to_relation.get(verb_root, RelationType.ACTION)

        # Find agent (subject of nested clause) - usually before the verb
        agent = None
        for i in range(max(0, verb_offset - 5), verb_offset):
            elem = elements[i]
            if isinstance(elem, dict):
                # Look for proper nouns or substantivo
                plena = elem.get('plena_vorto', '')
                if plena and plena[0].isupper():
                    agent = elem.get('radiko', '').lower()
                elif elem.get('vortspeco') == 'substantivo' and not agent:
                    agent = elem.get('radiko', '').lower()

        # Find object (usually after the verb)
        entity = None
        for i in range(verb_offset + 1, min(verb_offset + 8, len(elements))):
            elem = elements[i]
            if isinstance(elem, dict):
                if elem.get('vortspeco') == 'substantivo':
                    entity = elem.get('radiko', '')
                    break

        # Extract temporal modifiers
        modifiers = {}
        for elem in elements:
            if isinstance(elem, dict):
                if elem.get('vortspeco') == 'numero':
                    root = elem.get('radiko', '')
                    if len(root) == 4 and root.isdigit():
                        modifiers['tempo'] = root

        # Build fact based on relation type
        if relation in [RelationType.CREATED_BY, RelationType.PUBLISHED, RelationType.FOUNDED]:
            if not entity:
                return None

            arguments = {}
            if agent:
                arguments['aganto'] = agent

            return Fact(
                entity=entity,
                relation=relation,
                arguments=arguments,
                modifiers=modifiers,
                source_sentence=source_sentence,
                source_ast=None,  # Subsequence, not full AST
                confidence=0.8  # Slightly lower confidence for nested extraction
            )

        return None

    def _extract_modifiers(self, aliaj: List[Dict]) -> Dict[str, Any]:
        """Extract modifiers from aliaj (time, place, manner, etc.)."""
        modifiers = {}

        for alia in aliaj:
            if not isinstance(alia, dict):
                continue

            vortspeco = alia.get('vortspeco', '')
            root = alia.get('radiko', '')

            # Time modifiers - prepositions like "en" with years
            if vortspeco == 'prepozicio' and root == 'en':
                # Next item in aliaj might be the year
                continue  # Will be picked up below
            elif vortspeco == 'numero':
                # Check if it's a year (4 digits)
                if len(root) == 4 and root.isdigit():
                    modifiers['tempo'] = root
                else:
                    modifiers['kvanto'] = root

            # Quantity modifiers
            elif 'milion' in root or vortspeco == 'nombro':
                modifiers['kvanto'] = self._get_entity_name(alia)

            # Manner modifiers (adverbs)
            elif vortspeco == 'adverbo':
                modifiers['maniero'] = self._get_entity_name(alia)

        return modifiers

    def _get_root(self, node: Dict) -> Optional[str]:
        """Get root from AST node."""
        if not isinstance(node, dict):
            return None

        if node.get('tipo') == 'vorto':
            return node.get('radiko', '').lower()
        elif node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            return self._get_root(kerno) if kerno else None

        return None

    def _get_entity_info(self, node: Dict) -> Optional[tuple]:
        """
        Get entity name and proper noun status from AST node.

        Returns:
            (entity_name: str, is_proper_noun: bool, capitalized_form: str) or None
        """
        if not isinstance(node, dict):
            return None

        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            plena_vorto = node.get('plena_vorto', root)
            vortspeco = node.get('vortspeco', '')

            # Check if proper noun (fix: use "propra_nomo" not "propranomo")
            is_proper = (vortspeco == 'propra_nomo' or
                        (root and root[0].isupper()) or
                        (plena_vorto and plena_vorto[0].isupper()))

            if is_proper:
                # Normalize to lowercase for lookup; capitalized_form preserves display form
                return (root.lower(), True, plena_vorto or root)
            else:
                # Lowercase for common nouns
                return (root.lower(), False, None)

        elif node.get('tipo') == 'vortgrupo':
            # For word groups, get the head (kerno)
            kerno = node.get('kerno')
            if kerno:
                return self._get_entity_info(kerno)

        return None

    def _get_entity_name(self, node: Dict) -> Optional[str]:
        """Get entity name from AST node (recursively for vortgrupo)."""
        info = self._get_entity_info(node)
        return info[0] if info else None

    def _get_entity_name_legacy(self, node: Dict) -> Optional[str]:
        """Legacy method - kept for backwards compatibility."""
        if not isinstance(node, dict):
            return None

        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            # Capitalize proper nouns
            vortspeco = node.get('vortspeco', '')
            if vortspeco == 'propra_nomo' or (root and root[0].isupper()):
                return root
            return root.lower()

        elif node.get('tipo') == 'vortgrupo':
            # For word groups, get the head (kerno)
            kerno = node.get('kerno')
            if kerno:
                return self._get_entity_name(kerno)
            # Fallback: try to reconstruct from parts
            priskriboj = node.get('priskriboj', [])
            if priskriboj and kerno:
                # Adjective + noun: "internacia planlingvo"
                adj = self._get_entity_name(priskriboj[0]) if priskriboj else None
                noun = self._get_entity_name(kerno)
                if adj and noun:
                    return f"{adj} {noun}"
                return noun

        return None
