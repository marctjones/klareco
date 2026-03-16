#!/usr/bin/env python3
"""
Fact Extractor - Extract Semantic Facts from ASTs

Extracts structured semantic facts from Esperanto ASTs. Uses deterministic
mapping from verb roots to semantic relation types.

Design Philosophy:
- Deterministic extraction using verb→relation mappings
- Exploit Esperanto's compositional morphology
- Extract from AST structure, not text
- No learned components (100% rule-based)

Example:
    ast = parse("Zamenhof kreis Esperanton en 1887")
    fact = extract_facts(ast)[0]
    # → Fact(entity="Esperanto", relation="CREATED-BY",
    #         arguments={"agent": "Zamenhof"},
    #         modifiers={"time": "1887"})
"""

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

        entity = self._get_entity_name(subjekto)

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
            arguments={'type': category},
            modifiers=modifiers,
            source_sentence=source_sentence,
            source_ast=frazo,
            confidence=1.0
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
            arguments={'agent': agent},
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
            arguments={'property': property_val},
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
            arguments={'location': location} if location else {},
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
            arguments['agent'] = agent

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
                arguments['object'] = obj_name

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
                        proper_nouns.append(alia.get('radiko', ''))
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
                    arguments['agent'] = agent

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

            # If we found the prep, next substantivo is the object
            if prep_found and alia.get('vortspeco') == 'substantivo':
                return alia.get('radiko', '')

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
                    modifiers['time'] = root
                else:
                    modifiers['quantity'] = root

            # Quantity modifiers
            elif 'milion' in root or vortspeco == 'nombro':
                modifiers['quantity'] = self._get_entity_name(alia)

            # Manner modifiers (adverbs)
            elif vortspeco == 'adverbo':
                modifiers['manner'] = self._get_entity_name(alia)

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

    def _get_entity_name(self, node: Dict) -> Optional[str]:
        """Get entity name from AST node (recursively for vortgrupo)."""
        if not isinstance(node, dict):
            return None

        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            # Capitalize proper nouns
            vortspeco = node.get('vortspeco', '')
            if vortspeco == 'propranomo' or (root and root[0].isupper()):
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
