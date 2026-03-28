"""
Semantic Query API - Clean interface for 4-layer semantic ontology

Provides high-level API for querying the semantic ontology in Kuzu database.
Replaces all hardcoded gazetteers, synonym lists, and pattern matching.

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (4-layer ontology)
STAGE: Infrastructure

Usage:
    from klareco.ontology import SemanticQuery

    semantic = SemanticQuery(kuzu_conn)

    # Get all verbs in a class
    creation_verbs = semantic.get_verb_class_members('kreado-26')
    # Returns: ['fond', 'kre', 'produk', 'far']

    # Check entity types
    if semantic.is_person('kuracist'):
        ...

    # Get importance weights
    importance = semantic.get_schema_slot_importance('ĉefa_realigo', 'biografia')
    # Returns: 0.95

Last Updated: 2026-03-28
"""

from typing import List, Dict, Optional, Set
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class SemanticQuery:
    """
    High-level API for querying 4-layer semantic ontology.

    Replaces all hardcoded gazetteers, synonym lists, and pattern matching
    with clean queries to the Kuzu database semantic ontology.

    Includes caching for frequently-used queries.
    """

    def __init__(self, kuzu_conn):
        """
        Initialize SemanticQuery with Kuzu connection.

        Args:
            kuzu_conn: Active Kuzu database connection
        """
        self.conn = kuzu_conn
        self._cache = {}
        logger.info("SemanticQuery initialized with caching enabled")

    # ===================================================================
    # LAYER 1: LEXICAL SEMANTICS - Verb Classes
    # ===================================================================

    def get_verb_class_members(self, klaso_id: str) -> List[str]:
        """
        Get all roots in a verb class.

        Args:
            klaso_id: Verb class ID (e.g., 'kreado-26')

        Returns:
            List of root strings in this class

        Example:
            >>> semantic.get_verb_class_members('kreado-26')
            ['fond', 'kre', 'produk', 'far']
        """
        cache_key = f"verb_class:{klaso_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                RETURN r.radiko
            """)

            members = []
            while result.has_next():
                members.append(result.get_next()[0])

            # Also get exemplar roots from class definition
            result = self.conn.execute(f"""
                MATCH (v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                RETURN v.ekzemplaj_radikoj
            """)

            if result.has_next():
                exemplars = result.get_next()[0]
                if exemplars:
                    members.extend(exemplars)

            # Remove duplicates
            members = list(set(members))

            self._cache[cache_key] = members
            return members

        except Exception as e:
            logger.error(f"Error querying verb class {klaso_id}: {e}")
            return []

    def get_verb_class_id(self, verb_root: str) -> Optional[str]:
        """
        Get the verb class ID for a given root.

        Args:
            verb_root: Root string (e.g., 'fond')

        Returns:
            Verb class ID or None if not found

        Example:
            >>> semantic.get_verb_class_id('fond')
            'kreado-26'
        """
        cache_key = f"verb_root_class:{verb_root}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{verb_root}'}})-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
                RETURN v.klaso_id
            """)

            if result.has_next():
                klaso_id = result.get_next()[0]
                self._cache[cache_key] = klaso_id
                return klaso_id

            return None

        except Exception as e:
            logger.error(f"Error querying verb class for {verb_root}: {e}")
            return None

    def get_verb_synonyms(self, verb_root: str) -> List[str]:
        """
        Get all verbs in the same class as this verb (synonyms).

        Args:
            verb_root: Root string (e.g., 'fond')

        Returns:
            List of synonym roots including the input verb

        Example:
            >>> semantic.get_verb_synonyms('fond')
            ['fond', 'kre', 'produk', 'far']
        """
        # First get the class
        klaso_id = self.get_verb_class_id(verb_root)

        if not klaso_id:
            # Not in any class, return just the verb itself
            return [verb_root]

        # Get all members of that class
        return self.get_verb_class_members(klaso_id)

    def get_aspectual_class(self, verb_root: str) -> Optional[str]:
        """
        Get aspectual class of a verb (stato, aktiveco, plenumigo, atingaĵo).

        Args:
            verb_root: Root string (e.g., 'fond')

        Returns:
            Aspectual class ID or None if not found

        Example:
            >>> semantic.get_aspectual_class('fond')
            'plenumigo'
        """
        cache_key = f"aspectual:{verb_root}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{verb_root}'}})-[:HAVAS_ASPEKTAN_KLASON]->(a:AspektaKlaso)
                RETURN a.klaso_id
            """)

            if result.has_next():
                klaso = result.get_next()[0]
                self._cache[cache_key] = klaso
                return klaso

            return None

        except Exception as e:
            logger.error(f"Error querying aspectual class for {verb_root}: {e}")
            return None

    # ===================================================================
    # LAYER 1: LEXICAL SEMANTICS - Entity Types
    # ===================================================================

    def get_entity_type_members(self, tipo_id: str) -> List[str]:
        """
        Get all roots of a specific entity type.

        Args:
            tipo_id: Entity type ID ('persono', 'loko', 'tempo', etc.)

        Returns:
            List of root strings with this entity type

        Example:
            >>> semantic.get_entity_type_members('persono')
            ['hom', 'vir', 'kuracist', 'instruist', ...]
        """
        cache_key = f"entity_type:{tipo_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                RETURN r.radiko
            """)

            members = []
            while result.has_next():
                members.append(result.get_next()[0])

            # Also get exemplar roots from entity type definition
            result = self.conn.execute(f"""
                MATCH (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                RETURN e.ekzemploj
            """)

            if result.has_next():
                exemplars = result.get_next()[0]
                if exemplars:
                    members.extend(exemplars)

            # Remove duplicates
            members = list(set(members))

            self._cache[cache_key] = members
            return members

        except Exception as e:
            logger.error(f"Error querying entity type {tipo_id}: {e}")
            return []

    def get_entity_type(self, root: str) -> Optional[str]:
        """
        Get the entity type of a root.

        Args:
            root: Root string (e.g., 'kuracist')

        Returns:
            Entity type ID or None if not found

        Example:
            >>> semantic.get_entity_type('kuracist')
            'profesio'
        """
        cache_key = f"root_entity_type:{root}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
                RETURN e.tipo_id
            """)

            if result.has_next():
                tipo = result.get_next()[0]
                self._cache[cache_key] = tipo
                return tipo

            return None

        except Exception as e:
            logger.error(f"Error querying entity type for {root}: {e}")
            return None

    # ===================================================================
    # HELPER METHODS - Entity Type Checking
    # ===================================================================

    def is_person(self, root: str) -> bool:
        """
        Check if root is entity type 'persono'.

        Args:
            root: Root string

        Returns:
            True if root is a person entity

        Example:
            >>> semantic.is_person('kuracist')
            True
            >>> semantic.is_person('tabl')
            False
        """
        entity_type = self.get_entity_type(root)
        return entity_type == 'persono' or entity_type == 'profesio'

    def is_place(self, root: str) -> bool:
        """
        Check if root is entity type 'loko'.

        Args:
            root: Root string

        Returns:
            True if root is a place entity

        Example:
            >>> semantic.is_place('Varsovio')
            True
        """
        entity_type = self.get_entity_type(root)
        return entity_type == 'loko'

    def is_time(self, root: str) -> bool:
        """
        Check if root is entity type 'tempo'.

        Args:
            root: Root string

        Returns:
            True if root is a time entity

        Example:
            >>> semantic.is_time('jaro')
            True
        """
        entity_type = self.get_entity_type(root)
        return entity_type == 'tempo'

    def is_organization(self, root: str) -> bool:
        """Check if root is entity type 'organizaĵo'."""
        entity_type = self.get_entity_type(root)
        return entity_type == 'organizaĵo'

    def is_event(self, root: str) -> bool:
        """Check if root is entity type 'evento'."""
        entity_type = self.get_entity_type(root)
        return entity_type == 'evento'

    # ===================================================================
    # LAYER 1: LEXICAL SEMANTICS - Thematic Roles
    # ===================================================================

    def get_thematic_role_filler(self, ast: Dict, verb_root: str, role_id: str) -> Optional[str]:
        """
        Find filler of thematic role in AST.

        Args:
            ast: Parsed AST dictionary
            verb_root: Verb root string
            role_id: Thematic role ID ('aganto', 'paciento', 'loko', 'tempo', etc.)

        Returns:
            Root string filling this role, or None

        Example:
            >>> ast = parse("Zamenhof fondis Esperanton en 1887")
            >>> semantic.get_thematic_role_filler(ast, 'fond', 'aganto')
            'Zamenhof'
            >>> semantic.get_thematic_role_filler(ast, 'fond', 'tempo')
            '1887'
        """
        if not ast or not isinstance(ast, dict):
            return None

        # Map thematic roles to AST positions
        role_mapping = {
            'aganto': 'subjekto',      # Agent typically in subject
            'paciento': 'objekto',      # Patient typically in object
            'temo': 'objekto',          # Theme typically in object
            'spertanto': 'subjekto',    # Experiencer typically in subject
            'instrumento': 'aliaj',     # Instrument in modifiers
            'fonto': 'aliaj',           # Source in modifiers
            'celo': 'aliaj',            # Goal in modifiers
            'loko': 'aliaj',            # Location in modifiers
            'tempo': 'aliaj',           # Time in modifiers
        }

        ast_position = role_mapping.get(role_id)

        if not ast_position:
            return None

        # Extract from AST
        if ast_position in ['subjekto', 'objekto']:
            node = ast.get(ast_position)
            if node and isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    return node.get('radiko')
                elif node.get('tipo') == 'vortgrupo':
                    kerno = node.get('kerno')
                    if kerno and isinstance(kerno, dict):
                        return kerno.get('radiko')

        elif ast_position == 'aliaj':
            # Search in modifiers for matching semantic type
            aliaj = ast.get('aliaj', [])
            for modifier in aliaj:
                if isinstance(modifier, dict):
                    if modifier.get('tipo') == 'vorto':
                        root = modifier.get('radiko')
                        if role_id == 'loko' and self.is_place(root):
                            return root
                        elif role_id == 'tempo' and self.is_time(root):
                            return root

        return None

    # ===================================================================
    # LAYER 4: SCHEMA SEMANTICS
    # ===================================================================

    def get_schema_slot_importance(self, sloto_id: str, skemo_id: str = 'biografia') -> float:
        """
        Get importance weight for a schema slot.

        Args:
            sloto_id: Schema slot ID ('identigo', 'ĉefa_realigo', etc.)
            skemo_id: Schema ID (default: 'biografia')

        Returns:
            Importance weight (0.0-1.0)

        Example:
            >>> semantic.get_schema_slot_importance('ĉefa_realigo', 'biografia')
            0.95
        """
        cache_key = f"slot_importance:{skemo_id}:{sloto_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (sl:SkemaSloto {{sloto_id: '{sloto_id}'}})
                RETURN sl.graveco_pezo
            """)

            if result.has_next():
                importance = result.get_next()[0]
                self._cache[cache_key] = importance
                return importance

            # Default importance if not found
            return 0.5

        except Exception as e:
            logger.error(f"Error querying schema slot importance {sloto_id}: {e}")
            return 0.5

    def classify_fact_into_slot(self, ast: Dict, skemo_id: str = 'biografia') -> Dict:
        """
        Classify an AST fact into a schema slot.

        Uses semantic classes to match fact against schema slot patterns.

        Args:
            ast: Parsed AST dictionary
            skemo_id: Schema ID (default: 'biografia')

        Returns:
            Dict with 'slot_id' and 'importance' keys

        Example:
            >>> ast = parse("Zamenhof fondis Esperanton")
            >>> semantic.classify_fact_into_slot(ast, 'biografia')
            {'slot_id': 'ĉefa_realigo', 'importance': 0.95}
        """
        if not ast or not isinstance(ast, dict):
            return {'slot_id': 'alia', 'importance': 0.5}

        # Extract verb from AST
        verb_node = ast.get('verbo')
        if not verb_node or not isinstance(verb_node, dict):
            return {'slot_id': 'alia', 'importance': 0.5}

        verb_root = verb_node.get('radiko')
        if not verb_root:
            return {'slot_id': 'alia', 'importance': 0.5}

        # Get verb class and aspectual class
        verb_class = self.get_verb_class_id(verb_root)
        aspectual_class = self.get_aspectual_class(verb_root)

        # Match against biographical schema patterns
        if skemo_id == 'biografia':
            # ĉefa_realigo: creation verbs with accomplishment aspect
            if verb_class == 'kreado-26' and aspectual_class == 'plenumigo':
                return {'slot_id': 'ĉefa_realigo', 'importance': 0.95}

            # identigo: "estas" + person/profession object
            if verb_root == 'est':
                objekto = ast.get('objekto')
                if objekto and isinstance(objekto, dict):
                    obj_root = objekto.get('radiko') if objekto.get('tipo') == 'vorto' else \
                               objekto.get('kerno', {}).get('radiko')
                    if obj_root and (self.is_person(obj_root) or
                                     self.get_entity_type(obj_root) == 'profesio'):
                        return {'slot_id': 'identigo', 'importance': 1.0}

            # naskiĝo_morto: life events
            if verb_class == 'vivo-48':
                return {'slot_id': 'naskiĝo_morto', 'importance': 0.85}

            # profesio: professional activity
            if verb_class == 'profesio-50':
                return {'slot_id': 'profesio', 'importance': 0.80}

            # loko: contains location
            if ast.get('aliaj'):
                for modifier in ast.get('aliaj', []):
                    if isinstance(modifier, dict) and modifier.get('tipo') == 'vorto':
                        root = modifier.get('radiko')
                        if root and self.is_place(root):
                            return {'slot_id': 'loko', 'importance': 0.70}

        # Default: unclassified
        return {'slot_id': 'alia', 'importance': 0.5}

    def get_schema_slots(self, skemo_id: str = 'biografia') -> List[Dict]:
        """
        Get all schema slots for a schema with their importance weights.

        Args:
            skemo_id: Schema ID (default: 'biografia')

        Returns:
            List of dicts with 'slot_id', 'slot_nomo', 'graveco_pezo'
        """
        cache_key = f"schema_slots:{skemo_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self.conn.execute(f"""
                MATCH (s:EnhavaSkemo {{skemo_id: '{skemo_id}'}})-[:SKEMO_HAVAS_SLOTON]->(sl:SkemaSloto)
                RETURN sl.sloto_id, sl.sloto_nomo, sl.graveco_pezo
                ORDER BY sl.graveco_pezo DESC
            """)

            slots = []
            while result.has_next():
                row = result.get_next()
                slots.append({
                    'slot_id': row[0],
                    'slot_nomo': row[1],
                    'graveco_pezo': row[2]
                })

            self._cache[cache_key] = slots
            return slots

        except Exception as e:
            logger.error(f"Error querying schema slots for {skemo_id}: {e}")
            return []

    # ===================================================================
    # UTILITY METHODS
    # ===================================================================

    def clear_cache(self):
        """Clear the query cache."""
        self._cache.clear()
        logger.info("SemanticQuery cache cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache size and hit info
        """
        return {
            'size': len(self._cache),
            'keys': list(self._cache.keys())
        }
