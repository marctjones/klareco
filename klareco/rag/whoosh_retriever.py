"""
Whoosh-based Full-Text Search Retriever

Fast keyword-based sentence retrieval using Whoosh FTS index.
Replaces slow Kuzu scanning with efficient BM25-ranked search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import kuzu
from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

from klareco.parser import parse
from klareco.rag.kuzu_ast_reconstructor import KuzuASTReconstructor

logger = logging.getLogger(__name__)


def expand_entity_noun(root: str) -> List[str]:
    """
    Expand entity/proper noun to ONLY noun forms.

    For entity nouns (like place names, language names, person names),
    we want ONLY the noun forms, not verb/adjective/adverb forms.

    Args:
        root: Esperanto root for entity (e.g., "esperant" for Esperanto language)

    Returns:
        List of noun forms: -o, -on, -oj, -ojn
    """
    return [
        root,           # bare root (sometimes used)
        root + 'o',     # nominative singular (Esperanto)
        root + 'on',    # accusative singular (Esperanton)
        root + 'oj',    # nominative plural (Esperantoj)
        root + 'ojn',   # accusative plural (Esperantojn)
    ]


def expand_esperanto_root(root: str, question_type: Optional[str] = None, is_entity: bool = False) -> List[str]:
    """
    Expand an Esperanto root to word forms, optimized by question type.

    SMART EXPANSION OPTIMIZATION:
    Instead of generating all 15 forms, prioritize forms based on question type.
    This reduces query size and improves precision.

    Args:
        root: Esperanto root (e.g., "fond")
        question_type: Optional question type (who/what/when/where) for prioritization

    Returns:
        List of expanded forms (4-6 forms for typed questions, 15 for unknown)
    """
    # If this is an entity (proper noun), expand ONLY to noun forms
    if is_entity:
        return expand_entity_noun(root)

    forms = [root]  # Always include bare root

    # TEMPORARILY DISABLED: Always use full expansion to test if smart expansion is causing issues
    # If no question type, use full expansion (backward compatibility)
    if True:  # question_type is None or question_type.lower() not in ['who', 'what', 'when', 'where', 'how', 'why']:
        # Full expansion (~15 forms)
        forms.extend([root + 'as', root + 'is', root + 'i'])
        forms.extend([root + 'anta', root + 'inta', root + 'ita'])
        forms.extend([root + 'o', root + 'on'])
        forms.extend([root + 'into', root + 'anto', root + 'onto'])
        forms.append(root + 'a')
        return forms

    # Smart expansion based on question type
    question_type = question_type.lower()

    if question_type == 'who':
        # For WHO questions, focus on agent nouns and past tense
        forms.extend([
            root + 'is',    # past tense (most common in Wikipedia)
            root + 'into',  # agent noun (the one who X'd)
            root + 'anto',  # present agent
        ])

    elif question_type == 'what':
        # For WHAT questions, focus on nouns and adjectives
        forms.extend([
            root + 'o',     # noun
            root + 'a',     # adjective
            root + 'i',     # infinitive (for definitions)
            root + 'aĵo',   # thing characterized by root
        ])

    elif question_type == 'when':
        # For WHEN questions, focus on verb forms
        forms.extend([
            root + 'is',    # past tense
            root + 'as',    # present tense
            root + 'os',    # future tense
            root + 'inta',  # past participle
        ])

    elif question_type in ['where', 'how', 'why']:
        # For other questions, use medium expansion
        forms.extend([
            root + 'is',    # past tense
            root + 'o',     # noun
            root + 'a',     # adjective
            root + 'i',     # infinitive
        ])

    return forms  # 4-6 forms per root (3-4x smaller than full expansion)


class WhooshRetriever:
    """
    Retrieve sentences using Whoosh full-text search.

    Combines Whoosh FTS (for fast keyword retrieval) with Kuzu (for AST metadata).
    """

    def __init__(
        self,
        whoosh_index_dir: Path,
        kuzu_db_path: Path
    ):
        """
        Initialize retriever.

        Args:
            whoosh_index_dir: Path to Whoosh index directory
            kuzu_db_path: Path to Kuzu database (for AST retrieval)
        """
        self.whoosh_index_dir = Path(whoosh_index_dir)
        self.kuzu_db_path = Path(kuzu_db_path)

        # Open Whoosh index
        logger.info(f"Loading Whoosh index from {whoosh_index_dir}")
        self.ix = open_dir(str(whoosh_index_dir))

        # Connect to Kuzu for AST retrieval
        logger.info(f"Connecting to Kuzu database at {kuzu_db_path}")
        self.kuzu_db = kuzu.Database(str(kuzu_db_path))
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)

        # Initialize AST reconstructor for fast precomputed AST fetching
        logger.info("Initializing KuzuASTReconstructor for 10x faster AST retrieval")
        self.ast_reconstructor = KuzuASTReconstructor(self.kuzu_conn)

    def retrieve_with_ast_roles(
        self,
        query_ast: Dict,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve sentences matching GRAMMATICAL ROLE constraints from query AST.

        This is the PRIMARY retrieval method - uses AST structure, not text matching.

        Supports all question types with appropriate grammatical patterns:
        - WHO: verb + object → extract subject
        - WHERE: verb + entity → extract location prepositional phrases
        - WHEN: verb + entity → extract temporal expressions
        - WHAT: definition patterns (estas + predicate)
        - WHY: causal patterns (ĉar, pro, por)
        - HOW: manner patterns (per, laŭ)
        - HOW_MANY: numeric patterns

        Args:
            query_ast: Parsed query AST with grammatical structure
            top_k: Number of results to return

        Returns:
            List of sentence dicts with 'text', 'ast', 'id', 'score'
        """
        # Detect question type from query AST
        question_type = self._detect_question_type(query_ast)
        logger.info(f"Detected question type: {question_type}")

        # Dispatch to appropriate retrieval pattern
        if question_type == 'KIU':  # WHO
            return self._retrieve_who_pattern(query_ast, top_k)
        elif question_type == 'KIE':  # WHERE
            return self._retrieve_where_pattern(query_ast, top_k)
        elif question_type == 'KIAM':  # WHEN
            return self._retrieve_when_pattern(query_ast, top_k)
        elif question_type == 'KIO':  # WHAT
            return self._retrieve_what_pattern(query_ast, top_k)
        elif question_type == 'KIAL':  # WHY
            return self._retrieve_why_pattern(query_ast, top_k)
        elif question_type == 'KIEL':  # HOW
            return self._retrieve_how_pattern(query_ast, top_k)
        elif question_type == 'KIOM':  # HOW_MANY
            return self._retrieve_how_many_pattern(query_ast, top_k)
        else:
            logger.warning(f"Unknown question type: {question_type}, using generic pattern")
            return self._retrieve_generic_pattern(query_ast, top_k)

    def _detect_question_type(self, query_ast: Dict) -> str:
        """Detect question type from query AST correlative."""
        subjekto = query_ast.get('subjekto')
        if subjekto:
            if subjekto.get('tipo') == 'vortgrupo':
                kerno = subjekto.get('kerno', {})
            else:
                kerno = subjekto

            if kerno.get('vortspeco') == 'korelativo':
                radiko = kerno.get('radiko', '').upper()
                return radiko

        return 'UNKNOWN'

    def _retrieve_who_passive_pattern(self, verb_root: str, verb_synonyms: List[str],
                                       obj_root: str, top_k: int, query_ast: Dict) -> List[Dict]:
        """
        Retrieve WHO questions using passive voice pattern (Phase 2 improvement).

        For query "Kiu fondis Esperanton?" also matches:
        - "Esperanto estis fondita de Zamenhof" (passive voice)
        - Looks for: object as subject + "est" + past participle + "de" agent

        Args:
            verb_root: Main verb root (e.g., "fond")
            verb_synonyms: Synonym roots (e.g., ["kre"])
            obj_root: Object/patient root (e.g., "esperant")
            top_k: Number of results
            query_ast: Query AST for ranking

        Returns:
            List of documents with passive voice constructions
        """
        # Build list of verb roots to check (including synonyms)
        all_verbs = [verb_root] + list(verb_synonyms)

        logger.info(f"WHO passive pattern: verbs={all_verbs}, patient={obj_root}")

        # Passive pattern: Patient as subject + "est" + participle-ita
        # Look for sentences where:
        # - Subject contains object root (patient becomes subject in passive)
        # - Verb is "est" (estas, estis)
        # - Has participle in "aliaj" matching verb roots
        # - Ideally has "de" preposition introducing agent

        # We'll look for sentences with patient as subject and "est" verb
        # The participle and agent detection would require more complex pattern matching
        # For now, match subject=patient + verb="est" which often indicates passive

        kuzu_query_passive = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{obj_root}'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Also try with subject as single Vorto
        kuzu_query_passive_simple = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
            WHERE subj.radiko = '{obj_root}'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Execute both queries and merge
        results_vg = self._execute_kuzu_query(kuzu_query_passive, top_k, [obj_root], query_ast=query_ast)
        results_simple = self._execute_kuzu_query(kuzu_query_passive_simple, top_k, [obj_root], query_ast=query_ast)

        # Merge and deduplicate
        all_results = results_vg + results_simple
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.get('id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)

        logger.info(f"WHO passive pattern found {len(unique_results)} documents")

        return unique_results

    def _retrieve_who_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """
        Retrieve WHO questions with Phase 2 improvement: passive voice support.

        Patterns:
        1. Active: "Kiu fondis Esperanton?" → "X fondis Esperanton"
        2. Passive: "Kiu fondis Esperanton?" → "Esperanto estis fondita de X"
        3. Identity: "Kiu estis X?" → sentences about X
        """
        from klareco.knowledge import get_synonyms

        # Extract verb and object constraints
        verb_root, obj_root = self._extract_verb_and_object(query_ast)

        if not verb_root:
            logger.debug("WHO question: No verb found")
            return []

        if not obj_root:
            logger.debug("WHO question: No object found")
            return []

        # Identity question pattern: "Kiu estas/estis X?"
        # Use simpler entity-based query without verb synonym expansion
        if verb_root in ['est']:
            logger.info(f"WHO identity pattern: entity={obj_root}")

            # Match sentences mentioning the entity (any role)
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{obj_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """

            return self._execute_kuzu_query(kuzu_query, top_k, [obj_root], query_ast=query_ast)

        # Action question pattern: "Kiu VERB-is OBJECT-n?"
        # Use verb synonym expansion for recall
        verb_synonyms = get_synonyms(verb_root, max_count=3)
        verb_constraint = [verb_root] + list(verb_synonyms)
        verb_list_str = ','.join(f"'{v}'" for v in verb_constraint)

        logger.info(f"WHO action pattern: verb={verb_constraint}, object={obj_root}")

        # Phase 2.1: Active voice pattern (original)
        kuzu_query_active = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko IN [{verb_list_str}]
            MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj_kerno:Vorto)
            WHERE obj_kerno.radiko = '{obj_root}' AND obj_kerno.kazo = 'akuzativo'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        active_results = self._execute_kuzu_query(kuzu_query_active, top_k, verb_constraint + [obj_root], query_ast=query_ast)

        # Phase 2.2: Passive voice pattern (new)
        passive_results = self._retrieve_who_passive_pattern(
            verb_root, verb_synonyms, obj_root, top_k, query_ast
        )

        # Merge active + passive results, deduplicating by id
        seen_ids = {doc.get('id') for doc in active_results}
        merged_results = active_results[:]
        for doc in passive_results:
            if doc.get('id') not in seen_ids:
                merged_results.append(doc)
                seen_ids.add(doc.get('id'))

        logger.info(f"WHO pattern: {len(active_results)} active + {len(passive_results)} passive = {len(merged_results)} total")

        # Phase 4: Add grammatical variant results (participial, relative clause, appositive)
        from klareco.rag.grammatical_variants import GrammaticalVariantGenerator

        variant_gen = GrammaticalVariantGenerator()
        variants = variant_gen.generate_who_variants(
            verb_root=verb_root,
            verb_synonyms=verb_synonyms,
            obj_root=obj_root,
            top_k=top_k // 2  # Get fewer results per variant
        )

        variant_results = self._execute_variant_queries(
            variants=variants,
            top_k=top_k // 2,
            matching_roots=verb_constraint + [obj_root],
            query_ast=query_ast,
            question_type='KIU',
            query_entity=obj_root
        )

        # Merge variant results
        for doc in variant_results:
            if doc.get('id') not in seen_ids:
                merged_results.append(doc)
                seen_ids.add(doc.get('id'))

        logger.info(f"WHO pattern with variants: {len(merged_results)} total results")

        # Re-sort by score and return top_k
        merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return merged_results[:top_k]

    def _retrieve_where_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Retrieve WHERE questions: location prepositional phrases."""
        from klareco.knowledge import get_synonyms

        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not verb_root:
            logger.debug("WHERE question: No verb found")
            return []

        # Get verb synonyms
        verb_synonyms = get_synonyms(verb_root, max_count=3)
        verb_constraint = [verb_root] + list(verb_synonyms)
        verb_list_str = ','.join(f"'{v}'" for v in verb_constraint)

        logger.info(f"WHERE pattern: verb={verb_constraint}, entity={entity_root or 'any'}")

        # Kuzu query: Match verb + location prepositions (en, de, ĉe, etc.)
        # Look for sentences with location words in aliaj
        if entity_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
        else:
            # No entity specified, just match verb
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """

        base_results = self._execute_kuzu_query(kuzu_query, top_k, verb_constraint + ([entity_root] if entity_root else []), query_ast=query_ast)

        # Phase 4: Add grammatical variant results (participial, nominalization)
        if entity_root:
            from klareco.rag.grammatical_variants import GrammaticalVariantGenerator

            variant_gen = GrammaticalVariantGenerator()
            variants = variant_gen.generate_where_variants(
                verb_root=verb_root,
                verb_synonyms=verb_synonyms,
                entity_root=entity_root,
                top_k=top_k // 2
            )

            variant_results = self._execute_variant_queries(
                variants=variants,
                top_k=top_k // 2,
                matching_roots=verb_constraint + [entity_root],
                query_ast=query_ast,
                question_type='KIE',
                query_entity=entity_root
            )

            # Merge variant results
            seen_ids = {doc.get('id') for doc in base_results}
            merged_results = base_results[:]
            for doc in variant_results:
                if doc.get('id') not in seen_ids:
                    merged_results.append(doc)
                    seen_ids.add(doc.get('id'))

            logger.info(f"WHERE pattern with variants: {len(merged_results)} total results")

            # Re-sort by score and return top_k
            merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return merged_results[:top_k]
        else:
            return base_results

    def _retrieve_when_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Retrieve WHEN questions: temporal expressions."""
        from klareco.knowledge import get_synonyms

        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not verb_root:
            logger.debug("WHEN question: No verb found")
            return []

        # Get verb synonyms
        verb_synonyms = get_synonyms(verb_root, max_count=3)
        verb_constraint = [verb_root] + list(verb_synonyms)
        verb_list_str = ','.join(f"'{v}'" for v in verb_constraint)

        logger.info(f"WHEN pattern: verb={verb_constraint}, entity={entity_root or 'any'}")

        # Kuzu query: Match verb + any entity mention
        if entity_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
        else:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """

        base_results = self._execute_kuzu_query(kuzu_query, top_k, verb_constraint + ([entity_root] if entity_root else []), query_ast=query_ast)

        # Phase 4: Add grammatical variant results (nominalization, participial)
        if entity_root:
            from klareco.rag.grammatical_variants import GrammaticalVariantGenerator

            variant_gen = GrammaticalVariantGenerator()
            variants = variant_gen.generate_when_variants(
                verb_root=verb_root,
                verb_synonyms=verb_synonyms,
                entity_root=entity_root,
                top_k=top_k // 2
            )

            variant_results = self._execute_variant_queries(
                variants=variants,
                top_k=top_k // 2,
                matching_roots=verb_constraint + [entity_root],
                query_ast=query_ast,
                question_type='KIAM',
                query_entity=entity_root
            )

            # Merge variant results
            seen_ids = {doc.get('id') for doc in base_results}
            merged_results = base_results[:]
            for doc in variant_results:
                if doc.get('id') not in seen_ids:
                    merged_results.append(doc)
                    seen_ids.add(doc.get('id'))

            logger.info(f"WHEN pattern with variants: {len(merged_results)} total results")

            # Re-sort by score and return top_k
            merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return merged_results[:top_k]
        else:
            return base_results

    def _retrieve_is_a_pattern(self, entity_root: str, top_k: int, query_ast: Dict) -> List[Dict]:
        """
        Retrieve IS-A facts for an entity (Phase 1 improvement).

        Matches definitional patterns in priority order:
        1. Direct IS-A: "X estas Y" (entity as subject)
        2. Reverse IS-A: "Y estas X" (entity as predicate nominative)

        This fixes WHAT questions which were matching narrative sentences
        instead of definitional IS-A relations.

        Args:
            entity_root: Root form of entity (e.g., "hund" for "hundo")
            top_k: Number of results to return
            query_ast: Query AST for semantic ranking

        Returns:
            List of documents with IS-A relations
        """
        logger.info(f"IS-A pattern: entity={entity_root}")

        # Priority 1: Direct IS-A (entity as subject)
        # Pattern: "Hundo estas besto" → entity=hund IS-A besto
        kuzu_query_direct = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Also try with subject as single Vorto (not in Vortgrupo)
        kuzu_query_direct_simple = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Priority 2: Reverse IS-A (entity as predicate nominative in objekto)
        # Pattern: "Besto estas hundo" → besto IS-A hund
        kuzu_query_reverse = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj:Vorto)
            WHERE obj.radiko = '{entity_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Also try with objekto as single Vorto
        kuzu_query_reverse_simple = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
            WHERE obj.radiko = '{entity_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        # Try all IS-A patterns and merge results
        results_direct = self._execute_kuzu_query(kuzu_query_direct, top_k, [entity_root], query_ast=query_ast)
        results_direct_simple = self._execute_kuzu_query(kuzu_query_direct_simple, top_k, [entity_root], query_ast=query_ast)
        results_reverse = self._execute_kuzu_query(kuzu_query_reverse, top_k, [entity_root], query_ast=query_ast)
        results_reverse_simple = self._execute_kuzu_query(kuzu_query_reverse_simple, top_k, [entity_root], query_ast=query_ast)

        # Merge and deduplicate by id
        all_results = results_direct + results_direct_simple + results_reverse + results_reverse_simple
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.get('id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)

        logger.info(f"IS-A pattern found {len(unique_results)} unique documents")

        return unique_results[:top_k]

    def _retrieve_what_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """
        Retrieve WHAT questions: prioritize IS-A definitional patterns.

        Phase 1 improvement: Use explicit IS-A pattern matching instead of
        generic entity mentions. This fixes WHAT questions which were 0% accurate.
        """
        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not entity_root:
            logger.debug("WHAT question: No entity found")
            return []

        logger.info(f"WHAT pattern: entity={entity_root}")

        # Phase 1: Try IS-A pattern first (highest priority)
        is_a_results = self._retrieve_is_a_pattern(entity_root, top_k, query_ast)

        if len(is_a_results) >= top_k // 2:
            # IS-A retrieval successful, return these results
            logger.info(f"IS-A retrieval successful: {len(is_a_results)} results")
            return is_a_results

        # Fallback: Generic entity mention (original behavior)
        # Only used if IS-A pattern returns too few results
        logger.info(f"IS-A returned only {len(is_a_results)} results, adding generic mentions")

        kuzu_query_generic = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
            WHERE alia.radiko = '{entity_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 3}
        """

        generic_results = self._execute_kuzu_query(kuzu_query_generic, top_k, [entity_root], query_ast=query_ast)

        # Merge IS-A + generic, deduplicating by id
        seen_ids = {doc.get('id') for doc in is_a_results}
        merged_results = is_a_results[:]
        for doc in generic_results:
            if doc.get('id') not in seen_ids:
                merged_results.append(doc)
                seen_ids.add(doc.get('id'))

        # Phase 4: Add grammatical variant results (appositive, relative clause)
        from klareco.rag.grammatical_variants import GrammaticalVariantGenerator

        variant_gen = GrammaticalVariantGenerator()
        variants = variant_gen.generate_what_variants(
            entity_root=entity_root,
            top_k=top_k // 2
        )

        variant_results = self._execute_variant_queries(
            variants=variants,
            top_k=top_k // 2,
            matching_roots=[entity_root],
            query_ast=query_ast,
            question_type='KIO',
            query_entity=entity_root
        )

        # Merge variant results
        for doc in variant_results:
            if doc.get('id') not in seen_ids:
                merged_results.append(doc)
                seen_ids.add(doc.get('id'))

        logger.info(f"WHAT pattern with variants: {len(merged_results)} total results")

        # Re-sort by score and return top_k
        merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return merged_results[:top_k]

    def _retrieve_why_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Retrieve WHY questions: causal markers (ĉar, pro, por)."""
        from klareco.knowledge import get_synonyms

        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not verb_root:
            logger.debug("WHY question: No verb found")
            return []

        verb_synonyms = get_synonyms(verb_root, max_count=3)
        verb_constraint = [verb_root] + list(verb_synonyms)
        verb_list_str = ','.join(f"'{v}'" for v in verb_constraint)

        logger.info(f"WHY pattern: verb={verb_constraint}, entity={entity_root or 'any'}")

        # Kuzu query: Match verb + entity
        if entity_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
        else:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """

        return self._execute_kuzu_query(kuzu_query, top_k, verb_constraint + ([entity_root] if entity_root else []), query_ast=query_ast)

    def _retrieve_how_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Retrieve HOW questions: manner patterns."""
        from klareco.knowledge import get_synonyms

        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not verb_root:
            logger.debug("HOW question: No verb found")
            return []

        verb_synonyms = get_synonyms(verb_root, max_count=3)
        verb_constraint = [verb_root] + list(verb_synonyms)
        verb_list_str = ','.join(f"'{v}'" for v in verb_constraint)

        logger.info(f"HOW pattern: verb={verb_constraint}, entity={entity_root or 'any'}")

        # Kuzu query: Match verb + entity
        if entity_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
        else:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko IN [{verb_list_str}]
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """

        return self._execute_kuzu_query(kuzu_query, top_k, verb_constraint + ([entity_root] if entity_root else []), query_ast=query_ast)

    def _retrieve_how_many_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Retrieve HOW_MANY questions: numeric patterns."""
        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not entity_root:
            logger.debug("HOW_MANY question: No entity found")
            return []

        logger.info(f"HOW_MANY pattern: entity={entity_root}")

        # Kuzu query: Match entity mentions (likely with numbers)
        kuzu_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
            WHERE alia.radiko = '{entity_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k * 5}
        """

        return self._execute_kuzu_query(kuzu_query, top_k, [entity_root], query_ast=query_ast)

    def _retrieve_generic_pattern(self, query_ast: Dict, top_k: int) -> List[Dict]:
        """Generic fallback: match verb + entity."""
        verb_root, entity_root = self._extract_verb_and_object(query_ast)

        if not verb_root and not entity_root:
            logger.debug("Generic pattern: No verb or entity found")
            return []

        logger.info(f"Generic pattern: verb={verb_root}, entity={entity_root}")

        if verb_root and entity_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko = '{verb_root}'
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
            return self._execute_kuzu_query(kuzu_query, top_k, [verb_root, entity_root], query_ast=query_ast)
        elif verb_root:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
                WHERE verb.radiko = '{verb_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
            return self._execute_kuzu_query(kuzu_query, top_k, [verb_root], query_ast=query_ast)
        else:
            kuzu_query = f"""
                MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
                MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
                WHERE alia.radiko = '{entity_root}'
                RETURN ft.id AS id, ft.teksto AS text
                LIMIT {top_k * 5}
            """
            return self._execute_kuzu_query(kuzu_query, top_k, [entity_root], query_ast=query_ast)

    def _extract_verb_and_object(self, query_ast: Dict) -> tuple:
        """Extract verb and object roots from query AST."""
        verb_root = None
        obj_root = None

        # First check the verbo field (most common location)
        verbo = query_ast.get('verbo')
        if verbo:
            if isinstance(verbo, dict):
                verb_root = verbo.get('radiko')

        # Check aliaj for entities and fallback verb
        aliaj = query_ast.get('aliaj', [])
        for alia in aliaj:
            if not isinstance(alia, dict):
                continue

            # Fallback verb check (if not found in verbo field)
            if alia.get('vortspeco') == 'verbo' and not verb_root:
                verb_root = alia.get('radiko')

            # Extract entity (any noun, regardless of case)
            if alia.get('vortspeco') == 'substantivo':
                if alia.get('kazo') == 'akuzativo' and not obj_root:
                    obj_root = alia.get('radiko')
                elif not obj_root:  # Nominative or other case
                    obj_root = alia.get('radiko')

            # Also check for unknown words (proper names like "Zamenhof")
            if alia.get('vortspeco') == 'nekonata' and not obj_root:
                obj_root = alia.get('radiko')

        # Try objekto field as fallback
        if not obj_root:
            obj = query_ast.get('objekto')
            if obj:
                if obj.get('tipo') == 'vortgrupo':
                    obj_root = obj.get('kerno', {}).get('radiko')
                else:
                    obj_root = obj.get('radiko')

        return verb_root, obj_root

    def _execute_kuzu_query(
        self,
        kuzu_query: str,
        top_k: int,
        matching_roots: List[str],
        query_ast: Optional[Dict] = None,
        question_type: Optional[str] = None,
        query_entity: Optional[str] = None
    ) -> List[Dict]:
        """
        Execute Kuzu query and return formatted documents with semantic ranking.

        NEW: Semantic AST ranking (Issue #713)
        - Parse ASTs for all candidates (not just top_k)
        - Score by structural and semantic similarity
        - Return top_k after ranking

        Args:
            kuzu_query: Cypher query to execute
            top_k: Number of top results to return
            matching_roots: Roots that matched (for metadata)
            query_ast: Query AST for semantic ranking (if None, uses rank order)
            question_type: Question type for importance scoring (Phase 3)
            query_entity: Query entity for importance scoring (Phase 3)
        """
        try:
            result = self.kuzu_conn.execute(kuzu_query)
        except Exception as e:
            logger.error(f"Kuzu query failed: {e}")
            return []

        # Build documents
        documents = []
        while result.has_next():
            row = result.get_next()
            sentence_id = str(row[0])
            text = row[1]

            if not text:
                continue

            documents.append({
                'text': text,
                'ast': None,  # Will parse below
                'id': sentence_id,
                'score': 100.0 - len(documents),  # Temporary rank order score
                'matching_roots': matching_roots,
                'num_matches': len(matching_roots),
                'doc_title': '',
                'metadata': '',
                'retrieval_method': 'ast_role_query'
            })

        logger.info(f"Kuzu query returned {len(documents)} sentences")

        if not documents:
            return []

        # Parse ASTs for ALL candidates (needed for semantic ranking)
        # Limit parsing to reasonable number to avoid performance issues
        max_parse = min(len(documents), top_k * 5)  # Parse at most 5x top_k
        from klareco.parser import parse

        for doc in documents[:max_parse]:
            try:
                doc['ast'] = parse(doc['text'])
            except Exception as e:
                logger.warning(f"Failed to parse sentence {doc['id']}: {e}")
                doc['ast'] = None

        # === NEW: Semantic AST Ranking (Issue #713) ===
        if query_ast:
            logger.info("Applying semantic AST ranking...")
            from klareco.rag.ast_semantic_ranker import rank_ast_matches

            # Rank candidates by semantic similarity
            # Phase 3: Enable importance-aware ranking
            ranked_documents = rank_ast_matches(
                query_ast=query_ast,
                candidates=documents[:max_parse],
                use_embeddings=True,  # Now enabled - uses 64D root embeddings
                use_importance_scoring=True,  # Phase 3: Integrate importance scoring
                question_type=question_type,   # Phase 3: Pass question type for importance
                query_entity=query_entity,     # Phase 3: Pass query entity
                query_roots=matching_roots     # Phase 3: Pass query roots
            )

            logger.info(f"Semantic ranking complete. Top score: {ranked_documents[0]['score']:.2f}")
            return ranked_documents[:top_k]
        else:
            # Fallback: Use rank order (old behavior)
            logger.warning("No query_ast provided for semantic ranking, using rank order")
            return documents[:top_k]

    def _execute_variant_queries(
        self,
        variants: List,  # List[GrammaticalVariant]
        top_k: int,
        matching_roots: List[str],
        query_ast: Optional[Dict] = None,
        question_type: Optional[str] = None,
        query_entity: Optional[str] = None
    ) -> List[Dict]:
        """
        Execute grammatical variant queries and merge results.

        Phase 4: Grammatical Variant Framework
        Execute Cypher queries for participial, nominalization, relative clause,
        and appositive constructions. Merge with confidence weighting.

        Args:
            variants: List of GrammaticalVariant objects with cypher_query and confidence
            top_k: Number of results to return per variant
            matching_roots: Roots that matched (for metadata)
            query_ast: Query AST for semantic ranking
            question_type: Question type for importance scoring
            query_entity: Query entity for importance scoring

        Returns:
            Merged list of documents with confidence-weighted scores
        """
        all_results = []
        seen_ids = set()

        for variant in variants:
            logger.info(f"Executing variant: {variant.description} (confidence={variant.confidence})")

            try:
                # Execute variant query
                variant_results = self._execute_kuzu_query(
                    kuzu_query=variant.cypher_query,
                    top_k=top_k,
                    matching_roots=matching_roots,
                    query_ast=query_ast,
                    question_type=question_type,
                    query_entity=query_entity
                )

                # Apply confidence weighting to scores
                for doc in variant_results:
                    if doc.get('id') not in seen_ids:
                        # Weight score by variant confidence
                        doc['score'] = doc['score'] * variant.confidence
                        doc['variant_type'] = variant.pattern_type.value
                        doc['variant_confidence'] = variant.confidence
                        all_results.append(doc)
                        seen_ids.add(doc.get('id'))

                logger.info(f"Variant {variant.pattern_type.value} returned {len(variant_results)} results")

            except Exception as e:
                logger.warning(f"Variant query failed: {e}")
                continue

        # Sort by score (already confidence-weighted)
        all_results.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Total unique results from {len(variants)} variants: {len(all_results)}")
        return all_results[:top_k]

    def retrieve(
        self,
        query_roots: List[str],
        top_k: int = 20,
        retrieval_limit: int = 200,
        question_type: Optional[str] = None,
        query_entity: Optional[str] = None,
        query_ast: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve sentences using AST role-based grammatical constraints.

        STRATEGY:
        - REQUIRED: query_ast must be provided
        - Uses Kuzu graph queries to match grammatical structure (verb + object roles)
        - Returns ONLY sentences with correct grammatical patterns
        - NO fallback to BM25 text matching (pure AST-first approach)

        If query_ast is not provided, returns empty list (fail loudly).
        This enforces AST-first architecture and prevents silent fallback to inferior text matching.

        Args:
            query_roots: List of root words (used for logging, not retrieval)
            top_k: Number of top results to return
            retrieval_limit: Ignored (kept for API compatibility)
            question_type: Ignored (kept for API compatibility)
            query_entity: Ignored (kept for API compatibility)
            query_ast: Parsed query AST for grammatical role constraints (REQUIRED)

        Returns:
            List of sentence dicts with 'text', 'ast', 'id', 'score', 'matching_roots'
            Empty list if query_ast not provided (enforces AST-first approach)
        """
        if not query_ast:
            logger.error("❌ query_ast is REQUIRED for AST-first retrieval. Refusing to fall back to BM25.")
            logger.error("   This is intentional: AST-first architecture requires query_ast.")
            logger.error("   If you're seeing this, fix the caller to pass query_ast.")
            return []

        logger.info(f"Query roots (for reference): {query_roots}")

        # === AST role-based retrieval (ONLY retrieval method) ===
        logger.info("Using AST role-based retrieval (grammatical constraints)...")
        ast_results = self.retrieve_with_ast_roles(query_ast, top_k)

        if ast_results:
            logger.info(f"✓ AST role retrieval: {len(ast_results)} sentences found")
        else:
            logger.warning(f"✗ AST role retrieval: 0 sentences found (grammatical pattern not in corpus)")

        return ast_results

        # === OLD PHASE 2: BM25 fallback (REMOVED - pure AST-first now) ===
        # logger.info("Using BM25 text matching with word form expansion...")

        # AND QUERY OPTIMIZATION: Separate proper names from common words
        # Proper names (capitalized) should be required via AND, not expanded
        # Common words should be expanded to word forms
        proper_names = []
        common_words = []

        for root in query_roots:
            # Detect proper names: capitalized AND length > 2 (avoid "Mi", "Li", etc.)
            if root and len(root) > 2 and root[0].isupper():
                proper_names.append(root.lower())  # Lowercase for case-insensitive search
            else:
                common_words.append(root)

        # Build Whoosh query based on what we have
        if proper_names and common_words:
            # BEST CASE: Have both proper names and common words
            # Query: (name1 OR name2) AND (expanded_word_forms)
            # Example: "Lincoln AND (estis OR estinta OR estanto)"

            name_part = ' OR '.join(proper_names)

            word_forms = []
            for word in common_words:
                # Check if this word is an entity (query_entity or place name)
                is_entity_word = (word == query_entity)
                word_forms.extend(expand_esperanto_root(word, question_type=None, is_entity=is_entity_word))
            word_part = ' OR '.join(word_forms)

            query_str = f"({name_part}) AND ({word_part})"
            logger.info(f"AND query: {len(proper_names)} names + {len(word_forms)} forms")

        elif proper_names:
            # Only proper names - OR them together
            # Example: "Lincoln OR Roosevelt"
            query_str = ' OR '.join(proper_names)
            logger.info(f"Proper names only: {len(proper_names)} names")

        else:
            # Only common words - expand all (current behavior)
            all_forms = []
            for root in query_roots:
                # Check if this root is an entity
                is_entity_word = (root == query_entity)
                all_forms.extend(expand_esperanto_root(root, question_type=None, is_entity=is_entity_word))
            query_str = ' OR '.join(all_forms)
            logger.info(f"Common words only: {len(all_forms)} forms")

        logger.debug(f"Whoosh query: {query_str[:150]}...")

        # Search Whoosh index
        with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
            query = QueryParser("text_lower", self.ix.schema, group=OrGroup).parse(query_str)
            results = searcher.search(query, limit=retrieval_limit)

            logger.info(f"Whoosh found {len(results)} matching sentences")
            id_to_score = {hit['id']: hit.score for hit in results}
            sentence_ids = list(id_to_score.keys())

        if not sentence_ids:
            return []

        # Fetch ASTs from Kuzu for matched sentences
        # Build query with IN clause (IDs must be integers, not strings)
        ids_str = ','.join(str(sid) for sid in sentence_ids)
        kuzu_query = f"""
            MATCH (ft:Frazoteksto)
            WHERE ft.id IN [{ids_str}]
            RETURN ft.id AS id, ft.teksto AS text
        """

        result = self.kuzu_conn.execute(kuzu_query)

        # Build sentence dict WITHOUT parsing ASTs yet (lazy parsing optimization)
        documents = []
        while result.has_next():
            row = result.get_next()
            sentence_id = str(row[0])
            text = row[1]

            if not text:
                continue

            # Count matching roots (for metadata)
            text_lower = text.lower()
            matching_roots = [r for r in query_roots if r in text_lower]

            # Get Whoosh BM25 score
            bm25_score = id_to_score.get(sentence_id, 0.0)

            documents.append({
                'text': text,
                'ast': None,  # Lazy parsing - parse later
                'id': sentence_id,
                'score': bm25_score,  # Use actual BM25 score from Whoosh
                'matching_roots': matching_roots,
                'num_matches': len(matching_roots),
                'doc_title': '',  # Not available in current schema
                'metadata': ''
            })

        # Meta-content filtering: penalize dictionary definitions, word lists, disambiguation pages
        # This addresses the problem where BM25 retrieves meta-content instead of factual sentences
        documents = self._filter_meta_content(documents)
        penalized_count = sum(1 for d in documents if d.get('meta_penalty', 1.0) < 1.0)
        if penalized_count > 0:
            logger.info(f"Meta-content penalty applied to {penalized_count}/{len(documents)} documents")

        # Sort by score (BM25 + meta-content penalty) BEFORE parsing
        documents.sort(key=lambda d: d['score'], reverse=True)

        # Fetch precomputed ASTs from graph (10x faster than parsing!)
        # OLD APPROACH: parse(text) for each sentence = 50ms per sentence
        # NEW APPROACH: fetch from graph in batch = <5ms total (10x speedup)
        parse_limit = min(50, len(documents))
        sentence_ids_to_parse = [int(documents[i]['id']) for i in range(parse_limit)]

        logger.debug(f"Fetching {len(sentence_ids_to_parse)} precomputed ASTs from graph")
        reconstructed_asts = self.ast_reconstructor.reconstruct_ast_batch(sentence_ids_to_parse)
        logger.debug(f"Retrieved {len(reconstructed_asts)} ASTs from graph")

        for i in range(parse_limit):
            sentence_id = int(documents[i]['id'])
            documents[i]['ast'] = reconstructed_asts.get(sentence_id)

            # Fallback to parsing if AST not found in graph (shouldn't happen in v2.1+)
            if documents[i]['ast'] is None:
                logger.warning(f"AST not found for sentence {sentence_id}, falling back to parsing")
                documents[i]['ast'] = parse(documents[i]['text'])

        # AST-aware filtering: boost documents where entity appears in correct grammatical role
        # Only applies to parsed documents (top 50)
        if question_type and query_entity:
            logger.info(f"Applying AST filtering: question_type={question_type}, entity={query_entity}")
            documents = self._apply_ast_filtering(documents[:parse_limit], question_type, query_entity)

            # Log boost statistics
            boosted_count = sum(1 for d in documents if d.get('ast_boost', 1.0) > 1.0)
            if boosted_count > 0:
                logger.info(f"AST boosted {boosted_count}/{len(documents)} documents")

        # Final sort by score (BM25 + AST boost)
        documents.sort(key=lambda d: d['score'], reverse=True)

        return documents[:top_k]

    def _filter_meta_content(self, documents: List[Dict]) -> List[Dict]:
        """
        Penalize meta-content (dictionaries, word lists, disambiguation pages).

        PROBLEM: BM25 retrieves meta-content with high keyword frequency instead of
        factual sentences:
        - "La rusa adjektivo учредительный signifas ankaŭ fonda..." (dictionary)
        - "Kre aŭ KRE havas plurajn signifojn: *kre estas kodo..." (disambiguation)
        - "*Esperant' *Sociolekta Triopo..." (word list)

        SOLUTION: Detect meta-content patterns and apply score penalty.

        Args:
            documents: Retrieved documents

        Returns:
            Documents with meta-content penalties applied
        """
        meta_content_patterns = [
            # Dictionary definitions (foreign language explanations)
            ('учредительный', 0.2),  # Cyrillic (Russian dictionary entries)
            ('signifo', 0.4),         # "meaning" - often in dictionary entries
            ('signifas', 0.4),        # "means" - dictionary definitions

            # Disambiguation pages
            ('havas plurajn signifojn', 0.1),  # "has multiple meanings"
            ('estas kodo de lingvo', 0.1),      # "is a language code"
            ('laŭ normo iso', 0.2),             # "according to ISO standard"

            # Word lists / bullet lists
            ('* ', 0.5),  # Lines starting with bullet points
            ('*Esperant', 0.3),  # Word lists with asterisks

            # Meta-linguistic content
            ('adjektivo', 0.4),   # "adjective" - grammar terminology
            ('substantivo', 0.4), # "noun" - grammar terminology
            ('verbo estas', 0.4), # Verb conjugation tables

            # Generic fragments / truncated sentences
            (' ... ', 0.5),  # Ellipsis indicating truncation
            ('(UTC)', 0.3),  # Timestamps from talk pages

            # ReVo dictionary structure
            ('|| Vidu la artikolon:', 0.3),  # ReVo cross-references
            ('Tabulo omaĝe al', 0.3),        # Monument/plaque descriptions
        ]

        for doc in documents:
            text = doc['text']
            penalty = 1.0  # No penalty by default

            # Check each pattern
            for pattern, pattern_penalty in meta_content_patterns:
                if pattern in text:
                    penalty = min(penalty, pattern_penalty)
                    # Take the MOST severe penalty (lowest multiplier)

            # Apply penalty to score
            if penalty < 1.0:
                doc['score'] *= penalty
                doc['meta_penalty'] = penalty

        return documents

    def _apply_ast_filtering(
        self,
        documents: List[Dict],
        question_type: str,
        query_entity: str
    ) -> List[Dict]:
        """
        Apply AST-aware filtering by checking grammatical roles.

        For "Kiu kreis Esperanton?" - boost sentences where "esperant" is OBJECT (accusative)
        For "Kio estas Esperanto?" - boost sentences where "esperant" is SUBJECT
        For "Kie estas Pollando?" - boost sentences where "pol" is SUBJECT

        Args:
            documents: Retrieved documents with ASTs
            question_type: Question type (who/what/where/when)
            query_entity: Entity root to check (e.g., "esperant")

        Returns:
            Documents with AST-boosted scores
        """
        question_type_lower = question_type.lower()

        for doc in documents:
            ast = doc.get('ast')
            if not ast or ast.get('tipo') != 'frazo':
                continue

            # Check if entity appears in correct grammatical role
            boost_multiplier = 1.0

            # WHO questions: "Kiu kreis Esperanton?" → esperant should be OBJECT (accusative)
            if question_type_lower == 'who':
                if self._entity_in_object(ast, query_entity):
                    boost_multiplier = 3.0  # Strong boost for correct grammatical role
                    logger.debug(f"AST boost: {query_entity} in OBJECT position")

            # WHAT questions: "Kio estas Esperanto?" → esperant should be SUBJECT
            elif question_type_lower == 'what':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.5
                    logger.debug(f"AST boost: {query_entity} in SUBJECT position")
                elif self._entity_in_object(ast, query_entity):
                    boost_multiplier = 1.5  # Moderate boost for object position

            # WHERE questions: "Kie estas Pollando?" → pol should be SUBJECT
            elif question_type_lower == 'where':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.5
                elif self._entity_in_location_modifier(ast, query_entity):
                    boost_multiplier = 2.0  # "en Pollando" is also valid

            # WHEN questions: entity could be subject or object
            elif question_type_lower == 'when':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.0
                elif self._entity_in_object(ast, query_entity):
                    boost_multiplier = 2.0

            # Apply boost to BM25 score
            doc['score'] *= boost_multiplier
            if boost_multiplier > 1.0:
                doc['ast_boost'] = boost_multiplier

        return documents

    def _entity_in_subject(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears in subject position."""
        subjekto = ast.get('subjekto')
        if not subjekto:
            return False

        return self._contains_root(subjekto, entity_root)

    def _entity_in_object(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears in object position (accusative case)."""
        objekto = ast.get('objekto')
        if not objekto:
            return False

        # Check if object contains the root AND has accusative case (-n)
        if not self._contains_root(objekto, entity_root):
            return False

        # Verify accusative case
        return self._has_accusative_case(objekto)

    def _entity_in_location_modifier(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears as location modifier (en X, al X, etc.)."""
        aliaj = ast.get('aliaj', [])
        for alia in aliaj:
            if self._contains_root(alia, entity_root):
                # Check if it's a prepositional phrase with location preposition
                if isinstance(alia, dict):
                    if alia.get('tipo') == 'vorto':
                        radiko = alia.get('radiko', '')
                        if radiko in ['en', 'al', 'de', 'ĉe']:  # Location prepositions
                            return True
        return False

    def _contains_root(self, node: Dict, root: str) -> bool:
        """Recursively check if AST node contains a specific root."""
        if not isinstance(node, dict):
            return False

        # Check if this node has the root
        if node.get('radiko', '').lower() == root.lower():
            return True

        # Recursively check vortgrupo structure
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno and self._contains_root(kerno, root):
                return True

            priskriboj = node.get('priskriboj', [])
            for priskribo in priskriboj:
                if self._contains_root(priskribo, root):
                    return True

        return False

    def _has_accusative_case(self, node: Dict) -> bool:
        """Check if node has accusative case (-n ending)."""
        if not isinstance(node, dict):
            return False

        # Check direct accusative
        if node.get('kazo') == 'akuzativo':
            return True

        # Check vortgrupo kerno
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno and self._has_accusative_case(kerno):
                return True

        return False
