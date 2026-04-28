"""
Grammatical Variant Generator for AST-First Retrieval

Generates grammatical variants of query patterns using Esperanto morphology.
This enables robust retrieval while maintaining grammatical precision.

Phase 4 of AST-First Retrieval Improvements:
- Active/passive voice (already implemented in whoosh_retriever.py)
- Participial constructions: "Zamenhof, la fondinto de Esperanto"
- Nominalizations: "La fondado de Esperanto fare de Zamenhof"
- Relative clauses: "Zamenhof, kiu fondis Esperanton"
- Appositives: "Zamenhof, la kreanto de Esperanto"

Expected Impact: +3% retrieval recall
See: docs/AST_FIRST_RETRIEVAL_DESIGN.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Types of grammatical variants for query patterns."""
    ACTIVE = "active"              # "Zamenhof fondis Esperanton" (1.0 confidence)
    PASSIVE = "passive"            # "Esperanto estis fondita de Zamenhof" (0.9 confidence)
    PARTICIPIAL = "participial"    # "Zamenhof, la fondinto de Esperanto" (0.8 confidence)
    NOMINALIZATION = "nominalization"  # "La fondado de Esperanto" (0.7 confidence)
    RELATIVE_CLAUSE = "relative_clause"  # "Zamenhof, kiu fondis Esperanton" (0.85 confidence)
    APPOSITIVE = "appositive"      # "Zamenhof, la kreanto" (0.75 confidence)


@dataclass
class GrammaticalVariant:
    """
    A grammatical variant of a query pattern with its Kuzu query.

    Attributes:
        pattern_type: Type of grammatical variant
        cypher_query: Kuzu graph pattern (Cypher query)
        confidence: How likely this variant matches intent (0.0-1.0)
        description: Human-readable description of what this variant matches
    """
    pattern_type: VariantType
    cypher_query: str
    confidence: float
    description: str


class GrammaticalVariantGenerator:
    """
    Generates grammatical variants for AST-based retrieval queries.

    Key Innovation: Instead of falling back to BM25 keywords, we expand
    to grammatical variants of the query pattern while maintaining
    grammatical precision.

    Example:
        Query: "Kiu fondis Esperanton?"

        Variants:
        1. Active (1.0): "Zamenhof fondis Esperanton"
        2. Passive (0.9): "Esperanto estis fondita de Zamenhof"
        3. Participial (0.8): "Zamenhof, la fondinto de Esperanto"
        4. Relative (0.85): "Zamenhof, kiu fondis Esperanton"
    """

    def __init__(self):
        """Initialize the grammatical variant generator."""
        pass

    def generate_who_variants(
        self,
        verb_root: str,
        verb_synonyms: List[str],
        obj_root: str,
        top_k: int = 200
    ) -> List[GrammaticalVariant]:
        """
        Generate grammatical variants for WHO questions.

        Args:
            verb_root: Main verb root (e.g., "fond")
            verb_synonyms: List of synonym verb roots (e.g., ["kre", "establ"])
            obj_root: Object entity root (e.g., "esperant")
            top_k: Limit for Kuzu queries

        Returns:
            List of grammatical variants with Cypher queries

        Example:
            Query: "Kiu fondis Esperanton?"
            Variants:
            1. Active: "X fondis Esperanton" (handled by retriever)
            2. Passive: "Esperanto estis fondita" (handled by retriever)
            3. Participial: "X, la fondinto de Esperanto"
            4. Relative clause: "X, kiu fondis Esperanton"
            5. Appositive: "X, la kreanto de Esperanto"
        """
        variants = []
        all_verbs = [verb_root] + list(verb_synonyms)
        verb_list = "', '".join(all_verbs)

        # Variant 3: Participial construction
        # Pattern: [agent], la VERB-into de [patient]
        # Example: "Zamenhof, la fondinto de Esperanto"
        participial_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_ALIAJN]->(participle:Vorto)
            WHERE participle.radiko IN ['{verb_list}']
                AND participle.sufiksoj CONTAINS 'int'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(prep_de:Vorto)
            WHERE prep_de.radiko = 'de'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(obj_mention:Vorto)
            WHERE obj_mention.radiko = '{obj_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.PARTICIPIAL,
            cypher_query=participial_query,
            confidence=0.8,
            description=f"Participial: 'X, la {verb_root}into de {obj_root}'"
        ))

        # Variant 4: Relative clause with "kiu"
        # Pattern: [agent], kiu VERB-is [patient]
        # Example: "Zamenhof, kiu fondis Esperanton"
        relative_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_ALIAJN]->(kiu:Vorto)
            WHERE kiu.radiko = 'kiu'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko IN ['{verb_list}']
            MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj:Vorto)
            WHERE obj.radiko = '{obj_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.RELATIVE_CLAUSE,
            cypher_query=relative_query,
            confidence=0.85,
            description=f"Relative clause: 'X, kiu {verb_root}is {obj_root}on'"
        ))

        # Variant 5: Appositive with synonym verbs
        # Pattern: [agent], la VERB-into
        # Example: "Zamenhof, la kreanto de Esperanto"
        appositive_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_ALIAJN]->(noun:Vorto)
            WHERE noun.radiko IN ['{verb_list}']
                AND (noun.sufiksoj CONTAINS 'ant' OR noun.sufiksoj CONTAINS 'int')
            MATCH (frazo)-[:HAVAS_ALIAJN]->(obj_mention:Vorto)
            WHERE obj_mention.radiko = '{obj_root}'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.APPOSITIVE,
            cypher_query=appositive_query,
            confidence=0.75,
            description=f"Appositive: 'X, la {verb_root}anto/into de {obj_root}'"
        ))

        logger.info(f"Generated {len(variants)} grammatical variants for WHO question")
        return variants

    def generate_what_variants(
        self,
        entity_root: str,
        top_k: int = 200
    ) -> List[GrammaticalVariant]:
        """
        Generate grammatical variants for WHAT questions.

        Args:
            entity_root: Entity to define (e.g., "hund")
            top_k: Limit for Kuzu queries

        Returns:
            List of grammatical variants

        Example:
            Query: "Kio estas hundo?"
            Variants:
            1. IS-A: "Hundo estas besto" (handled by retriever)
            2. Appositive: "Hundo, besto kiu..."
            3. Relative clause: "Hundo, kiu estas besto"
        """
        variants = []

        # Variant 2: Appositive definition
        # Pattern: [entity], [definition] kiu...
        # Example: "Hundo, besto kiu havas..."
        appositive_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(kiu:Vorto)
            WHERE kiu.radiko = 'kiu'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.APPOSITIVE,
            cypher_query=appositive_query,
            confidence=0.7,
            description=f"Appositive: '{entity_root}, [tipo] kiu...'"
        ))

        # Variant 3: Relative clause definition
        # Pattern: [entity], kiu estas [definition]
        # Example: "Hundo, kiu estas besto"
        relative_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            WHERE verb.radiko = 'est'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(kiu:Vorto)
            WHERE kiu.radiko = 'kiu'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.RELATIVE_CLAUSE,
            cypher_query=relative_query,
            confidence=0.75,
            description=f"Relative clause: '{entity_root}, kiu estas [tipo]'"
        ))

        logger.info(f"Generated {len(variants)} grammatical variants for WHAT question")
        return variants

    def generate_where_variants(
        self,
        verb_root: str,
        verb_synonyms: List[str],
        entity_root: str,
        top_k: int = 200
    ) -> List[GrammaticalVariant]:
        """
        Generate grammatical variants for WHERE questions.

        Args:
            verb_root: Main verb root (e.g., "nask")
            verb_synonyms: List of synonym verb roots
            entity_root: Entity subject root (e.g., "zamenhof")
            top_k: Limit for Kuzu queries

        Returns:
            List of grammatical variants

        Example:
            Query: "Kie naskiĝis Zamenhof?"
            Variants:
            1. Active: "Zamenhof naskiĝis en [loko]" (handled by retriever)
            2. Participial: "Zamenhof, naskita en [loko]"
            3. Nominalization: "La naskiĝo de Zamenhof en [loko]"
        """
        variants = []
        all_verbs = [verb_root] + list(verb_synonyms)
        verb_list = "', '".join(all_verbs)

        # Variant 2: Participial with location
        # Pattern: [entity], VERB-ita en [loko]
        # Example: "Zamenhof, naskita en Bjalistoko"
        participial_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(participle:Vorto)
            WHERE participle.radiko IN ['{verb_list}']
                AND participle.sufiksoj CONTAINS 'it'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(prep_en:Vorto)
            WHERE prep_en.radiko = 'en'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.PARTICIPIAL,
            cypher_query=participial_query,
            confidence=0.8,
            description=f"Participial: '{entity_root}, {verb_root}ita en [loko]'"
        ))

        # Variant 3: Nominalization
        # Pattern: La VERB-o de [entity] en [loko]
        # Example: "La naskiĝo de Zamenhof en Bjalistoko"
        nominalization_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko IN ['{verb_list}']
                AND subj.vortspeco = 'substantivo'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(entity_mention:Vorto)
            WHERE entity_mention.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(prep_en:Vorto)
            WHERE prep_en.radiko = 'en'
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.NOMINALIZATION,
            cypher_query=nominalization_query,
            confidence=0.7,
            description=f"Nominalization: 'La {verb_root}o de {entity_root} en [loko]'"
        ))

        logger.info(f"Generated {len(variants)} grammatical variants for WHERE question")
        return variants

    def generate_when_variants(
        self,
        verb_root: str,
        verb_synonyms: List[str],
        entity_root: str,
        top_k: int = 200
    ) -> List[GrammaticalVariant]:
        """
        Generate grammatical variants for WHEN questions.

        Args:
            verb_root: Main verb root (e.g., "fond")
            verb_synonyms: List of synonym verb roots
            entity_root: Entity subject/object root (e.g., "esperant")
            top_k: Limit for Kuzu queries

        Returns:
            List of grammatical variants

        Example:
            Query: "Kiam estis fondita Esperanto?"
            Variants:
            1. Passive: "Esperanto estis fondita en 1887" (handled by retriever)
            2. Nominalization: "La fondado de Esperanto en 1887"
            3. Participial: "Esperanto, fondita en 1887"
        """
        variants = []
        all_verbs = [verb_root] + list(verb_synonyms)
        verb_list = "', '".join(all_verbs)

        # Variant 2: Nominalization with temporal
        # Pattern: La VERB-o de [entity] en [tempo]
        # Example: "La fondado de Esperanto en 1887"
        nominalization_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko IN ['{verb_list}']
                AND subj.vortspeco = 'substantivo'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(entity_mention:Vorto)
            WHERE entity_mention.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(temporal:Vorto)
            WHERE temporal.radiko IN ['jar', 'jarcent', 'dato', 'temp']
                OR regexp_matches(temporal.plena_vorto, '\\\\d{{4}}')
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.NOMINALIZATION,
            cypher_query=nominalization_query,
            confidence=0.75,
            description=f"Nominalization: 'La {verb_root}o de {entity_root} en [tempo]'"
        ))

        # Variant 3: Participial with temporal
        # Pattern: [entity], VERB-ita en [tempo]
        # Example: "Esperanto, fondita en 1887"
        participial_query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
            WHERE subj.radiko = '{entity_root}'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(participle:Vorto)
            WHERE participle.radiko IN ['{verb_list}']
                AND participle.sufiksoj CONTAINS 'it'
            MATCH (frazo)-[:HAVAS_ALIAJN]->(temporal:Vorto)
            WHERE temporal.radiko IN ['jar', 'jarcent', 'dato', 'temp']
                OR regexp_matches(temporal.plena_vorto, '\\\\d{{4}}')
            RETURN ft.id AS id, ft.teksto AS text
            LIMIT {top_k}
        """

        variants.append(GrammaticalVariant(
            pattern_type=VariantType.PARTICIPIAL,
            cypher_query=participial_query,
            confidence=0.8,
            description=f"Participial: '{entity_root}, {verb_root}ita en [tempo]'"
        ))

        logger.info(f"Generated {len(variants)} grammatical variants for WHEN question")
        return variants
