"""
Semantic Bridge - Connects old knowledge API to new semantic ontology

This module provides backward-compatible access to semantic ontology data
while maintaining the same API as the old hardcoded gazetteers.

Version: v2.2
Created: 2026-03-28
"""

from typing import Dict, List, Set, Optional
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# Global connection (lazy-loaded)
_semantic_query = None


def get_semantic_query():
    """Lazy-load SemanticQuery connection."""
    global _semantic_query

    if _semantic_query is None:
        try:
            import kuzu
            from klareco.ontology import SemanticQuery
            from klareco.utils.kuzu_open import open_kuzu

            # Connect to default database. Honor KLARECO_KUZU_DB_PATH so
            # callers running outside the project root (e.g. Modal containers
            # with a volume-mounted index) can point at the real location.
            db_path = Path(os.environ.get(
                'KLARECO_KUZU_DB_PATH',
                'data/indexes/v2.1_kuzu_index_full',
            ))
            if not db_path.exists():
                logger.warning(f"Database not found at {db_path}, semantic features disabled")
                return None

            db = open_kuzu(db_path)
            conn = kuzu.Connection(db)
            _semantic_query = SemanticQuery(conn)

            logger.info("✓ SemanticQuery bridge initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize SemanticQuery: {e}")
            return None

    return _semantic_query


def get_place_names_from_ontology() -> Set[str]:
    """
    Get place names from semantic ontology.

    Returns:
        Set of place name roots (lowercase)
    """
    sq = get_semantic_query()
    if sq is None:
        return set()

    try:
        places = sq.get_entity_type_members('loko')
        return set(places)
    except Exception as e:
        logger.debug(f"Failed to get places from ontology: {e}")
        return set()


def get_person_roots_from_ontology() -> Set[str]:
    """
    Get person roots from semantic ontology.

    Returns:
        Set of person roots (lowercase)
    """
    sq = get_semantic_query()
    if sq is None:
        return set()

    try:
        persons = sq.get_entity_type_members('persono')
        return set(persons)
    except Exception as e:
        logger.debug(f"Failed to get persons from ontology: {e}")
        return set()


def get_temporal_roots_from_ontology() -> Set[str]:
    """
    Get temporal roots from semantic ontology.

    Returns:
        Set of temporal roots (lowercase)
    """
    sq = get_semantic_query()
    if sq is None:
        return set()

    try:
        temporal = sq.get_entity_type_members('tempo')
        return set(temporal)
    except Exception as e:
        logger.debug(f"Failed to get temporal words from ontology: {e}")
        return set()


def get_verb_synonyms_from_ontology(root: str) -> List[str]:
    """
    Get verb synonyms from semantic ontology.

    Args:
        root: Verb root

    Returns:
        List of synonym roots (including the root itself)
    """
    sq = get_semantic_query()
    if sq is None:
        return [root]

    try:
        synonyms = sq.get_verb_synonyms(root)
        return synonyms if synonyms else [root]
    except Exception as e:
        logger.debug(f"Failed to get verb synonyms for '{root}': {e}")
        return [root]


def is_person_from_ontology(root: str) -> bool:
    """
    Check if root is a person using semantic ontology.

    Args:
        root: Root to check (lowercase)

    Returns:
        True if root is classified as person
    """
    sq = get_semantic_query()
    if sq is None:
        return False

    try:
        return sq.is_person(root)
    except Exception as e:
        logger.debug(f"Failed to check if '{root}' is person: {e}")
        return False


def is_place_from_ontology(root: str) -> bool:
    """
    Check if root is a place using semantic ontology.

    Args:
        root: Root to check (lowercase)

    Returns:
        True if root is classified as place
    """
    sq = get_semantic_query()
    if sq is None:
        return False

    try:
        return sq.is_place(root)
    except Exception as e:
        logger.debug(f"Failed to check if '{root}' is place: {e}")
        return False


def is_time_from_ontology(root: str) -> bool:
    """
    Check if root is temporal using semantic ontology.

    Args:
        root: Root to check (lowercase)

    Returns:
        True if root is classified as temporal
    """
    sq = get_semantic_query()
    if sq is None:
        return False

    try:
        return sq.is_time(root)
    except Exception as e:
        logger.debug(f"Failed to check if '{root}' is time: {e}")
        return False


# Cache for merged data (combines ontology + fallback)
_merged_places = None
_merged_persons = None
_merged_temporal = None


def get_merged_places(fallback_places: Set[str]) -> Set[str]:
    """
    Get places from ontology merged with fallback data.

    Args:
        fallback_places: Hardcoded fallback places

    Returns:
        Merged set of places
    """
    global _merged_places

    if _merged_places is None:
        ontology_places = get_place_names_from_ontology()
        _merged_places = ontology_places | fallback_places

        logger.info(f"Place names: {len(ontology_places)} from ontology + "
                   f"{len(fallback_places)} fallback = {len(_merged_places)} total")

    return _merged_places


def get_merged_persons(fallback_persons: Set[str]) -> Set[str]:
    """
    Get persons from ontology merged with fallback data.

    Args:
        fallback_persons: Hardcoded fallback persons

    Returns:
        Merged set of persons
    """
    global _merged_persons

    if _merged_persons is None:
        ontology_persons = get_person_roots_from_ontology()
        _merged_persons = ontology_persons | fallback_persons

        logger.info(f"Person roots: {len(ontology_persons)} from ontology + "
                   f"{len(fallback_persons)} fallback = {len(_merged_persons)} total")

    return _merged_persons


def get_merged_temporal(fallback_temporal: Set[str]) -> Set[str]:
    """
    Get temporal words from ontology merged with fallback data.

    Args:
        fallback_temporal: Hardcoded fallback temporal words

    Returns:
        Merged set of temporal words
    """
    global _merged_temporal

    if _merged_temporal is None:
        ontology_temporal = get_temporal_roots_from_ontology()
        _merged_temporal = ontology_temporal | fallback_temporal

        logger.info(f"Temporal words: {len(ontology_temporal)} from ontology + "
                   f"{len(fallback_temporal)} fallback = {len(_merged_temporal)} total")

    return _merged_temporal
