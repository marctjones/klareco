"""
Synonym Relations for Esperanto Verbs and Nouns

This module provides unified synonym dictionaries used for:
- Answer extraction (verb matching in ASTAnswerExtractor)
- Query expansion (retrieval in Whoosh)

Sources:
- Hand-curated MANUAL_VERB_SYNONYMS (historical: drawn from the old
  rag/answer_extractor.py during the v2.2 schema-first migration)
- Semantic ontology verb classes (v2.2+) — queried via
  klareco.knowledge.semantic_bridge.get_verb_synonyms_from_ontology
- Merged and deduplicated for consistency

Format: Dict[str, Set[str]]
  - Key: root (radiko)
  - Value: Set of synonym roots

Version: v2.2 (Now uses semantic ontology + fallback)
Created: 2026-03-25
Updated: 2026-03-28 - Integrated with semantic ontology
"""

from typing import Dict, Set
from .semantic_bridge import get_verb_synonyms_from_ontology
from .synonym_ranking import get_top_synonyms

# Verb Synonyms (for answer extraction and query expansion)
verb_synonyms: Dict[str, Set[str]] = {
    # Create/found verbs
    'fond': {'kre', 'establ', 'komenc', 'startig'},
    'kre': {'fond', 'establ', 'far', 'produk', 'aŭtor', 'verk', 'invent'},
    'establ': {'fond', 'kre', 'startig', 'komenc'},
    'komenc': {'fond', 'start', 'ekig', 'establ'},
    'invent': {'kre', 'far', 'kre'},

    # Birth/death/life verbs
    'naski': {'nask', 'genat'},
    'nask': {'naski', 'genat'},
    'mort': {'perdiĝ', 'forpas'},
    'viv': {'ekzist', 'log', 'rest'},

    # Make/produce verbs
    'far': {'kre', 'produk', 'fabrik'},
    'produk': {'far', 'kre', 'fabrik'},

    # Write/publish verbs
    'skrib': {'redakt', 'kompoz', 'ver', 'verk', 'aŭtor'},
    'verk': {'skrib', 'kre', 'aŭtor', 'publik'},
    'publik': {'eldone', 'aperig', 'verk'},
    'redakt': {'skrib', 'verk'},

    # Say/speak verbs
    'dir': {'parol', 'ekster', 'ekspr'},
    'parol': {'dir', 'ekster', 'ekspr', 'ling', 'idiom', 'lingv'},

    # Think/know/understand verbs
    'pens': {'opini', 'kred', 'konsider'},
    'sci': {'kon', 'kompren'},
    'kon': {'sci', 'kompren'},

    # See/observe verbs
    'vid': {'rimark', 'konsider', 'observ', 'rigard', 'pert'},
    'rimark': {'vid', 'observ', 'konsider'},
    'observ': {'vid', 'rimark', 'konsider'},

    # Definition/description verbs (WHAT questions - Quick Win #1)
    'est': {'difin', 'signif', 'konsist', 'represent', 'nomiĝ', 'konstitu'},
    'difin': {'est', 'signif', 'klarig', 'priskrib'},
    'signif': {'est', 'difin', 'vol_dir', 'reprezent'},
    'konsist': {'est', 'komponiĝ', 'enhavas', 'konsist_el'},
    'represent': {'est', 'signif', 'simbol', 'egal'},
    'priskrib': {'difin', 'klarig', 'prezent', 'raport'},
    'klarig': {'difin', 'eksplik', 'priskrib'},

    # Design/invention verbs (expanded)
    'desegn': {'kre', 'invent', 'plan', 'far'},
    'konstruk': {'far', 'kre', 'edif', 'startig'},
    'edif': {'konstruk', 'far', 'kre'},

    # Temporal verbs (expanded)
    'komenciĝ': {'komenc', 'startiĝ', 'okaz', 'ekest'},
    'startiĝ': {'komenc', 'komenciĝ', 'ekest'},
    'finiĝ': {'fin', 'ĉes', 'komplet'},
    'daŭr': {'kontinu', 'persist', 'rest'},
    'okaz': {'pas', 'event', 'far', 'realiĝ'},

    # Location/situation verbs
    'situ': {'est', 'trov', 'lok', 'kuŝ'},
    'trov': {'est', 'situ', 'lok', 'ekzist'},
    'lok': {'situ', 'trov', 'pozici'},
    'kuŝ': {'situ', 'est', 'lok'},
}

# Noun Synonyms (for query expansion)
noun_synonyms: Dict[str, Set[str]] = {
    # Language-related
    'ling': {'parol', 'idiom', 'lingv'},
    'lingv': {'ling', 'parol', 'idiom'},
    'idiom': {'ling', 'lingv', 'parol'},

    # Book/document
    'libr': {'dokument', 'verk', 'skribaĵ'},
    'dokument': {'libr', 'skribaĵ', 'tekst'},
    'tekst': {'dokument', 'skribaĵ'},

    # Person/human
    'person': {'hom', 'individu'},
    'hom': {'person', 'individu'},
    'individu': {'person', 'hom'},

    # Learn/study
    'lern': {'stud', 'eduk'},
    'stud': {'lern', 'eduk'},
    'eduk': {'lern', 'stud'},

    # Sports/games (Quick Win #1 - domain knowledge)
    'sport': {'lud', 'atletik', 'konkurs'},
    'lud': {'sport', 'ĝem', 'konkurs'},
    'basketbal': {'sport', 'lud', 'pilkosport'},
    'futbal': {'sport', 'lud', 'pilkosport'},
    'volbal': {'sport', 'lud', 'pilkosport'},
    'besbal': {'sport', 'lud', 'pilkosport'},

    # Science/field
    'scienç': {'fak', 'stud', 'disciplin'},
    'fak': {'scienç', 'disciplin', 'branĉ'},
    'fizik': {'scienç', 'fak', 'natur'},
    'kemi': {'scienç', 'fak'},
    'biologi': {'scienç', 'fak', 'vivscienç'},

    # Invention/creation
    'invent': {'kre', 'eltrov', 'nov', 'kresk'},
    'eltrov': {'invent', 'malkovr', 'trov'},
    'malkovr': {'eltrov', 'trov', 'skov'},
}


def are_synonyms(root1: str, root2: str) -> bool:
    """
    Check if two roots are synonyms (in either verb or noun dictionaries).

    Args:
        root1: First root
        root2: Second root

    Returns:
        True if roots are synonyms
    """
    if root1 == root2:
        return True

    # Check verb synonyms
    if root1 in verb_synonyms and root2 in verb_synonyms[root1]:
        return True
    if root2 in verb_synonyms and root1 in verb_synonyms[root2]:
        return True

    # Check noun synonyms
    if root1 in noun_synonyms and root2 in noun_synonyms[root1]:
        return True
    if root2 in noun_synonyms and root1 in noun_synonyms[root2]:
        return True

    return False


def get_synonyms(root: str, max_count: int = 3) -> Set[str]:
    """
    Get top N semantically closest synonyms for a root.

    Uses semantic ontology with intelligent ranking to limit to most relevant synonyms.
    This prevents query expansion dilution (v2.2 issue: 8x expansion caused 24% accuracy).

    Args:
        root: Root to look up
        max_count: Maximum number of synonyms to return (default: 3)

    Returns:
        Set of top N closest synonym roots (empty if none found)
    """
    synonyms = set()

    # First try semantic ontology for verb synonyms (verb class members)
    ontology_synonyms = get_verb_synonyms_from_ontology(root)
    if ontology_synonyms and len(ontology_synonyms) > 1:  # More than just root itself
        # Rank by semantic closeness, take only top N
        ranked = get_top_synonyms(root, list(ontology_synonyms), max_count=max_count)
        synonyms.update(ranked)
    else:
        # Fallback: use hardcoded synonyms (but still limit to max_count)
        if root in verb_synonyms:
            # Take up to max_count from hardcoded synonyms
            hardcoded = list(verb_synonyms[root])[:max_count]
            synonyms.update(hardcoded)

        if root in noun_synonyms:
            # Take up to max_count from hardcoded synonyms
            hardcoded = list(noun_synonyms[root])[:max_count]
            synonyms.update(hardcoded)

    # Remove the root itself from synonyms (we don't want 'fond' in synonyms of 'fond')
    synonyms.discard(root)

    return synonyms
