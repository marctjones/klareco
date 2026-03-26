"""
Synonym Relations for Esperanto Verbs and Nouns

This module provides unified synonym dictionaries used for:
- Answer extraction (verb matching in ASTAnswerExtractor)
- Query expansion (retrieval in Whoosh)

Sources:
- MANUAL_VERB_SYNONYMS from klareco/rag/answer_extractor.py
- MANUAL_SYNONYMS from scripts/demo_extractive_qa.py
- Merged and deduplicated for consistency

Format: Dict[str, Set[str]]
  - Key: root (radiko)
  - Value: Set of synonym roots

Version: v2.1
Created: 2026-03-25
"""

from typing import Dict, Set

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


def get_synonyms(root: str) -> Set[str]:
    """
    Get all synonyms for a root (from both verb and noun dictionaries).

    Args:
        root: Root to look up

    Returns:
        Set of synonym roots (empty if none found)
    """
    synonyms = set()

    if root in verb_synonyms:
        synonyms.update(verb_synonyms[root])

    if root in noun_synonyms:
        synonyms.update(noun_synonyms[root])

    return synonyms
