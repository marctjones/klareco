"""
Synonym Ranking - Rank verb class members by semantic closeness

This module provides functionality to rank verb synonyms from the semantic
ontology by their semantic distance to the query term.

Since all members of a verb class are treated equally in the ontology,
we use heuristics to rank them by likely semantic closeness:

1. **Core creation verbs**: fond, kre, establ, konstru
2. **Production verbs**: far, produk, fabrik
3. **Design/invention verbs**: desegn, invent, develop
4. **Composition verbs**: kompon, aŭtor, form
5. **Starting verbs**: komenc, startig

Version: v2.2
Created: 2026-03-28
"""

from typing import List, Set

# Manual ranking of verb class members by semantic closeness
# These are ordered from most general/core to most specific/domain-specific

VERB_CLASS_CORE_MEMBERS = {
    'kreado-26': [
        # Tier 1: Core creation (most general)
        ['kre', 'fond', 'establ'],
        # Tier 2: Construction/making
        ['konstru', 'far', 'produk'],
        # Tier 3: Design/invention
        ['desegn', 'invent', 'develop'],
        # Tier 4: Specific domains (writing, manufacturing, starting)
        ['kompon', 'aŭtor', 'fabrik', 'form', 'komenc', 'startig'],
    ],

    'movo-51': [
        # Tier 1: General movement
        ['mov', 'ir', 'ven'],
        # Tier 2: Travel
        ['vetur', 'vojaĝ', 'migr'],
        # Tier 3: Specific movements
        ['salt', 'mar', 'kur', 'flug'],
    ],

    # Add more verb classes as needed
}


def rank_synonyms(root: str, synonyms: List[str], max_synonyms: int = 3) -> List[str]:
    """
    Rank verb synonyms by semantic closeness to root.

    Args:
        root: Original query root
        synonyms: List of all verb class members (from semantic ontology)
        max_synonyms: Maximum number of synonyms to return

    Returns:
        List of top N synonyms, ranked by semantic closeness
    """
    if not synonyms:
        return []

    # Find which verb class this root belongs to
    for klaso_id, tiers in VERB_CLASS_CORE_MEMBERS.items():
        for tier in tiers:
            if root in tier:
                # Root found! Rank synonyms by tier proximity
                ranked = []

                # Add synonyms from same tier first
                for syn in tier:
                    if syn != root and syn in synonyms:
                        ranked.append(syn)

                # Then add from adjacent tiers
                tier_idx = tiers.index(tier)

                # Check tier above (more general)
                if tier_idx > 0:
                    for syn in tiers[tier_idx - 1]:
                        if syn != root and syn in synonyms and syn not in ranked:
                            ranked.append(syn)

                # Check tier below (more specific)
                if tier_idx < len(tiers) - 1:
                    for syn in tiers[tier_idx + 1]:
                        if syn != root and syn in synonyms and syn not in ranked:
                            ranked.append(syn)

                # Add remaining synonyms (from distant tiers)
                for syn in synonyms:
                    if syn != root and syn not in ranked:
                        ranked.append(syn)

                return ranked[:max_synonyms]

    # Fallback: root not in manual rankings, return first N alphabetically
    # (alphabetical sorting ensures consistency)
    return sorted([s for s in synonyms if s != root])[:max_synonyms]


def get_top_synonyms(root: str, all_synonyms: List[str], max_count: int = 3) -> List[str]:
    """
    Convenience function to get top N semantically close synonyms.

    Args:
        root: Query root
        all_synonyms: All members of the verb class (from semantic ontology)
        max_count: Maximum number to return (default: 3)

    Returns:
        List of top N closest synonyms
    """
    return rank_synonyms(root, all_synonyms, max_synonyms=max_count)
