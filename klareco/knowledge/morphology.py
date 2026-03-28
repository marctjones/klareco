"""
Morphological Normalization for Esperanto Roots

This module handles systematic morphological relationships in Esperanto,
particularly reflexive ↔ transitive verb pairs that BM25 keyword matching cannot capture.

KEY PROBLEM SOLVED:
- Query: "naskiĝis" (was born - reflexive) → root: naskiĝ
- Corpus: "naskita en Bjalistoko" (born - from transitive root "nask")
- Without normalization: ZERO OVERLAP! Query fails to find answer.
- With normalization: naskiĝ → {naskiĝ, nask} → MATCHES corpus!

ESPERANTO LINGUISTIC PATTERN:
- Transitive: "nask/i" (to give birth to) → "Mi naskis infanon" (I gave birth to a child)
- Reflexive: "naskiĝ/i" (to be born) → "Mi naskiĝis" (I was born)
- The -iĝ suffix systematically creates reflexive/middle voice from transitive verbs
- Both forms share the same semantic core but have different grammatical roles

This is the #1 priority fix identified in failure analysis (70% of failures are vocabulary mismatch).

Version: v2.1
Created: 2026-03-25
Expected Impact: +10-15% accuracy improvement
"""

from typing import Set, Dict

# Manual mapping of common reflexive ↔ transitive pairs
# These are high-frequency verbs where the relationship is semantically strong
REFLEXIVE_TRANSITIVE_PAIRS: Dict[str, str] = {
    # Birth/creation (CRITICAL for "Kie naskiĝis Zamenhof?" failure)
    'naskiĝ': 'nask',      # be born ↔ give birth
    'kreiĝ': 'kre',        # be created ↔ create
    'fondiĝ': 'fond',      # be founded ↔ found

    # Publication/appearance
    'publikitĝ': 'publik', # be published ↔ publish
    'apeitĝ': 'aper',      # appear (intrans) ↔ make appear

    # Construction/establishment
    'konstruiĝ': 'konstru', # be built ↔ build
    'establiĝ': 'establ',   # be established ↔ establish
    'formiĝ': 'form',       # be formed ↔ form

    # Change/transformation
    'ŝanĝiĝ': 'ŝanĝ',      # change (intrans) ↔ change (trans)
    'modiĝ': 'modif',      # be modified ↔ modify

    # Location/placement
    'troiĝ': 'trov',       # be found/located ↔ find
    'situiĝ': 'situ',      # be situated ↔ situate

    # Death/ending
    'mortiĝ': 'mort',      # die ↔ kill
    'finiĝ': 'fin',        # end (intrans) ↔ end (trans)

    # Opening/closing
    'fermiĝ': 'ferm',      # close (intrans) ↔ close (trans)
    'malfermiĝ': 'malferm', # open (intrans) ↔ open (trans)

    # Development/growth
    'disvolviĝ': 'disvolvl', # develop (intrans) ↔ develop (trans)
    'kresιĝ': 'kresk',      # grow (intrans) ↔ grow (trans)

    # Begin/start
    'komeciĝ': 'komenc',   # begin (intrans) ↔ begin (trans)
    'startiĝ': 'start',    # start (intrans) ↔ start (trans)
}


def normalize_reflexive_root(root: str) -> Set[str]:
    """
    Given a root, return all morphological variants including reflexive ↔ transitive.

    This function handles the systematic Esperanto pattern where:
    - Adding -iĝ to a transitive verb creates a reflexive/middle voice
    - Removing -iĝ from a reflexive verb reveals the transitive base

    Args:
        root: An Esperanto root (e.g., "naskiĝ", "nask", "kre")

    Returns:
        Set of morphologically related roots

    Examples:
        >>> normalize_reflexive_root("naskiĝ")
        {'naskiĝ', 'nask'}  # Reflexive + transitive base

        >>> normalize_reflexive_root("nask")
        {'nask', 'naskiĝ'}  # Transitive + reflexive form

        >>> normalize_reflexive_root("kre")
        {'kre', 'kreiĝ'}    # Create + be created

        >>> normalize_reflexive_root("hund")
        {'hund'}            # No morphological variants (not a verb)
    """
    roots = {root}  # Always include the original root

    # Check manual mapping (both directions)
    if root in REFLEXIVE_TRANSITIVE_PAIRS:
        # Root is reflexive (-iĝ form), add transitive base
        transitive = REFLEXIVE_TRANSITIVE_PAIRS[root]
        roots.add(transitive)

    # Reverse lookup: root is transitive, find reflexive
    reverse_map = {v: k for k, v in REFLEXIVE_TRANSITIVE_PAIRS.items()}
    if root in reverse_map:
        reflexive = reverse_map[root]
        roots.add(reflexive)

    # Systematic pattern: if root ends in -iĝ, also try base form
    # This catches reflexive verbs not in the manual mapping
    if root.endswith('iĝ'):
        base = root[:-2]  # Remove -iĝ suffix
        if len(base) >= 2:  # Sanity check (avoid single-letter roots)
            roots.add(base)

    # Systematic pattern: for any root, also try adding -iĝ
    # This catches transitive verbs not in the manual mapping
    else:
        reflexive_form = root + 'iĝ'
        # Only add if this looks like a valid verb root
        # (This is a heuristic - we don't want to add -iĝ to nouns)
        # We'll be conservative and only add for roots in the manual mapping
        if root in reverse_map:
            roots.add(reflexive_form)

    return roots


def expand_with_morphology(roots: Set[str]) -> Set[str]:
    """
    Expand a set of roots with all morphological variants.

    This is the main entry point for query expansion. Given a set of query roots,
    it returns an expanded set including all reflexive ↔ transitive variants.

    Args:
        roots: Set of query roots

    Returns:
        Expanded set with morphological variants

    Example:
        >>> expand_with_morphology({'naskiĝ', 'zamenhof'})
        {'naskiĝ', 'nask', 'zamenhof'}
        # naskiĝ expanded to include nask, zamenhof unchanged (not a verb)
    """
    expanded = set()

    for root in roots:
        # Add original root plus all morphological variants
        expanded.update(normalize_reflexive_root(root))

    return expanded


def is_reflexive_verb(root: str) -> bool:
    """
    Check if a root is a reflexive verb (ends in -iĝ or is in manual mapping).

    Args:
        root: Esperanto root

    Returns:
        True if root is a reflexive verb

    Examples:
        >>> is_reflexive_verb("naskiĝ")
        True
        >>> is_reflexive_verb("nask")
        False
        >>> is_reflexive_verb("hund")
        False
    """
    return root.endswith('iĝ') or root in REFLEXIVE_TRANSITIVE_PAIRS


def get_transitive_base(reflexive_root: str) -> str:
    """
    Get the transitive base form of a reflexive verb.

    Args:
        reflexive_root: A reflexive verb root (ending in -iĝ)

    Returns:
        Transitive base root, or original root if not reflexive

    Examples:
        >>> get_transitive_base("naskiĝ")
        'nask'
        >>> get_transitive_base("kreiĝ")
        'kre'
        >>> get_transitive_base("hund")
        'hund'  # Not a reflexive verb, return unchanged
    """
    # Check manual mapping first (more accurate)
    if reflexive_root in REFLEXIVE_TRANSITIVE_PAIRS:
        return REFLEXIVE_TRANSITIVE_PAIRS[reflexive_root]

    # Fallback: systematic pattern
    if reflexive_root.endswith('iĝ'):
        return reflexive_root[:-2]

    # Not a reflexive verb
    return reflexive_root


def get_reflexive_form(transitive_root: str) -> str:
    """
    Get the reflexive form of a transitive verb.

    Args:
        transitive_root: A transitive verb root

    Returns:
        Reflexive form (with -iĝ), or original root if not in mapping

    Examples:
        >>> get_reflexive_form("nask")
        'naskiĝ'
        >>> get_reflexive_form("kre")
        'kreiĝ'
        >>> get_reflexive_form("hund")
        'hund'  # Not a verb, return unchanged
    """
    # Check reverse manual mapping
    reverse_map = {v: k for k, v in REFLEXIVE_TRANSITIVE_PAIRS.items()}
    if transitive_root in reverse_map:
        return reverse_map[transitive_root]

    # Not in mapping - return unchanged (don't guess)
    return transitive_root
