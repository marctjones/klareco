"""
Semantic signature extraction from ASTs.

Extracts (agent, action, patient) tuples from parsed sentences
for role-based semantic retrieval.

Example:
    "La hundo vidas la katon." -> (hund, vid, kat)
    "Frodo amas Samon." -> (Frod, am, Sam)
"""

from typing import Optional, Tuple, Dict, Any


def extract_signature(ast: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract semantic signature (agent, action, patient) from AST.

    Args:
        ast: Parsed sentence AST

    Returns:
        Tuple of (agent_root, action_root, patient_root)
        None values indicate missing components
    """
    if not ast or ast.get('tipo') != 'frazo':
        return (None, None, None)

    # Extract components
    subjekto = ast.get('subjekto')
    verbo = ast.get('verbo')
    objekto = ast.get('objekto')

    # Get roots
    aganto = get_root(subjekto) if subjekto else None
    ago = get_root(verbo) if verbo else None
    paciento = get_root(objekto) if objekto else None

    return (aganto, ago, paciento)


def get_root(node: dict) -> Optional[str]:
    """
    Extract root/radiko from a word or word group.

    Handles:
    - vorto: direct word with radiko
    - vortgrupo: word group with kerno (head)
    - Nested structures
    """
    if not node or not isinstance(node, dict):
        return None

    tipo = node.get('tipo')

    if tipo == 'vorto':
        return node.get('radiko')

    elif tipo == 'vortgrupo':
        # Word group - get the head (kerno)
        kerno = node.get('kerno')
        if kerno:
            return get_root(kerno)
        return None

    # Fallback - try to find radiko directly
    if 'radiko' in node:
        return node.get('radiko')

    return None


def signature_to_string(sig: Tuple[Optional[str], Optional[str], Optional[str]]) -> str:
    """
    Convert signature tuple to string for display/debugging.

    Example: (hund, vid, kat) -> "hund:vid:kat"
    """
    parts = [s if s else '*' for s in sig]
    return ':'.join(parts)


def signature_from_string(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse signature string back to tuple.

    Example: "hund:vid:kat" -> (hund, vid, kat)
             "*:vid:kat" -> (None, vid, kat)
    """
    parts = s.split(':')
    if len(parts) != 3:
        return (None, None, None)

    return tuple(p if p != '*' else None for p in parts)


def match_signature(
    query: Tuple[Optional[str], Optional[str], Optional[str]],
    candidate: Tuple[Optional[str], Optional[str], Optional[str]],
) -> float:
    """
    Score how well a candidate signature matches a query.

    Args:
        query: Query signature (None = wildcard, matches anything)
        candidate: Candidate signature to compare

    Returns:
        Match score 0.0 to 1.0
    """
    if not query or not candidate:
        return 0.0

    score = 0.0
    weights = [0.3, 0.4, 0.3]  # agent, action, patient

    for i, (q, c) in enumerate(zip(query, candidate)):
        if q is None:
            # Wildcard - matches anything, but doesn't add to score
            continue
        elif q == c:
            # Exact match
            score += weights[i]

    return score


def extract_all_signatures(ast: dict) -> list:
    """
    Extract all semantic signatures from an AST.

    For complex sentences with multiple clauses, this extracts
    a signature for each clause.

    Args:
        ast: Parsed AST (can be sentence or higher level)

    Returns:
        List of signature tuples
    """
    signatures = []

    if not ast or not isinstance(ast, dict):
        return signatures

    tipo = ast.get('tipo')

    if tipo == 'frazo':
        sig = extract_signature(ast)
        if any(sig):  # At least one non-None component
            signatures.append(sig)

    # Recurse into nested structures
    for key, value in ast.items():
        if isinstance(value, dict):
            signatures.extend(extract_all_signatures(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    signatures.extend(extract_all_signatures(item))

    return signatures


def get_signature_components(ast: dict) -> Dict[str, Any]:
    """
    Extract detailed signature components with metadata.

    Returns more than just roots - includes full word info.

    Args:
        ast: Parsed sentence AST

    Returns:
        Dict with agent, action, patient details
    """
    if not ast or ast.get('tipo') != 'frazo':
        return {}

    result = {}

    # Agent (subject)
    subjekto = ast.get('subjekto')
    if subjekto:
        result['agent'] = {
            'root': get_root(subjekto),
            'full_word': subjekto.get('plena_vorto', ''),
            'case': subjekto.get('kazo', 'nominativo'),
            'number': subjekto.get('nombro', 'singularo'),
        }

    # Action (verb)
    verbo = ast.get('verbo')
    if verbo:
        result['action'] = {
            'root': get_root(verbo),
            'full_word': verbo.get('plena_vorto', ''),
            'tense': verbo.get('tempo', ''),
        }

    # Patient (object)
    objekto = ast.get('objekto')
    if objekto:
        result['patient'] = {
            'root': get_root(objekto),
            'full_word': objekto.get('plena_vorto', ''),
            'case': objekto.get('kazo', 'akuzativo'),
            'number': objekto.get('nombro', 'singularo'),
        }

    return result
