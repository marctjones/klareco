"""
AST Semantic Ranker - Rank AST Matches by Structural and Semantic Similarity

Addresses the "no ranking" problem: Kuzu graph queries return grammatically valid
matches in arbitrary order. This ranker scores matches by relevance.

Architecture:
- Input: Query AST + List of candidate ASTs (from graph query)
- Output: Relevance scores for each candidate
- Approach: Deterministic structural features + minimal learned (root embeddings)

Scoring Components:
1. Verb synonym distance (40%): How close is the verb synonym?
2. Object match quality (30%): Does the object match exactly?
3. Entity prominence (20%): Is the answer entity the subject (vs buried in modifier)?
4. Root similarity (10%): Embedding similarity (optional, learned component)

Aligns with Pure Esperanto AI thesis:
- Deterministic structure (AST comparison)
- Minimal learned (only root embeddings for similarity)
- Explainable (can show breakdown of scores)
"""

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_ast_verb_root(ast: Dict) -> Optional[str]:
    """
    Extract verb root from AST.

    For questions, verb is in 'aliaj' field (e.g., "Kiu fondis Esperanton?")
    For statements, verb is in 'verbo' field (e.g., "Zamenhof fondis Esperanton")
    """
    if not ast or ast.get('tipo') != 'frazo':
        return None

    # Check main verb field first
    verbo = ast.get('verbo')
    if verbo:
        return verbo.get('radiko')

    # For questions, check aliaj field
    aliaj = ast.get('aliaj', [])
    for item in aliaj:
        if item.get('vortspeco') == 'verbo':
            return item.get('radiko')

    return None


def get_ast_object_root(ast: Dict) -> Optional[str]:
    """
    Extract object root from AST.

    For questions, object is in 'aliaj' with kazo='akuzativo'
    For statements, object is in 'objekto' field
    """
    if not ast or ast.get('tipo') != 'frazo':
        return None

    # Check main object field first
    objekto = ast.get('objekto')
    if objekto:
        # Object can be vortgrupo (word group) or single vorto
        if objekto.get('tipo') == 'vortgrupo':
            kerno = objekto.get('kerno')
            if kerno:
                return kerno.get('radiko')
        else:
            return objekto.get('radiko')

    # For questions, check aliaj field for accusative nouns
    aliaj = ast.get('aliaj', [])
    for item in aliaj:
        if item.get('kazo') == 'akuzativo' and item.get('vortspeco') in ['substantivo', 'propra_nomo']:
            return item.get('radiko')

    return None


def get_ast_subject_root(ast: Dict) -> Optional[str]:
    """Extract subject root from AST."""
    if not ast or ast.get('tipo') != 'frazo':
        return None

    subjekto = ast.get('subjekto')
    if not subjekto:
        return None

    # Subject can be vortgrupo or single vorto
    if subjekto.get('tipo') == 'vortgrupo':
        kerno = subjekto.get('kerno')
        if kerno:
            return kerno.get('radiko')
    else:
        return subjekto.get('radiko')

    return None


def is_proper_noun(ast_node: Dict) -> bool:
    """Check if AST node represents a proper noun (entity)."""
    if not ast_node:
        return False

    # Check if it's marked as proper noun
    parse_status = ast_node.get('parse_status')
    if parse_status in ['proper_name', 'proper_name_unknown']:
        return True

    # Check if first letter is capitalized (heuristic)
    radiko = ast_node.get('radiko', '')
    if radiko and radiko[0].isupper():
        return True

    return False


def get_synonym_distance(query_root: str, candidate_root: str) -> float:
    """
    Calculate synonym distance between query root and candidate root.

    Uses knowledge module synonym dictionary to determine how "close" two roots are.

    Returns:
        1.0 = exact match
        0.8 = direct synonym (e.g., fond → kre)
        0.5 = indirect synonym (e.g., fond → establ → firme)
        0.0 = unrelated
    """
    from klareco.knowledge import get_synonyms

    if query_root == candidate_root:
        return 1.0

    # Check if candidate is in query's direct synonyms
    query_synonyms = get_synonyms(query_root, max_count=5)
    if candidate_root in query_synonyms:
        return 0.8

    # Check if they share a common synonym (indirect)
    candidate_synonyms = get_synonyms(candidate_root, max_count=5)
    if query_synonyms & candidate_synonyms:  # Set intersection
        return 0.5

    return 0.0


def score_structural_match(query_ast: Dict, candidate_ast: Dict) -> float:
    """
    Score structural match between query and candidate ASTs.

    Checks:
    - Object match (30%): Does candidate have same object as query?
    - Subject prominence (20%): Is subject a proper noun (likely the answer)?

    Returns:
        Score from 0.0 to 0.5 (0.3 for object match + 0.2 for subject prominence)
    """
    score = 0.0

    # Object match (30%)
    query_obj = get_ast_object_root(query_ast)
    cand_obj = get_ast_object_root(candidate_ast)

    if query_obj and cand_obj:
        if query_obj == cand_obj:
            score += 0.3
            logger.debug(f"Object match: {query_obj} == {cand_obj} (+0.3)")
        else:
            logger.debug(f"Object mismatch: {query_obj} != {cand_obj} (+0.0)")

    # Subject prominence (20%): For WHO questions, prefer proper noun subjects
    cand_subj_node = candidate_ast.get('subjekto')
    if cand_subj_node:
        if cand_subj_node.get('tipo') == 'vortgrupo':
            kerno = cand_subj_node.get('kerno')
            if kerno and is_proper_noun(kerno):
                score += 0.2
                logger.debug(f"Subject is proper noun: {kerno.get('radiko')} (+0.2)")
        elif is_proper_noun(cand_subj_node):
            score += 0.2
            logger.debug(f"Subject is proper noun: {cand_subj_node.get('radiko')} (+0.2)")

    return score


def score_verb_similarity(query_ast: Dict, candidate_ast: Dict) -> float:
    """
    Score verb similarity using synonym distance.

    Returns:
        Score from 0.0 to 0.4 (synonym distance * 0.4)
    """
    query_verb = get_ast_verb_root(query_ast)
    cand_verb = get_ast_verb_root(candidate_ast)

    if not query_verb or not cand_verb:
        return 0.0

    distance = get_synonym_distance(query_verb, cand_verb)
    score = distance * 0.4  # Weight verb similarity as 40%

    logger.debug(f"Verb similarity: {query_verb} ~ {cand_verb} = {distance:.2f} → score={score:.2f}")

    return score


def rank_ast_matches(
    query_ast: Dict,
    candidates: List[Dict],
    use_embeddings: bool = False,
    embedding_weight: float = 0.1
) -> List[Dict]:
    """
    Rank AST matches by semantic and structural similarity.

    Scoring breakdown:
    - Verb synonym distance: 40%
    - Object exact match: 30%
    - Subject prominence: 20%
    - Root embedding similarity: 10% (optional, if use_embeddings=True)

    Args:
        query_ast: Parsed query AST
        candidates: List of candidate documents with 'ast' field
        use_embeddings: Whether to use learned root embeddings (default: False)
        embedding_weight: Weight for embedding similarity (default: 0.1)

    Returns:
        List of candidates with updated 'score' field, sorted by score descending
    """
    if not query_ast:
        logger.warning("No query AST provided, cannot rank")
        return candidates

    logger.info(f"Ranking {len(candidates)} AST matches...")

    for cand in candidates:
        cand_ast = cand.get('ast')
        if not cand_ast:
            cand['score'] = 0.0
            cand['score_breakdown'] = {'error': 'No AST'}
            continue

        # Component scores
        verb_score = score_verb_similarity(query_ast, cand_ast)
        struct_score = score_structural_match(query_ast, cand_ast)

        # Total score (without embeddings)
        total_score = verb_score + struct_score

        # Optional: Add embedding similarity
        emb_score = 0.0
        if use_embeddings:
            # TODO: Implement root embedding similarity
            # This would load root embeddings and compute cosine similarity
            # between query roots and candidate roots
            emb_score = 0.0

        total_score += emb_score

        # Update candidate with score
        cand['score'] = total_score
        cand['score_breakdown'] = {
            'verb_similarity': verb_score,
            'structural_match': struct_score,
            'embedding_similarity': emb_score,
            'total': total_score
        }

        logger.debug(f"Candidate: {cand.get('text', '')[:50]}... → score={total_score:.2f}")

    # Sort by score descending
    ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)

    logger.info(f"Ranked results: scores range {ranked[0]['score']:.2f} to {ranked[-1]['score']:.2f}")

    return ranked


def explain_ranking(candidate: Dict) -> str:
    """
    Generate human-readable explanation of why a candidate was ranked.

    Useful for debugging and transparency.
    """
    if 'score_breakdown' not in candidate:
        return "No score breakdown available"

    breakdown = candidate['score_breakdown']

    lines = [
        f"Total Score: {breakdown.get('total', 0.0):.2f}",
        f"  Verb Similarity: {breakdown.get('verb_similarity', 0.0):.2f} (40% weight)",
        f"  Structural Match: {breakdown.get('structural_match', 0.0):.2f} (object + subject)",
        f"  Embedding Similarity: {breakdown.get('embedding_similarity', 0.0):.2f} (10% weight)"
    ]

    return "\n".join(lines)
