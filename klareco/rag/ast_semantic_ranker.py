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
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Global cache for embeddings
_EMBEDDING_CACHE = None


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


def load_embeddings(embedding_path: Path = None) -> Dict[str, torch.Tensor]:
    """
    Load root embeddings from checkpoint.

    Caches embeddings globally to avoid repeated loading.

    Args:
        embedding_path: Path to embedding checkpoint (default: models/root_embeddings/best_model.pt)

    Returns:
        Dict mapping root → embedding vector
    """
    global _EMBEDDING_CACHE

    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE

    if embedding_path is None:
        embedding_path = Path('models/root_embeddings/best_model.pt')

    if not embedding_path.exists():
        logger.warning(f"Embedding file not found: {embedding_path}")
        return {}

    try:
        checkpoint = torch.load(embedding_path, map_location='cpu', weights_only=False)

        # Extract root embeddings (64D vectors)
        # Try multiple checkpoint formats
        if 'root_embeddings' in checkpoint:
            root_embeddings = checkpoint['root_embeddings']
        elif 'model_state_dict' in checkpoint:
            # Check for different embedding weight keys
            state_dict = checkpoint['model_state_dict']
            if 'root_embeddings.weight' in state_dict:
                root_embeddings = state_dict['root_embeddings.weight']
            elif 'embeddings.weight' in state_dict:
                root_embeddings = state_dict['embeddings.weight']
            else:
                logger.warning(f"Could not find embeddings in model_state_dict. Keys: {list(state_dict.keys())}")
                return {}
        else:
            logger.warning(f"Could not find embeddings in checkpoint. Keys: {list(checkpoint.keys())}")
            return {}

        # Get vocabulary
        # Try multiple vocabulary key formats
        if 'root_vocab' in checkpoint:
            root_vocab = checkpoint['root_vocab']
        elif 'root_to_idx' in checkpoint:
            root_vocab = checkpoint['root_to_idx']
        elif 'vocab' in checkpoint:
            root_vocab = checkpoint['vocab']
        else:
            logger.warning(f"Could not find vocabulary in checkpoint. Keys: {list(checkpoint.keys())}")
            return {}

        # Build dict: root → embedding
        embeddings = {}
        for root, idx in root_vocab.items():
            embeddings[root] = root_embeddings[idx]

        _EMBEDDING_CACHE = embeddings
        logger.info(f"Loaded {len(embeddings)} root embeddings from {embedding_path}")

        return embeddings

    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return {}


def get_all_roots_from_ast(ast: Dict) -> List[str]:
    """
    Extract all roots from an AST (for embedding similarity).

    Returns all content word roots (verbs, nouns, adjectives).
    """
    roots = []

    if not ast or ast.get('tipo') != 'frazo':
        return roots

    # Helper to extract root from node
    def extract_root(node):
        if not node:
            return None
        if isinstance(node, dict):
            return node.get('radiko')
        return None

    # Get verb root
    verb_root = get_ast_verb_root(ast)
    if verb_root:
        roots.append(verb_root)

    # Get object root
    obj_root = get_ast_object_root(ast)
    if obj_root:
        roots.append(obj_root)

    # Get subject root
    subj_root = get_ast_subject_root(ast)
    if subj_root:
        roots.append(subj_root)

    # Get roots from aliaj (modifiers)
    aliaj = ast.get('aliaj', [])
    for item in aliaj:
        root = extract_root(item)
        if root and root not in roots:
            # Skip function words (correlatives, pronouns)
            if item.get('vortspeco') not in ['korelativo', 'pronomo']:
                roots.append(root)

    return roots


def compute_embedding_similarity(
    query_roots: List[str],
    candidate_roots: List[str],
    embeddings: Dict[str, torch.Tensor]
) -> float:
    """
    Compute cosine similarity between query and candidate root embeddings.

    Strategy: Average query vectors, average candidate vectors, compute cosine similarity.

    Args:
        query_roots: List of roots from query
        candidate_roots: List of roots from candidate
        embeddings: Dict mapping root → embedding vector

    Returns:
        Cosine similarity score [0, 1]
    """
    if not embeddings or not query_roots or not candidate_roots:
        return 0.0

    # Get query embeddings
    query_vecs = []
    for root in query_roots:
        if root in embeddings:
            query_vecs.append(embeddings[root])

    # Get candidate embeddings
    cand_vecs = []
    for root in candidate_roots:
        if root in embeddings:
            cand_vecs.append(embeddings[root])

    if not query_vecs or not cand_vecs:
        return 0.0

    # Average vectors
    query_avg = torch.stack(query_vecs).mean(dim=0)
    cand_avg = torch.stack(cand_vecs).mean(dim=0)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        query_avg.unsqueeze(0),
        cand_avg.unsqueeze(0)
    ).item()

    # Normalize to [0, 1] (cosine similarity is [-1, 1])
    normalized = (cos_sim + 1.0) / 2.0

    return normalized


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
    use_embeddings: bool = True,  # Now enabled by default
    embedding_weight: float = 0.1,
    embedding_path: Path = None,
    use_importance_scoring: bool = True,  # Phase 3: importance-aware ranking
    question_type: Optional[str] = None,
    query_entity: Optional[str] = None,
    query_roots: Optional[List[str]] = None
) -> List[Dict]:
    """
    Rank AST matches by semantic and structural similarity.

    Phase 3 improvement: Integrate importance scoring into retrieval ranking.

    Scoring breakdown (without importance):
    - Verb synonym distance: 40%
    - Object exact match: 30%
    - Subject prominence: 20%
    - Root embedding similarity: 10% (learned component)

    Scoring breakdown (with importance, Phase 3):
    - Grammatical match: 30% (verb + object)
    - Fact importance: 40% (IS-A priority, entity centrality, etc.)
    - Subject prominence: 20%
    - Root embedding similarity: 10%

    Args:
        query_ast: Parsed query AST
        candidates: List of candidate documents with 'ast' field
        use_embeddings: Whether to use learned root embeddings (default: True)
        embedding_weight: Weight for embedding similarity (default: 0.1)
        embedding_path: Path to embeddings (default: models/root_embeddings/best_model.pt)
        use_importance_scoring: Whether to use importance scoring (Phase 3, default: True)
        question_type: Question type for importance scoring (e.g., "who", "what")
        query_entity: Entity being queried (e.g., "hund")
        query_roots: Query roots for importance scoring

    Returns:
        List of candidates with updated 'score' field, sorted by score descending
    """
    if not query_ast:
        logger.warning("No query AST provided, cannot rank")
        return candidates

    logger.info(f"Ranking {len(candidates)} AST matches (use_embeddings={use_embeddings}, use_importance={use_importance_scoring})...")

    # Load embeddings if needed
    embeddings = {}
    if use_embeddings:
        embeddings = load_embeddings(embedding_path)
        if embeddings and not query_roots:
            query_roots = get_all_roots_from_ast(query_ast)
            logger.debug(f"Query roots for embedding: {query_roots}")

    # Initialize importance scorer if needed (Phase 3)
    importance_scorer = None
    from klareco.rag.importance_scorer import QuestionType
    qt_enum = None
    if use_importance_scoring:
        try:
            from klareco.rag.importance_scorer import FactImportanceScorer, classify_question_type
            importance_scorer = FactImportanceScorer(use_embeddings=False)  # Embeddings handled separately
            # Convert question type string to enum
            if question_type:
                qt_str = question_type.upper() if isinstance(question_type, str) else str(question_type).split('.')[-1].upper()
                qt_enum = QuestionType[qt_str] if hasattr(QuestionType, qt_str) else QuestionType.OTHER
            logger.debug(f"Importance scoring enabled with question_type={qt_enum}")
        except Exception as e:
            logger.warning(f"Could not initialize importance scorer: {e}")
            use_importance_scoring = False

    # Build the fact extractor once, not per candidate (each init re-loads ReVo CSV)
    extractor = None
    if use_importance_scoring and importance_scorer and qt_enum:
        from klareco.rag.unified_extractor import UnifiedASTExtractor
        extractor = UnifiedASTExtractor()

    for cand in candidates:
        cand_ast = cand.get('ast')
        if not cand_ast:
            cand['score'] = 0.0
            cand['score_breakdown'] = {'error': 'No AST'}
            continue

        if use_importance_scoring and importance_scorer and qt_enum:
            # Phase 3: Use importance-aware ranking
            verb_score = score_verb_similarity(query_ast, cand_ast) * 0.3  # Reduced from 40% to 30%
            struct_score = score_structural_match(query_ast, cand_ast)  # Still 30% (20% subject + 10% from object reduction)

            # Extract fact from candidate AST and score importance (40% weight)
            try:
                facts = extractor.extract(cand_ast, source_sentence=cand.get('text', ''))

                # Score the first fact (most relevant)
                importance_score = 0.0
                if facts:
                    fact_breakdown = importance_scorer.score(
                        facts[0], qt_enum, query_entity, query_roots or [],
                        source_metadata=cand.get('metadata', {})
                    )
                    importance_score = fact_breakdown.final_score * 0.4  # 40% weight
                    logger.debug(f"Importance score: {importance_score:.2f} from {fact_breakdown}")
            except Exception as e:
                logger.debug(f"Could not extract/score fact: {e}")
                importance_score = 0.0

            # Embedding similarity (10%)
            emb_score = 0.0
            if use_embeddings and embeddings and query_roots:
                cand_roots = get_all_roots_from_ast(cand_ast)
                similarity = compute_embedding_similarity(query_roots, cand_roots, embeddings)
                emb_score = similarity * embedding_weight

            total_score = verb_score + struct_score + importance_score + emb_score

            cand['score'] = total_score
            cand['score_breakdown'] = {
                'verb_similarity': verb_score,
                'structural_match': struct_score,
                'importance': importance_score,
                'embedding_similarity': emb_score,
                'total': total_score
            }
        else:
            # Original ranking (no importance scoring)
            verb_score = score_verb_similarity(query_ast, cand_ast)
            struct_score = score_structural_match(query_ast, cand_ast)

            emb_score = 0.0
            if use_embeddings and embeddings and query_roots:
                cand_roots = get_all_roots_from_ast(cand_ast)
                similarity = compute_embedding_similarity(query_roots, cand_roots, embeddings)
                emb_score = similarity * embedding_weight

            total_score = verb_score + struct_score + emb_score

            cand['score'] = total_score
            cand['score_breakdown'] = {
                'verb_similarity': verb_score,
                'structural_match': struct_score,
                'embedding_similarity': emb_score,
                'total': total_score
            }

        logger.debug(f"Candidate: {cand.get('text', '')[:50]}... → score={cand['score']:.2f}")

    # Sort by score descending
    ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)

    if ranked:
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
