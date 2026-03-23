#!/usr/bin/env python3
"""
Add antonym pairs to root embedding training.

This script generates systematic antonym pairs from the mal- prefix in Esperanto.

Usage:
    from add_antonym_pairs import generate_antonym_pairs

    antonym_pairs = generate_antonym_pairs(root_to_idx, FUNCTION_WORDS)
    pairs.extend(antonym_pairs)
"""

from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def generate_antonym_pairs(
    root_to_idx: Dict[str, int],
    function_words: Set[str],
    target_similarity: float = -0.7,
    weight: float = 20.0
) -> Tuple[List[Tuple[int, int, float]], List[float]]:
    """
    Generate antonym pairs from Esperanto mal- prefix.

    Esperanto has systematic negation: malbon (bad) = antonym of bon (good).
    This is productive for ~30% of the vocabulary.

    Args:
        root_to_idx: Vocabulary mapping
        function_words: Set of function words to exclude
        target_similarity: Target similarity for antonyms (negative!)
        weight: Training weight for antonym pairs (high priority)

    Returns:
        (pairs, weights) where pairs = [(idx1, idx2, target_sim), ...]

    Examples:
        bon (good) ↔ malbon (bad) → target = -0.7
        long (long) ↔ mallong (short) → target = -0.7
        varm (warm) ↔ malvarm (cold) → target = -0.7
    """
    pairs = []
    weights = []

    antonym_count = 0
    skipped_function = 0
    skipped_missing = 0

    logger.info("Generating antonym pairs from mal- prefix...")

    # Find all mal- prefixed roots
    for root in root_to_idx:
        if not root.startswith('mal'):
            continue

        # Extract positive form (remove mal-)
        positive_root = root[3:]  # 'malbon' → 'bon'

        # Skip if positive root is too short (likely not a real antonym)
        if len(positive_root) < 2:
            continue

        # Skip if either is a function word (shouldn't happen, but safety check)
        if root in function_words or positive_root in function_words:
            skipped_function += 1
            continue

        # Check if positive root exists in vocabulary
        if positive_root not in root_to_idx:
            skipped_missing += 1
            continue

        # Create antonym pair with NEGATIVE similarity
        idx1 = root_to_idx[root]
        idx2 = root_to_idx[positive_root]

        pairs.append((idx1, idx2, target_similarity))
        weights.append(weight)
        antonym_count += 1

    logger.info(f"Generated {antonym_count} antonym pairs:")
    logger.info(f"  Target similarity: {target_similarity} (negative = antonyms)")
    logger.info(f"  Weight: {weight} (high priority)")
    logger.info(f"  Skipped: {skipped_function} function words, {skipped_missing} missing pairs")

    # Log examples
    if antonym_count > 0:
        examples = []
        for (idx1, idx2, _) in pairs[:5]:
            root1 = [k for k, v in root_to_idx.items() if v == idx1][0]
            root2 = [k for k, v in root_to_idx.items() if v == idx2][0]
            examples.append(f"({root2}, {root1})")
        logger.info(f"  Examples: {', '.join(examples)}")

    return pairs, weights


def validate_antonyms(
    model,
    root_to_idx: Dict[str, int],
    idx_to_root: Dict[int, str],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Validate that antonym pairs have negative similarity.

    Returns metrics:
        - mean_antonym_sim: Average similarity of antonym pairs
        - antonym_negative_rate: % of antonyms with sim < 0
        - examples: List of (root1, root2, similarity) tuples
    """
    import torch

    antonym_sims = []
    examples = []

    for root in root_to_idx:
        if not root.startswith('mal'):
            continue

        positive_root = root[3:]
        if len(positive_root) < 2 or positive_root not in root_to_idx:
            continue

        idx1 = torch.tensor([root_to_idx[root]], device=device)
        idx2 = torch.tensor([root_to_idx[positive_root]], device=device)

        with torch.no_grad():
            sim = model.similarity(idx1, idx2).item()

        antonym_sims.append(sim)

        if len(examples) < 10:
            examples.append((positive_root, root, sim))

    if not antonym_sims:
        return {
            'mean_antonym_sim': 0.0,
            'antonym_negative_rate': 0.0,
            'count': 0,
            'examples': []
        }

    mean_sim = sum(antonym_sims) / len(antonym_sims)
    negative_rate = sum(1 for s in antonym_sims if s < 0) / len(antonym_sims)

    return {
        'mean_antonym_sim': mean_sim,
        'antonym_negative_rate': negative_rate,
        'count': len(antonym_sims),
        'examples': examples
    }


# Example integration into existing training script
"""
# In build_similarity_pairs(), after ReVo pairs, add:

# =========================================================================
# 4. Systematic antonym pairs (mal- prefix)
# =========================================================================
from improvements.add_antonym_pairs import generate_antonym_pairs

antonym_pairs, antonym_weights = generate_antonym_pairs(
    root_to_idx,
    FUNCTION_WORDS,
    target_similarity=-0.7,  # Negative = antonyms
    weight=20.0  # High priority
)

pairs.extend(antonym_pairs)
weights.extend(antonym_weights)

logger.info(f"Total pairs after antonyms: {len(pairs)}")
"""
