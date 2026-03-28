#!/usr/bin/env python3
"""
Semi-Automated Root Annotation (Aggressive Mode)

Enhanced heuristics that ALWAYS make a decision, even for mixed-usage roots.
Uses relative ratios and linguistic patterns to infer semantics.

Strategy:
1. High-confidence rules (>70% usage in one role)
2. Medium-confidence rules (40-70% usage)
3. Low-confidence rules (use highest ratio)
4. Linguistic fallbacks (word patterns)
"""

import argparse
import jsonlines
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.morphology.root_lexicon import ROOT_LEXICON

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED HEURISTICS FOR AUTOMATIC ANNOTATION
# ============================================================================

def infer_animacy_type_from_usage(root: str, stats: Dict) -> Tuple[str, str, float]:
    """
    Infer animacy and type from usage statistics.

    AGGRESSIVE MODE: Always makes a decision, even for ambiguous cases.

    Returns:
        (animacy, type, confidence)
    """
    total = stats['total_count']
    subj_ratio = stats['subject_count'] / total if total > 0 else 0
    verb_ratio = stats['verb_count'] / total if total > 0 else 0
    obj_ratio = stats['object_count'] / total if total > 0 else 0

    # High confidence: verb-only usage (>95%)
    if verb_ratio > 0.95:
        return 'abstract', 'action', 0.9

    # High confidence: clear verb (>70%)
    if verb_ratio > 0.7:
        # Check for agent-requiring verbs (common patterns)
        if any(root.startswith(pfx) for pfx in ['manĝ', 'vid', 'aŭd', 'lern', 'pens', 'parol']):
            return 'abstract', 'action', 0.85
        return 'abstract', 'action', 0.85

    # High confidence: primarily subject (>70%)
    if subj_ratio > 0.7:
        # Check for common person/animal roots
        if any(pattern in root for pattern in ['hom', 'vir', 'person', 'infan']):
            return 'animate', 'person', 0.85
        if any(pattern in root for pattern in ['hund', 'kat', 'bird', 'best', 'fiŝ']):
            return 'animate', 'animal', 0.85
        # Check for place names
        if any(pattern in root for pattern in ['urb', 'land', 'lok', 'reg']):
            return 'inanimate', 'place', 0.75
        # Default to inanimate object for subject-heavy
        return 'inanimate', 'object', 0.7

    # High confidence: primarily object (>70%)
    if obj_ratio > 0.7:
        # Check for consumables
        if any(pattern in root for pattern in ['manĝ', 'pom', 'pan', 'vian']):
            return 'inanimate', 'food', 0.8
        # Check for artifacts
        if any(pattern in root for pattern in ['libr', 'paper', 'skribaĵ', 'dokument']):
            return 'inanimate', 'artifact', 0.75
        return 'inanimate', 'object', 0.7

    # Medium confidence: moderate verb usage (40-70%)
    if verb_ratio > 0.4:
        return 'abstract', 'action', 0.6

    # Medium confidence: moderate subject usage (40-70%)
    if subj_ratio > 0.4:
        # Check for person/animate indicators
        if any(pattern in root for pattern in ['hom', 'vir', 'person', 'infan', 'animat']):
            return 'animate', 'person', 0.6
        # Default to inanimate
        return 'inanimate', 'object', 0.55

    # Medium confidence: moderate object usage (40-70%)
    if obj_ratio > 0.4:
        return 'inanimate', 'object', 0.55

    # Low confidence: mixed usage - use highest ratio
    if verb_ratio >= subj_ratio and verb_ratio >= obj_ratio:
        # Verb-leaning but not dominant
        return 'abstract', 'action', 0.45
    elif subj_ratio >= obj_ratio:
        # Subject-leaning
        # Check for temporal indicators
        if any(pattern in root for pattern in ['temp', 'jar', 'hor', 'dat', 'monat']):
            return 'inanimate', 'temporal', 0.45
        return 'inanimate', 'object', 0.45
    else:
        # Object-leaning
        return 'inanimate', 'object', 0.45

    # Fallback: if all else fails, guess based on linguistic patterns
    # This should rarely be reached
    if root.endswith('i') or root.endswith('as') or root.endswith('is'):
        # Looks like a verb form
        return 'abstract', 'action', 0.3
    else:
        # Default to inanimate object (most common)
        return 'inanimate', 'object', 0.3


def check_verb_constraints(root: str, stats: Dict) -> Optional[Dict]:
    """
    Infer verb selectional constraints from root patterns.

    Returns verb constraint dict or None if not a verb.
    """
    verb_ratio = stats['verb_count'] / stats['total_count'] if stats['total_count'] > 0 else 0

    if verb_ratio < 0.4:  # Lowered threshold from 0.5
        return None  # Not primarily a verb

    constraints = {}

    # Perception verbs require sentient agent
    if any(root.startswith(pfx) for pfx in ['vid', 'aŭd', 'sent', 'gustu']):
        constraints['requires_sentient'] = True
        constraints['requires_animate_agent'] = True

    # Cognition verbs require sentient agent
    elif any(root.startswith(pfx) for pfx in ['pens', 'lern', 'sci', 'komprenu', 'memor']):
        constraints['requires_sentient'] = True
        constraints['requires_animate_agent'] = True

    # Communication verbs require animate agent
    elif any(root.startswith(pfx) for pfx in ['parol', 'dir', 'demand', 'respond', 'rakon']):
        constraints['requires_animate_agent'] = True

    # Consumption verbs
    elif any(root.startswith(pfx) for pfx in ['manĝ', 'trink']):
        constraints['requires_animate_agent'] = True
        constraints['requires_physical_patient'] = True

    # Motion verbs
    elif any(root.startswith(pfx) for pfx in ['ir', 'ven', 'kur', 'salt', 'naĝ', 'flug']):
        constraints['requires_animate_agent'] = True

    return constraints if constraints else None


def annotate_root(root_data: Dict) -> Dict:
    """
    Generate annotation for a single root using enhanced heuristics.
    """
    root = root_data['root']

    # Check if already annotated
    if root in ROOT_LEXICON:
        return {
            **root_data,
            'animacy': ROOT_LEXICON[root].get('animacy', 'unknown'),
            'type': ROOT_LEXICON[root].get('type', 'unknown'),
            'confidence': 1.0,
            'source': 'existing_lexicon',
            'needs_review': False
        }

    # Infer from usage statistics
    animacy, type_val, confidence = infer_animacy_type_from_usage(root, root_data)

    # Check for verb constraints
    verb_constraints = check_verb_constraints(root, root_data)

    annotation = {
        **root_data,
        'animacy': animacy,
        'type': type_val,
        'confidence': confidence,
        'source': 'heuristic',
        'needs_review': confidence < 0.5,  # Only very low confidence needs review
    }

    if verb_constraints:
        annotation['verb_constraints'] = verb_constraints

    return annotation


# ============================================================================
# BATCH ANNOTATION
# ============================================================================

def annotate_batch(input_path: Path, output_path: Path, limit: Optional[int] = None):
    """
    Annotate a batch of roots with aggressive heuristics.
    """
    logger.info(f"Loading roots from {input_path}")

    roots = []
    with jsonlines.open(input_path) as reader:
        for i, root_data in enumerate(reader):
            if limit and i >= limit:
                break
            roots.append(root_data)

    logger.info(f"Loaded {len(roots)} roots")

    # Annotate each root
    annotated = []
    already_annotated = 0
    high_confidence = 0  # ≥0.7
    medium_confidence = 0  # 0.5-0.7
    low_confidence = 0  # <0.5

    for root_data in roots:
        annotation = annotate_root(root_data)
        annotated.append(annotation)

        if annotation['source'] == 'existing_lexicon':
            already_annotated += 1
        elif annotation['confidence'] >= 0.7:
            high_confidence += 1
        elif annotation['confidence'] >= 0.5:
            medium_confidence += 1
        else:
            low_confidence += 1

    # Save annotated roots
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(annotated)

    logger.info(f"\nAnnotation complete!")
    logger.info(f"  Already annotated: {already_annotated}")
    logger.info(f"  High confidence (≥0.7): {high_confidence}")
    logger.info(f"  Medium confidence (0.5-0.7): {medium_confidence}")
    logger.info(f"  Low confidence (<0.5): {low_confidence}")
    logger.info(f"  TOTAL: {len(annotated)} (all roots annotated)")
    logger.info(f"\nOutput: {output_path}")

    # Print sample of low confidence roots
    low_conf_roots = [r for r in annotated if r['confidence'] < 0.5][:10]
    if low_conf_roots:
        logger.info(f"\nSample low-confidence roots (may need manual review):")
        for r in low_conf_roots:
            logger.info(f"  {r['root']:15s}: {r['animacy']:10s}/{r['type']:15s} (confidence: {r['confidence']:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Semi-automatic root annotation (aggressive mode)')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input JSONL with roots to annotate')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL with annotations')
    parser.add_argument('--limit', type=int,
                       help='Limit number of roots to annotate')

    args = parser.parse_args()

    annotate_batch(args.input, args.output, args.limit)


if __name__ == '__main__':
    main()
