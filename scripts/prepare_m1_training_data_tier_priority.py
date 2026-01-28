#!/usr/bin/env python3
"""
Prepare M1 Training Data with Tier Priority + Role-Swap Negatives

FIXES BUG: Tier0 data was excluded because max_triples limit was reached
before tier0 sentences appeared in corpus.

This version processes tiers in priority order:
1. Tier 0 (ALL) - authoritative grammar texts (PMEG, Krestomatio, etc.)
2. Tier 2 (ALL) - Fundamento and born-digital high quality
3. Tier 5 (sample) - Wikipedia
4. Tier 6 (sample) - Gutenberg

This guarantees tier0 is included even if it appears late in corpus.

NEGATIVE GENERATION STRATEGIES:
1. Semantic distance corruption: Replace subject/verb/object with semantically DISTANT root
2. Smart role-swap corruption (NEW): Swap subject ↔ object to teach role-dependent selectional restrictions
   - Checks corpus first: only swaps if swapped version doesn't exist (asymmetric relations)
   - Skips symmetric relations: if "man fucks woman" AND "woman fucks man" both exist, no swap
   - Creates negatives for asymmetric relations: "dog eats food" exists, "food eats dog" doesn't → swap
   - Addresses synonym expansion issue where roots are valid but roles are wrong

Usage:
    python scripts/prepare_m1_training_data_tier_priority.py \
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
        --stage1-model models/root_embeddings/best_model.pt \
        --output-dir data/training/m1_semantic_tier_priority \
        --max-triples 500000 \
        --priority-qualities GOLD SILVER \
        --include-role-swaps  # Default: enabled
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from klareco.utils.ast_utils import extract_word_structure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


class SemanticDistanceCorruptor:
    """Corrupts triples using semantic distance to ensure distinguishability."""

    def __init__(self, stage1_checkpoint_path: Path, similarity_threshold: float = 0.15):
        logger.info(f"Loading Stage 1 embeddings from {stage1_checkpoint_path}")

        checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu', weights_only=False)
        self.embeddings = checkpoint['model_state_dict']['embeddings.weight']
        self.root_to_idx = checkpoint['root_to_idx']
        self.idx_to_root = checkpoint['idx_to_root']

        # Normalize embeddings for fast cosine similarity
        self.normalized_embeddings = F.normalize(self.embeddings, dim=-1)

        logger.info(f"Loaded {len(self.root_to_idx):,} root embeddings")
        logger.info(f"Similarity threshold: {similarity_threshold}")

        self.similarity_threshold = similarity_threshold

    def get_embedding(self, root: str) -> Optional[torch.Tensor]:
        """Get normalized embedding for a root."""
        idx = self.root_to_idx.get(root.lower())
        if idx is None:
            return None
        return self.normalized_embeddings[idx]

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        return (emb1 @ emb2).item()

    def find_distant_candidates_fast(
        self,
        original_root: str,
        candidate_names: List[str],
        candidate_indices: List[int],
        candidate_embs: torch.Tensor,
        other_roots: List[str],
        max_candidates: int = 100
    ) -> List[Tuple[str, float]]:
        """Find candidate words that are semantically DISTANT from original and other roots.

        OPTIMIZED: Pre-built candidate indices and embeddings for maximum speed.
        """
        original_emb = self.get_embedding(original_root)
        if original_emb is None:
            return [(c, 0.0) for c in random.sample(candidate_names, min(max_candidates, len(candidate_names)))]

        # Get embeddings for other roots in triple
        other_embs = []
        for root in other_roots:
            emb = self.get_embedding(root)
            if emb is not None:
                other_embs.append(emb)

        # Use all candidates (filtering original is not critical for performance)
        # The semantic distance check will naturally avoid selecting the original
        filtered_embs = candidate_embs
        filtered_names = candidate_names

        if len(filtered_embs) == 0:
            return []

        # VECTORIZED: Compute similarity to original for ALL candidates at once
        # [num_candidates] = [num_candidates, emb_dim] @ [emb_dim]
        sims_to_original = filtered_embs @ original_emb

        # VECTORIZED: Compute similarity to other components
        if other_embs:
            # Stack other embeddings [num_others, emb_dim]
            other_embs_stacked = torch.stack(other_embs)
            # Compute similarities [num_candidates, num_others]
            sims_to_others = filtered_embs @ other_embs_stacked.T
            # Average across others [num_candidates]
            avg_sims_to_others = sims_to_others.mean(dim=1)
            # Combined average
            avg_sims = (sims_to_original + avg_sims_to_others) / 2.0
        else:
            avg_sims = sims_to_original

        # VECTORIZED: Filter by threshold using boolean indexing
        distant_mask = avg_sims < self.similarity_threshold
        distant_indices = torch.where(distant_mask)[0]

        if len(distant_indices) == 0:
            return []

        # Get distant candidates and their scores
        distant_sims = avg_sims[distant_indices]
        distant_names = [filtered_names[i] for i in distant_indices.tolist()]

        # Sort by distance (lowest similarity first)
        sorted_indices = torch.argsort(distant_sims)[:max_candidates]

        scored_candidates = [
            (distant_names[i], distant_sims[i].item())
            for i in sorted_indices.tolist()
        ]

        return scored_candidates

    def corrupt_triple_fast(
        self,
        subject: str,
        verb: str,
        obj: str,
        noun_names: List[str],
        noun_indices: List[int],
        noun_embs: torch.Tensor,
        verb_names: List[str],
        verb_indices: List[int],
        verb_embs: torch.Tensor,
        corruption_type: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """Corrupt a triple using semantic distance (fast version with pre-built embeddings)."""
        if corruption_type == 'subject':
            candidates = self.find_distant_candidates_fast(
                subject, noun_names, noun_indices, noun_embs, [verb, obj], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_subj = random.choice(candidates[:10])[0]
            return (corrupted_subj, verb, obj, 'subject')

        elif corruption_type == 'object':
            candidates = self.find_distant_candidates_fast(
                obj, noun_names, noun_indices, noun_embs, [subject, verb], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_obj = random.choice(candidates[:10])[0]
            return (subject, verb, corrupted_obj, 'object')

        elif corruption_type == 'verb':
            candidates = self.find_distant_candidates_fast(
                verb, verb_names, verb_indices, verb_embs, [subject, obj], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_verb = random.choice(candidates[:10])[0]
            return (subject, corrupted_verb, obj, 'verb')

        return None


def extract_svo_structures(ast: Dict) -> Optional[Dict]:
    """
    Extract subject-verb-object word structures from AST.

    Returns dict with full morphological structure (case-normalized).
    """
    if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
        return None

    subjekto = ast['subjekto']
    verbo = ast['verbo']
    objekto = ast['objekto']

    try:
        # Extract words (handle vortgrupo with kerno)
        subj_word = subjekto.get('kerno') if isinstance(subjekto, dict) and subjekto.get('tipo') == 'vortgrupo' else subjekto
        verb_word = verbo
        obj_word = objekto.get('kerno') if isinstance(objekto, dict) and objekto.get('tipo') == 'vortgrupo' else objekto

        if all(w and isinstance(w, dict) for w in [subj_word, verb_word, obj_word]):
            return {
                'subject': extract_word_structure(subj_word, strip_case=True),
                'verb': extract_word_structure(verb_word, strip_case=True),
                'object': extract_word_structure(obj_word, strip_case=True)
            }
    except (AttributeError, KeyError, TypeError):
        pass

    return None


def load_corpus_triples_prioritized(
    corpus_path: Path,
    max_triples: Optional[int] = None,
    min_parse_rate: float = 0.0,
    priority_qualities: List[str] = ['GOLD'],
    fill_qualities: List[str] = ['BRONZE', 'COPPER']
) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """
    Load positive triples from corpus with quality priority.

    Process qualities in priority order to ensure high-quality data is included first.

    Args:
        corpus_path: Path to corpus JSONL file
        max_triples: Maximum triples to extract (None = all)
        min_parse_rate: Minimum parse rate filter
        priority_qualities: Qualities to include fully (default: ['GOLD'])
        fill_qualities: Qualities to sample from to fill remaining quota (default: ['BRONZE', 'COPPER'])

    Returns:
        (triples, vocabularies)
    """
    logger.info(f"Loading triples from {corpus_path} with quality priority")
    logger.info(f"  Priority qualities (include all): {priority_qualities}")
    logger.info(f"  Fill qualities (sample to quota): {fill_qualities}")

    # First pass: Collect all triples by quality
    logger.info("Pass 1: Collecting triples by quality...")
    triples_by_quality = defaultdict(list)
    nouns = set()
    verbs = set()
    triple_counts = Counter()

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                logger.info(f"  Processed {i:,} sentences")

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse rate filter
            if entry.get('parse_rate', 0) < min_parse_rate:
                continue

            # Extract triple structures
            ast = entry.get('ast')
            if not ast:
                continue

            structures = extract_svo_structures(ast)
            if not structures:
                continue

            # Get quality
            quality = entry.get('source', {}).get('quality')
            if quality not in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
                continue

            # Store full structures instead of just roots
            triple_dict = {
                'subject': structures['subject'],
                'verb': structures['verb'],
                'object': structures['object'],
                'label': 1.0,
                'corruption': None,
                'source': entry.get('source', {}),
                'original_text': entry.get('text', ''),
                'frequency': 1
            }

            triples_by_quality[quality].append(triple_dict)

            # Update vocabulary collection (use roots)
            nouns.add(structures['subject']['root'])
            nouns.add(structures['object']['root'])
            verbs.add(structures['verb']['root'])

            # Track triple frequency by roots (for deduplication)
            triple_key = (structures['subject']['root'], structures['verb']['root'], structures['object']['root'])
            triple_counts[triple_key] += 1

    logger.info("Pass 1 complete:")
    for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
        if quality in triples_by_quality:
            logger.info(f"  {quality}: {len(triples_by_quality[quality]):,} triples")

    # Second pass: Build result with quality priority
    logger.info("")
    logger.info("Pass 2: Building result with quality priority...")
    result = []

    # Add all priority quality triples
    for quality in priority_qualities:
        if quality in triples_by_quality:
            quality_triples = triples_by_quality[quality]
            result.extend(quality_triples)
            logger.info(f"  Added ALL {len(quality_triples):,} triples from {quality}")

    # Calculate remaining quota
    if max_triples:
        remaining = max_triples - len(result)
        logger.info(f"  Remaining quota: {remaining:,} triples")

        if remaining > 0:
            # Sample from fill qualities
            fill_triples = []
            for quality in fill_qualities:
                if quality in triples_by_quality:
                    fill_triples.extend(triples_by_quality[quality])

            if fill_triples:
                # Sample randomly from fill qualities
                n_sample = min(remaining, len(fill_triples))
                sampled = random.sample(fill_triples, n_sample)
                result.extend(sampled)
                logger.info(f"  Sampled {len(sampled):,} triples from fill qualities {fill_qualities}")
    else:
        # No limit, add all triples
        for quality in fill_qualities:
            if quality in triples_by_quality:
                quality_triples = triples_by_quality[quality]
                result.extend(quality_triples)
                logger.info(f"  Added ALL {len(quality_triples):,} triples from {quality}")

    # Update frequencies (use roots as key)
    for triple_dict in result:
        key = (triple_dict['subject']['root'], triple_dict['verb']['root'], triple_dict['object']['root'])
        triple_dict['frequency'] = triple_counts[key]

    logger.info("")
    logger.info(f"Final: {len(result):,} positive triples")
    logger.info(f"  Unique nouns: {len(nouns):,}")
    logger.info(f"  Unique verbs: {len(verbs):,}")

    # Print quality distribution
    quality_dist = Counter(t['source']['quality'] for t in result)
    logger.info("")
    logger.info("Quality distribution in training data:")
    for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
        if quality in quality_dist:
            count = quality_dist[quality]
            pct = 100 * count / len(result) if len(result) > 0 else 0
            logger.info(f"  {quality}: {count:,} ({pct:.1f}%)")

    vocabularies = {'nouns': nouns, 'verbs': verbs}

    return result, vocabularies


def generate_semantic_negatives(
    positive_triples: List[Dict],
    vocabularies: Dict[str, Set[str]],
    corruptor: SemanticDistanceCorruptor,
    negatives_per_positive: int = 1,
    include_role_swaps: bool = True
) -> List[Dict]:
    """Generate negative samples using semantic distance corruption + smart role swaps.

    OPTIMIZED: Uses vectorized similarity computation for 100-1000x speedup.

    Role-Swap Logic (SMART):
    - Before creating a role-swap negative, checks if swapped triple exists in corpus
    - If (A, verb, B) and (B, verb, A) both exist → symmetric relation → skip role-swap
    - If only (A, verb, B) exists → asymmetric → create (B, verb, A) as negative
    - Examples:
      * "man fucks woman" ↔ "woman fucks man" → BOTH in corpus → symmetric → no role-swap
      * "dog eats food" in corpus, "food eats dog" NOT → asymmetric → create role-swap

    Args:
        positive_triples: List of positive (subject, verb, object) triples
        vocabularies: Dict with 'nouns' and 'verbs' sets
        corruptor: SemanticDistanceCorruptor for finding distant candidates
        negatives_per_positive: Number of negatives per positive
        include_role_swaps: If True, include smart role-swap negatives (checks corpus first)

    Returns:
        List of negative triples with corruption metadata
    """
    logger.info(f"Generating {negatives_per_positive} negatives per positive")

    nouns = list(vocabularies['nouns'])
    verbs = list(vocabularies['verbs'])

    negatives = []
    corruption_types = ['subject', 'object', 'verb']
    if include_role_swaps:
        corruption_types.append('role_swap')
        logger.info(f"  Corruption types: {corruption_types} (equal probability)")
    else:
        logger.info(f"  Corruption types: {corruption_types} (no role swaps)")

    total_to_generate = len(positive_triples) * negatives_per_positive
    failed_corruptions = 0

    logger.info(f"  Processing {len(positive_triples):,} positive triples...")
    logger.info(f"  Will generate {total_to_generate:,} total negatives")
    logger.info(f"  Progress updates every 1,000 triples")
    logger.info(f"  Using VECTORIZED semantic distance computation (100-1000x faster)")
    logger.info("")

    # PRE-BUILD candidate embeddings once (major speedup!)
    logger.info(f"  Pre-building candidate embeddings for {len(nouns):,} nouns and {len(verbs):,} verbs...")
    noun_names = []
    noun_indices = []
    for noun in nouns:
        idx = corruptor.root_to_idx.get(noun.lower())
        if idx is not None:
            noun_names.append(noun)
            noun_indices.append(idx)

    verb_names = []
    verb_indices = []
    for verb in verbs:
        idx = corruptor.root_to_idx.get(verb.lower())
        if idx is not None:
            verb_names.append(verb)
            verb_indices.append(idx)

    # Get all embeddings at once
    noun_embs = corruptor.normalized_embeddings[noun_indices]
    verb_embs = corruptor.normalized_embeddings[verb_indices]

    logger.info(f"  Pre-built: {len(noun_names):,} noun embeddings, {len(verb_names):,} verb embeddings")
    logger.info("")

    # Build set of positive triples for role-swap checking (use roots)
    # This prevents creating role-swap negatives for symmetric relations
    # (e.g., "man fucks woman" and "woman fucks man" are both valid)
    positive_triple_set = None
    role_swaps_skipped = 0
    if include_role_swaps:
        logger.info(f"  Building positive triple set for role-swap validation...")
        positive_triple_set = {
            (t['subject']['root'].lower(), t['verb']['root'].lower(), t['object']['root'].lower())
            for t in positive_triples
        }
        logger.info(f"  Indexed {len(positive_triple_set):,} unique positive triples")
        logger.info("")

    start_time = time.time()
    last_log_time = start_time

    for idx, pos_triple in enumerate(positive_triples):
        # Show progress every 1000 triples
        if idx > 0 and idx % 1000 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            rate = idx / elapsed  # triples per second
            remaining = total_to_generate - len(negatives)
            eta_sec = remaining / rate if rate > 0 else 0
            eta_min = eta_sec / 60

            logger.info(f"  Generated {len(negatives):,} / {total_to_generate:,} negatives ({100*len(negatives)/total_to_generate:.1f}%) - Rate: {rate:.0f}/sec - ETA: ~{eta_min:.0f} min")

        # Extract word structures
        subj_struct = pos_triple['subject']
        verb_struct = pos_triple['verb']
        obj_struct = pos_triple['object']

        # Get roots for semantic distance computation
        subj_root = subj_struct['root']
        verb_root = verb_struct['root']
        obj_root = obj_struct['root']

        for _ in range(negatives_per_positive):
            corruption = random.choice(corruption_types)

            # Handle role-swap separately (no semantic distance needed)
            if corruption == 'role_swap':
                # Check if swapped triple exists in corpus (symmetric relation)
                # E.g., both "man fucks woman" and "woman fucks man" are valid
                swapped_triple = (obj_root.lower(), verb_root.lower(), subj_root.lower())

                if swapped_triple in positive_triple_set:
                    # Swapped version exists in corpus - this is a symmetric relation
                    # Skip role-swap and fall back to semantic corruption instead
                    role_swaps_skipped += 1
                    corruption = random.choice(['subject', 'object', 'verb'])
                    # Fall through to semantic corruption below
                else:
                    # Swapped version NOT in corpus - create role-swap negative
                    # E.g., "dog eats food" exists, but "food eats dog" doesn't
                    # Swap full structures, not just roots
                    neg_triple = {
                        'subject': obj_struct,     # Object structure in subject position
                        'verb': verb_struct,       # Same verb
                        'object': subj_struct,     # Subject structure in object position
                        'label': 0.0,
                        'corruption': 'role_swap',
                        'source': pos_triple['source'],
                        'original_text': pos_triple['original_text']
                    }
                    negatives.append(neg_triple)
                    continue

            # Semantic distance corruption - find distant root, keep morphology
            result = corruptor.corrupt_triple_fast(
                subj_root, verb_root, obj_root,
                noun_names, noun_indices, noun_embs,
                verb_names, verb_indices, verb_embs,
                corruption
            )

            if result is None:
                # Fallback to random if no distant candidate found
                failed_corruptions += 1
                if corruption == 'subject':
                    candidates = [n for n in nouns if n != subj_root]
                    corrupted_root = random.choice(candidates) if candidates else subj_root
                    result = (corrupted_root, verb_root, obj_root, 'subject')
                elif corruption == 'object':
                    candidates = [n for n in nouns if n != obj_root]
                    corrupted_root = random.choice(candidates) if candidates else obj_root
                    result = (subj_root, verb_root, corrupted_root, 'object')
                else:  # verb
                    candidates = [v for v in verbs if v != verb_root]
                    corrupted_root = random.choice(candidates) if candidates else verb_root
                    result = (subj_root, corrupted_root, obj_root, 'verb')

            corrupted_subj_root, corrupted_verb_root, corrupted_obj_root, corruption_type = result

            # Build corrupted structures: replace root, keep morphology
            if corruption_type == 'subject':
                neg_triple = {
                    'subject': {
                        'root': corrupted_subj_root,
                        'prefixes': subj_struct['prefixes'],  # Keep morphology
                        'suffixes': subj_struct['suffixes'],
                        'ending': subj_struct['ending']
                    },
                    'verb': verb_struct,
                    'object': obj_struct,
                    'label': 0.0,
                    'corruption': corruption_type,
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }
            elif corruption_type == 'object':
                neg_triple = {
                    'subject': subj_struct,
                    'verb': verb_struct,
                    'object': {
                        'root': corrupted_obj_root,
                        'prefixes': obj_struct['prefixes'],  # Keep morphology
                        'suffixes': obj_struct['suffixes'],
                        'ending': obj_struct['ending']
                    },
                    'label': 0.0,
                    'corruption': corruption_type,
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }
            else:  # verb
                neg_triple = {
                    'subject': subj_struct,
                    'verb': {
                        'root': corrupted_verb_root,
                        'prefixes': verb_struct['prefixes'],  # Keep morphology
                        'suffixes': verb_struct['suffixes'],
                        'ending': verb_struct['ending']
                    },
                    'object': obj_struct,
                    'label': 0.0,
                    'corruption': corruption_type,
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }

            negatives.append(neg_triple)

    logger.info(f"Generated {len(negatives):,} negative samples")
    if failed_corruptions > 0:
        logger.info(f"  Fell back to random for {failed_corruptions:,} cases ({100*failed_corruptions/total_to_generate:.1f}%)")

    # Show role-swap statistics
    if include_role_swaps and role_swaps_skipped > 0:
        logger.info(f"  Role-swaps skipped (symmetric relations): {role_swaps_skipped:,}")
        logger.info(f"    These triples exist in both directions in corpus")
        logger.info(f"    Examples: 'man fucks woman' ↔ 'woman fucks man' (both valid)")

    # Show corruption type distribution
    corruption_counts = Counter(neg['corruption'] for neg in negatives)
    logger.info("  Corruption type distribution:")
    for corruption_type, count in sorted(corruption_counts.items()):
        pct = 100 * count / len(negatives)
        logger.info(f"    {corruption_type:12s}: {count:6,} ({pct:.1f}%)")

    return negatives, role_swaps_skipped


def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    logger.info(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    return train, val, test


def save_splits(train: List[Dict], val: List[Dict], test: List[Dict], output_dir: Path, args,
                role_swaps_skipped: int = 0):
    """Save train/val/test splits to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(split_data):,} examples to {output_path}")

    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"
    nouns = set()
    verbs = set()
    for item in train:
        nouns.add(item['subject']['root'])
        nouns.add(item['object']['root'])
        verbs.add(item['verb']['root'])

    vocab = {
        'nouns': sorted(list(nouns)),
        'verbs': sorted(list(verbs)),
        'num_nouns': len(nouns),
        'num_verbs': len(verbs)
    }

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved vocabulary to {vocab_path}")
    logger.info(f"  Nouns: {len(nouns):,}")
    logger.info(f"  Verbs: {len(verbs):,}")

    # Save metadata
    metadata = {
        'description': 'M1 training data with semantic-distance corruption and quality priority',
        'corpus': str(args.corpus),
        'stage1_model': str(args.stage1_model),
        'similarity_threshold': args.similarity_threshold,
        'include_role_swaps': args.include_role_swaps,
        'role_swaps_skipped': role_swaps_skipped,
        'priority_qualities': args.priority_qualities,
        'fill_qualities': args.fill_qualities,
        'total_examples': len(train) + len(val) + len(test),
        'train_examples': len(train),
        'val_examples': len(val),
        'test_examples': len(test),
        'plausible_count': len([x for x in train if x['label'] == 1.0]),
        'implausible_count': len([x for x in train if x['label'] == 0.0]),
        'negatives_per_positive': args.negatives_per_positive,
        'max_triples': args.max_triples,
        'min_parse_rate': args.min_parse_rate
    }

    # Count quality distribution
    quality_dist = Counter(item['source']['quality'] for item in train)
    metadata['quality_distribution'] = dict(quality_dist)

    # Count corruption type distribution in negatives
    neg_train = [x for x in train if x['label'] == 0.0]
    corruption_dist = Counter(item.get('corruption', 'unknown') for item in neg_train)
    metadata['corruption_distribution'] = dict(corruption_dist)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {metadata_path}")


def save_checkpoint(output_dir: Path, positive_triples: List[Dict] = None,
                   negative_triples: List[Dict] = None, vocabularies: Dict = None,
                   stage: str = 'triples'):
    """Save checkpoint for restartability."""
    checkpoint_path = output_dir / 'data_generation_checkpoint.json'
    temp_path = output_dir / 'data_generation_checkpoint.json.tmp'

    checkpoint = {
        'stage': stage,
        'timestamp': str(Path(__file__).stat().st_mtime),
    }

    if positive_triples is not None:
        checkpoint['positive_triples'] = positive_triples
    if negative_triples is not None:
        checkpoint['negative_triples'] = negative_triples
    if vocabularies is not None:
        # Convert sets to lists for JSON serialization
        checkpoint['vocabularies'] = {
            'nouns': list(vocabularies['nouns']) if isinstance(vocabularies['nouns'], set) else vocabularies['nouns'],
            'verbs': list(vocabularies['verbs']) if isinstance(vocabularies['verbs'], set) else vocabularies['verbs']
        }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        temp_path.rename(checkpoint_path)
        logger.info(f"✓ Checkpoint saved: {checkpoint_path} (stage: {stage})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(output_dir: Path):
    """Load checkpoint if available."""
    checkpoint_path = output_dir / 'data_generation_checkpoint.json'

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Convert vocabulary lists back to sets
        if 'vocabularies' in checkpoint:
            checkpoint['vocabularies'] = {
                'nouns': set(checkpoint['vocabularies']['nouns']),
                'verbs': set(checkpoint['vocabularies']['verbs'])
            }

        logger.info(f"✓ Loaded checkpoint from: {checkpoint_path} (stage: {checkpoint['stage']})")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare M1 training data with quality priority (GOLD first)"
    )
    parser.add_argument(
        '--corpus', type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Path to parsed corpus'
    )
    parser.add_argument(
        '--stage1-model', type=Path,
        default=Path('models/root_embeddings/best_model.pt'),
        help='Path to Stage 1 embeddings'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('data/training/m1_semantic_tier_priority'),
        help='Output directory for training data'
    )
    parser.add_argument(
        '--max-triples', type=int,
        default=200000,
        help='Maximum positive triples to extract'
    )
    parser.add_argument(
        '--priority-qualities', type=str, nargs='+',
        default=['GOLD'],
        help='Qualities to include fully (default: GOLD)'
    )
    parser.add_argument(
        '--fill-qualities', type=str, nargs='+',
        default=['BRONZE', 'COPPER'],
        help='Qualities to sample from to fill quota (default: BRONZE COPPER)'
    )
    parser.add_argument(
        '--negatives-per-positive', type=int,
        default=1,
        help='Number of negative samples per positive'
    )
    parser.add_argument(
        '--similarity-threshold', type=float,
        default=0.15,
        help='Maximum similarity for corruption'
    )
    parser.add_argument(
        '--include-role-swaps', action='store_true',
        default=True,
        help='Include role-swap negatives (swap subject ↔ object) - default: enabled'
    )
    parser.add_argument(
        '--no-role-swaps', dest='include_role_swaps', action='store_false',
        help='Disable role-swap negatives'
    )
    parser.add_argument(
        '--min-parse-rate', type=float,
        default=0.0,
        help='Minimum parse rate to include (0.0 = all)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--fresh', action='store_true',
        help='Start fresh (ignore checkpoints)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    if not args.stage1_model.exists():
        logger.error(f"Stage 1 model not found: {args.stage1_model}")
        return 1

    logger.info("=" * 70)
    logger.info("M1 Training Data Generation with Tier Priority")
    logger.info("=" * 70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max triples: {args.max_triples:,}")
    logger.info(f"Priority qualities: {args.priority_qualities} (include ALL)")
    logger.info(f"Fill qualities: {args.fill_qualities} (sample to fill quota)")
    logger.info(f"Negative generation: semantic-distance + role-swaps={'ENABLED' if args.include_role_swaps else 'DISABLED'}")
    logger.info("")

    # Handle checkpoint flags
    checkpoint_path = args.output_dir / 'data_generation_checkpoint.json'
    if args.fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - removing checkpoint")
        checkpoint_path.unlink()
        logger.info("")

    # Try to load checkpoint
    checkpoint = None
    if args.resume or (not args.fresh and checkpoint_path.exists()):
        checkpoint = load_checkpoint(args.output_dir)
        logger.info("")

    # Initialize semantic distance corruptor
    corruptor = SemanticDistanceCorruptor(
        args.stage1_model,
        similarity_threshold=args.similarity_threshold
    )
    logger.info("")

    # Load corpus triples with quality priority (or from checkpoint)
    if checkpoint and checkpoint['stage'] in ['triples', 'negatives', 'complete']:
        logger.info("Resuming from checkpoint - loading triples...")
        positive_triples = checkpoint['positive_triples']
        vocabularies = checkpoint['vocabularies']
        logger.info(f"✓ Loaded {len(positive_triples):,} positive triples from checkpoint")
        logger.info("")
    else:
        positive_triples, vocabularies = load_corpus_triples_prioritized(
            args.corpus,
            max_triples=args.max_triples,
            min_parse_rate=args.min_parse_rate,
            priority_qualities=args.priority_qualities,
            fill_qualities=args.fill_qualities
        )
        logger.info("")

        # Save checkpoint after loading triples
        save_checkpoint(args.output_dir, positive_triples=positive_triples,
                       vocabularies=vocabularies, stage='triples')
        logger.info("")

    # Generate negative samples (or from checkpoint)
    role_swaps_skipped = 0  # Default for checkpoint loading
    if checkpoint and checkpoint['stage'] in ['negatives', 'complete']:
        logger.info("Resuming from checkpoint - loading negatives...")
        negative_triples = checkpoint['negative_triples']
        logger.info(f"✓ Loaded {len(negative_triples):,} negative triples from checkpoint")
        logger.info("")
    else:
        negative_triples, role_swaps_skipped = generate_semantic_negatives(
            positive_triples,
            vocabularies,
            corruptor,
            negatives_per_positive=args.negatives_per_positive,
            include_role_swaps=args.include_role_swaps
        )
        logger.info("")

        # Save checkpoint after generating negatives
        save_checkpoint(args.output_dir, positive_triples=positive_triples,
                       negative_triples=negative_triples, vocabularies=vocabularies,
                       stage='negatives')
        logger.info("")

    # Combine and split
    all_data = positive_triples + negative_triples
    logger.info(f"Total examples: {len(all_data):,} ({len(positive_triples):,} positive, {len(negative_triples):,} negative)")

    train, val, test = split_data(all_data, seed=args.seed)
    logger.info("")

    # Save splits
    save_splits(train, val, test, args.output_dir, args, role_swaps_skipped)
    logger.info("")

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("✓ Checkpoint cleaned up (data generation complete)")
        logger.info("")

    logger.info("=" * 70)
    logger.info("✓ Data generation complete!")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
