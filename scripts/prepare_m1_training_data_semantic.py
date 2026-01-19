#!/usr/bin/env python3
"""
Prepare M1 Training Data with Semantic-Distance-Based Corruption

FIXES BUG #2: Random corruption doesn't create distinguishable negatives.

This version uses Stage 1 embeddings to ensure corrupted words are semantically
DISTANT from the original, creating a learnable signal for M1.

Corruption strategy:
1. Load Stage 1 embeddings to compute semantic similarity
2. When corrupting a word, select a replacement that:
   - Has LOW similarity (< 0.15) to the original word
   - Has LOW similarity to other components in the triple
   - Is from vocabulary (not unknown)

This ensures corrupted triples have LOWER embedding similarity than plausible ones,
giving the model a clear signal to learn from.

Usage:
    python scripts/prepare_m1_training_data_semantic.py \
        --corpus data/enhanced_corpus/corpus_with_tier0.jsonl \
        --stage1-model models/root_embeddings_tier0/best_model.pt \
        --output-dir data/training/m1_semantic \
        --min-parse-rate 0.0
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


class SemanticDistanceCorruptor:
    """Corrupts triples using semantic distance to ensure distinguishability."""

    def __init__(self, stage1_checkpoint_path: Path, similarity_threshold: float = 0.15):
        """
        Initialize with Stage 1 embeddings.

        Args:
            stage1_checkpoint_path: Path to Stage 1 best_model.pt
            similarity_threshold: Maximum similarity for corruption (default 0.15)
        """
        logger.info(f"Loading Stage 1 embeddings from {stage1_checkpoint_path}")

        checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
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

    def find_distant_candidates(
        self,
        original_root: str,
        candidates: List[str],
        other_roots: List[str],
        max_candidates: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Find candidate words that are semantically DISTANT from original and other roots.

        Args:
            original_root: Root to replace
            candidates: List of possible replacements
            other_roots: Other roots in the triple (should also be distant from these)
            max_candidates: Maximum number of candidates to return

        Returns:
            List of (root, avg_similarity) tuples sorted by distance (lowest similarity first)
        """
        original_emb = self.get_embedding(original_root)
        if original_emb is None:
            # If original has no embedding, use random
            return [(c, 0.0) for c in random.sample(candidates, min(max_candidates, len(candidates)))]

        # Get embeddings for other roots in triple
        other_embs = []
        for root in other_roots:
            emb = self.get_embedding(root)
            if emb is not None:
                other_embs.append(emb)

        # Evaluate each candidate
        scored_candidates = []
        for candidate in candidates:
            if candidate == original_root:
                continue

            cand_emb = self.get_embedding(candidate)
            if cand_emb is None:
                continue

            # Compute similarity to original
            sim_to_original = self.cosine_similarity(cand_emb, original_emb)

            # Compute similarity to other components
            sims_to_others = [
                self.cosine_similarity(cand_emb, other_emb)
                for other_emb in other_embs
            ]

            # Average similarity (penalize if similar to ANY component)
            avg_sim = (sim_to_original + sum(sims_to_others)) / (1 + len(sims_to_others))

            # Only accept if sufficiently distant
            if avg_sim < self.similarity_threshold:
                scored_candidates.append((candidate, avg_sim))

        # Sort by distance (lowest similarity first)
        scored_candidates.sort(key=lambda x: x[1])

        return scored_candidates[:max_candidates]

    def corrupt_triple(
        self,
        subject: str,
        verb: str,
        obj: str,
        nouns: List[str],
        verbs: List[str],
        corruption_type: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Corrupt a triple using semantic distance.

        Returns:
            (corrupted_subj, corrupted_verb, corrupted_obj, corruption_type) or None if no valid corruption
        """
        if corruption_type == 'subject':
            candidates = self.find_distant_candidates(
                subject, nouns, [verb, obj], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_subj = random.choice(candidates[:10])[0]  # Pick from top 10 distant
            return (corrupted_subj, verb, obj, 'subject')

        elif corruption_type == 'object':
            candidates = self.find_distant_candidates(
                obj, nouns, [subject, verb], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_obj = random.choice(candidates[:10])[0]
            return (subject, verb, corrupted_obj, 'object')

        elif corruption_type == 'verb':
            candidates = self.find_distant_candidates(
                verb, verbs, [subject, obj], max_candidates=50
            )
            if not candidates:
                return None
            corrupted_verb = random.choice(candidates[:10])[0]
            return (subject, corrupted_verb, obj, 'verb')

        return None


def extract_svo_triple(ast: Dict) -> Optional[Tuple[str, str, str]]:
    """Extract (subject_root, verb_root, object_root) from AST."""
    if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
        return None

    subjekto = ast['subjekto']
    verbo = ast['verbo']
    objekto = ast['objekto']

    try:
        subj_root = subjekto.get('kerno', {}).get('radiko') if isinstance(subjekto, dict) else None
        verb_root = verbo.get('radiko') if isinstance(verbo, dict) else None
        obj_root = objekto.get('kerno', {}).get('radiko') if isinstance(objekto, dict) else None

        if subj_root and verb_root and obj_root:
            return (subj_root.lower(), verb_root.lower(), obj_root.lower())
    except (AttributeError, KeyError, TypeError):
        pass

    return None


def load_corpus_triples(
    corpus_path: Path,
    max_triples: Optional[int] = None,
    min_parse_rate: float = 0.7
) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """Load positive triples from corpus."""
    logger.info(f"Loading triples from {corpus_path}")

    triples = []
    nouns = set()
    verbs = set()
    triple_counts = Counter()

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                logger.info(f"  Processed {i:,} sentences, found {len(triples):,} triples")

            if max_triples and len(triples) >= max_triples:
                break

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get('parse_rate', 0) < min_parse_rate:
                continue

            ast = entry.get('ast')
            if not ast:
                continue

            triple = extract_svo_triple(ast)
            if not triple:
                continue

            subj, verb, obj = triple

            nouns.add(subj)
            nouns.add(obj)
            verbs.add(verb)

            triple_counts[triple] += 1

            triples.append({
                'subject_root': subj,
                'verb_root': verb,
                'object_root': obj,
                'label': 1.0,
                'corruption': None,
                'source': entry.get('source', 'unknown'),
                'original_text': entry.get('text', ''),
                'frequency': 1
            })

    # Update frequencies
    for triple_dict in triples:
        key = (triple_dict['subject_root'], triple_dict['verb_root'], triple_dict['object_root'])
        triple_dict['frequency'] = triple_counts[key]

    logger.info(f"Loaded {len(triples):,} positive triples")
    logger.info(f"  Unique nouns: {len(nouns):,}")
    logger.info(f"  Unique verbs: {len(verbs):,}")

    vocabularies = {'nouns': nouns, 'verbs': verbs}

    return triples, vocabularies


def generate_semantic_negatives(
    positive_triples: List[Dict],
    vocabularies: Dict[str, Set[str]],
    corruptor: SemanticDistanceCorruptor,
    negatives_per_positive: int = 1
) -> List[Dict]:
    """Generate negative samples using semantic distance corruption."""
    logger.info(f"Generating {negatives_per_positive} semantic-distance negatives per positive")

    nouns = list(vocabularies['nouns'])
    verbs = list(vocabularies['verbs'])

    negatives = []
    corruption_types = ['subject', 'object', 'verb']

    total_to_generate = len(positive_triples) * negatives_per_positive
    failed_corruptions = 0

    for idx, pos_triple in enumerate(positive_triples):
        if idx > 0 and idx % 100000 == 0:
            logger.info(f"  Generated {len(negatives):,} / {total_to_generate:,} negatives ({100*len(negatives)/total_to_generate:.1f}%)")
            if failed_corruptions > 0:
                logger.info(f"  Failed to find distant candidates: {failed_corruptions:,} times (fell back to random)")

        subj = pos_triple['subject_root']
        verb = pos_triple['verb_root']
        obj = pos_triple['object_root']

        for _ in range(negatives_per_positive):
            corruption = random.choice(corruption_types)

            result = corruptor.corrupt_triple(subj, verb, obj, nouns, verbs, corruption)

            if result is None:
                # Fallback to random if no distant candidate found
                failed_corruptions += 1
                if corruption == 'subject':
                    candidates = [n for n in nouns if n != subj]
                    corrupted_subj = random.choice(candidates) if candidates else subj
                    result = (corrupted_subj, verb, obj, 'subject')
                elif corruption == 'object':
                    candidates = [n for n in nouns if n != obj]
                    corrupted_obj = random.choice(candidates) if candidates else obj
                    result = (subj, verb, corrupted_obj, 'object')
                else:  # verb
                    candidates = [v for v in verbs if v != verb]
                    corrupted_verb = random.choice(candidates) if candidates else verb
                    result = (subj, corrupted_verb, obj, 'verb')

            corrupted_subj, corrupted_verb, corrupted_obj, corruption_type = result

            neg_triple = {
                'subject_root': corrupted_subj,
                'verb_root': corrupted_verb,
                'object_root': corrupted_obj,
                'label': 0.0,
                'corruption': corruption_type,
                'source': pos_triple['source'],
                'original_text': pos_triple['original_text']
            }

            negatives.append(neg_triple)

    logger.info(f"Generated {len(negatives):,} negative samples")
    if failed_corruptions > 0:
        logger.info(f"  Fell back to random for {failed_corruptions:,} cases ({100*failed_corruptions/total_to_generate:.1f}%)")

    return negatives


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


def save_splits(train: List[Dict], val: List[Dict], test: List[Dict], output_dir: Path):
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
        nouns.add(item['subject_root'])
        nouns.add(item['object_root'])
        verbs.add(item['verb_root'])

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


def main():
    parser = argparse.ArgumentParser(
        description="Prepare M1 training data with semantic-distance-based corruption"
    )
    parser.add_argument(
        '--corpus', type=Path,
        default=Path('data/enhanced_corpus/corpus_with_tier0.jsonl'),
        help='Path to parsed corpus'
    )
    parser.add_argument(
        '--stage1-model', type=Path,
        default=Path('models/root_embeddings_tier0/best_model.pt'),
        help='Path to Stage 1 embeddings'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('data/training/m1_semantic'),
        help='Output directory for training data'
    )
    parser.add_argument(
        '--max-triples', type=int,
        default=None,
        help='Maximum positive triples to extract (default: all)'
    )
    parser.add_argument(
        '--negatives-per-positive', type=int,
        default=1,
        help='Number of negative samples per positive (default: 1)'
    )
    parser.add_argument(
        '--similarity-threshold', type=float,
        default=0.15,
        help='Maximum similarity for corruption (default: 0.15)'
    )
    parser.add_argument(
        '--min-parse-rate', type=float,
        default=0.7,
        help='Minimum parse rate to include (default: 0.7, use 0.0 for tier0)'
    )
    parser.add_argument(
        '--train-ratio', type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio', type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio', type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    if not args.stage1_model.exists():
        logger.error(f"Stage 1 model not found: {args.stage1_model}")
        logger.error("Train Stage 1 first: ./scripts/train_roots.sh")
        return 1

    # Initialize semantic distance corruptor
    corruptor = SemanticDistanceCorruptor(
        args.stage1_model,
        similarity_threshold=args.similarity_threshold
    )

    # Load corpus triples
    positive_triples, vocabularies = load_corpus_triples(
        args.corpus,
        max_triples=args.max_triples,
        min_parse_rate=args.min_parse_rate
    )

    if not positive_triples:
        logger.error("No valid triples found in corpus")
        return 1

    # Generate semantic-distance negatives
    negative_triples = generate_semantic_negatives(
        positive_triples,
        vocabularies,
        corruptor,
        negatives_per_positive=args.negatives_per_positive
    )

    # Combine and shuffle
    all_data = positive_triples + negative_triples

    logger.info(f"Total examples: {len(all_data):,} ({len(positive_triples):,} positive, {len(negative_triples):,} negative)")

    # Split into train/val/test
    train, val, test = split_data(
        all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Save splits
    save_splits(train, val, test, args.output_dir)

    # Save metadata
    metadata = {
        'description': 'M1 training data with semantic-distance-based corruption',
        'corpus': str(args.corpus),
        'stage1_model': str(args.stage1_model),
        'similarity_threshold': args.similarity_threshold,
        'total_examples': len(all_data),
        'train_examples': len(train),
        'val_examples': len(val),
        'test_examples': len(test),
        'plausible_count': len(positive_triples),
        'implausible_count': len(negative_triples),
        'negatives_per_positive': args.negatives_per_positive,
        'max_triples': args.max_triples,
        'min_parse_rate': args.min_parse_rate
    }

    metadata_path = args.output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
