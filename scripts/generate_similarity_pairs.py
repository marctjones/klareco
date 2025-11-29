#!/usr/bin/env python3
"""
Generate Semantic Similarity Training Pairs from Parallel Corpus.

Uses English as a similarity oracle to identify semantically similar
Esperanto sentence pairs, then outputs training data containing ONLY
Esperanto text (English is discarded after labeling).

Strategy:
1. Load Esperanto-English parallel corpus
2. Group by English sentence to find Esperanto paraphrases (same meaning)
3. Compute English embeddings to find semantically similar sentences
4. Generate training triplets: (esperanto_a, esperanto_b, similarity_score)

The model will only ever see Esperanto - English is just for labeling.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm


def load_parallel_corpus(en_path: Path, eo_path: Path) -> List[Tuple[str, str]]:
    """Load aligned English-Esperanto sentence pairs."""
    pairs = []
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(eo_path, 'r', encoding='utf-8') as f_eo:
        for en, eo in zip(f_en, f_eo):
            en = en.strip()
            eo = eo.strip()
            if en and eo:
                pairs.append((en, eo))
    return pairs


def group_by_english(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Group Esperanto sentences by their English translation.

    This finds natural paraphrases - different Esperanto expressions
    of the same English meaning.
    """
    en_to_eo = defaultdict(list)
    for en, eo in pairs:
        en_to_eo[en].append(eo)
    return dict(en_to_eo)


def compute_english_embeddings(
    sentences: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 128,
) -> np.ndarray:
    """Compute embeddings for English sentences."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(sentences):,} sentences...")
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings


def generate_paraphrase_pairs(
    en_to_eo: Dict[str, List[str]],
    max_pairs_per_group: int = 10,
) -> List[Tuple[str, str, float]]:
    """Generate positive pairs from Esperanto paraphrases.

    These are sentences with the same English translation,
    so they have the same meaning.
    """
    pairs = []
    for en, eo_list in tqdm(en_to_eo.items(), desc="Generating paraphrase pairs"):
        if len(eo_list) < 2:
            continue

        # Generate pairs from this group
        unique_eo = list(set(eo_list))  # Remove duplicates
        if len(unique_eo) < 2:
            continue

        # Sample pairs if too many
        n_pairs = 0
        for i in range(len(unique_eo)):
            for j in range(i + 1, len(unique_eo)):
                if n_pairs >= max_pairs_per_group:
                    break
                # These are paraphrases - similarity = 1.0
                pairs.append((unique_eo[i], unique_eo[j], 1.0))
                n_pairs += 1
            if n_pairs >= max_pairs_per_group:
                break

    return pairs


def generate_similarity_pairs(
    pairs: List[Tuple[str, str]],
    embeddings: np.ndarray,
    en_sentences: List[str],
    en_to_idx: Dict[str, int],
    high_sim_threshold: float = 0.8,
    medium_sim_threshold: float = 0.5,
    low_sim_threshold: float = 0.3,
    max_pairs_per_type: int = 100000,
) -> List[Tuple[str, str, float]]:
    """Generate similarity pairs using English embedding similarity.

    For each Esperanto sentence pair, compute similarity based on
    their English translations' embedding similarity.
    """

    # Build Esperanto -> English mapping
    eo_to_en = {}
    for en, eo in pairs:
        if eo not in eo_to_en:
            eo_to_en[eo] = en

    unique_eo = list(eo_to_en.keys())
    n = len(unique_eo)
    print(f"Finding similar pairs among {n:,} unique Esperanto sentences...")

    high_sim_pairs = []
    medium_sim_pairs = []
    low_sim_pairs = []

    # Sample random pairs and compute similarity
    n_samples = min(n * 100, 5000000)  # Sample up to 5M pairs

    for _ in tqdm(range(n_samples), desc="Sampling pairs"):
        i, j = random.sample(range(n), 2)
        eo_a, eo_b = unique_eo[i], unique_eo[j]
        en_a, en_b = eo_to_en[eo_a], eo_to_en[eo_b]

        # Skip if same English (those are paraphrases, handled separately)
        if en_a == en_b:
            continue

        # Get English embeddings
        idx_a = en_to_idx.get(en_a)
        idx_b = en_to_idx.get(en_b)
        if idx_a is None or idx_b is None:
            continue

        # Cosine similarity (embeddings are normalized)
        sim = float(np.dot(embeddings[idx_a], embeddings[idx_b]))

        # Categorize
        if sim >= high_sim_threshold and len(high_sim_pairs) < max_pairs_per_type:
            high_sim_pairs.append((eo_a, eo_b, sim))
        elif medium_sim_threshold <= sim < high_sim_threshold and len(medium_sim_pairs) < max_pairs_per_type:
            medium_sim_pairs.append((eo_a, eo_b, sim))
        elif sim < low_sim_threshold and len(low_sim_pairs) < max_pairs_per_type:
            low_sim_pairs.append((eo_a, eo_b, sim))

        # Early exit if we have enough
        if (len(high_sim_pairs) >= max_pairs_per_type and
            len(medium_sim_pairs) >= max_pairs_per_type and
            len(low_sim_pairs) >= max_pairs_per_type):
            break

    print(f"  High similarity (>{high_sim_threshold}): {len(high_sim_pairs):,}")
    print(f"  Medium similarity ({medium_sim_threshold}-{high_sim_threshold}): {len(medium_sim_pairs):,}")
    print(f"  Low similarity (<{low_sim_threshold}): {len(low_sim_pairs):,}")

    return high_sim_pairs + medium_sim_pairs + low_sim_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic similarity training pairs from parallel corpus"
    )
    parser.add_argument(
        "--en-file",
        type=Path,
        default=Path("data/tatoeba/Tatoeba.en-eo.en"),
        help="Path to English sentences file",
    )
    parser.add_argument(
        "--eo-file",
        type=Path,
        default=Path("data/tatoeba/Tatoeba.en-eo.eo"),
        help="Path to Esperanto sentences file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/similarity_pairs.jsonl"),
        help="Output file for training pairs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="Sentence transformer model for English similarity",
    )
    parser.add_argument(
        "--max-paraphrase-pairs",
        type=int,
        default=200000,
        help="Maximum paraphrase pairs to generate",
    )
    parser.add_argument(
        "--max-similarity-pairs",
        type=int,
        default=100000,
        help="Maximum pairs per similarity category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load parallel corpus
    print(f"\nLoading parallel corpus...")
    print(f"  English: {args.en_file}")
    print(f"  Esperanto: {args.eo_file}")
    pairs = load_parallel_corpus(args.en_file, args.eo_file)
    print(f"  Loaded {len(pairs):,} sentence pairs")

    # Group by English to find paraphrases
    print(f"\nGrouping by English sentence...")
    en_to_eo = group_by_english(pairs)
    multi_translation = {k: v for k, v in en_to_eo.items() if len(v) > 1}
    print(f"  Unique English sentences: {len(en_to_eo):,}")
    print(f"  With multiple Esperanto translations: {len(multi_translation):,}")

    # Generate paraphrase pairs (same meaning)
    print(f"\n=== Phase 1: Paraphrase Pairs ===")
    paraphrase_pairs = generate_paraphrase_pairs(en_to_eo)
    print(f"Generated {len(paraphrase_pairs):,} paraphrase pairs (similarity=1.0)")

    # Compute English embeddings for similarity
    print(f"\n=== Phase 2: Similarity Pairs ===")
    unique_en = list(en_to_eo.keys())
    en_to_idx = {en: i for i, en in enumerate(unique_en)}

    embeddings = compute_english_embeddings(unique_en, model_name=args.model)

    # Generate similarity pairs
    similarity_pairs = generate_similarity_pairs(
        pairs,
        embeddings,
        unique_en,
        en_to_idx,
        max_pairs_per_type=args.max_similarity_pairs,
    )

    # Combine all pairs
    all_pairs = paraphrase_pairs + similarity_pairs
    random.shuffle(all_pairs)

    # Split into train/val/test
    n = len(all_pairs)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    # Write output
    print(f"\n=== Writing Output ===")

    def write_pairs(pairs: List[Tuple[str, str, float]], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            for eo_a, eo_b, sim in pairs:
                record = {
                    "sentence_a": eo_a,
                    "sentence_b": eo_b,
                    "similarity": round(sim, 4),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Write train/val/test splits
    output_base = args.output.stem
    output_dir = args.output.parent

    train_path = output_dir / f"{output_base}_train.jsonl"
    val_path = output_dir / f"{output_base}_val.jsonl"
    test_path = output_dir / f"{output_base}_test.jsonl"

    write_pairs(train_pairs, train_path)
    write_pairs(val_pairs, val_path)
    write_pairs(test_pairs, test_path)

    print(f"  Train: {len(train_pairs):,} pairs -> {train_path}")
    print(f"  Val:   {len(val_pairs):,} pairs -> {val_path}")
    print(f"  Test:  {len(test_pairs):,} pairs -> {test_path}")

    # Statistics
    print(f"\n=== Summary ===")
    print(f"Total pairs: {len(all_pairs):,}")

    sim_counts = defaultdict(int)
    for _, _, sim in all_pairs:
        if sim >= 0.9:
            sim_counts["0.9-1.0"] += 1
        elif sim >= 0.7:
            sim_counts["0.7-0.9"] += 1
        elif sim >= 0.5:
            sim_counts["0.5-0.7"] += 1
        elif sim >= 0.3:
            sim_counts["0.3-0.5"] += 1
        else:
            sim_counts["0.0-0.3"] += 1

    print("Similarity distribution:")
    for bucket in ["0.9-1.0", "0.7-0.9", "0.5-0.7", "0.3-0.5", "0.0-0.3"]:
        print(f"  {bucket}: {sim_counts[bucket]:,}")

    print("\nDone! Training data contains ONLY Esperanto sentences.")
    print("English was used only to determine similarity labels.")


if __name__ == "__main__":
    main()
