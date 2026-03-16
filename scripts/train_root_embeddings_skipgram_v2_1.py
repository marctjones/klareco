#!/usr/bin/env python3
"""
Train Root Embeddings with Skip-Gram and Negative Sampling (Phase 1)

VERSION: v2.1
COMPATIBLE WITH: v2.1 training pairs (from extract_embedding_training_pairs.py)
DEPENDENCIES: None (standalone embedding model)
STAGE: Training

Description:
    Trains skip-gram embeddings for Esperanto content roots with negative sampling.
    Focuses ONLY on semantic similarity - grammar is handled deterministically.
    Includes early stopping and embedding collapse detection.

Pipeline Position:
    training_pairs.jsonl → [THIS SCRIPT] → root_embeddings.pt → retriever integration

Usage:
    python scripts/train_root_embeddings_skipgram_v2_1.py \
        --training-pairs data/training/root_embedding_pairs.jsonl \
        --vocabulary data/training/root_embedding_pairs_vocab.json \
        --output models/root_embeddings_phase1 \
        --embedding-dim 64 \
        --epochs 10 \
        --batch-size 1024 \
        --learning-rate 0.025 \
        --negative-samples 5

Inputs:
    - training_pairs.jsonl: {"target": "hund", "context": "kat", "weight": 1.0}
    - vocabulary JSON: ["hund", "kat", "arb", ...]

Outputs:
    - root_embeddings.pt: Model checkpoint with embeddings
    - Contains: {
        'embeddings': tensor (vocab_size x embedding_dim),
        'vocab': list of roots,
        'root_to_idx': dict mapping root → index,
        'embedding_dim': 64,
        'training_stats': {...}
      }

Quality Checks:
    - Early stopping: patience=3 epochs, min_delta=0.001
    - Collapse detection: mean_similarity < 0.7 threshold
    - Loss validation: converging to reasonable values
    - Checkpoint saving: best model preserved

Last Updated: 2026-03-09
Author: Claude + Marc
Related Issues: Phase 1 Root Embeddings
See Also: docs/ROOT_EMBEDDINGS_DESIGN.md
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkipGramDataset(Dataset):
    """
    Dataset for skip-gram training with weighted pairs and adaptive negative sampling.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, float]],
        root_to_idx: Dict[str, int],
        negative_samples: int = 5,
        vocab_size: int = 5000
    ):
        self.pairs = pairs
        self.root_to_idx = root_to_idx
        self.negative_samples = negative_samples
        self.vocab_size = vocab_size

        # Convert to indices
        self.indexed_pairs = [
            (root_to_idx[target], root_to_idx[context], weight)
            for target, context, weight in pairs
            if target in root_to_idx and context in root_to_idx
        ]

        logger.info(f"Dataset: {len(self.indexed_pairs)} pairs")

    def set_negative_samples(self, num_negatives: int):
        """Dynamically adjust number of negative samples (for adaptive sampling)."""
        self.negative_samples = num_negatives

    def __len__(self):
        return len(self.indexed_pairs)

    def __getitem__(self, idx):
        target_idx, context_idx, weight = self.indexed_pairs[idx]

        # Generate negative samples (random indices, excluding target and context)
        negative_indices = []
        while len(negative_indices) < self.negative_samples:
            neg_idx = random.randint(0, self.vocab_size - 1)
            if neg_idx != target_idx and neg_idx != context_idx:
                negative_indices.append(neg_idx)

        return (
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            torch.tensor(negative_indices, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32)
        )


class SkipGramModel(nn.Module):
    """
    Skip-gram model with negative sampling for root embeddings.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Target embeddings (what we actually use)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Context embeddings (for training only)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        nn.init.uniform_(self.target_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, target_idx, context_idx, negative_indices):
        """
        Compute skip-gram loss with negative sampling.

        Args:
            target_idx: (batch_size,) target word indices
            context_idx: (batch_size,) positive context word indices
            negative_indices: (batch_size, num_negatives) negative sample indices

        Returns:
            loss: scalar loss value
        """
        batch_size = target_idx.size(0)
        num_negatives = negative_indices.size(1)

        # Get embeddings
        target_emb = self.target_embeddings(target_idx)  # (batch_size, embedding_dim)
        context_emb = self.context_embeddings(context_idx)  # (batch_size, embedding_dim)
        negative_emb = self.context_embeddings(negative_indices)  # (batch_size, num_negatives, embedding_dim)

        # Positive loss: log(sigmoid(target · context))
        positive_score = torch.sum(target_emb * context_emb, dim=1)  # (batch_size,)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)

        # Negative loss: sum_i log(sigmoid(-target · negative_i))
        negative_scores = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2)  # (batch_size, num_negatives)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores) + 1e-10), dim=1)

        # Total loss
        loss = positive_loss + negative_loss

        return loss.mean()

    def get_embeddings(self):
        """
        Get the trained target embeddings (for downstream use).
        """
        return self.target_embeddings.weight.data


def load_training_data(pairs_path: Path, vocab_path: Path) -> Tuple[List, List, Dict]:
    """
    Load training pairs and vocabulary.

    Returns:
        (pairs, vocabulary, root_to_idx)
    """
    logger.info(f"Loading training pairs from {pairs_path}...")

    pairs = []
    with open(pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            pairs.append((record['target'], record['context'], record['weight']))

    logger.info(f"Loaded {len(pairs)} training pairs")

    logger.info(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    root_to_idx = {root: idx for idx, root in enumerate(vocabulary)}

    logger.info(f"Loaded vocabulary with {len(vocabulary)} roots")

    return pairs, vocabulary, root_to_idx


def detect_collapse(embeddings: torch.Tensor, threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Detect embedding collapse by computing mean pairwise similarity.

    Returns:
        (is_collapsed, mean_similarity)
    """
    # Normalize embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute pairwise similarities (sample 1000 random pairs for efficiency)
    n = embeddings_norm.size(0)
    num_samples = min(1000, n)

    indices = torch.randperm(n)[:num_samples]
    sample_emb = embeddings_norm[indices]

    # Compute similarity matrix
    similarity_matrix = torch.mm(sample_emb, sample_emb.t())

    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(num_samples, dtype=torch.bool)
    similarities = similarity_matrix[mask]

    mean_sim = similarities.mean().item()
    is_collapsed = mean_sim > threshold

    return is_collapsed, mean_sim


def train_epoch(
    model: SkipGramModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train one epoch.

    Returns:
        Average loss
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    for target_idx, context_idx, negative_indices, weights in tqdm(dataloader, desc="Training"):
        target_idx = target_idx.to(device)
        context_idx = context_idx.to(device)
        negative_indices = negative_indices.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()

        # Compute loss
        loss = model(target_idx, context_idx, negative_indices)

        # Apply weights
        loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: SkipGramModel,
    optimizer: optim.Optimizer,
    vocabulary: List[str],
    root_to_idx: Dict[str, int],
    stats: Dict,
    epoch: int,
    best_loss: float,
    patience_counter: int,
    output_path: Path
) -> None:
    """
    Save model checkpoint with embeddings, optimizer state, and training state.
    """
    logger.info(f"Saving checkpoint to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'embeddings': model.get_embeddings(),
        'vocab': vocabulary,
        'root_to_idx': root_to_idx,
        'embedding_dim': model.embedding_dim,
        'vocab_size': model.vocab_size,
        'training_stats': stats,
        'epoch': epoch,
        'best_loss': best_loss,
        'patience_counter': patience_counter
    }

    # Atomic save (write to temp file, then rename)
    temp_path = output_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(output_path)

    logger.info(f"Checkpoint saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def load_checkpoint(
    checkpoint_path: Path,
    model: SkipGramModel,
    optimizer: optim.Optimizer
) -> Tuple[int, float, int, Dict]:
    """
    Load checkpoint and restore training state.

    Returns:
        (start_epoch, best_loss, patience_counter, training_stats)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Restore model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_loss = checkpoint['best_loss']
    patience_counter = checkpoint['patience_counter']
    training_stats = checkpoint['training_stats']

    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    logger.info(f"Best loss so far: {best_loss:.4f}")
    logger.info(f"Patience counter: {patience_counter}")

    return start_epoch, best_loss, patience_counter, training_stats


def get_adaptive_negative_samples(epoch: int, total_epochs: int, initial_k: int = 10) -> int:
    """
    Compute adaptive number of negative samples based on training progress.

    Start with more negatives (10) for diverse learning early on,
    reduce to fewer negatives (3) as training progresses for speed.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        initial_k: Initial number of negative samples (default: 10)

    Returns:
        Number of negative samples for this epoch
    """
    progress = epoch / total_epochs

    if progress < 0.3:  # First 30% of training
        return initial_k
    elif progress < 0.6:  # Middle 30%
        return max(5, initial_k // 2)
    else:  # Final 40%
        return 3




def main():
    parser = argparse.ArgumentParser(
        description='Train skip-gram root embeddings with negative sampling (Phase 1)'
    )
    parser.add_argument(
        '--training-pairs',
        type=Path,
        required=True,
        help='Path to training pairs JSONL file'
    )
    parser.add_argument(
        '--vocabulary',
        type=Path,
        required=True,
        help='Path to vocabulary JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for model checkpoint'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=64,
        help='Embedding dimension (default: 64)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size (default: 1024)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.025,
        help='Learning rate (default: 0.025)'
    )
    parser.add_argument(
        '--negative-samples',
        type=int,
        default=5,
        help='Number of negative samples per positive pair (default: 5)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience (default: 3)'
    )
    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.001,
        help='Minimum improvement for early stopping (default: 0.001)'
    )
    parser.add_argument(
        '--collapse-threshold',
        type=float,
        default=0.7,
        help='Embedding collapse detection threshold (default: 0.7)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu only, default: cpu)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint if it exists'
    )
    parser.add_argument(
        '--dataset-fraction',
        type=float,
        default=1.0,
        help='Fraction of training data to use (0.0-1.0, default: 1.0 = all data). Use 0.33 for 3x speedup'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Root Embedding Training (Skip-Gram, Phase 1)")
    logger.info("=" * 80)
    logger.info(f"Training pairs: {args.training_pairs}")
    logger.info(f"Vocabulary: {args.vocabulary}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Embedding dim: {args.embedding_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Negative samples: {args.negative_samples}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    device = torch.device(args.device)

    # Load data
    pairs, vocabulary, root_to_idx = load_training_data(args.training_pairs, args.vocabulary)

    # Subsample dataset if requested (for faster training)
    if args.dataset_fraction < 1.0:
        import random
        original_size = len(pairs)
        sample_size = int(original_size * args.dataset_fraction)
        random.seed(42)  # Reproducible sampling
        pairs = random.sample(pairs, sample_size)
        logger.info(f"Subsampled dataset: {len(pairs):,} / {original_size:,} pairs ({args.dataset_fraction*100:.0f}%)")
        logger.info(f"Estimated speedup: {1/args.dataset_fraction:.1f}x faster")

    vocab_size = len(vocabulary)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create dataset and dataloader
    dataset = SkipGramDataset(
        pairs,
        root_to_idx,
        negative_samples=args.negative_samples,
        vocab_size=vocab_size
    )

    # Optimize DataLoader for CPU
    import os
    # Memory-aware worker selection:
    # Ultra-large (>80M pairs): 0 workers (single-threaded, avoids OOM)
    # Large (>50M pairs): 2 workers
    # Normal: 4 workers
    if len(pairs) > 80_000_000:
        num_workers = 0
        use_persistent = False
        logger.warning(f"Ultra-large dataset ({len(pairs):,} pairs) - using single-threaded loading to avoid OOM")
    elif len(pairs) > 50_000_000:
        num_workers = 2
        use_persistent = False
    else:
        num_workers = 4
        use_persistent = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )

    logger.info(f"DataLoader: {num_workers} workers, batch_size={args.batch_size}, persistent={use_persistent}")

    # Create model
    model = SkipGramModel(vocab_size, args.embedding_dim).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Check for existing checkpoint and resume if requested
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0
    training_stats = {
        'epochs': [],
        'losses': [],
        'mean_similarities': [],
    }

    checkpoint_path = args.output / 'root_embeddings_checkpoint.pt'
    if args.resume and checkpoint_path.exists():
        try:
            start_epoch, best_loss, patience_counter, training_stats = load_checkpoint(
                checkpoint_path, model, optimizer
            )
            logger.info("=" * 80)
            logger.info(f"RESUMING from epoch {start_epoch}")
            logger.info("=" * 80)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.warning("Starting training from scratch")
    elif args.resume:
        logger.info("No checkpoint found, starting fresh")
    else:
        logger.info("Starting fresh training (use --resume to continue from checkpoint)")

    for epoch in range(start_epoch, args.epochs + 1):
        # Adaptive negative sampling
        num_negatives = get_adaptive_negative_samples(epoch, args.epochs, args.negative_samples)
        dataset.set_negative_samples(num_negatives)

        logger.info(f"Epoch {epoch}/{args.epochs} (neg_samples={num_negatives})")

        # Train
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        # Check for collapse
        is_collapsed, mean_sim = detect_collapse(model.get_embeddings(), args.collapse_threshold)

        logger.info(f"Loss: {avg_loss:.4f} | Mean similarity: {mean_sim:.3f} | Collapsed: {is_collapsed}")

        training_stats['epochs'].append(epoch)
        training_stats['losses'].append(avg_loss)
        training_stats['mean_similarities'].append(mean_sim)

        # Early stopping check
        if avg_loss < best_loss - args.min_delta:
            best_loss = avg_loss
            patience_counter = 0

            # Save best model
            output_path = args.output / 'root_embeddings_best.pt'
            save_checkpoint(model, optimizer, vocabulary, root_to_idx, training_stats,
                          epoch, best_loss, patience_counter, output_path)
            logger.info(f"New best model (loss: {best_loss:.4f})")

            # Also save as checkpoint for resume
            checkpoint_output = args.output / 'root_embeddings_checkpoint.pt'
            save_checkpoint(model, optimizer, vocabulary, root_to_idx, training_stats,
                          epoch, best_loss, patience_counter, checkpoint_output)
        else:
            patience_counter += 1
            logger.info(f"No improvement (patience: {patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                logger.info("Early stopping triggered")
                break

        # Collapse check
        if is_collapsed:
            logger.warning(f"Embedding collapse detected (mean similarity: {mean_sim:.3f} > {args.collapse_threshold})")
            logger.warning("Stopping training to prevent further collapse")
            break

    # Save final model
    output_path = args.output / 'root_embeddings_final.pt'
    save_checkpoint(model, optimizer, vocabulary, root_to_idx, training_stats,
                  epoch, best_loss, patience_counter, output_path)

    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final mean similarity: {mean_sim:.3f}")
    logger.info(f"Total epochs: {epoch}")
    logger.info(f"Best model: {args.output / 'root_embeddings_best.pt'}")
    logger.info(f"Final model: {args.output / 'root_embeddings_final.pt'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
