#!/usr/bin/env python3
"""
Train Semantic Embeddings with Contrastive Learning.

Uses triplet margin loss to train embeddings where:
- Synonyms/related words have high similarity
- Unrelated words have low similarity

This captures semantic relationships, not just distributional co-occurrence.

Usage:
    python scripts/training/train_semantic_embeddings.py
    python scripts/training/train_semantic_embeddings.py --resume
    python scripts/training/train_semantic_embeddings.py --fresh

Features:
    - Triplet margin loss for contrastive learning
    - Checkpoint saving every epoch
    - Restartable from checkpoint
    - Progress logging to file and screen
    - Early stopping on validation loss
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"semantic_training_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("semantic_trainer")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


class TripletDataset(Dataset):
    """Dataset for triplet training data."""

    def __init__(self, triplets_path: Path, root_to_idx: Dict[str, int]):
        self.triplets = []
        self.root_to_idx = root_to_idx

        with open(triplets_path) as f:
            for line in f:
                triplet = json.loads(line)
                anchor = triplet['anchor']
                positive = triplet['positive']
                negative = triplet['negative']

                # Only include triplets where all roots are in vocabulary
                if all(r in root_to_idx for r in [anchor, positive, negative]):
                    self.triplets.append((
                        root_to_idx[anchor],
                        root_to_idx[positive],
                        root_to_idx[negative],
                    ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return (
            torch.tensor(anchor, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negative, dtype=torch.long),
        )


class SemanticEmbedding(nn.Module):
    """Simple embedding model for semantic training."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def get_embedding(self, idx: int) -> torch.Tensor:
        """Get embedding for a single root index."""
        with torch.no_grad():
            return self.embedding.weight[idx]


class SemanticTrainer:
    """Trainer for semantic embeddings with contrastive learning."""

    def __init__(
        self,
        model: SemanticEmbedding,
        learning_rate: float = 0.001,
        margin: float = 0.5,
        device: str = 'cpu',
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            # Get embeddings
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)

            # Compute triplet loss
            loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / num_batches
                self.logger.info(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate_similarities(
        self,
        test_pairs: List[Tuple[str, str]],
        root_to_idx: Dict[str, int],
    ) -> Dict[str, float]:
        """Evaluate similarity on test pairs."""
        self.model.eval()

        similarities = []
        with torch.no_grad():
            for a, b in test_pairs:
                if a not in root_to_idx or b not in root_to_idx:
                    continue

                emb_a = self.model.get_embedding(root_to_idx[a])
                emb_b = self.model.get_embedding(root_to_idx[b])

                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    emb_a.unsqueeze(0),
                    emb_b.unsqueeze(0),
                ).item()
                similarities.append((a, b, sim))

        return similarities

    def save_checkpoint(self, path: Path, root_to_idx: Dict[str, int]):
        """Save checkpoint atomically."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'root_to_idx': root_to_idx,
        }

        temp_path = path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(path)

    def load_checkpoint(self, path: Path) -> Dict[str, int]:
        """Load checkpoint and return root_to_idx."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.patience_counter = checkpoint.get('patience_counter', 0)

        return checkpoint['root_to_idx']


def build_vocabulary(triplets_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from triplet data."""
    roots = set()

    with open(triplets_path) as f:
        for line in f:
            triplet = json.loads(line)
            roots.add(triplet['anchor'])
            roots.add(triplet['positive'])
            roots.add(triplet['negative'])

    # Sort for reproducibility
    roots = sorted(roots)
    root_to_idx = {root: idx for idx, root in enumerate(roots)}
    idx_to_root = {idx: root for idx, root in enumerate(roots)}

    return root_to_idx, idx_to_root


def main():
    parser = argparse.ArgumentParser(description="Train semantic embeddings")
    parser.add_argument(
        "--triplets",
        type=Path,
        default=PROJECT_ROOT / "data/training/semantic_triplets.jsonl",
        help="Path to triplet training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models/semantic_embeddings",
        help="Output directory for model",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Triplet margin loss margin",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = PROJECT_ROOT / "logs" / "training"
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("Training Semantic Embeddings")
    logger.info("=" * 60)

    # Check input files
    if not args.triplets.exists():
        logger.error(f"Triplets file not found: {args.triplets}")
        logger.error("Run scripts/prepare_semantic_training_data.py first")
        sys.exit(1)

    # Checkpoint path
    checkpoint_path = args.output_dir / "checkpoint.pt"
    best_model_path = args.output_dir / "best_model.pt"

    # Handle resume/fresh
    resume_from_checkpoint = False
    if args.fresh and checkpoint_path.exists():
        logger.info("Fresh start requested, removing checkpoint...")
        checkpoint_path.unlink()
    elif args.resume and checkpoint_path.exists():
        resume_from_checkpoint = True
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    elif checkpoint_path.exists():
        # Default: resume if checkpoint exists
        resume_from_checkpoint = True
        logger.info(f"Found checkpoint, resuming: {checkpoint_path}")

    # Build vocabulary
    logger.info(f"Building vocabulary from {args.triplets}")
    root_to_idx, idx_to_root = build_vocabulary(args.triplets)
    vocab_size = len(root_to_idx)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create model
    model = SemanticEmbedding(vocab_size, args.embedding_dim)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = SemanticTrainer(
        model=model,
        learning_rate=args.learning_rate,
        margin=args.margin,
        device=args.device,
        logger=logger,
    )

    # Load checkpoint if resuming
    if resume_from_checkpoint:
        loaded_root_to_idx = trainer.load_checkpoint(checkpoint_path)
        # Verify vocabulary matches
        if loaded_root_to_idx != root_to_idx:
            logger.warning("Vocabulary changed since checkpoint, starting fresh")
            resume_from_checkpoint = False
            trainer.epoch = 0
            trainer.best_loss = float('inf')
        else:
            logger.info(f"Resumed at epoch {trainer.epoch + 1}")

    # Create dataset and dataloader
    logger.info("Loading triplet dataset...")
    dataset = TripletDataset(args.triplets, root_to_idx)
    logger.info(f"Dataset size: {len(dataset)} triplets")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )

    # Test pairs for evaluation
    test_pairs = [
        ('esperant', 'lingv'),
        ('fond', 'kre'),
        ('zamenhof', 'aŭtor'),
        ('hund', 'kat'),
        ('manĝ', 'trink'),
        ('grand', 'vast'),
        ('bon', 'bel'),
        ('leg', 'skrib'),
        ('parol', 'dir'),
        ('vid', 'aŭd'),
    ]

    # Training loop
    logger.info("")
    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Margin: {args.margin}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    start_epoch = trainer.epoch
    for epoch in range(start_epoch, args.epochs):
        trainer.epoch = epoch

        epoch_start = time.time()
        loss = trainer.train_epoch(dataloader, epoch)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Evaluate on test pairs
        similarities = trainer.evaluate_similarities(test_pairs, root_to_idx)
        logger.info("  Test similarities:")
        for a, b, sim in similarities[:5]:
            logger.info(f"    {a} ~ {b}: {sim:.3f}")

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, root_to_idx)
        logger.info(f"  Checkpoint saved")

        # Check for improvement
        if loss < trainer.best_loss:
            trainer.best_loss = loss
            trainer.patience_counter = 0
            trainer.save_checkpoint(best_model_path, root_to_idx)
            logger.info(f"  New best model saved (loss: {loss:.4f})")
        else:
            trainer.patience_counter += 1
            logger.info(f"  No improvement ({trainer.patience_counter}/{args.patience})")

        # Early stopping
        if trainer.patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best loss: {trainer.best_loss:.4f}")
    logger.info(f"Model saved to: {best_model_path}")

    # Save vocabulary
    vocab_path = args.output_dir / "vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump(root_to_idx, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulary saved to: {vocab_path}")

    # Final evaluation
    logger.info("")
    logger.info("Final test similarities:")
    trainer.load_checkpoint(best_model_path)
    similarities = trainer.evaluate_similarities(test_pairs, root_to_idx)
    for a, b, sim in similarities:
        logger.info(f"  {a} ~ {b}: {sim:.3f}")


if __name__ == "__main__":
    main()
