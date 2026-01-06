#!/usr/bin/env python3
"""
Train topical embeddings independently.

Trains ONLY topical embeddings on skip-gram pairs from corpus.
Does NOT depend on linguistic embeddings - fully independent.

Input: Skip-gram training pairs (data/training/topical_pairs_smart.jsonl)
Output: Topical embedding model (models/topical_embeddings/best_model.pt)

Training:
- MSE loss on cosine similarity
- Adam optimizer
- Batch processing
- Checkpoint resume support
"""

import argparse
import json
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from klareco.embeddings.topical_embeddings import TopicalEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TopicalTrainer:
    """Trainer for topical embeddings."""

    def __init__(
        self,
        model: TopicalEmbeddings,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(
        self,
        data_path: Path,
        batch_size: int = 1024
    ) -> Tuple[float, int]:
        """
        Train one epoch.

        Args:
            data_path: Path to training pairs (JSONL)
            batch_size: Batch size

        Returns:
            (average_loss, num_batches)
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Batch accumulation
        batch_idx1 = []
        batch_idx2 = []
        batch_targets = []

        with open(data_path) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,} pairs, avg loss: {total_loss / max(num_batches, 1):.4f}")

                pair = json.loads(line)
                idx1 = pair['idx1']
                idx2 = pair['idx2']
                target = pair['target_similarity']

                batch_idx1.append(idx1)
                batch_idx2.append(idx2)
                batch_targets.append(target)

                # Process batch
                if len(batch_idx1) >= batch_size:
                    loss = self._process_batch(batch_idx1, batch_idx2, batch_targets)
                    total_loss += loss
                    num_batches += 1

                    # Clear batch
                    batch_idx1 = []
                    batch_idx2 = []
                    batch_targets = []

        # Process remaining pairs
        if batch_idx1:
            loss = self._process_batch(batch_idx1, batch_idx2, batch_targets)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, num_batches

    def _process_batch(
        self,
        idx1: list,
        idx2: list,
        targets: list
    ) -> float:
        """Process a single batch."""
        # Convert to tensors
        idx1_t = torch.tensor(idx1, dtype=torch.long, device=self.device)
        idx2_t = torch.tensor(idx2, dtype=torch.long, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Forward pass
        emb1 = self.model(idx1_t)
        emb2 = self.model(idx2_t)

        # Cosine similarity
        similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1)

        # Loss
        loss = self.criterion(similarity, targets_t)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def save_checkpoint(
    output_dir: Path,
    model: TopicalEmbeddings,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    is_best: bool = False
):
    """Save training checkpoint atomically."""
    filename = 'best_model.pt' if is_best else f'checkpoint_epoch{epoch+1}.pt'
    checkpoint_path = output_dir / filename
    temp_path = checkpoint_path.with_suffix('.tmp')

    try:
        model.save_checkpoint(
            temp_path,
            epoch=epoch,
            loss=loss,
            optimizer_state_dict=optimizer.state_dict()
        )
        temp_path.rename(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def train_topical_embeddings(
    training_data_path: Path,
    vocab_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    Train topical embeddings.

    Args:
        training_data_path: Path to skip-gram training pairs
        vocab_path: Path to vocabulary JSON
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cpu or cuda)
    """
    logger.info("=" * 60)
    logger.info("Topical Embeddings Training")
    logger.info("=" * 60)
    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Vocabulary: {vocab_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Load vocabulary
    logger.info(f"\nLoading vocabulary from {vocab_path}")
    with open(vocab_path) as f:
        vocab = json.load(f)

    vocab_size = len(vocab['root_to_idx'])
    logger.info(f"Vocabulary size: {vocab_size:,}")

    # Create model
    model = TopicalEmbeddings(
        vocab_size=vocab_size,
        embedding_dim=64,
        root_to_idx=vocab['root_to_idx'],
        idx_to_root=vocab.get('idx_to_root', {idx: root for root, idx in vocab['root_to_idx'].items()})
    )

    logger.info(f"Model created: {vocab_size:,} roots, 64d embeddings")

    # Create trainer
    trainer = TopicalTrainer(model, learning_rate=learning_rate, device=device)

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'=' * 60}")

        loss, num_batches = trainer.train_epoch(
            training_data_path,
            batch_size=batch_size
        )

        logger.info(f"Epoch {epoch + 1}: Loss = {loss:.4f} ({num_batches:,} batches)")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(
                output_dir,
                model,
                trainer.optimizer,
                epoch,
                loss,
                is_best=True
            )

        # Save periodic checkpoint
        if (epoch + 1) % 2 == 0:
            save_checkpoint(
                output_dir,
                model,
                trainer.optimizer,
                epoch,
                loss,
                is_best=False
            )

    # Save final model
    final_path = output_dir / 'final_model.pt'
    model.save_checkpoint(final_path, epochs=epochs, final_loss=best_loss)

    logger.info(f"\n{'=' * 60}")
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Best model: {output_dir / 'best_model.pt'}")
    logger.info(f"Final model: {final_path}")
    logger.info(f"Training log: {log_file}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Train topical embeddings')
    parser.add_argument('--training-data', type=Path,
                        default=Path('data/training/topical_pairs_smart.jsonl'),
                        help='Path to training pairs (JSONL)')
    parser.add_argument('--vocab', type=Path,
                        default=Path('data/vocabularies/topical_vocab.json'),
                        help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/topical_embeddings'),
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Validate inputs
    if not args.training_data.exists():
        logger.error(f"Training data not found: {args.training_data}")
        return 1

    if not args.vocab.exists():
        logger.error(f"Vocabulary not found: {args.vocab}")
        return 1

    # Train
    train_topical_embeddings(
        training_data_path=args.training_data,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
