#!/usr/bin/env python3
"""
Train Tree-LSTM encoder using contrastive learning.

Usage:
    python scripts/train_tree_lstm.py --training-data data/training_pairs --output models/tree_lstm --epochs 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.dataloader import create_dataloader
from klareco.logging_config import setup_logging


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training Tree-LSTM encoder.

    Encourages similar pairs to have small distance and dissimilar pairs to have large distance.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            emb1: Embeddings for first graph (batch_size, embed_dim)
            emb2: Embeddings for second graph (batch_size, embed_dim)
            labels: Labels (1 = similar, 0 = dissimilar)

        Returns:
            Loss scalar
        """
        # Compute L2 distance
        distances = F.pairwise_distance(emb1, emb2)

        # Positive pairs: minimize distance
        positive_loss = labels * torch.pow(distances, 2)

        # Negative pairs: maximize distance (up to margin)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        # Average loss
        loss = torch.mean(positive_loss + negative_loss)

        return loss


def train_epoch(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ContrastiveLoss,
    device: torch.device,
    logger
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: TreeLSTMEncoder model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        logger: Logger

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")

    for graphs1, graphs2, labels in pbar:
        # Move labels to device
        labels = labels.to(device)

        # Forward pass for both graphs
        embeddings1 = []
        embeddings2 = []

        for g1, g2 in zip(graphs1, graphs2):
            g1 = g1.to(device)
            g2 = g2.to(device)

            emb1 = model(g1)
            emb2 = model(g2)

            embeddings1.append(emb1)
            embeddings2.append(emb2)

        # Stack embeddings
        embeddings1 = torch.stack(embeddings1)
        embeddings2 = torch.stack(embeddings2)

        # Compute loss
        loss = criterion(embeddings1, embeddings2, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total += labels.size(0)

        # Compute accuracy (distance < threshold = similar)
        distances = F.pairwise_distance(embeddings1, embeddings2)
        threshold = 0.5  # Tune this
        predictions = (distances < threshold).float()
        correct += (predictions == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate_epoch(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
    logger
) -> Tuple[float, float]:
    """
    Validate for one epoch.

    Args:
        model: TreeLSTMEncoder model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        logger: Logger

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for graphs1, graphs2, labels in pbar:
            # Move labels to device
            labels = labels.to(device)

            # Forward pass for both graphs
            embeddings1 = []
            embeddings2 = []

            for g1, g2 in zip(graphs1, graphs2):
                g1 = g1.to(device)
                g2 = g2.to(device)

                emb1 = model(g1)
                emb2 = model(g2)

                embeddings1.append(emb1)
                embeddings2.append(emb2)

            # Stack embeddings
            embeddings1 = torch.stack(embeddings1)
            embeddings2 = torch.stack(embeddings2)

            # Compute loss
            loss = criterion(embeddings1, embeddings2, labels)

            # Statistics
            total_loss += loss.item()
            total += labels.size(0)

            # Compute accuracy
            distances = F.pairwise_distance(embeddings1, embeddings2)
            threshold = 0.5
            predictions = (distances < threshold).float()
            correct += (predictions == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: TreeLSTMEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    output_dir: Path,
    logger
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    checkpoint_file = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"  Saved checkpoint: {checkpoint_file}")

    # Also save as "best" if this is the best validation loss
    best_file = output_dir / 'best_model.pt'
    if not best_file.exists() or val_loss < torch.load(best_file)['val_loss']:
        torch.save(checkpoint, best_file)
        logger.info(f"  Saved best model: {best_file}")


def load_checkpoint(checkpoint_path: Path, model, optimizer, logger):
    """
    Load checkpoint and return start epoch.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        logger: Logger

    Returns:
        Starting epoch number
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    logger.info(f"  Resumed from epoch {checkpoint['epoch']}")
    logger.info(f"  Train loss: {checkpoint['train_loss']:.4f}")

    return start_epoch


def find_latest_checkpoint(output_dir: Path) -> Path:
    """Find the latest checkpoint in the output directory."""
    checkpoints = list(output_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None

    # Sort by epoch number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return checkpoints[-1]


def main():
    """Train Tree-LSTM encoder."""
    # Warn if in web environment (browser-based Claude Code)
    try:
        from klareco.environment import warn_if_web_training
        warn_if_web_training()  # Will prompt user or exit if in web
    except ImportError:
        # environment module not available, continue
        pass

    parser = argparse.ArgumentParser(description='Train Tree-LSTM with contrastive learning')
    parser.add_argument('--training-data', type=str, default='data/training_pairs',
                        help='Training data directory')
    parser.add_argument('--output', type=str, default='models/tree_lstm',
                        help='Output directory for model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path or "auto" for latest)')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='LSTM hidden dimension')
    parser.add_argument('--output-dim', type=int, default=512,
                        help='Output embedding dimension')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Contrastive loss margin')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum training pairs to use')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device)

    logger.info("="*70)
    logger.info("TREE-LSTM TRAINING - PHASE 3")
    logger.info("="*70)
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model config:")
    logger.info(f"  Vocab size: {args.vocab_size}")
    logger.info(f"  Embed dim: {args.embed_dim}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Output dim: {args.output_dim}")
    logger.info(f"Training config:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Margin: {args.margin}")
    logger.info(f"  Device: {device}")
    logger.info("")

    try:
        # Load training data
        data_dir = Path(args.training_data)
        positive_file = data_dir / 'positive_pairs.jsonl'
        negative_file = data_dir / 'negative_pairs.jsonl'

        if not positive_file.exists() or not negative_file.exists():
            raise FileNotFoundError(f"Training data not found in {data_dir}")

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader = create_dataloader(
            positive_pairs_file=positive_file,
            negative_pairs_file=negative_file,
            batch_size=args.batch_size,
            max_pairs=args.max_pairs,
            shuffle=True
        )
        logger.info(f"  Training pairs: {len(train_dataloader.dataset)}")
        logger.info("")

        # Create model
        logger.info("Creating Tree-LSTM model...")
        model = TreeLSTMEncoder(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {num_params:,}")
        logger.info("")

        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = ContrastiveLoss(margin=args.margin)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle checkpoint resumption
        start_epoch = 1
        training_history = []

        if args.resume:
            if args.resume == 'auto':
                # Find latest checkpoint
                checkpoint_path = find_latest_checkpoint(output_dir)
                if checkpoint_path:
                    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, logger)
                    # Load training history if exists
                    history_file = output_dir / 'training_history.json'
                    if history_file.exists():
                        with open(history_file, 'r') as f:
                            training_history = json.load(f)
                        logger.info(f"  Loaded training history ({len(training_history)} epochs)")
                else:
                    logger.info("No checkpoint found, starting from scratch")
            else:
                # Load specific checkpoint
                checkpoint_path = Path(args.resume)
                if checkpoint_path.exists():
                    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, logger)
                    # Load training history if exists
                    history_file = output_dir / 'training_history.json'
                    if history_file.exists():
                        with open(history_file, 'r') as f:
                            training_history = json.load(f)
                        logger.info(f"  Loaded training history ({len(training_history)} epochs)")
                else:
                    logger.error(f"Checkpoint not found: {checkpoint_path}")
                    return 1

        # Training loop
        logger.info("Starting training...")
        logger.info(f"  Epochs: {start_epoch} to {args.epochs}")
        logger.info("")

        best_val_loss = float('inf')

        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f"Epoch {epoch}/{args.epochs}")
            logger.info("-" * 70)

            # Train
            train_loss, train_acc = train_epoch(
                model, train_dataloader, optimizer, criterion, device, logger
            )
            logger.info(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

            # Save checkpoint every epoch
            save_checkpoint(
                model, optimizer, epoch, train_loss, 0.0, output_dir, logger
            )

            # Track history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
            })

            logger.info("")

        # Save final model
        final_model_file = output_dir / 'final_model.pt'
        torch.save(model.state_dict(), final_model_file)
        logger.info(f"Saved final model: {final_model_file}")

        # Save training history
        history_file = output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Saved training history: {history_file}")

        # Summary
        logger.info("")
        logger.info("="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Final train loss: {train_loss:.4f}")
        logger.info(f"Final train acc: {train_acc:.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("")
        logger.info("✅ Tree-LSTM training complete!")

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
