#!/usr/bin/env python3
"""
Train M1 Selectional Preference Model

Learns compatibility between roots in grammatical roles:
- Subject-verb compatibility
- Verb-object compatibility
- Triple plausibility (subject, verb, object)

Uses Stage 1 root embeddings as input features.

Training data: data/training/m1_selectional_hard/
Output: models/m1_selectional/

Usage:
    python scripts/train_m1_selectional.py --fresh
    python scripts/train_m1_selectional.py --resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import M1 model
sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.models.m1_selectional import M1SelectionalPreference, M1Loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


# =============================================================================
# Dataset
# =============================================================================

class M1Dataset(Dataset):
    """Dataset for M1 selectional preference training."""

    def __init__(self, data_path: Path, root_embeddings: torch.Tensor,
                 root_to_idx: Dict[str, int]):
        """
        Initialize dataset.

        Args:
            data_path: Path to train/val/test.jsonl
            root_embeddings: Stage 1 embeddings [vocab_size, embedding_dim]
            root_to_idx: Root to index mapping
        """
        self.root_embeddings = root_embeddings
        self.root_to_idx = root_to_idx
        self.examples = []

        # Load examples
        with open(data_path) as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Get embeddings for each role
        subj = example['subject_root'].lower()
        verb = example['verb_root'].lower()
        obj = example['object_root'].lower()
        label = float(example['label'])

        # Lookup embeddings (use zero vector if not found)
        subj_idx = self.root_to_idx.get(subj, 0)
        verb_idx = self.root_to_idx.get(verb, 0)
        obj_idx = self.root_to_idx.get(obj, 0)

        subj_emb = self.root_embeddings[subj_idx]
        verb_emb = self.root_embeddings[verb_idx]
        obj_emb = self.root_embeddings[obj_idx]

        return {
            'subject_emb': subj_emb,
            'verb_emb': verb_emb,
            'object_emb': obj_emb,
            'label': torch.tensor([label], dtype=torch.float32)
        }


# =============================================================================
# Training Functions
# =============================================================================

def load_root_embeddings(model_path: Path) -> Tuple[torch.Tensor, Dict, Dict]:
    """Load Stage 1 root embeddings."""
    logger.info(f"Loading Stage 1 embeddings from {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    embeddings = checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = checkpoint['idx_to_root']

    logger.info(f"Loaded {len(root_to_idx):,} root embeddings (dim={embeddings.shape[1]})")

    return embeddings, root_to_idx, idx_to_root


def create_dataloaders(train_path: Path, val_path: Path, test_path: Path,
                       root_embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    train_dataset = M1Dataset(train_path, root_embeddings, root_to_idx)
    val_dataset = M1Dataset(val_path, root_embeddings, root_to_idx)
    test_dataset = M1Dataset(test_path, root_embeddings, root_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train examples: {len(train_dataset):,}")
    logger.info(f"Val examples: {len(val_dataset):,}")
    logger.info(f"Test examples: {len(test_dataset):,}")

    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: M1Loss,
                optimizer: torch.optim.Optimizer, device: str) -> Dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_subj_verb_loss = 0.0
    total_verb_obj_loss = 0.0
    total_triple_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        subject_emb = batch['subject_emb'].to(device)
        verb_emb = batch['verb_emb'].to(device)
        object_emb = batch['object_emb'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(subject_emb, verb_emb, object_emb)

        # Compute loss
        losses = criterion(outputs, labels)

        # Backward pass
        losses['loss'].backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses['loss'].item()
        total_subj_verb_loss += losses['subj_verb_loss'].item()
        total_verb_obj_loss += losses['verb_obj_loss'].item()
        total_triple_loss += losses['triple_loss'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'subj_verb_loss': total_subj_verb_loss / num_batches,
        'verb_obj_loss': total_verb_obj_loss / num_batches,
        'triple_loss': total_triple_loss / num_batches
    }


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: M1Loss,
             device: str) -> Dict:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_subj_verb_loss = 0.0
    total_verb_obj_loss = 0.0
    total_triple_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            subject_emb = batch['subject_emb'].to(device)
            verb_emb = batch['verb_emb'].to(device)
            object_emb = batch['object_emb'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(subject_emb, verb_emb, object_emb)

            # Compute loss
            losses = criterion(outputs, labels)

            # Accumulate losses
            total_loss += losses['loss'].item()
            total_subj_verb_loss += losses['subj_verb_loss'].item()
            total_verb_obj_loss += losses['verb_obj_loss'].item()
            total_triple_loss += losses['triple_loss'].item()

            # Accuracy (threshold at 0.5)
            predictions = (outputs['triple_score'] > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    num_batches = len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    return {
        'loss': total_loss / num_batches,
        'subj_verb_loss': total_subj_verb_loss / num_batches,
        'verb_obj_loss': total_verb_obj_loss / num_batches,
        'triple_loss': total_triple_loss / num_batches,
        'accuracy': accuracy
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    best_accuracy: float, output_dir: Path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim
    }

    # Atomic save
    temp_path = output_dir / 'best_model.pt.tmp'
    try:
        torch.save(checkpoint, temp_path)
        temp_path.rename(output_dir / 'best_model.pt')
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    checkpoint_path: Path) -> Tuple[int, float]:
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']

    logger.info(f"Resumed from epoch {epoch}, best accuracy: {best_accuracy:.4f}")

    return epoch, best_accuracy


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train M1 Selectional Preference Model')

    # Paths
    parser.add_argument('--stage1-model', type=str,
                        default='models/root_embeddings/best_model.pt',
                        help='Path to Stage 1 root embeddings')
    parser.add_argument('--data-dir', type=str,
                        default='data/training/m1_selectional_hard',
                        help='Directory with train/val/test.jsonl')
    parser.add_argument('--output-dir', type=str,
                        default='models/m1_selectional',
                        help='Output directory for model checkpoints')
    parser.add_argument('--log-dir', type=str,
                        default='logs/training',
                        help='Directory for training logs')

    # Model hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Control flags
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore checkpoints)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to train on (cpu or cuda)')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"m1_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_file_logging(log_file)

    logger.info("=" * 60)
    logger.info("M1 Selectional Preference Training")
    logger.info("=" * 60)
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    # Load Stage 1 root embeddings
    stage1_path = Path(args.stage1_model)
    if not stage1_path.exists():
        logger.error(f"Stage 1 model not found: {stage1_path}")
        logger.error("Train Stage 1 first: ./scripts/train_roots.sh")
        return 1

    root_embeddings, root_to_idx, idx_to_root = load_root_embeddings(stage1_path)
    embedding_dim = root_embeddings.shape[1]

    # Load training data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'val.jsonl'
    test_path = data_dir / 'test.jsonl'

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Generate M1 data first: python scripts/prepare_m1_training_data_hard_negatives.py")
        return 1

    train_loader, val_loader, test_loader = create_dataloaders(
        train_path, val_path, test_path,
        root_embeddings, root_to_idx,
        batch_size=args.batch_size
    )

    # Initialize model
    model = M1SelectionalPreference(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model = model.to(args.device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Initialize training
    criterion = M1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    best_accuracy = 0.0
    patience_counter = 0

    # Resume from checkpoint if requested
    checkpoint_path = output_dir / 'best_model.pt'
    if args.resume and checkpoint_path.exists():
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1  # Continue from next epoch
    elif args.fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - ignoring checkpoint")
    elif checkpoint_path.exists():
        logger.info(f"Checkpoint found: {checkpoint_path}")
        logger.info("Use --resume to continue or --fresh to start over")
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1

    # Training loop
    logger.info("")
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device)

        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, best_accuracy, output_dir)
            logger.info(f"Saved new best model (accuracy: {best_accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break

    # Final evaluation on test set
    logger.info("")
    logger.info("Training complete!")
    logger.info(f"Loading best model (accuracy: {best_accuracy:.4f})")

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, args.device)

    logger.info("")
    logger.info("=== Test Set Results ===")
    logger.info(f"Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Subject-verb loss: {test_metrics['subj_verb_loss']:.4f}")
    logger.info(f"Verb-object loss: {test_metrics['verb_obj_loss']:.4f}")
    logger.info(f"Triple loss: {test_metrics['triple_loss']:.4f}")

    logger.info("")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Log saved to: {log_file}")

    return 0


if __name__ == '__main__':
    exit(main())
