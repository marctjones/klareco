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

# Import M1 model and CompositionalEmbedding
sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.models.m1_selectional import M1SelectionalPreference, M1Loss
from klareco.embeddings.compositional import CompositionalEmbedding

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
    """
    Memory-efficient streaming dataset for M1 selectional preference training.

    Uses CompositionalEmbedding to encode full word structures on-the-fly.
    This leverages morphological information (suffixes, endings) for better
    semantic plausibility learning.

    Instead of loading all examples into RAM, stores only file offsets
    and reads lines on-demand. This reduces memory usage.
    """

    def __init__(self, data_path: Path, compositional_emb: CompositionalEmbedding):
        """
        Initialize dataset.

        Args:
            data_path: Path to train/val/test.jsonl with word structures
            compositional_emb: CompositionalEmbedding for encoding words
        """
        self.data_path = data_path
        self.comp_emb = compositional_emb

        # Build line offset index (minimal memory: ~8 bytes per example)
        self.offsets = []
        logger.info(f"Building offset index for {data_path.name}...")
        with open(data_path, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

        logger.info(f"  Indexed {len(self.offsets):,} examples (offsets: {len(self.offsets)*8/1024/1024:.1f}MB)")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        # Read single line on-demand (memory efficient)
        with open(self.data_path, 'r') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            example = json.loads(line)

        # Extract word structures (already case-normalized in data)
        subj = example['subject']
        verb = example['verb']
        obj = example['object']
        label = float(example['label'])

        # Encode with CompositionalEmbedding (on-the-fly)
        with torch.no_grad():
            subj_emb = self.comp_emb.encode_word(
                root=subj['root'],
                prefixes=subj['prefixes'],
                suffixes=subj['suffixes'],
                ending=subj['ending']
            )
            verb_emb = self.comp_emb.encode_word(
                root=verb['root'],
                prefixes=verb['prefixes'],
                suffixes=verb['suffixes'],
                ending=verb['ending']
            )
            obj_emb = self.comp_emb.encode_word(
                root=obj['root'],
                prefixes=obj['prefixes'],
                suffixes=obj['suffixes'],
                ending=obj['ending']
            )

        return {
            'subject_emb': subj_emb,
            'verb_emb': verb_emb,
            'object_emb': obj_emb,
            'label': torch.tensor([label], dtype=torch.float32)
        }


# =============================================================================
# Training Functions
# =============================================================================

def load_compositional_embedding(comp_model_path: Path) -> CompositionalEmbedding:
    """Load pre-trained CompositionalEmbedding."""
    logger.info(f"Loading CompositionalEmbedding from {comp_model_path}")

    checkpoint = torch.load(comp_model_path, map_location='cpu', weights_only=False)

    # Check if this is a new-format CompositionalEmbedding or old Stage 1 checkpoint
    if 'root_vocab' in checkpoint:
        # New format - direct load
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint['embed_dim'],
            composition_method=checkpoint.get('composition_method', 'sum'),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old Stage 1 format - build vocabularies from root_to_idx
        logger.info("  Converting Stage 1 checkpoint to CompositionalEmbedding format...")

        root_vocab = checkpoint['root_to_idx']
        embed_dim = checkpoint['embedding_dim']

        # Build standard Esperanto affix vocabularies
        prefix_vocab = {
            '<NONE>': 0,
            'mal': 1,   # opposite
            're': 2,    # again/back
            'dis': 3,   # apart/asunder
            'ge': 4,    # both sexes
            'pra': 5,   # ancient/primeval
            'bo': 6,    # in-law
            'ek': 7,    # sudden action
        }

        suffix_vocab = {
            '<NONE>': 0,
            'aĉ': 1,    # pejorative
            'ad': 2,    # continuous action
            'aĵ': 3,    # thing/concrete
            'an': 4,    # member
            'ar': 5,    # collection
            'ebl': 6,   # possible
            'ec': 7,    # quality
            'eg': 8,    # augmentative
            'ej': 9,    # place
            'em': 10,   # tendency
            'end': 11,  # must/should
            'er': 12,   # fragment
            'estr': 13, # leader
            'et': 14,   # diminutive
            'id': 15,   # offspring
            'ig': 16,   # cause to be
            'iĝ': 17,   # become
            'il': 18,   # tool
            'in': 19,   # feminine
            'ind': 20,  # worthy
            'ing': 21,  # holder
            'ism': 22,  # doctrine
            'ist': 23,  # professional
            'obl': 24,  # multiple
            'on': 25,   # fraction
            'op': 26,   # collective
            'uj': 27,   # container
            'ul': 28,   # person characterized by
            'um': 29,   # indefinite relation
        }

        # Create CompositionalEmbedding with standard vocabularies
        comp_emb = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=embed_dim,
            composition_method='sum',
        )

        # Load only the root embeddings from Stage 1 checkpoint
        # Other embeddings (prefix, suffix, ending) will be randomly initialized
        state_dict = comp_emb.state_dict()
        state_dict['root_embed.weight'] = checkpoint['model_state_dict']['embeddings.weight']
        comp_emb.load_state_dict(state_dict)

        logger.info(f"  Initialized from Stage 1: {len(root_vocab):,} roots")
        logger.info(f"  Added affixes: {len(prefix_vocab)} prefixes, {len(suffix_vocab)} suffixes")

    comp_emb.eval()  # Freeze for M1 training

    logger.info(f"Loaded CompositionalEmbedding: {comp_emb.embed_dim}D, {len(comp_emb.root_vocab):,} roots")
    return comp_emb


def create_dataloaders(train_path: Path, val_path: Path, test_path: Path,
                       comp_emb: CompositionalEmbedding,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    train_dataset = M1Dataset(train_path, comp_emb)
    val_dataset = M1Dataset(val_path, comp_emb)
    test_dataset = M1Dataset(test_path, comp_emb)

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

    # Progress tracking
    total_batches = len(dataloader)
    log_interval = max(1, total_batches // 10)  # Log ~10 times per epoch

    for batch_idx, batch in enumerate(dataloader):
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

        # Gradient clipping to prevent extreme weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += losses['loss'].item()
        total_subj_verb_loss += losses['subj_verb_loss'].item()
        total_verb_obj_loss += losses['verb_obj_loss'].item()
        total_triple_loss += losses['triple_loss'].item()
        num_batches += 1

        # Log progress periodically
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            avg_loss = total_loss / num_batches
            pct = 100.0 * (batch_idx + 1) / total_batches
            logger.info(
                f"  Batch {batch_idx + 1}/{total_batches} ({pct:.1f}%) - "
                f"loss={avg_loss:.4f}"
            )

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
    all_scores = []  # Track score distribution

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

            # Collect scores for distribution analysis
            all_scores.extend(outputs['triple_score'].squeeze().cpu().tolist())

    num_batches = len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    # Compute score statistics
    score_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 0.0
    score_std = (sum((s - score_mean) ** 2 for s in all_scores) / len(all_scores)) ** 0.5 if all_scores else 0.0

    return {
        'loss': total_loss / num_batches,
        'subj_verb_loss': total_subj_verb_loss / num_batches,
        'verb_obj_loss': total_verb_obj_loss / num_batches,
        'triple_loss': total_triple_loss / num_batches,
        'accuracy': accuracy,
        'score_mean': score_mean,
        'score_min': score_min,
        'score_max': score_max,
        'score_std': score_std
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    best_accuracy: float, output_dir: Path, comp_model_path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'comp_model_path': comp_model_path  # Store path for inference
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
    parser = argparse.ArgumentParser(description='Train M1 Selectional Preference Model v2 (Compositional)')

    # Paths
    parser.add_argument('--comp-model', type=str,
                        default='models/root_embeddings_tier0/best_model.pt',
                        help='Path to CompositionalEmbedding model')
    parser.add_argument('--train-data', type=str,
                        default='data/training/m1_compositional/train.jsonl',
                        help='Path to training data')
    parser.add_argument('--val-data', type=str,
                        default='data/training/m1_compositional/val.jsonl',
                        help='Path to validation data')
    parser.add_argument('--test-data', type=str,
                        default='data/training/m1_compositional/test.jsonl',
                        help='Path to test data')
    parser.add_argument('--output-dir', type=str,
                        default='models/m1_compositional',
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
    logger.info("M1 Selectional Preference Training v2 (Compositional)")
    logger.info("=" * 60)
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    # Load CompositionalEmbedding
    comp_model_path = Path(args.comp_model)
    if not comp_model_path.exists():
        logger.error(f"CompositionalEmbedding not found: {comp_model_path}")
        logger.error("Train compositional embeddings first: ./scripts/train_roots.sh")
        return 1

    comp_emb = load_compositional_embedding(comp_model_path)
    embedding_dim = comp_emb.embed_dim

    # Load training data
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    test_path = Path(args.test_data)

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Generate M1 data first: python scripts/prepare_m1_training_data_tier_priority.py")
        return 1

    train_loader, val_loader, test_loader = create_dataloaders(
        train_path, val_path, test_path,
        comp_emb,
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
        # Log epoch start
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Training...")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device)

        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"scores=[{val_metrics['score_min']:.3f}, {val_metrics['score_max']:.3f}], "
            f"mean={val_metrics['score_mean']:.3f}, std={val_metrics['score_std']:.3f}"
        )

        # Check for model collapse (scores stuck near 0 or 1)
        if val_metrics['score_std'] < 0.05:
            logger.warning(
                f"⚠️  Low score variance (std={val_metrics['score_std']:.4f}) - "
                f"model may be collapsing!"
            )

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, best_accuracy, output_dir, str(comp_model_path))
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
