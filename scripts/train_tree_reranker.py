#!/usr/bin/env python3
"""
Train TreeMatchReranker

VERSION: v2.1
COMPATIBLE WITH: v2.1 database, compositional embeddings v3
DEPENDENCIES: Compositional embeddings (frozen), AST parser
STAGE: Training

Description:
    Trains TreeMatchReranker with multi-level matching:
    - 70% Syntax matching (deterministic, 0 params)
    - 20% Compositional matching (hybrid)
    - 10% Semantic matching (learned, ~20K params)

Pipeline Position:
    Training Data → [THIS SCRIPT] → Trained TreeMatchReranker → Evaluate

Usage:
    python scripts/train_tree_reranker.py \\
        --data data/training/tree_reranker_train.jsonl \\
        --output models/tree_reranker \\
        --epochs 20 \\
        --batch-size 64

Inputs:
    - Training data: JSONL with (query, doc, label, query_ast, doc_ast)

Outputs:
    - Trained model: models/tree_reranker/best_model.pt
    - Training log: models/tree_reranker/training.log
    - Checkpoints: models/tree_reranker/checkpoint_*.pt

Quality Checks:
    - Validation accuracy > 80%
    - Syntax weight stays ≥ 0.5 (favor deterministic)
    - No overfitting (train/val gap < 5%)

Training Time:
    - 2K examples: ~5 minutes on CPU
    - 5K examples: ~10 minutes on CPU
    - 10K examples: ~20 minutes on CPU

Last Updated: 2026-03-26
Author: Claude + Marc
Related Issues: #704
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.tree_match_reranker import TreeMatchReranker, count_parameters
from klareco.embeddings import CompositionalEmbedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RerankerDataset(Dataset):
    """Dataset for TreeMatchReranker training."""

    def __init__(self, data_path: Path):
        """Load training data from JSONL file."""
        self.examples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'query_ast': example['query_ast'],
            'doc_ast': example['doc_ast'],
            'label': torch.tensor(example['label'], dtype=torch.float32)
        }


def collate_fn(batch):
    """Custom collate function for batch loading."""
    query_asts = [item['query_ast'] for item in batch]
    doc_asts = [item['doc_ast'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    return query_asts, doc_asts, labels


class TreeRerankerTrainer:
    """Trainer for TreeMatchReranker."""

    def __init__(
        self,
        model: TreeMatchReranker,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        lr: float = 1e-3,
        weight_decay: float = 0.01
    ):
        """
        Initialize trainer.

        Args:
            model: TreeMatchReranker model
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Output directory for checkpoints
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=20,  # Will be updated
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        logger.info("TreeRerankerTrainer initialized")
        trainable, total = count_parameters(model)
        logger.info(f"  Trainable params: {trainable:,}")
        logger.info(f"  Total params: {total:,}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for query_asts, doc_asts, labels in pbar:
            # Forward pass
            batch_loss = 0.0
            batch_correct = 0

            for i in range(len(query_asts)):
                score, breakdown = self.model(query_asts[i], doc_asts[i])
                label = labels[i]

                # Compute loss
                loss = self.criterion(score.unsqueeze(0), label.unsqueeze(0))
                batch_loss += loss

                # Accuracy
                pred = (score > 0.5).float()
                batch_correct += (pred == label).float().item()

            # Average over batch
            batch_loss = batch_loss / len(query_asts)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += batch_loss.item()
            correct += batch_correct
            total += len(query_asts)

            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'acc': f'{correct/total:.3f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for query_asts, doc_asts, labels in pbar:
                batch_loss = 0.0
                batch_correct = 0

                for i in range(len(query_asts)):
                    score, breakdown = self.model(query_asts[i], doc_asts[i])
                    label = labels[i]

                    # Compute loss
                    loss = self.criterion(score.unsqueeze(0), label.unsqueeze(0))
                    batch_loss += loss

                    # Accuracy
                    pred = (score > 0.5).float()
                    batch_correct += (pred == label).float().item()

                # Average over batch
                batch_loss = batch_loss / len(query_asts)

                total_loss += batch_loss.item()
                correct += batch_correct
                total += len(query_asts)

                pbar.set_postfix({
                    'loss': batch_loss.item(),
                    'acc': f'{correct/total:.3f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def train(self, num_epochs: int, patience: int = 3):
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
        """
        logger.info(f"\n{'='*60}")
        logger.info("Starting Training")
        logger.info(f"{'='*60}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Early stopping patience: {patience}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            # Validate
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Log mixing weights
            weights = self.model._normalize_weights()
            logger.info(f"Mixing weights: Syntax={weights['syntax']:.3f}, "
                       f"Comp={weights['compositional']:.3f}, "
                       f"Semantic={weights['semantic']:.3f}")

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info("✓ New best model!")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{patience}")

                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info(f"\n{'='*60}")
        logger.info("Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save checkpoint atomically."""
        checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch+1}.pt.tmp'
        final_path = self.output_dir / f'checkpoint_epoch{epoch+1}.pt'

        # Save to temp file
        self.model.save(checkpoint_path)

        # Atomic rename
        checkpoint_path.rename(final_path)

        # If this is the best model, save as best_model.pt
        if val_loss <= self.best_val_loss:
            best_path = self.output_dir / 'best_model.pt'
            prev_best = self.output_dir / 'best_model.prev.pt'

            # Rotate: best → prev, new → best
            if best_path.exists():
                best_path.rename(prev_best)
            final_path.rename(best_path)

        # Keep only last 3 checkpoints
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch*.pt'))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()


def load_compositional_embeddings(checkpoint_path: Path):
    """Load compositional embeddings from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        # Full CompositionalEmbedding checkpoint
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Simple root embeddings with model_state_dict
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 128)
        )
        # Load state dict (contains root_embeddings.weight)
        comp_emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")

    return comp_emb


def main():
    parser = argparse.ArgumentParser(description="Train TreeMatchReranker")
    parser.add_argument('--data', type=Path, required=True,
                       help='Training data JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--comp-emb', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to compositional embedding model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    logger.info("Loading compositional embeddings...")
    comp_emb = load_compositional_embeddings(args.comp_emb)
    logger.info("✓ Compositional embeddings loaded")

    logger.info(f"Loading training data from {args.data}...")
    full_dataset = RerankerDataset(args.data)

    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # CPU training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    logger.info("Initializing TreeMatchReranker...")
    model = TreeMatchReranker(
        compositional_embedding=comp_emb,
        freeze_embedding=True,
        root_dim=64,
        hidden_dim=32,
        num_heads=2
    )
    logger.info("✓ Model initialized")

    # Create trainer
    trainer = TreeRerankerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output,
        lr=args.lr
    )

    # Train
    trainer.train(num_epochs=args.epochs, patience=args.patience)

    logger.info(f"\nModel saved to {args.output / 'best_model.pt'}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
