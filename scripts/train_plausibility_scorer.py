#!/usr/bin/env python3
"""
Train Plausibility Scorer for Semantic Fact Validator

VERSION: v2.1
COMPATIBLE WITH: v2.1 plausibility training dataset, hybrid root embeddings
DEPENDENCIES: klareco.embeddings.hybrid, torch
STAGE: Training

Description:
    Train simple concatenation-based plausibility scorer.

    Architecture:
    - Frozen hybrid root embeddings (128D each)
    - Concatenate subject + verb + object embeddings (384D)
    - 3-layer MLP: 384 → 256 → 128 → 1
    - Sigmoid output for plausibility score
    - Total: ~98K trainable parameters

Pipeline Position:
    Quality Dataset (200K) → [THIS SCRIPT] → Trained Scorer → Task #4 Integration

Usage:
    # Train with default settings
    python scripts/train_plausibility_scorer.py \
        --train-data data/plausibility_training_quality/train.jsonl \
        --val-data data/plausibility_training_quality/val.jsonl \
        --output-dir models/plausibility_scorer

    # Resume training
    python scripts/train_plausibility_scorer.py \
        --train-data data/plausibility_training_quality/train.jsonl \
        --val-data data/plausibility_training_quality/val.jsonl \
        --output-dir models/plausibility_scorer \
        --resume

    # With custom hyperparameters
    python scripts/train_plausibility_scorer.py \
        --train-data data/plausibility_training_quality/train.jsonl \
        --val-data data/plausibility_training_quality/val.jsonl \
        --output-dir models/plausibility_scorer \
        --batch-size 256 \
        --learning-rate 0.001 \
        --epochs 50

Inputs:
    - Training JSONL (from generate_plausibility_training_data_quality.py)
    - Validation JSONL

Outputs:
    - models/plausibility_scorer/model_best.pt - Best model checkpoint
    - models/plausibility_scorer/model_final.pt - Final model checkpoint
    - models/plausibility_scorer/training_log.json - Training history
    - models/plausibility_scorer/config.json - Model configuration

Quality Checks:
    - Validation accuracy >85%
    - Balanced precision/recall for both classes
    - F1 score >82%
    - No overfitting (train/val loss gap <0.1)

Last Updated: 2026-03-22
Author: Claude Code
Related Issues: #699
See Also: /tmp/plausibility_scorer_design.md, Task #1
"""

import argparse
import json
import jsonlines
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import time

from klareco.embeddings import load_hybrid_embedder


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class PlausibilityScorer(nn.Module):
    """
    Simple concatenation-based plausibility scorer.

    Architecture:
    - Frozen hybrid embeddings (128D per root)
    - Concatenate (subject, verb, object) → 384D
    - MLP: 384 → 256 → 128 → 1
    - Sigmoid output

    Parameters: ~98K
    """

    def __init__(self, embedder):
        super().__init__()

        # Frozen embeddings
        self.embedder = embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

        # MLP scorer
        self.scorer = nn.Sequential(
            nn.Linear(128 * 3, 256),  # 384D → 256D
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),      # 256D → 128D
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),        # 128D → 1
            nn.Sigmoid()
        )

    def forward(self, subject_roots, verb_roots, object_roots):
        """
        Args:
            subject_roots: List of subject root strings
            verb_roots: List of verb root strings
            object_roots: List of object root strings

        Returns:
            plausibility: Tensor of shape (batch_size,) with scores [0, 1]
        """
        batch_size = len(subject_roots)

        # Get embeddings (frozen)
        subj_embs = []
        verb_embs = []
        obj_embs = []

        for subj, verb, obj in zip(subject_roots, verb_roots, object_roots):
            subj_emb = self.embedder.get_embedding(subj)
            verb_emb = self.embedder.get_embedding(verb)
            obj_emb = self.embedder.get_embedding(obj)

            # Handle unknown roots (not in embedder vocabulary)
            if subj_emb is None or verb_emb is None or obj_emb is None:
                # Use zero embedding for unknown roots
                device = next(self.parameters()).device
                zero_emb = torch.zeros(128, device=device)
                if subj_emb is None:
                    subj_emb = zero_emb
                if verb_emb is None:
                    verb_emb = zero_emb
                if obj_emb is None:
                    obj_emb = zero_emb

            subj_embs.append(subj_emb)
            verb_embs.append(verb_emb)
            obj_embs.append(obj_emb)

        # Stack to tensors
        subj_embs = torch.stack(subj_embs)  # (batch, 128)
        verb_embs = torch.stack(verb_embs)  # (batch, 128)
        obj_embs = torch.stack(obj_embs)    # (batch, 128)

        # Concatenate
        combined = torch.cat([subj_embs, verb_embs, obj_embs], dim=-1)  # (batch, 384)

        # Score
        plausibility = self.scorer(combined).squeeze(-1)  # (batch,)

        return plausibility

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATASET
# ============================================================================

class PlausibilityDataset(Dataset):
    """Dataset for plausibility training."""

    def __init__(self, jsonl_path: Path):
        self.examples = []

        with jsonlines.open(jsonl_path) as reader:
            for example in reader:
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'subject_root': example['subject_root'],
            'verb_root': example['verb_root'],
            'object_root': example['object_root'],
            'plausible': float(example['plausible'])
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    subject_roots = [item['subject_root'] for item in batch]
    verb_roots = [item['verb_root'] for item in batch]
    object_roots = [item['object_root'] for item in batch]
    plausible = torch.tensor([item['plausible'] for item in batch])

    return subject_roots, verb_roots, object_roots, plausible


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for subject_roots, verb_roots, object_roots, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)

        # Forward pass
        outputs = model(subject_roots, verb_roots, object_roots)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * len(labels)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # For precision/recall
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for subject_roots, verb_roots, object_roots, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)

            # Forward pass
            outputs = model(subject_roots, verb_roots, object_roots)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * len(labels)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            # Precision/recall stats
            for pred, label in zip(predictions, labels):
                if pred == 1 and label == 1:
                    true_positives += 1
                elif pred == 1 and label == 0:
                    false_positives += 1
                elif pred == 0 and label == 0:
                    true_negatives += 1
                elif pred == 0 and label == 1:
                    false_negatives += 1

    avg_loss = total_loss / total
    accuracy = correct / total

    # Precision/Recall/F1
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train plausibility scorer')
    parser.add_argument('--train-data', type=Path, required=True,
                        help='Path to training JSONL')
    parser.add_argument('--val-data', type=Path, required=True,
                        help='Path to validation JSONL')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load embedder
    logging.info("Loading hybrid root embedder...")
    embedder = load_hybrid_embedder()
    logging.info(f"Embedder vocabulary: {len(embedder.root_to_idx)} roots")

    # Create model
    logging.info("Creating plausibility scorer...")
    model = PlausibilityScorer(embedder).to(args.device)
    trainable_params = model.count_parameters()
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Load datasets
    logging.info(f"Loading training data from {args.train_data}")
    train_dataset = PlausibilityDataset(args.train_data)
    logging.info(f"Training examples: {len(train_dataset):,}")

    logging.info(f"Loading validation data from {args.val_data}")
    val_dataset = PlausibilityDataset(args.val_data)
    logging.info(f"Validation examples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Must be 0 for custom collate with embeddings
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_f1 = 0
    patience_counter = 0
    training_history = []

    if args.resume:
        checkpoint_path = args.output_dir / 'model_best.pt'
        if checkpoint_path.exists():
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            logging.info(f"Resumed from epoch {checkpoint['epoch']}, best val F1: {best_val_f1:.4f}")

    # Training loop
    logging.info("\n" + "="*60)
    logging.info("TRAINING START")
    logging.info("="*60)

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, args.device)

        # Log
        logging.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        logging.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        logging.info(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1']
        })

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'config': {
                    'embed_dim': 128,
                    'hidden_dims': [256, 128],
                    'dropout': 0.2,
                    'trainable_params': trainable_params
                }
            }

            torch.save(checkpoint, args.output_dir / 'model_best.pt')
            logging.info(f"✓ Saved best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            logging.info(f"\nEarly stopping triggered (patience={args.patience})")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_f1': val_metrics['f1'],
        'val_metrics': val_metrics
    }, args.output_dir / 'model_final.pt')

    # Save training history
    with open(args.output_dir / 'training_log.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save config
    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump({
            'embed_dim': 128,
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'trainable_params': trainable_params,
            'training_examples': len(train_dataset),
            'validation_examples': len(val_dataset),
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs_trained': epoch + 1,
            'best_val_f1': best_val_f1
        }, f, indent=2)

    # Final summary
    logging.info("\n" + "="*60)
    logging.info("TRAINING COMPLETE")
    logging.info("="*60)
    logging.info(f"Best validation F1: {best_val_f1:.4f}")
    logging.info(f"Final validation metrics:")
    logging.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logging.info(f"  Precision: {val_metrics['precision']:.4f}")
    logging.info(f"  Recall: {val_metrics['recall']:.4f}")
    logging.info(f"  F1 Score: {val_metrics['f1']:.4f}")
    logging.info(f"\nModel saved to: {args.output_dir}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
