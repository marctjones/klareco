#!/usr/bin/env python3
"""
Train Hybrid Plausibility Scorer (Word-Level v2.0)

VERSION: v2.0
COMPATIBLE WITH: Word-level training dataset, HybridWordEncoder
DEPENDENCIES: klareco.embeddings.hybrid_word, torch
STAGE: Training

Description:
    Train hybrid plausibility scorer using word-level representations.

    Architecture:
    - HybridWordEncoder (172D per word):
      * 128D learned root embeddings (frozen)
      * 20D deterministic affix features
      * 24D deterministic lexicon features
    - Concatenate subject + verb + object (516D)
    - 3-layer MLP: 516 → 256 → 128 → 1
    - Sigmoid output for plausibility score
    - Total: ~131K trainable parameters (MLP only, embeddings frozen)

Pipeline Position:
    Word-Level Dataset → [THIS SCRIPT] → Trained Hybrid Scorer → Evaluation

Usage:
    # Train with 10K dataset (quick validation)
    python scripts/train_plausibility_scorer_word_level.py \
        --train-data data/plausibility_training_word_level_10k/train.jsonl \
        --val-data data/plausibility_training_word_level_10k/val.jsonl \
        --output-dir models/plausibility_word_level_10k

    # Train with 100K dataset (full training)
    python scripts/train_plausibility_scorer_word_level.py \
        --train-data data/plausibility_training_word_level/train.jsonl \
        --val-data data/plausibility_training_word_level/val.jsonl \
        --output-dir models/plausibility_word_level \
        --batch-size 256 \
        --epochs 30

Inputs:
    - Training JSONL (word-level format with decomposition)
    - Validation JSONL

Outputs:
    - models/.../model_best.pt - Best model checkpoint
    - models/.../training_log.json - Training history
    - models/.../config.json - Model configuration

Quality Checks:
    - Target F1: 85-95% (vs 66% for v1.0 root-level)
    - Balanced precision/recall
    - Zero-shot generalization to unseen word forms

Last Updated: 2026-03-23
Author: Claude Code
Related Issues: #9
See Also: docs/HYBRID_PLAUSIBILITY_V2_PROGRESS.md
"""

import argparse
import json
import jsonlines
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.hybrid import HybridRootEmbedder
from klareco.embeddings.hybrid_word import HybridWordEncoder


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class HybridPlausibilityScorer(nn.Module):
    """
    Hybrid plausibility scorer using word-level representations.

    Architecture:
    - HybridWordEncoder (172D per word, frozen)
    - Concatenate (subject, verb, object) → 516D
    - MLP: 516 → 256 → 128 → 1
    - Sigmoid output

    Parameters: ~131K trainable (MLP only)
    """

    def __init__(self, word_encoder):
        super().__init__()

        # Frozen word encoder
        self.word_encoder = word_encoder
        for param in self.word_encoder.parameters():
            param.requires_grad = False

        # MLP scorer
        input_dim = word_encoder.output_dim * 3  # 172 * 3 = 516
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 256),  # 516D → 256D
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),        # 256D → 128D
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),          # 128D → 1
            nn.Sigmoid()
        )

    def forward(self, subject_words, verb_words, object_words):
        """
        Args:
            subject_words: List of word data dicts
            verb_words: List of word data dicts
            object_words: List of word data dicts

        Returns:
            plausibility: Tensor of shape (batch_size,) with scores [0, 1]
        """
        # Get word embeddings (frozen)
        subj_embs = [self.word_encoder.encode(w) for w in subject_words]
        verb_embs = [self.word_encoder.encode(w) for w in verb_words]
        obj_embs = [self.word_encoder.encode(w) for w in object_words]

        # Stack to tensors
        subj_embs = torch.stack(subj_embs)  # (batch, 172)
        verb_embs = torch.stack(verb_embs)  # (batch, 172)
        obj_embs = torch.stack(obj_embs)    # (batch, 172)

        # Concatenate
        combined = torch.cat([subj_embs, verb_embs, obj_embs], dim=-1)  # (batch, 516)

        # Score
        plausibility = self.scorer(combined).squeeze(-1)  # (batch,)

        return plausibility

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATASET
# ============================================================================

class WordLevelPlausibilityDataset(Dataset):
    """Dataset for word-level plausibility training."""

    def __init__(self, jsonl_path: Path):
        self.examples = []

        with jsonlines.open(jsonl_path) as reader:
            for example in reader:
                # Validate required fields
                if all(key in example for key in ['subject', 'verb', 'object', 'plausible']):
                    self.examples.append(example)
                else:
                    logger.warning(f"Skipping example missing required fields")

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'subject': example['subject'],
            'verb': example['verb'],
            'object': example['object'],
            'plausible': float(example['plausible'])
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    subject_words = [item['subject'] for item in batch]
    verb_words = [item['verb'] for item in batch]
    object_words = [item['object'] for item in batch]
    plausible = torch.tensor([item['plausible'] for item in batch])

    return subject_words, verb_words, object_words, plausible


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for subject_words, verb_words, object_words, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)

        # Forward pass
        outputs = model(subject_words, verb_words, object_words)
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
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # For F1 calculation
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for subject_words, verb_words, object_words, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)

            # Forward pass
            outputs = model(subject_words, verb_words, object_words)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * len(labels)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            # F1 components
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    # Calculate F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, f1, precision, recall


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train hybrid plausibility scorer (word-level)')
    parser.add_argument('--train-data', type=Path, required=True, help='Training JSONL')
    parser.add_argument('--val-data', type=Path, required=True, help='Validation JSONL')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')

    # Model paths
    parser.add_argument('--production-embeddings', type=Path,
                       default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'),
                       help='Production root embeddings')
    parser.add_argument('--ast-embeddings', type=Path,
                       default=Path('models/root_embeddings_fundamento_ast/root_embeddings_best.pt'),
                       help='AST root embeddings')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    # Options
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    file_handler = logging.FileHandler(args.output_dir / 'training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("="*60)
    logger.info("HYBRID PLAUSIBILITY SCORER TRAINING (Word-Level v2.0)")
    logger.info("="*60)

    # Load root embedder
    logger.info("Loading root embedder...")
    if not args.production_embeddings.exists() or not args.ast_embeddings.exists():
        logger.error(f"Root embedding models not found!")
        logger.error(f"  Production: {args.production_embeddings} (exists: {args.production_embeddings.exists()})")
        logger.error(f"  AST: {args.ast_embeddings} (exists: {args.ast_embeddings.exists()})")
        return

    root_embedder = HybridRootEmbedder(
        production_path=str(args.production_embeddings),
        ast_path=str(args.ast_embeddings)
    )
    logger.info("Root embedder loaded")

    # Create hybrid word encoder
    logger.info("Creating hybrid word encoder...")
    word_encoder = HybridWordEncoder(
        root_embedder=root_embedder,
        embed_dim=128,
        use_lexicon=True,
        use_affix_rules=True
    )
    logger.info(f"Word encoder output dimension: {word_encoder.output_dim}D")

    # Create model
    logger.info("Creating hybrid plausibility scorer...")
    model = HybridPlausibilityScorer(word_encoder)
    model = model.to(args.device)

    trainable_params = model.count_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = WordLevelPlausibilityDataset(args.train_data)
    val_dataset = WordLevelPlausibilityDataset(args.val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Val examples: {len(val_dataset)}")

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    training_history = []

    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)

        # Evaluate
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(model, val_loader, criterion, args.device)

        epoch_time = time.time() - start_time

        # Log results
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        logger.info(f"  Val F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'time': epoch_time
        })

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_acc': val_acc
            }, args.output_dir / 'model_best.pt')

            logger.info(f"  ✓ New best F1: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save final model and history
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1
    }, args.output_dir / 'model_final.pt')

    with open(args.output_dir / 'training_log.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump({
            'model': 'HybridPlausibilityScorer',
            'version': '2.0',
            'word_encoder_dim': word_encoder.output_dim,
            'trainable_parameters': trainable_params,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs_trained': epoch + 1,
            'best_f1': best_f1,
            'best_epoch': best_epoch
        }, f, indent=2)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best F1: {best_f1:.4f} (epoch {best_epoch})")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
