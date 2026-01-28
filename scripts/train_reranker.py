#!/usr/bin/env python3
"""
Train query-document relevance reranker.

Usage:
    # Basic training
    python scripts/train_reranker.py \
        --train-data data/training/reranker/combined/train.jsonl \
        --val-data data/training/reranker/combined/val.jsonl \
        --compositional-model models/root_embeddings/best_model.pt \
        --output models/reranker/

    # Resume from checkpoint
    python scripts/train_reranker.py --resume

    # Start fresh (ignore checkpoint)
    python scripts/train_reranker.py --fresh
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings import CompositionalEmbedding
from klareco.models.reranker import ASTReranker, count_parameters
from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RerankerDataset(Dataset):
    """
    Dataset for reranker training.

    Each example is (query_ast, doc_ast, relevance_score).
    """

    def __init__(self, data_path: Path, cache_parsed: bool = True):
        """
        Load training data.

        Args:
            data_path: Path to JSONL file with training pairs
            cache_parsed: Cache parsed ASTs to speed up loading
        """
        self.data_path = data_path
        self.cache_parsed = cache_parsed
        self.examples = []

        logger.info(f"Loading dataset from {data_path}...")
        self._load_data()
        logger.info(f"Loaded {len(self.examples):,} examples")

    def _load_data(self):
        """Load and parse training examples."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)

                    # Parse query and document
                    query_text = example['query']
                    doc_text = example['doc_text']
                    relevance = float(example['relevance'])

                    # Parse to ASTs
                    try:
                        query_ast = parse(query_text)
                        doc_ast = parse(doc_text)

                        self.examples.append({
                            'query_ast': query_ast,
                            'doc_ast': doc_ast,
                            'relevance': relevance,
                            'query_text': query_text,
                            'doc_text': doc_text,
                        })

                    except Exception as e:
                        logger.warning(f"  Line {line_num}: Failed to parse, skipping")
                        continue

                    if line_num % 10000 == 0:
                        logger.info(f"  Loaded {line_num:,} lines...")

                except json.JSONDecodeError:
                    logger.warning(f"  Line {line_num}: Invalid JSON, skipping")
                    continue

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_fn(batch: List[Dict]) -> Tuple[List[Dict], List[Dict], torch.Tensor]:
    """
    Collate batch of examples.

    Returns:
        (query_asts, doc_asts, relevance_scores)
    """
    query_asts = [ex['query_ast'] for ex in batch]
    doc_asts = [ex['doc_ast'] for ex in batch]
    relevance = torch.tensor([ex['relevance'] for ex in batch], dtype=torch.float32)

    return query_asts, doc_asts, relevance


class RerankerTrainer:
    """Trainer for ASTReranker model."""

    def __init__(
        self,
        model: ASTReranker,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            model: ASTReranker model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory for checkpoints and logs
            learning_rate: Learning rate (default: 1e-3)
            weight_decay: Weight decay for regularization
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.device = device

        # Loss and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy for relevance scores
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging to file
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Output: {output_dir}")
        trainable, total = count_parameters(model)
        logger.info(f"  Trainable params: {trainable:,}")
        logger.info(f"  Total params: {total:,}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (query_asts, doc_asts, relevance) in enumerate(self.train_loader):
            relevance = relevance.to(self.device)

            # Forward pass
            batch_loss = 0.0
            for query_ast, doc_ast, target in zip(query_asts, doc_asts, relevance):
                # Score this pair
                predicted = self.model(query_ast, doc_ast)
                loss = self.criterion(predicted, target.unsqueeze(0))
                batch_loss += loss

            # Average loss for batch
            batch_loss = batch_loss / len(query_asts)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: loss = {avg_loss:.4f}")

        return total_loss / num_batches

    def validate(self) -> float:
        """
        Validate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for query_asts, doc_asts, relevance in self.val_loader:
                relevance = relevance.to(self.device)

                # Forward pass
                batch_loss = 0.0
                for query_ast, doc_ast, target in zip(query_asts, doc_asts, relevance):
                    predicted = self.model(query_ast, doc_ast)
                    loss = self.criterion(predicted, target.unsqueeze(0))
                    batch_loss += loss

                batch_loss = batch_loss / len(query_asts)
                total_loss += batch_loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        epochs: int = 20,
        early_stopping_patience: int = 3,
    ):
        """
        Full training loop.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Early stopping patience: {early_stopping_patience}")
        logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset):,}")
        logger.info("")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            logger.info(f"Epoch {epoch}/{epochs}")

            # Train
            train_loss = self.train_epoch()
            logger.info(f"  Train loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            logger.info(f"  Val loss:   {val_loss:.4f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                logger.info(f"  ✓ New best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{early_stopping_patience})")

                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(is_best=False)

            logger.info("")

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.output_dir / 'best_model.pt'}")

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / 'checkpoint.pt'

        if is_best:
            # Save best model
            best_path = self.output_dir / 'best_model.pt'

            # Rotate previous best
            if best_path.exists():
                prev_path = self.output_dir / 'best_model.prev.pt'
                best_path.rename(prev_path)

            # Save new best (atomic write)
            temp_path = self.output_dir / 'best_model.tmp'
            try:
                self.model.save(temp_path)
                temp_path.rename(best_path)
                logger.info(f"    Saved best model to {best_path}")
            except Exception as e:
                logger.error(f"    Failed to save best model: {e}")
                if temp_path.exists():
                    temp_path.unlink()

        # Save checkpoint with training state
        temp_path = self.output_dir / 'checkpoint.tmp'
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.scorer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
            }, temp_path)
            temp_path.rename(checkpoint_path)
        except Exception as e:
            logger.error(f"    Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if it exists.

        Returns:
            True if checkpoint was loaded
        """
        checkpoint_path = self.output_dir / 'checkpoint.pt'
        if not checkpoint_path.exists():
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.scorer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint['patience_counter']

            logger.info(f"Loaded checkpoint from epoch {self.epoch}")
            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Train reranker model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--train-data',
        type=Path,
        default=Path('data/training/reranker/combined/train.jsonl'),
        help='Training data file'
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        default=Path('data/training/reranker/combined/val.jsonl'),
        help='Validation data file'
    )
    parser.add_argument(
        '--compositional-model',
        type=Path,
        default=Path('models/root_embeddings/best_model.pt'),
        help='Pre-trained CompositionalEmbedding model (frozen)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/reranker/'),
        help='Output directory for model and logs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs (default: 20)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()

    # Check inputs
    if not args.train_data.exists():
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)

    if not args.val_data.exists():
        logger.error(f"Validation data not found: {args.val_data}")
        sys.exit(1)

    if not args.compositional_model.exists():
        logger.error(f"Compositional model not found: {args.compositional_model}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Reranker Training")
    logger.info("=" * 60)
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")
    logger.info(f"Compositional model: {args.compositional_model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Load datasets
    train_dataset = RerankerDataset(args.train_data)
    val_dataset = RerankerDataset(args.val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load compositional embedding model (frozen)
    logger.info("Loading compositional embedding model...")
    checkpoint = torch.load(args.compositional_model, map_location='cpu', weights_only=False)

    # Check if this is a CompositionalEmbedding or simple root embeddings
    if 'root_vocab' in checkpoint:
        # Full CompositionalEmbedding checkpoint
        compositional_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        compositional_emb.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Simple root embeddings - create minimal CompositionalEmbedding
        logger.info("  Converting simple root embeddings to CompositionalEmbedding format...")
        root_to_idx = checkpoint['root_to_idx']

        # Create minimal vocabularies
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        compositional_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 64),
        )

        # Load only the root embeddings
        # The checkpoint has 'embeddings.weight' key
        if 'embeddings.weight' in checkpoint['model_state_dict']:
            compositional_emb.root_embed.weight.data = checkpoint['model_state_dict']['embeddings.weight']
        elif 'weight' in checkpoint['model_state_dict']:
            compositional_emb.root_embed.weight.data = checkpoint['model_state_dict']['weight']
        else:
            raise KeyError(f"Could not find embeddings in checkpoint. Keys: {list(checkpoint['model_state_dict'].keys())}")

    compositional_emb.eval()  # Set to eval mode (frozen)
    logger.info("  ✓ Loaded")

    # Initialize reranker
    logger.info("Initializing reranker...")
    model = ASTReranker(
        compositional_embedding=compositional_emb,
        freeze_embedding=True,
        hidden_dims=[256, 256, 128],
        dropout=0.2,
    )
    logger.info("  ✓ Initialized")

    # Initialize trainer
    trainer = RerankerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Load checkpoint if requested
    if args.resume and not args.fresh:
        if trainer.load_checkpoint():
            logger.info("Resuming from checkpoint\n")
        else:
            logger.info("No checkpoint found, starting fresh\n")
    elif args.fresh:
        logger.info("Starting fresh (ignoring checkpoint)\n")

    # Train
    trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
    )


if __name__ == '__main__':
    main()
