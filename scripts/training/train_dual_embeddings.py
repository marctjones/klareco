#!/usr/bin/env python3
"""
Train dual parallel embeddings (linguistic + topical).

Phase 2 of the dual embeddings implementation:
1. Load pre-trained linguistic embeddings
2. Train topical embeddings on skip-gram pairs
3. Optionally fine-tune both jointly

Strategy:
- Sequential training: Freeze linguistic, train topical first
- Optional joint fine-tuning afterwards
- Save checkpoints for both stages
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
from typing import Dict, Tuple, Optional
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from klareco.embeddings.dual_root_embeddings import DualRootEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TopicalTrainer:
    """Trainer for dual embeddings with sequential/joint training."""

    def __init__(
        self,
        model: DualRootEmbeddings,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate

        # Separate optimizers for linguistic and topical
        self.linguistic_optimizer = None
        self.topical_optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'topical' in n],
            lr=learning_rate
        )

        # Loss function
        self.criterion = nn.MSELoss()

    def freeze_linguistic(self):
        """Freeze linguistic embeddings for topical-only training."""
        for name, param in self.model.named_parameters():
            if 'linguistic' in name:
                param.requires_grad = False
        logger.info("Froze linguistic embeddings")

    def unfreeze_all(self):
        """Unfreeze all parameters for joint fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

        # Create optimizer for all parameters
        self.linguistic_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr * 0.1  # Lower learning rate for fine-tuning
        )
        logger.info("Unfroze all parameters for joint fine-tuning")

    def train_epoch(
        self,
        data_path: Path,
        batch_size: int = 1024,
        mode: str = 'topical'
    ) -> Tuple[float, int]:
        """
        Train one epoch.

        Args:
            data_path: Path to training pairs (JSONL)
            batch_size: Batch size
            mode: 'topical' (freeze linguistic) or 'joint' (train both)

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
                    loss = self._process_batch(batch_idx1, batch_idx2, batch_targets, mode)
                    total_loss += loss
                    num_batches += 1

                    # Clear batch
                    batch_idx1 = []
                    batch_idx2 = []
                    batch_targets = []

        # Process remaining pairs
        if batch_idx1:
            loss = self._process_batch(batch_idx1, batch_idx2, batch_targets, mode)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, num_batches

    def _process_batch(
        self,
        idx1: list,
        idx2: list,
        targets: list,
        mode: str
    ) -> float:
        """Process a single batch."""
        # Convert to tensors
        idx1_t = torch.tensor(idx1, dtype=torch.long, device=self.device)
        idx2_t = torch.tensor(idx2, dtype=torch.long, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Forward pass
        emb1 = self.model(idx1_t, mode=mode)
        emb2 = self.model(idx2_t, mode=mode)

        # Cosine similarity
        similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1)

        # Loss
        loss = self.criterion(similarity, targets_t)

        # Backward pass
        if mode == 'topical':
            self.topical_optimizer.zero_grad()
            loss.backward()
            self.topical_optimizer.step()
        else:  # joint
            self.linguistic_optimizer.zero_grad()
            loss.backward()
            self.linguistic_optimizer.step()

        return loss.item()


def load_linguistic_embeddings(
    linguistic_model_path: Path,
    vocab_path: Path,
    embedding_dim: int = 64
) -> DualRootEmbeddings:
    """
    Load pre-trained linguistic embeddings into dual model.

    Handles vocabulary size mismatch by:
    1. Creating model with larger topical vocab size
    2. Loading pretrained weights for roots that exist in both
    3. Initializing remaining roots randomly

    Args:
        linguistic_model_path: Path to trained linguistic model
        vocab_path: Path to vocabulary JSON (topical vocab, may be larger)
        embedding_dim: Embedding dimension (64 for each mode)

    Returns:
        Initialized DualRootEmbeddings with loaded linguistic weights
    """
    logger.info(f"Loading linguistic embeddings from {linguistic_model_path}")

    # Load topical vocabulary (may be larger than linguistic vocab)
    with open(vocab_path) as f:
        topical_vocab = json.load(f)

    topical_vocab_size = len(topical_vocab['root_to_idx'])
    logger.info(f"Topical vocabulary size: {topical_vocab_size:,}")

    # Load linguistic model checkpoint
    checkpoint = torch.load(linguistic_model_path, map_location='cpu')

    linguistic_vocab_size = checkpoint.get('vocab_size', 0)
    linguistic_root_to_idx = checkpoint.get('root_to_idx', {})

    logger.info(f"Linguistic model vocabulary size: {linguistic_vocab_size:,}")

    # Extract linguistic embeddings from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Get pretrained weights
    pretrained_weights = None
    if 'embeddings.weight' in state_dict:
        pretrained_weights = state_dict['embeddings.weight']
    elif 'embedding.weight' in state_dict:
        pretrained_weights = state_dict['embedding.weight']

    if pretrained_weights is None:
        logger.warning("No pretrained weights found, using random initialization")
        # Create model with random init
        model = DualRootEmbeddings(
            num_roots=topical_vocab_size,
            linguistic_dim=embedding_dim,
            topical_dim=embedding_dim
        )
        return model

    # Create model with topical vocab size
    model = DualRootEmbeddings(
        num_roots=topical_vocab_size,
        linguistic_dim=embedding_dim,
        topical_dim=embedding_dim
    )

    # Map linguistic embeddings to topical vocabulary
    # For roots that exist in both, copy pretrained weights
    # For new roots, keep random initialization
    loaded_count = 0
    new_count = 0

    for root, topical_idx in topical_vocab['root_to_idx'].items():
        if root in linguistic_root_to_idx:
            linguistic_idx = linguistic_root_to_idx[root]
            if linguistic_idx < pretrained_weights.shape[0]:
                # Copy pretrained embedding
                model.linguistic_embedding.weight.data[topical_idx] = pretrained_weights[linguistic_idx]
                loaded_count += 1
            else:
                new_count += 1
        else:
            # Root not in linguistic vocab, keep random init
            new_count += 1

    logger.info(f"Loaded linguistic embeddings:")
    logger.info(f"  Pretrained roots: {loaded_count:,}")
    logger.info(f"  New roots (random init): {new_count:,}")
    logger.info(f"  Total roots: {topical_vocab_size:,}")
    logger.info(f"  Coverage: {loaded_count / topical_vocab_size * 100:.1f}%")

    return model


def save_checkpoint(
    output_dir: Path,
    model: DualRootEmbeddings,
    optimizer_topical: optim.Optimizer,
    optimizer_joint: Optional[optim.Optimizer],
    epoch: int,
    loss: float,
    stage: str
):
    """Save training checkpoint atomically."""
    checkpoint_path = output_dir / f'checkpoint_{stage}.pt'
    temp_path = checkpoint_path.with_suffix('.tmp')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_topical_state_dict': optimizer_topical.state_dict(),
        'loss': loss,
        'stage': stage
    }

    if optimizer_joint:
        checkpoint['optimizer_joint_state_dict'] = optimizer_joint.state_dict()

    try:
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def train_dual_embeddings(
    training_data_path: Path,
    linguistic_model_path: Path,
    vocab_path: Path,
    output_dir: Path,
    topical_epochs: int = 10,
    joint_epochs: int = 5,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    skip_joint: bool = False
):
    """
    Train dual embeddings with sequential training.

    Phase 1: Train topical embeddings (linguistic frozen)
    Phase 2: Joint fine-tuning (optional)
    """
    logger.info("=" * 60)
    logger.info("Dual Embeddings Training")
    logger.info("=" * 60)
    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Linguistic model: {linguistic_model_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Topical epochs: {topical_epochs}")
    logger.info(f"Joint epochs: {joint_epochs if not skip_joint else 'SKIPPED'}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Load model with linguistic embeddings
    model = load_linguistic_embeddings(linguistic_model_path, vocab_path)

    # Create trainer
    trainer = TopicalTrainer(model, learning_rate=learning_rate, device=device)

    # Phase 1: Train topical embeddings (linguistic frozen)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Training topical embeddings (linguistic frozen)")
    logger.info("=" * 60)

    trainer.freeze_linguistic()

    best_topical_loss = float('inf')

    for epoch in range(topical_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{topical_epochs}")

        loss, num_batches = trainer.train_epoch(
            training_data_path,
            batch_size=batch_size,
            mode='topical'
        )

        logger.info(f"Topical training - Epoch {epoch + 1}: Loss = {loss:.4f} ({num_batches:,} batches)")

        # Save best model
        if loss < best_topical_loss:
            best_topical_loss = loss
            save_checkpoint(
                output_dir,
                model,
                trainer.topical_optimizer,
                None,
                epoch,
                loss,
                'topical_best'
            )

        # Save periodic checkpoint
        if (epoch + 1) % 2 == 0:
            save_checkpoint(
                output_dir,
                model,
                trainer.topical_optimizer,
                None,
                epoch,
                loss,
                f'topical_epoch{epoch+1}'
            )

    # Phase 2: Joint fine-tuning (optional)
    if not skip_joint and joint_epochs > 0:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Joint fine-tuning (both embeddings)")
        logger.info("=" * 60)

        trainer.unfreeze_all()

        best_joint_loss = float('inf')

        for epoch in range(joint_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{joint_epochs}")

            loss, num_batches = trainer.train_epoch(
                training_data_path,
                batch_size=batch_size,
                mode='combined'
            )

            logger.info(f"Joint training - Epoch {epoch + 1}: Loss = {loss:.4f} ({num_batches:,} batches)")

            # Save best model
            if loss < best_joint_loss:
                best_joint_loss = loss
                save_checkpoint(
                    output_dir,
                    model,
                    trainer.topical_optimizer,
                    trainer.linguistic_optimizer,
                    epoch,
                    loss,
                    'joint_best'
                )

    # Save final model
    final_path = output_dir / 'dual_embeddings_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_roots': model.num_roots,
            'linguistic_dim': model.linguistic_dim,
            'topical_dim': model.topical_dim
        }
    }, final_path)

    logger.info(f"\n{'=' * 60}")
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"Training log: {log_file}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Train dual parallel embeddings')
    parser.add_argument('--training-data', type=Path,
                        default=Path('data/training/topical_pairs_smart.jsonl'),
                        help='Path to training pairs (JSONL)')
    parser.add_argument('--linguistic-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'),
                        help='Path to pre-trained linguistic model')
    parser.add_argument('--vocab', type=Path,
                        default=Path('data/vocabularies/topical_vocab.json'),
                        help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/dual_embeddings'),
                        help='Output directory for trained model')
    parser.add_argument('--topical-epochs', type=int, default=10,
                        help='Number of epochs for topical training (default: 10)')
    parser.add_argument('--joint-epochs', type=int, default=5,
                        help='Number of epochs for joint fine-tuning (default: 5)')
    parser.add_argument('--skip-joint', action='store_true',
                        help='Skip joint fine-tuning phase')
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

    if not args.linguistic_model.exists():
        logger.error(f"Linguistic model not found: {args.linguistic_model}")
        return 1

    if not args.vocab.exists():
        logger.error(f"Vocabulary not found: {args.vocab}")
        return 1

    # Train
    train_dual_embeddings(
        training_data_path=args.training_data,
        linguistic_model_path=args.linguistic_model,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        topical_epochs=args.topical_epochs,
        joint_epochs=args.joint_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        skip_joint=args.skip_joint
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
