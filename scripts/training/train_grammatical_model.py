#!/usr/bin/env python3
"""
Train Stage 2 grammatical model on minimal pairs.

This script trains transforms that capture how grammatical features (negation,
tense, mood, sentence type) modify sentence meaning. It uses frozen Stage 1
embeddings and learns from minimal pairs with target similarities.

Training approach:
- For each minimal pair, compute Stage 1 embeddings for both sentences
- Learn feature-specific transforms that produce correct similarity
- Contrastive loss: predicted similarity should match target

Output: models/grammatical_transforms/best_model.pt

Usage:
    python scripts/training/train_grammatical_model.py \
        --stage1-model models/root_embeddings/best_model.pt \
        --affix-model models/affix_transforms_v2/best_model.pt \
        --training-data data/training/stage2_pairs.jsonl \
        --output-dir models/grammatical_transforms \
        --epochs 50
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from klareco.semantic_pipeline import SemanticPipeline, SemanticModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Grammatical features and their expected effects
GRAMMATICAL_FEATURES = {
    'negation': {
        'transforms': ['add_negation', 'remove_negation'],
        'expected_effect': 'polarity_flip',  # Should reverse meaning
    },
    'tense': {
        'transforms': ['present_to_past', 'past_to_present', 'present_to_future',
                      'future_to_present', 'past_to_future'],
        'expected_effect': 'temporal_shift',  # Related but different time
    },
    'mood': {
        'transforms': ['present_to_conditional', 'past_to_conditional', 'future_to_conditional'],
        'expected_effect': 'hypothetical',  # Factual vs counterfactual
    },
    'sentence_type': {
        'transforms': ['statement_to_question', 'question_to_statement'],
        'expected_effect': 'illocution_change',  # Same content, different act
    },
}


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


@dataclass
class MinimalPair:
    """Training pair with sentences and target similarity."""
    sentence1: str
    sentence2: str
    feature_type: str
    feature_detail: str
    similarity: float
    source_id: Optional[str] = None


class GrammaticalTransformModel(nn.Module):
    """
    Stage 2 grammatical model that transforms sentence embeddings.

    Architecture: For each grammatical feature, learn a transform that
    modifies the sentence embedding to reflect the grammatical change.

    The model learns:
    - How negation inverts meaning (polarity flip)
    - How tense shifts temporal context
    - How mood changes factuality
    - How sentence type changes illocutionary force
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Negation transform: flip polarity
        # Uses a reflection-like transform
        self.negation_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Tense transforms: temporal shift
        self.tense_transforms = nn.ModuleDict({
            'past': nn.Linear(embed_dim, embed_dim, bias=True),
            'present': nn.Linear(embed_dim, embed_dim, bias=True),
            'future': nn.Linear(embed_dim, embed_dim, bias=True),
        })

        # Mood transforms: factuality
        self.mood_transforms = nn.ModuleDict({
            'indicative': nn.Linear(embed_dim, embed_dim, bias=True),
            'conditional': nn.Linear(embed_dim, embed_dim, bias=True),
        })

        # Sentence type transforms: illocution
        self.sentence_type_transforms = nn.ModuleDict({
            'statement': nn.Linear(embed_dim, embed_dim, bias=True),
            'question': nn.Linear(embed_dim, embed_dim, bias=True),
        })

        # Initialize transforms close to identity
        self._init_transforms()

    def _init_transforms(self):
        """Initialize linear transforms close to identity."""
        for transform in [self.tense_transforms, self.mood_transforms,
                         self.sentence_type_transforms]:
            for name, linear in transform.items():
                # Start close to identity
                nn.init.eye_(linear.weight)
                # Small perturbation
                linear.weight.data += torch.randn_like(linear.weight.data) * 0.1
                nn.init.zeros_(linear.bias)

        # Negation needs stronger initialization (not identity)
        for layer in self.negation_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def apply_negation(self, emb: torch.Tensor, add: bool = True) -> torch.Tensor:
        """Apply negation transform (or its inverse)."""
        delta = self.negation_transform(emb)
        if add:
            return emb + delta  # Add negation
        else:
            return emb - delta  # Remove negation

    def apply_tense_change(self, emb: torch.Tensor,
                           from_tense: str, to_tense: str) -> torch.Tensor:
        """Apply tense transformation."""
        # First project out of from_tense, then into to_tense
        if from_tense in self.tense_transforms:
            # Use pseudo-inverse for reversal
            intermediate = emb
        else:
            intermediate = emb

        if to_tense in self.tense_transforms:
            return self.tense_transforms[to_tense](intermediate)
        return intermediate

    def apply_mood_change(self, emb: torch.Tensor,
                          to_conditional: bool = True) -> torch.Tensor:
        """Apply mood transformation."""
        if to_conditional:
            return self.mood_transforms['conditional'](emb)
        else:
            return self.mood_transforms['indicative'](emb)

    def apply_sentence_type_change(self, emb: torch.Tensor,
                                   to_question: bool = True) -> torch.Tensor:
        """Apply sentence type transformation."""
        if to_question:
            return self.sentence_type_transforms['question'](emb)
        else:
            return self.sentence_type_transforms['statement'](emb)

    def transform_by_feature(self, emb: torch.Tensor,
                             feature_type: str,
                             feature_detail: str) -> torch.Tensor:
        """Apply transformation based on feature type and detail."""
        if feature_type == 'negation':
            if feature_detail == 'add_negation':
                return self.apply_negation(emb, add=True)
            else:  # remove_negation
                return self.apply_negation(emb, add=False)

        elif feature_type == 'tense':
            # Parse "present_to_past" -> from_tense=present, to_tense=past
            parts = feature_detail.split('_to_')
            if len(parts) == 2:
                from_tense, to_tense = parts
                return self.apply_tense_change(emb, from_tense, to_tense)
            return emb

        elif feature_type == 'mood':
            # All mood transforms go to conditional
            return self.apply_mood_change(emb, to_conditional=True)

        elif feature_type == 'sentence_type':
            if feature_detail == 'statement_to_question':
                return self.apply_sentence_type_change(emb, to_question=True)
            else:  # question_to_statement
                return self.apply_sentence_type_change(emb, to_question=False)

        return emb

    def forward(self, emb1: torch.Tensor, feature_type: str,
                feature_detail: str) -> torch.Tensor:
        """Transform embedding according to grammatical feature."""
        return self.transform_by_feature(emb1, feature_type, feature_detail)


class MinimalPairsDataset(Dataset):
    """Dataset of minimal pairs for training."""

    def __init__(self, pairs: List[MinimalPair]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'sentence1': pair.sentence1,
            'sentence2': pair.sentence2,
            'feature_type': pair.feature_type,
            'feature_detail': pair.feature_detail,
            'similarity': pair.similarity,
        }


def load_training_data(path: Path) -> List[MinimalPair]:
    """Load minimal pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            pairs.append(MinimalPair(
                sentence1=data['sentence1'],
                sentence2=data['sentence2'],
                feature_type=data['feature_type'],
                feature_detail=data['feature_detail'],
                similarity=data['similarity'],
                source_id=data.get('source_id'),
            ))
    return pairs


def compute_sentence_embedding(pipeline: SemanticPipeline,
                                text: str) -> Optional[torch.Tensor]:
    """Compute Stage 1 sentence embedding."""
    try:
        enriched = pipeline.for_retrieval(text)
        if enriched.sentence_embedding is not None:
            return enriched.sentence_embedding.detach()
    except Exception as e:
        logger.debug(f"Failed to embed: {e}")
    return None


def train_epoch(
    model: GrammaticalTransformModel,
    pipeline: SemanticPipeline,
    pairs: List[MinimalPair],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    loss_by_feature = defaultdict(float)
    count_by_feature = defaultdict(int)
    num_batches = 0

    # Shuffle pairs
    random.shuffle(pairs)

    total_batches = (len(pairs) + batch_size - 1) // batch_size

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 50 == 0 or batch_num == total_batches:
            pct = 100 * batch_num / total_batches
            avg_loss = total_loss / max(num_batches, 1)
            print(f"\r  Batch {batch_num}/{total_batches} ({pct:.1f}%) loss={avg_loss:.4f}",
                  end='', flush=True)

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_pairs = 0

        for pair in batch:
            # Get Stage 1 embeddings for both sentences
            emb1 = compute_sentence_embedding(pipeline, pair.sentence1)
            emb2 = compute_sentence_embedding(pipeline, pair.sentence2)

            if emb1 is None or emb2 is None:
                continue

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)

            # Transform emb1 according to the feature
            transformed = model(emb1, pair.feature_type, pair.feature_detail)

            # Compute predicted similarity
            pred_sim = F.cosine_similarity(transformed.unsqueeze(0),
                                            emb2.unsqueeze(0))

            # Target similarity from training data
            target_sim = torch.tensor([pair.similarity], device=device)

            # MSE loss between predicted and target similarity
            pair_loss = F.mse_loss(pred_sim, target_sim)
            batch_loss = batch_loss + pair_loss

            loss_by_feature[pair.feature_type] += pair_loss.item()
            count_by_feature[pair.feature_type] += 1
            valid_pairs += 1

        if valid_pairs > 0:
            batch_loss = batch_loss / valid_pairs

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

    print()  # Newline after progress

    # Average loss by feature
    avg_by_feature = {
        ft: loss_by_feature[ft] / max(count_by_feature[ft], 1)
        for ft in loss_by_feature
    }

    return total_loss / max(num_batches, 1), avg_by_feature


def evaluate(
    model: GrammaticalTransformModel,
    pipeline: SemanticPipeline,
    pairs: List[MinimalPair],
    device: torch.device,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Evaluate model on validation pairs."""
    model.eval()
    metrics = {}

    # Sample if too many
    if len(pairs) > max_samples:
        eval_pairs = random.sample(pairs, max_samples)
    else:
        eval_pairs = pairs

    predictions_by_feature = defaultdict(list)
    targets_by_feature = defaultdict(list)

    with torch.no_grad():
        for pair in eval_pairs:
            emb1 = compute_sentence_embedding(pipeline, pair.sentence1)
            emb2 = compute_sentence_embedding(pipeline, pair.sentence2)

            if emb1 is None or emb2 is None:
                continue

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)

            # Without transform
            raw_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

            # With transform
            transformed = model(emb1, pair.feature_type, pair.feature_detail)
            pred_sim = F.cosine_similarity(transformed.unsqueeze(0),
                                            emb2.unsqueeze(0)).item()

            predictions_by_feature[pair.feature_type].append(pred_sim)
            targets_by_feature[pair.feature_type].append(pair.similarity)

    # Compute metrics per feature
    for feature_type in predictions_by_feature:
        preds = np.array(predictions_by_feature[feature_type])
        targets = np.array(targets_by_feature[feature_type])

        if len(preds) > 0:
            # Mean absolute error
            mae = np.mean(np.abs(preds - targets))

            # Correlation
            if len(preds) > 1:
                corr = np.corrcoef(preds, targets)[0, 1]
            else:
                corr = 0.0

            # For negation: check if we get negative similarity
            if feature_type == 'negation':
                neg_correct = np.mean(preds < 0)  # Should be negative
                metrics[f'{feature_type}_neg_rate'] = neg_correct

            metrics[f'{feature_type}_mae'] = mae
            metrics[f'{feature_type}_corr'] = corr
            metrics[f'{feature_type}_n'] = len(preds)

    return metrics


def save_checkpoint(path: Path, model: GrammaticalTransformModel,
                    optimizer, epoch: int, loss: float, metrics: Dict):
    """Atomically save checkpoint."""
    temp_path = path.with_suffix('.tmp')
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'embed_dim': model.embed_dim,
            'hidden_dim': model.hidden_dim,
        }, temp_path)
        temp_path.rename(path)
        logger.info(f"  Saved checkpoint to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Train Stage 2 grammatical model on minimal pairs'
    )
    parser.add_argument('--stage1-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'),
                        help='Path to Stage 1 root embeddings')
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_transforms_v2/best_model.pt'),
                        help='Path to affix transforms model')
    parser.add_argument('--training-data', type=Path,
                        default=Path('data/training/stage2_pairs.jsonl'),
                        help='Path to minimal pairs training data')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/grammatical_transforms'),
                        help='Output directory')
    parser.add_argument('--log-dir', type=Path, default=None,
                        help='Log directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"train_grammatical_{datetime.now():%Y%m%d_%H%M%S}.log"
        setup_file_logging(log_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check for training data
    if not args.training_data.exists():
        logger.error(f"Training data not found: {args.training_data}")
        logger.info("Run generate_stage2_pairs.py first to create training data")
        return 1

    # Load training data
    logger.info(f"Loading training data from {args.training_data}")
    all_pairs = load_training_data(args.training_data)
    logger.info(f"Loaded {len(all_pairs):,} minimal pairs")

    # Log distribution
    feature_counts = defaultdict(int)
    for pair in all_pairs:
        feature_counts[pair.feature_type] += 1
    for ft, count in sorted(feature_counts.items()):
        logger.info(f"  {ft}: {count:,}")

    # Split into train/val
    random.seed(42)
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * args.val_split)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]
    logger.info(f"Train: {len(train_pairs):,}, Val: {len(val_pairs):,}")

    # Load Stage 1 pipeline (frozen)
    logger.info("Loading Stage 1 pipeline...")
    if not args.stage1_model.exists():
        logger.error(f"Stage 1 model not found: {args.stage1_model}")
        return 1

    pipeline = SemanticPipeline.load(
        root_model_path=args.stage1_model,
        affix_model_path=args.affix_model if args.affix_model.exists() else None,
    )

    # Get embedding dimension from pipeline
    embed_dim = pipeline.semantic.embedding_dim
    logger.info(f"Embedding dimension: {embed_dim}")

    # Create model
    model = GrammaticalTransformModel(embed_dim=embed_dim, hidden_dim=embed_dim)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Check for checkpoint
    checkpoint_path = args.output_dir / 'best_model.pt'
    start_epoch = 0
    best_loss = float('inf')

    if checkpoint_path.exists() and not args.fresh:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        logger.info(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, loss_by_feature = train_epoch(
            model, pipeline, train_pairs, optimizer, device, args.batch_size
        )

        # Log per-feature loss
        logger.info(f"Train loss: {train_loss:.4f}")
        for ft, loss in sorted(loss_by_feature.items()):
            logger.info(f"  {ft}: {loss:.4f}")

        # Evaluate
        metrics = evaluate(model, pipeline, val_pairs, device)

        logger.info("Validation metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        # Check for improvement
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1,
                          train_loss, metrics)
        else:
            patience_counter += 1
            logger.info(f"  No improvement, patience {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info("\nTraining complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
