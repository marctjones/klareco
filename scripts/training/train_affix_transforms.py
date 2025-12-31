#!/usr/bin/env python3
"""
Train affix transformations using low-rank matrices.

This is a NEW approach where affixes are learned as TRANSFORMATIONS
rather than static embeddings. Each affix learns a low-rank matrix
that transforms root embeddings.

Key insight: mal- doesn't have a "meaning" on its own - it's a FUNCTION
that maps any root to its opposite. This is better modeled as a transformation.

Training approach:
1. Load frozen root embeddings from Stage 1
2. Extract (root, prefixes, suffixes) tuples from corpus ASTs
3. For each affix, learn a low-rank transformation matrix
4. Training signal: transformed embeddings should cluster appropriately
   - mal(bon) should be far from bon (opposite polarity)
   - mal(bon) should be similar to mal(bel) (both negated)
   - ej(lern) should be similar to ej(labor) (both places)

Output: models/affix_transforms/best_model.pt
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Official Esperanto affixes
PREFIXES = ['mal', 're', 'ge', 'eks', 'ek', 'pra', 'dis', 'mis', 'bo', 'fi', 'for', 'vic']
SUFFIXES = [
    'ul', 'in', 'et', 'eg', 'ej', 'ar', 'an', 'ist', 'il', 'ad', 'aĵ', 'ec',
    'ig', 'iĝ', 'ebl', 'em', 'end', 'ind', 'id', 'er', 'um', 'ĉj', 'nj',
    'ant', 'int', 'ont', 'at', 'it', 'ot'  # Participles
]


def setup_file_logging(log_path: Path):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


class LowRankTransform(nn.Module):
    """Low-rank affine transformation: x + up(down(x))"""

    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        # Initialize to near-identity
        nn.init.xavier_normal_(self.down.weight, gain=0.1)
        nn.init.xavier_normal_(self.up.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class AffixTransformModel(nn.Module):
    """Learnable transformations for Esperanto prefixes and suffixes."""

    def __init__(self, embedding_dim: int = 64, rank: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank = rank

        # Create transform for each prefix
        self.prefix_transforms = nn.ModuleDict({
            p: LowRankTransform(embedding_dim, rank) for p in PREFIXES
        })

        # Create transform for each suffix
        self.suffix_transforms = nn.ModuleDict({
            s: LowRankTransform(embedding_dim, rank) for s in SUFFIXES
        })

    def transform(self, root_emb: torch.Tensor,
                  prefixes: List[str],
                  suffixes: List[str]) -> torch.Tensor:
        """Apply affix transformations to root embedding."""
        emb = root_emb
        # Apply prefixes (left to right, outermost first)
        for p in prefixes:
            if p in self.prefix_transforms:
                emb = self.prefix_transforms[p](emb)
        # Apply suffixes (left to right)
        for s in suffixes:
            if s in self.suffix_transforms:
                emb = self.suffix_transforms[s](emb)
        return emb

    def forward_prefix(self, root_emb: torch.Tensor, prefix: str) -> torch.Tensor:
        """Apply single prefix transform."""
        if prefix in self.prefix_transforms:
            return self.prefix_transforms[prefix](root_emb)
        return root_emb

    def forward_suffix(self, root_emb: torch.Tensor, suffix: str) -> torch.Tensor:
        """Apply single suffix transform."""
        if suffix in self.suffix_transforms:
            return self.suffix_transforms[suffix](root_emb)
        return root_emb


class AffixTrainingDataset(Dataset):
    """Dataset of (root, affixes, context_roots) tuples."""

    def __init__(self, samples: List[Tuple[str, List[str], List[str], List[str]]]):
        """
        samples: [(root, prefixes, suffixes, context_roots), ...]
        context_roots: other roots in the same sentence (for contrastive learning)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_word_info(node: dict) -> Optional[Tuple[str, List[str], List[str]]]:
    """Extract root and affixes from a word node."""
    if not isinstance(node, dict) or node.get('tipo') != 'vorto':
        return None

    root = node.get('radiko')
    if not root:
        return None

    # Get prefixes (handle both old 'prefikso' and new 'prefiksoj' format)
    prefixes = node.get('prefiksoj', [])
    if not prefixes:
        p = node.get('prefikso')
        if p:
            prefixes = [p]

    # Get suffixes
    suffixes = node.get('sufiksoj', [])

    # Filter to known affixes
    prefixes = [p for p in prefixes if p in PREFIXES]
    suffixes = [s for s in suffixes if s in SUFFIXES]

    return (root, prefixes, suffixes)


def extract_sentence_words(ast: dict) -> List[Tuple[str, List[str], List[str]]]:
    """Extract all word tuples from an AST."""
    words = []

    def visit(node):
        if isinstance(node, dict):
            info = extract_word_info(node)
            if info:
                words.append(info)
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return words


def build_training_data(
    corpus_path: Path,
    root_to_idx: Dict[str, int],
    max_samples: int = 500000
) -> List[Tuple[str, List[str], List[str], List[str]]]:
    """Build training samples from corpus."""
    logger.info(f"Building training data from {corpus_path}")

    samples = []
    affix_counts = defaultdict(int)

    with open(corpus_path) as f:
        for line_num, line in enumerate(f, 1):
            if len(samples) >= max_samples:
                break

            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num:,} lines, {len(samples):,} samples")

            try:
                entry = json.loads(line)
                ast = entry.get('ast')
                if not ast:
                    continue

                words = extract_sentence_words(ast)
                if len(words) < 2:
                    continue

                # Get all roots in sentence
                all_roots = [w[0] for w in words if w[0] in root_to_idx]

                for root, prefixes, suffixes in words:
                    # Only use words with known roots and at least one affix
                    if root not in root_to_idx:
                        continue
                    if not prefixes and not suffixes:
                        continue

                    # Context roots = other roots in same sentence
                    context = [r for r in all_roots if r != root]
                    if not context:
                        continue

                    samples.append((root, prefixes, suffixes, context))

                    for p in prefixes:
                        affix_counts[f'pre:{p}'] += 1
                    for s in suffixes:
                        affix_counts[f'suf:{s}'] += 1

            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"Collected {len(samples):,} training samples")
    logger.info("Affix distribution:")
    for affix, count in sorted(affix_counts.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {affix}: {count:,}")

    return samples


def train_epoch(
    model: AffixTransformModel,
    root_embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        roots, prefixes_list, suffixes_list, contexts_list = batch

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        for root, prefixes, suffixes, contexts in zip(roots, prefixes_list, suffixes_list, contexts_list):
            if root not in root_to_idx:
                continue

            root_idx = root_to_idx[root]
            root_emb = root_embeddings[root_idx].to(device)

            # Apply transformations
            transformed = model.transform(root_emb, prefixes, suffixes)

            # Contrastive loss: transformed should be different from original
            # but still in same semantic neighborhood as context

            # Loss 1: Transformed should be different from original (for prefixes like mal-)
            if 'mal' in prefixes:
                # mal- should flip polarity - push away from original
                dist_from_orig = 1 - F.cosine_similarity(
                    transformed.unsqueeze(0),
                    root_emb.unsqueeze(0)
                )
                batch_loss = batch_loss + F.relu(0.5 - dist_from_orig).mean()

            # Loss 2: Similar affixes should produce similar transformations
            # (This is implicit in the shared transform matrices)

            # Loss 3: Transformed word should still relate to context
            context_embs = []
            for ctx in contexts:
                if ctx in root_to_idx:
                    context_embs.append(root_embeddings[root_to_idx[ctx]])

            if context_embs:
                context_embs = torch.stack(context_embs).to(device)
                # Should maintain some similarity to context
                ctx_sim = F.cosine_similarity(
                    transformed.unsqueeze(0).expand(len(context_embs), -1),
                    context_embs
                ).mean()
                # Soft target: maintain moderate similarity
                batch_loss = batch_loss + F.mse_loss(ctx_sim, torch.tensor(0.3, device=device))

            valid_samples += 1

        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: AffixTransformModel,
    root_embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    device: torch.device
) -> Dict[str, float]:
    """Evaluate transform quality."""
    model.eval()
    metrics = {}

    with torch.no_grad():
        # Test mal- polarity flip
        test_pairs = [
            ('bon', 'mal'),  # good -> bad
            ('grand', 'mal'),  # big -> small
            ('bel', 'mal'),  # beautiful -> ugly
            ('fort', 'mal'),  # strong -> weak
        ]

        polarity_dists = []
        for root, prefix in test_pairs:
            if root not in root_to_idx:
                continue
            root_emb = root_embeddings[root_to_idx[root]].to(device)
            transformed = model.forward_prefix(root_emb, prefix)
            dist = 1 - F.cosine_similarity(
                transformed.unsqueeze(0),
                root_emb.unsqueeze(0)
            ).item()
            polarity_dists.append(dist)

        if polarity_dists:
            metrics['mal_distance'] = np.mean(polarity_dists)

        # Test suffix consistency (e.g., all -ej words should cluster)
        ej_roots = ['lern', 'labor', 'preĝ', 'kuir']  # school, workplace, church, kitchen
        ej_embs = []
        for root in ej_roots:
            if root in root_to_idx:
                root_emb = root_embeddings[root_to_idx[root]].to(device)
                transformed = model.forward_suffix(root_emb, 'ej')
                ej_embs.append(transformed)

        if len(ej_embs) >= 2:
            ej_embs = torch.stack(ej_embs)
            # Compute pairwise similarities
            sims = []
            for i in range(len(ej_embs)):
                for j in range(i + 1, len(ej_embs)):
                    sim = F.cosine_similarity(
                        ej_embs[i].unsqueeze(0),
                        ej_embs[j].unsqueeze(0)
                    ).item()
                    sims.append(sim)
            metrics['ej_cluster_sim'] = np.mean(sims)

    return metrics


def custom_collate(batch):
    """Custom collate for variable-length lists."""
    roots = [b[0] for b in batch]
    prefixes = [b[1] for b in batch]
    suffixes = [b[2] for b in batch]
    contexts = [b[3] for b in batch]
    return roots, prefixes, suffixes, contexts


def save_checkpoint(path: Path, model: AffixTransformModel, optimizer, epoch: int,
                    loss: float, metrics: Dict):
    """Atomically save checkpoint."""
    temp_path = path.with_suffix('.tmp')
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'embedding_dim': model.embedding_dim,
            'rank': model.rank,
            'prefixes': PREFIXES,
            'suffixes': SUFFIXES,
        }, temp_path)
        temp_path.rename(path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train affix transformations')
    parser.add_argument('--root-embeddings', type=Path, required=True,
                        help='Path to trained root embeddings')
    parser.add_argument('--corpus', type=Path, required=True,
                        help='Path to corpus JSONL')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for model')
    parser.add_argument('--log-dir', type=Path, default=None,
                        help='Log directory')
    parser.add_argument('--rank', type=int, default=4,
                        help='Rank for low-rank transforms')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Max training samples')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    # Setup logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"train_affix_transforms_{datetime.now():%Y%m%d_%H%M%S}.log"
        setup_file_logging(log_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load root embeddings
    logger.info(f"Loading root embeddings from {args.root_embeddings}")
    root_checkpoint = torch.load(args.root_embeddings, map_location='cpu')

    root_to_idx = root_checkpoint['root_to_idx']
    embedding_dim = root_checkpoint['embedding_dim']

    # Reconstruct root embeddings
    class RootEmbeddings(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, dim)

    root_model = RootEmbeddings(root_checkpoint['vocab_size'], embedding_dim)
    root_model.load_state_dict(root_checkpoint['model_state_dict'])
    root_embeddings = root_model.embeddings.weight.detach()

    logger.info(f"Loaded {len(root_to_idx):,} root embeddings ({embedding_dim}d)")

    # Build training data
    samples = build_training_data(args.corpus, root_to_idx, args.max_samples)

    if not samples:
        logger.error("No training samples found!")
        return

    # Split train/val
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = AffixTrainingDataset(train_samples)
    val_dataset = AffixTrainingDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate
    )

    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # Create model
    model = AffixTransformModel(embedding_dim=embedding_dim, rank=args.rank)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Check for existing checkpoint
    checkpoint_path = args.output_dir / 'best_model.pt'
    start_epoch = 0
    best_metric = float('inf')

    if checkpoint_path.exists() and not args.fresh:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['loss']
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, root_embeddings, root_to_idx, train_loader, optimizer, device
        )

        # Evaluate
        metrics = evaluate(model, root_embeddings, root_to_idx, device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"mal_dist={metrics.get('mal_distance', 0):.4f}, "
            f"ej_sim={metrics.get('ej_cluster_sim', 0):.4f}"
        )

        # Save if improved
        if train_loss < best_metric:
            best_metric = train_loss
            patience_counter = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch + 1, train_loss, metrics
            )
            logger.info(f"  Saved new best model")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break

    logger.info("\nTraining complete!")
    logger.info(f"Best loss: {best_metric:.4f}")
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
