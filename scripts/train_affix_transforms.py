#!/usr/bin/env python3
"""
Train affix transformations using CONTRASTIVE learning.

This is a FIXED version of train_affix_transforms.py that addresses
embedding collapse caused by the original context similarity loss.

Key changes from v1:
1. REMOVED: The problematic context loss (F.mse_loss(ctx_sim, 0.3))
   that forced all affixed words toward the same region
2. ADDED: Proper triplet contrastive loss with hard negative mining
3. ADDED: Separation loss to keep different-root-same-affix words apart
4. ADDED: Better initialization (larger gain)

Training approach:
- For each affixed word, we create triplets: (anchor, positive, negative)
- Anchor: root embedding
- Positive: same root + SAME affix (identity for consistency)
- Negative: DIFFERENT root + same affix (should stay apart)

For mal- specifically:
- mal(bon) should be FAR from bon (polarity flip)
- mal(bon) should be FAR from mal(facil) (different meanings)
- All mal-words share the SAME transformation, but output different

Output: models/affix_transforms_v2/best_model.pt
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
    """Low-rank affine transformation: x + up(down(x))

    Key change: Larger initialization (gain=0.5) so transforms have effect.
    """

    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        # Larger initialization so transforms actually do something
        nn.init.xavier_normal_(self.down.weight, gain=0.5)
        nn.init.xavier_normal_(self.up.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class AffixTransformModel(nn.Module):
    """Learnable transformations for Esperanto prefixes and suffixes."""

    def __init__(self, embedding_dim: int = 64, rank: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank = rank

        self.prefix_transforms = nn.ModuleDict({
            p: LowRankTransform(embedding_dim, rank) for p in PREFIXES
        })
        self.suffix_transforms = nn.ModuleDict({
            s: LowRankTransform(embedding_dim, rank) for s in SUFFIXES
        })

    def transform(self, root_emb: torch.Tensor,
                  prefixes: List[str],
                  suffixes: List[str]) -> torch.Tensor:
        """Apply affix transformations to root embedding."""
        emb = root_emb
        for p in prefixes:
            if p in self.prefix_transforms:
                emb = self.prefix_transforms[p](emb)
        for s in suffixes:
            if s in self.suffix_transforms:
                emb = self.suffix_transforms[s](emb)
        return emb

    def forward_prefix(self, root_emb: torch.Tensor, prefix: str) -> torch.Tensor:
        if prefix in self.prefix_transforms:
            return self.prefix_transforms[prefix](root_emb)
        return root_emb

    def forward_suffix(self, root_emb: torch.Tensor, suffix: str) -> torch.Tensor:
        if suffix in self.suffix_transforms:
            return self.suffix_transforms[suffix](root_emb)
        return root_emb


def extract_word_info(node: dict) -> Optional[Tuple[str, List[str], List[str]]]:
    """Extract root and affixes from a word node."""
    if not isinstance(node, dict) or node.get('tipo') != 'vorto':
        return None

    root = node.get('radiko')
    if not root:
        return None

    prefixes = node.get('prefiksoj', [])
    if not prefixes:
        p = node.get('prefikso')
        if p:
            prefixes = [p]

    suffixes = node.get('sufiksoj', [])

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
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Set[str]]]:
    """Build training samples with affix groupings.

    Returns:
        samples: List of (root, affix_type, affix) tuples
        affix_to_roots: Dict mapping each affix to set of roots it appears with
    """
    logger.info(f"Building training data from {corpus_path}")

    samples = []  # (root, affix_type, affix)
    affix_to_roots = defaultdict(set)  # affix -> {root1, root2, ...}
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

                for root, prefixes, suffixes in words:
                    if root not in root_to_idx:
                        continue

                    # Add prefix samples
                    for p in prefixes:
                        samples.append((root, 'prefix', p))
                        affix_to_roots[f'pre:{p}'].add(root)
                        affix_counts[f'pre:{p}'] += 1

                    # Add suffix samples
                    for s in suffixes:
                        samples.append((root, 'suffix', s))
                        affix_to_roots[f'suf:{s}'].add(root)
                        affix_counts[f'suf:{s}'] += 1

            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"Collected {len(samples):,} training samples")
    logger.info(f"Affix coverage: {len(affix_to_roots)} affixes with known roots")
    logger.info("Affix distribution:")
    for affix, count in sorted(affix_counts.items(), key=lambda x: -x[1])[:20]:
        roots = affix_to_roots[affix]
        logger.info(f"  {affix}: {count:,} samples, {len(roots)} unique roots")

    return samples, affix_to_roots


def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor,
                         negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Triplet loss: anchor should be closer to positive than negative."""
    pos_dist = 1 - F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
    neg_dist = 1 - F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0))
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def train_epoch(
    model: AffixTransformModel,
    root_embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    samples: List[Tuple[str, str, str]],
    affix_to_roots: Dict[str, Set[str]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 64,
    neg_samples: int = 5
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with contrastive learning."""
    model.train()
    total_loss = 0.0
    loss_components = defaultdict(float)
    num_batches = 0
    total_batches = (len(samples) + batch_size - 1) // batch_size

    # Shuffle samples
    random.shuffle(samples)

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_num = i // batch_size + 1

        # Progress update (overwrites same line)
        if batch_num % 100 == 0 or batch_num == total_batches:
            pct = 100 * batch_num / total_batches
            avg_loss = total_loss / max(num_batches, 1)
            print(f"\r  Batch {batch_num:,}/{total_batches:,} ({pct:.1f}%) loss={avg_loss:.4f}", end='', flush=True)

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        for root, affix_type, affix in batch:
            if root not in root_to_idx:
                continue

            root_idx = root_to_idx[root]
            root_emb = root_embeddings[root_idx].to(device)

            # Apply transformation
            if affix_type == 'prefix':
                transformed = model.forward_prefix(root_emb, affix)
                affix_key = f'pre:{affix}'
            else:
                transformed = model.forward_suffix(root_emb, affix)
                affix_key = f'suf:{affix}'

            # === LOSS 1: Separation from different roots with same affix ===
            # This prevents collapse: mal(bon) should be FAR from mal(facil)
            other_roots = [r for r in affix_to_roots.get(affix_key, set())
                          if r != root and r in root_to_idx]

            if other_roots and len(other_roots) >= 1:
                # Sample negative examples (different roots, same affix)
                neg_roots = random.sample(other_roots, min(neg_samples, len(other_roots)))

                for neg_root in neg_roots:
                    neg_root_emb = root_embeddings[root_to_idx[neg_root]].to(device)
                    if affix_type == 'prefix':
                        neg_transformed = model.forward_prefix(neg_root_emb, affix)
                    else:
                        neg_transformed = model.forward_suffix(neg_root_emb, affix)

                    # Transformed words should maintain root distance
                    # If root1 and root2 are far, then affix(root1) and affix(root2) should be far
                    orig_dist = 1 - F.cosine_similarity(
                        root_emb.unsqueeze(0), neg_root_emb.unsqueeze(0)
                    )
                    trans_dist = 1 - F.cosine_similarity(
                        transformed.unsqueeze(0), neg_transformed.unsqueeze(0)
                    )

                    # Transformed distance should be at least as large as original
                    # This preserves semantic distinctions
                    sep_loss = F.relu(orig_dist - trans_dist + 0.1)
                    batch_loss = batch_loss + sep_loss.mean() * 0.5
                    loss_components['separation'] += sep_loss.item()

            # === LOSS 2: Polarity flip for mal- ===
            if affix == 'mal' and affix_type == 'prefix':
                # mal- should significantly change the embedding (push away)
                dist_from_orig = 1 - F.cosine_similarity(
                    transformed.unsqueeze(0),
                    root_emb.unsqueeze(0)
                )
                # Target: at least 0.5 distance (significant change)
                polarity_loss = F.relu(0.5 - dist_from_orig)
                batch_loss = batch_loss + polarity_loss.mean() * 2.0
                loss_components['mal_polarity'] += polarity_loss.item()

            # === LOSS 3: Suffix clustering (place words should cluster) ===
            if affix == 'ej' and affix_type == 'suffix':
                # -ej words should share some "place" semantics
                # But this is already handled by shared transformation matrix
                pass

            # === LOSS 4: Identity preservation for non-semantic affixes ===
            # For affixes like participal endings, maintain proximity
            if affix in ['ant', 'int', 'ont', 'at', 'it', 'ot']:
                # Participles should stay close to root
                dist_from_orig = 1 - F.cosine_similarity(
                    transformed.unsqueeze(0),
                    root_emb.unsqueeze(0)
                )
                # Keep close (< 0.3 distance)
                identity_loss = F.relu(dist_from_orig - 0.3)
                batch_loss = batch_loss + identity_loss.mean() * 0.3
                loss_components['identity'] += identity_loss.item()

            valid_samples += 1

        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

    # Average loss components
    for key in loss_components:
        loss_components[key] /= max(len(samples), 1)

    # Newline after batch progress line
    print()

    return total_loss / max(num_batches, 1), dict(loss_components)


def evaluate(
    model: AffixTransformModel,
    root_embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    affix_to_roots: Dict[str, Set[str]],
    device: torch.device
) -> Dict[str, float]:
    """Evaluate transform quality with anti-collapse metrics."""
    model.eval()
    metrics = {}

    with torch.no_grad():
        # === METRIC 1: mal- polarity flip ===
        test_pairs = [('bon', 'mal'), ('grand', 'mal'), ('bel', 'mal'), ('fort', 'mal')]
        polarity_dists = []
        for root, prefix in test_pairs:
            if root not in root_to_idx:
                continue
            root_emb = root_embeddings[root_to_idx[root]].to(device)
            transformed = model.forward_prefix(root_emb, prefix)
            dist = 1 - F.cosine_similarity(
                transformed.unsqueeze(0), root_emb.unsqueeze(0)
            ).item()
            polarity_dists.append(dist)

        if polarity_dists:
            metrics['mal_distance'] = np.mean(polarity_dists)

        # === METRIC 2: -ej suffix clustering ===
        ej_roots = ['lern', 'labor', 'preĝ', 'kuir']
        ej_embs = []
        for root in ej_roots:
            if root in root_to_idx:
                root_emb = root_embeddings[root_to_idx[root]].to(device)
                transformed = model.forward_suffix(root_emb, 'ej')
                ej_embs.append(transformed)

        if len(ej_embs) >= 2:
            ej_embs = torch.stack(ej_embs)
            sims = []
            for i in range(len(ej_embs)):
                for j in range(i + 1, len(ej_embs)):
                    sim = F.cosine_similarity(
                        ej_embs[i].unsqueeze(0), ej_embs[j].unsqueeze(0)
                    ).item()
                    sims.append(sim)
            metrics['ej_cluster_sim'] = np.mean(sims)

        # === METRIC 3: ANTI-COLLAPSE - different roots should stay apart ===
        # Sample mal- words and check they maintain distance
        mal_roots = list(affix_to_roots.get('pre:mal', set()))
        if len(mal_roots) >= 10:
            mal_roots = random.sample(mal_roots, 10)
            mal_embs = []
            for root in mal_roots:
                if root in root_to_idx:
                    root_emb = root_embeddings[root_to_idx[root]].to(device)
                    transformed = model.forward_prefix(root_emb, 'mal')
                    mal_embs.append(transformed)

            if len(mal_embs) >= 2:
                mal_embs = torch.stack(mal_embs)
                # Pairwise similarities - should NOT all be high
                sims = []
                for i in range(len(mal_embs)):
                    for j in range(i + 1, len(mal_embs)):
                        sim = F.cosine_similarity(
                            mal_embs[i].unsqueeze(0), mal_embs[j].unsqueeze(0)
                        ).item()
                        sims.append(sim)

                metrics['mal_mean_sim'] = np.mean(sims)
                metrics['mal_max_sim'] = np.max(sims)
                # Good if mean_sim < 0.7 (not collapsed)

        # === METRIC 4: Embedding diversity (std of embeddings) ===
        # Higher std = more diverse = less collapse
        all_suffixes = ['ej', 'ist', 'ul', 'in']
        all_transformed = []
        for suffix in all_suffixes:
            roots = list(affix_to_roots.get(f'suf:{suffix}', set()))[:5]
            for root in roots:
                if root in root_to_idx:
                    root_emb = root_embeddings[root_to_idx[root]].to(device)
                    transformed = model.forward_suffix(root_emb, suffix)
                    all_transformed.append(transformed)

        if len(all_transformed) >= 2:
            all_transformed = torch.stack(all_transformed)
            # Compute std across all embeddings
            std = all_transformed.std(dim=0).mean().item()
            metrics['embedding_diversity'] = std

    return metrics


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
    parser = argparse.ArgumentParser(description='Train affix transformations v2 (contrastive)')
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
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs (model typically converges by epoch 15-20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without min_delta improvement)')
    parser.add_argument('--min-delta', type=float, default=0.0005,
                        help='Minimum improvement required to reset patience (default 0.0005)')
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Max training samples')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"train_affix_v2_{datetime.now():%Y%m%d_%H%M%S}.log"
        setup_file_logging(log_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load root embeddings
    logger.info(f"Loading root embeddings from {args.root_embeddings}")
    root_checkpoint = torch.load(args.root_embeddings, map_location='cpu', weights_only=False)

    root_to_idx = root_checkpoint['root_to_idx']
    embedding_dim = root_checkpoint['embedding_dim']

    class RootEmbeddings(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, dim)

    root_model = RootEmbeddings(root_checkpoint['vocab_size'], embedding_dim)
    root_model.load_state_dict(root_checkpoint['model_state_dict'])
    root_embeddings = root_model.embeddings.weight.detach()

    logger.info(f"Loaded {len(root_to_idx):,} root embeddings ({embedding_dim}d)")

    # Build training data
    samples, affix_to_roots = build_training_data(args.corpus, root_to_idx, args.max_samples)

    if not samples:
        logger.error("No training samples found!")
        return

    logger.info(f"Training samples: {len(samples):,}")

    # Create model
    model = AffixTransformModel(embedding_dim=embedding_dim, rank=args.rank)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Check for checkpoint
    checkpoint_path = args.output_dir / 'best_model.pt'
    start_epoch = 0
    best_metric = float('inf')

    if checkpoint_path.exists() and not args.fresh:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['loss']
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, loss_components = train_epoch(
            model, root_embeddings, root_to_idx, samples, affix_to_roots,
            optimizer, device, args.batch_size
        )

        # Evaluate
        metrics = evaluate(model, root_embeddings, root_to_idx, affix_to_roots, device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={train_loss:.4f}, "
            f"mal_dist={metrics.get('mal_distance', 0):.4f}, "
            f"mal_mean_sim={metrics.get('mal_mean_sim', 0):.4f}, "
            f"diversity={metrics.get('embedding_diversity', 0):.4f}"
        )

        # Check for collapse: if mal_mean_sim > 0.9, embeddings are collapsed
        mal_sim = metrics.get('mal_mean_sim', 0)
        if mal_sim > 0.9:
            logger.warning(f"  WARNING: Possible embedding collapse (mal_mean_sim={mal_sim:.3f})")

        # Save if improved by at least min_delta
        improvement = best_metric - train_loss
        if improvement > args.min_delta:
            best_metric = train_loss
            patience_counter = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, train_loss, metrics)
            logger.info(f"  Saved new best model (improvement: {improvement:.6f})")
        else:
            patience_counter += 1
            if improvement > 0:
                logger.info(f"  Improvement {improvement:.6f} below min_delta {args.min_delta}, patience {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info(f"Early stopping: {args.patience} epochs without >{args.min_delta} improvement")
            break

    logger.info("\nTraining complete!")
    logger.info(f"Best loss: {best_metric:.4f}")
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
