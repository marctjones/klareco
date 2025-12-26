#!/usr/bin/env python3
"""
Train affix (prefix/suffix) embeddings using co-occurrence with roots.

Phase 2 of Fundamento-Centered Training (Issue #69)

Approach:
- Affixes that appear with similar roots should have similar embeddings
- Use Ekzercaro parsed sentences to find affix-root patterns
- Train with contrastive loss similar to root embeddings

Output: models/affix_embeddings/best_model.pt
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

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


def setup_file_logging(log_path: Path):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


class AffixEmbeddings(nn.Module):
    """Learnable embeddings for Esperanto prefixes and suffixes."""

    def __init__(self, num_prefixes: int, num_suffixes: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.prefix_embeddings = nn.Embedding(num_prefixes, embedding_dim)
        self.suffix_embeddings = nn.Embedding(num_suffixes, embedding_dim)
        nn.init.normal_(self.prefix_embeddings.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.suffix_embeddings.weight, mean=0.0, std=0.1)

    def forward_prefix(self, indices: torch.Tensor) -> torch.Tensor:
        return self.prefix_embeddings(indices)

    def forward_suffix(self, indices: torch.Tensor) -> torch.Tensor:
        return self.suffix_embeddings(indices)

    def prefix_similarity(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        emb1 = F.normalize(self.prefix_embeddings(idx1), dim=-1)
        emb2 = F.normalize(self.prefix_embeddings(idx2), dim=-1)
        return (emb1 * emb2).sum(dim=-1)

    def suffix_similarity(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        emb1 = F.normalize(self.suffix_embeddings(idx1), dim=-1)
        emb2 = F.normalize(self.suffix_embeddings(idx2), dim=-1)
        return (emb1 * emb2).sum(dim=-1)


class AffixPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int, float, str]]):
        """pairs = [(idx1, idx2, target, affix_type), ...]"""
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, target, affix_type = self.pairs[idx]
        return (
            torch.tensor(idx1, dtype=torch.long),
            torch.tensor(idx2, dtype=torch.long),
            torch.tensor(target, dtype=torch.float),
            affix_type
        )


def collate_fn(batch):
    """Custom collate to handle string affix_type."""
    idx1 = torch.stack([b[0] for b in batch])
    idx2 = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    types = [b[3] for b in batch]
    return idx1, idx2, targets, types


def build_affix_pairs(prefix_vocab: Dict, suffix_vocab: Dict,
                      ekzercaro: List[dict]) -> List[Tuple]:
    """Build training pairs based on affix co-occurrence patterns."""
    pairs = []

    # Track which affixes appear with which roots
    prefix_to_roots = defaultdict(set)
    suffix_to_roots = defaultdict(set)

    for sent in ekzercaro:
        roots = sent.get('roots', [])
        # Heuristic: check if roots contain known affixes
        for root in roots:
            for prefix in prefix_vocab:
                if prefix != '<NONE>' and root.startswith(prefix):
                    prefix_to_roots[prefix].add(root)
            for suffix in suffix_vocab:
                if suffix != '<NONE>' and root.endswith(suffix):
                    suffix_to_roots[suffix].add(root)

    logger.info(f"Found {len(prefix_to_roots)} prefixes with roots")
    logger.info(f"Found {len(suffix_to_roots)} suffixes with roots")

    # Prefixes sharing roots → similar
    prefixes = [p for p in prefix_vocab if p != '<NONE>']
    for i, p1 in enumerate(prefixes):
        roots1 = prefix_to_roots.get(p1, set())
        if not roots1:
            continue
        for p2 in prefixes[i+1:]:
            roots2 = prefix_to_roots.get(p2, set())
            if roots1 & roots2:  # Shared roots
                pairs.append((prefix_vocab[p1], prefix_vocab[p2], 1.0, 'prefix'))

    # Suffixes sharing roots → similar
    suffixes = [s for s in suffix_vocab if s != '<NONE>']
    for i, s1 in enumerate(suffixes):
        roots1 = suffix_to_roots.get(s1, set())
        if not roots1:
            continue
        for s2 in suffixes[i+1:]:
            roots2 = suffix_to_roots.get(s2, set())
            if roots1 & roots2:
                pairs.append((suffix_vocab[s1], suffix_vocab[s2], 1.0, 'suffix'))

    # Known semantic affix pairs (from Esperanto grammar)
    semantic_prefix_pairs = [
        ('mal', 're'),  # opposite, repeat - both modify meaning
        ('ek', 'dis'),  # begin, apart - both aspectual
        ('ge', 'pra'),  # both genders, ancestral - both categorical
    ]
    for p1, p2 in semantic_prefix_pairs:
        if p1 in prefix_vocab and p2 in prefix_vocab:
            pairs.append((prefix_vocab[p1], prefix_vocab[p2], 1.0, 'prefix'))

    semantic_suffix_pairs = [
        ('ig', 'iĝ'),   # causative, inchoative - both verbal
        ('ant', 'int'), # present, past participle
        ('ebl', 'ind'), # can be, should be - both modal
        ('ej', 'uj'),   # place, container - both locative
        ('et', 'eg'),   # diminutive, augmentative - both degree
    ]
    for s1, s2 in semantic_suffix_pairs:
        if s1 in suffix_vocab and s2 in suffix_vocab:
            pairs.append((suffix_vocab[s1], suffix_vocab[s2], 1.0, 'suffix'))

    positive_count = len(pairs)
    logger.info(f"Created {positive_count} positive affix pairs")

    # Add negative pairs
    for _ in range(positive_count):
        # Random prefix pair
        p1, p2 = random.sample(prefixes, 2)
        pairs.append((prefix_vocab[p1], prefix_vocab[p2], 0.0, 'prefix'))
        # Random suffix pair
        s1, s2 = random.sample(suffixes, 2)
        pairs.append((suffix_vocab[s1], suffix_vocab[s2], 0.0, 'suffix'))

    logger.info(f"Total pairs: {len(pairs)}")
    return pairs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for idx1, idx2, targets, types in dataloader:
        idx1, idx2, targets = idx1.to(device), idx2.to(device), targets.to(device)

        # Compute similarities based on type
        preds = []
        for i, t in enumerate(types):
            if t == 'prefix':
                pred = model.prefix_similarity(idx1[i:i+1], idx2[i:i+1])
            else:
                pred = model.suffix_similarity(idx1[i:i+1], idx2[i:i+1])
            preds.append(pred)

        preds = torch.cat(preds)
        loss = F.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx1, idx2, targets, types in dataloader:
            idx1, idx2, targets = idx1.to(device), idx2.to(device), targets.to(device)

            preds = []
            for i, t in enumerate(types):
                if t == 'prefix':
                    pred = model.prefix_similarity(idx1[i:i+1], idx2[i:i+1])
                else:
                    pred = model.suffix_similarity(idx1[i:i+1], idx2[i:i+1])
                preds.append(pred)

            preds = torch.cat(preds)
            loss = F.mse_loss(preds, targets)
            total_loss += loss.item()

            # Accuracy
            pred_class = (preds > 0.5).float()
            target_class = (targets > 0.5).float()
            correct += (pred_class == target_class).sum().item()
            total += len(targets)

    return total_loss / len(dataloader), correct / total if total > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, accuracy, prefix_vocab, suffix_vocab,
                    output_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'prefix_vocab': prefix_vocab,
        'suffix_vocab': suffix_vocab,
        'embedding_dim': model.embedding_dim,
    }
    temp_path = output_dir / 'checkpoint.pt.tmp'
    try:
        torch.save(checkpoint, temp_path)
        temp_path.rename(output_dir / 'checkpoint.pt')
        if is_best:
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"Saved new best model (accuracy: {accuracy:.4f})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train affix embeddings')
    parser.add_argument('--affix-vocab', type=Path,
                        default=Path('data/vocabularies/affix_vocabulary.json'))
    parser.add_argument('--ekzercaro', type=Path,
                        default=Path('data/training/ekzercaro_sentences.jsonl'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/affix_embeddings'))
    parser.add_argument('--log-dir', type=Path, default=Path('logs/training'))
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'train_affix_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 2: Affix Embedding Training")
    logger.info("=" * 60)

    # Load vocabularies
    with open(args.affix_vocab) as f:
        affix_data = json.load(f)
    prefix_vocab = affix_data['prefixes']
    suffix_vocab = affix_data['suffixes']
    logger.info(f"Prefixes: {len(prefix_vocab)}, Suffixes: {len(suffix_vocab)}")

    # Load Ekzercaro
    ekzercaro = []
    if args.ekzercaro.exists():
        with open(args.ekzercaro) as f:
            ekzercaro = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(ekzercaro)} Ekzercaro sentences")

    # Build pairs
    pairs = build_affix_pairs(prefix_vocab, suffix_vocab, ekzercaro)

    if args.dry_run:
        logger.info(f"\nDry run - would train on {len(pairs)} pairs")
        return

    # Split
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    train_dataset = AffixPairDataset(train_pairs)
    val_dataset = AffixPairDataset(val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AffixEmbeddings(len(prefix_vocab), len(suffix_vocab), args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info(f"\nStarting training...")
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, accuracy = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")

        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch+1, val_loss, accuracy,
                        prefix_vocab, suffix_vocab, args.output_dir, is_best)

        if epoch >= 20 and patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break

    logger.info(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
