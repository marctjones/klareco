#!/usr/bin/env python3
"""
Train affix (prefix/suffix) embeddings using co-occurrence with roots.

Phase 2 of Fundamento-Centered Training (Issue #69)

Approach:
- Extract affix-root co-occurrence from parsed ASTs in combined_training.jsonl
- Affixes that appear with similar roots should have similar embeddings
- Use trained root embeddings to bootstrap affix similarity
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


def extract_affixes_from_ast(node, prefix_to_roots: Dict[str, Set[str]],
                              suffix_to_roots: Dict[str, Set[str]],
                              prefix_vocab: Dict[str, int],
                              suffix_vocab: Dict[str, int]):
    """Recursively extract affix-root pairs from AST nodes."""
    if isinstance(node, dict):
        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            if root and len(root) >= 2:  # Skip very short roots
                # Handle prefixes (can be string or list)
                prefixes = node.get('prefiksoj', [])
                if not prefixes:
                    prefix = node.get('prefikso')
                    if prefix:
                        prefixes = [prefix]

                for p in prefixes:
                    if p and p in prefix_vocab and p != '<NONE>':
                        prefix_to_roots[p].add(root)

                # Handle suffixes
                suffixes = node.get('sufiksoj', [])
                for s in suffixes:
                    if s and s in suffix_vocab and s != '<NONE>':
                        suffix_to_roots[s].add(root)

        # Recurse into all values
        for v in node.values():
            extract_affixes_from_ast(v, prefix_to_roots, suffix_to_roots,
                                     prefix_vocab, suffix_vocab)
    elif isinstance(node, list):
        for item in node:
            extract_affixes_from_ast(item, prefix_to_roots, suffix_to_roots,
                                     prefix_vocab, suffix_vocab)


def load_root_embeddings(model_path: Path) -> Optional[Tuple[Dict[str, int], np.ndarray]]:
    """Load trained root embeddings for bootstrapping affix similarity."""
    if not model_path.exists():
        return None
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        root_to_idx = checkpoint.get('root_to_idx', {})
        embeddings = checkpoint['model_state_dict']['embeddings.weight'].numpy()
        return root_to_idx, embeddings
    except Exception as e:
        logger.warning(f"Could not load root embeddings: {e}")
        return None


def compute_affix_similarity_from_roots(
    affix1_roots: Set[str], affix2_roots: Set[str],
    root_to_idx: Dict[str, int], root_embeddings: np.ndarray
) -> float:
    """Compute similarity between affixes based on the roots they appear with."""
    # Get embeddings for roots of each affix
    def get_mean_embedding(roots):
        valid_indices = [root_to_idx[r] for r in roots if r in root_to_idx]
        if not valid_indices:
            return None
        embs = root_embeddings[valid_indices]
        mean_emb = embs.mean(axis=0)
        return mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    emb1 = get_mean_embedding(affix1_roots)
    emb2 = get_mean_embedding(affix2_roots)

    if emb1 is None or emb2 is None:
        return 0.0

    return float(np.dot(emb1, emb2))


def build_affix_pairs(prefix_vocab: Dict, suffix_vocab: Dict,
                      training_data_path: Path,
                      root_embeddings_path: Optional[Path] = None,
                      max_sentences: int = 500000) -> List[Tuple]:
    """Build training pairs from parsed ASTs in combined_training.jsonl."""
    pairs = []
    weights = []

    # Track which affixes appear with which roots
    prefix_to_roots = defaultdict(set)
    suffix_to_roots = defaultdict(set)

    # Load training data and extract affix-root co-occurrence
    logger.info(f"Loading training data from {training_data_path}...")
    sentence_count = 0
    with open(training_data_path) as f:
        for line in f:
            if sentence_count >= max_sentences:
                break
            data = json.loads(line)
            if 'ast' in data:
                extract_affixes_from_ast(data['ast'], prefix_to_roots, suffix_to_roots,
                                        prefix_vocab, suffix_vocab)
                sentence_count += 1

    logger.info(f"Processed {sentence_count} sentences")
    logger.info(f"Found {len(prefix_to_roots)} prefixes with roots")
    logger.info(f"Found {len(suffix_to_roots)} suffixes with roots")

    # Log top affixes by frequency
    prefix_counts = {p: len(roots) for p, roots in prefix_to_roots.items()}
    suffix_counts = {s: len(roots) for s, roots in suffix_to_roots.items()}
    logger.info(f"Top prefixes: {dict(sorted(prefix_counts.items(), key=lambda x: -x[1])[:5])}")
    logger.info(f"Top suffixes: {dict(sorted(suffix_counts.items(), key=lambda x: -x[1])[:5])}")

    # Load root embeddings for bootstrapping (optional but recommended)
    root_data = None
    if root_embeddings_path and root_embeddings_path.exists():
        root_data = load_root_embeddings(root_embeddings_path)
        if root_data:
            logger.info(f"Loaded {len(root_data[0])} root embeddings for bootstrapping")

    # ===== SEMANTIC AFFIX PAIRS (curated, high weight) =====
    # These are grammatically related affixes that should have similar embeddings

    # Prefix semantic groups
    SEMANTIC_PREFIX_GROUPS = [
        # Aspectual prefixes (modify how action unfolds)
        ['ek', 're', 'for'],  # begin, repeat, completely
        # Modification prefixes
        ['mal', 'mis', 'fi'],  # opposite, wrongly, morally bad
        # Relationship prefixes
        ['bo', 'ge', 'pra'],  # in-law, both genders, primordial
        # Separation prefixes
        ['dis', 'eks'],  # apart, former
    ]

    # Suffix semantic groups
    SEMANTIC_SUFFIX_GROUPS = [
        # Agent/person suffixes
        ['ul', 'ist', 'an'],  # person characterized by, professional, member
        # Verbal suffixes (causative/inchoative)
        ['ig', 'iĝ'],  # make/cause, become
        # Participial suffixes (active voice)
        ['ant', 'int', 'ont'],  # present, past, future active
        # Participial suffixes (passive voice)
        ['at', 'it', 'ot'],  # present, past, future passive
        # Modal suffixes
        ['ebl', 'ind', 'end'],  # can be, worthy of, must be
        # Degree suffixes
        ['et', 'eg'],  # diminutive, augmentative
        # Container/place suffixes
        ['ej', 'uj', 'ing'],  # place for, container, holder
        # Abstract suffixes
        ['ec', 'ism', 'aĵ'],  # quality, doctrine, concrete thing
        # Collection/unit suffixes
        ['ar', 'er'],  # collection, smallest unit
    ]

    # Add intra-group positive pairs
    for group in SEMANTIC_PREFIX_GROUPS:
        valid = [p for p in group if p in prefix_vocab]
        for i, p1 in enumerate(valid):
            for p2 in valid[i+1:]:
                pairs.append((prefix_vocab[p1], prefix_vocab[p2], 0.7, 'prefix'))
                weights.append(10.0)  # High weight for curated pairs

    for group in SEMANTIC_SUFFIX_GROUPS:
        valid = [s for s in group if s in suffix_vocab]
        for i, s1 in enumerate(valid):
            for s2 in valid[i+1:]:
                pairs.append((suffix_vocab[s1], suffix_vocab[s2], 0.7, 'suffix'))
                weights.append(10.0)

    semantic_count = len(pairs)
    logger.info(f"Created {semantic_count} semantic affix pairs (curated)")

    # ===== CO-OCCURRENCE BASED PAIRS (from data) =====
    # Affixes that appear with overlapping sets of roots → similar

    prefixes = [p for p in prefix_vocab if p != '<NONE>' and p in prefix_to_roots]
    suffixes = [s for s in suffix_vocab if s != '<NONE>' and s in suffix_to_roots]

    for i, p1 in enumerate(prefixes):
        roots1 = prefix_to_roots[p1]
        for p2 in prefixes[i+1:]:
            roots2 = prefix_to_roots[p2]
            overlap = len(roots1 & roots2)
            union = len(roots1 | roots2)
            if overlap > 0 and union > 0:
                jaccard = overlap / union
                if jaccard > 0.1:  # Meaningful overlap
                    # Use root embeddings to refine similarity if available
                    if root_data:
                        sim = compute_affix_similarity_from_roots(roots1, roots2, *root_data)
                        target = max(0.3, min(0.8, (jaccard + sim) / 2))
                    else:
                        target = max(0.3, min(0.7, jaccard))
                    pairs.append((prefix_vocab[p1], prefix_vocab[p2], target, 'prefix'))
                    weights.append(3.0)

    for i, s1 in enumerate(suffixes):
        roots1 = suffix_to_roots[s1]
        for s2 in suffixes[i+1:]:
            roots2 = suffix_to_roots[s2]
            overlap = len(roots1 & roots2)
            union = len(roots1 | roots2)
            if overlap > 0 and union > 0:
                jaccard = overlap / union
                if jaccard > 0.05:  # Lower threshold for suffixes (more common)
                    if root_data:
                        sim = compute_affix_similarity_from_roots(roots1, roots2, *root_data)
                        target = max(0.3, min(0.8, (jaccard + sim) / 2))
                    else:
                        target = max(0.3, min(0.7, jaccard))
                    pairs.append((suffix_vocab[s1], suffix_vocab[s2], target, 'suffix'))
                    weights.append(3.0)

    cooccurrence_count = len(pairs) - semantic_count
    logger.info(f"Created {cooccurrence_count} co-occurrence pairs")

    # ===== NEGATIVE PAIRS =====
    # Random pairs that should have low similarity
    positive_count = len(pairs)

    all_prefixes = [p for p in prefix_vocab if p != '<NONE>']
    all_suffixes = [s for s in suffix_vocab if s != '<NONE>']

    # Generate 2x negatives
    for _ in range(positive_count * 2):
        if len(all_prefixes) >= 2:
            p1, p2 = random.sample(all_prefixes, 2)
            pairs.append((prefix_vocab[p1], prefix_vocab[p2], 0.0, 'prefix'))
            weights.append(1.0)
        if len(all_suffixes) >= 2:
            s1, s2 = random.sample(all_suffixes, 2)
            pairs.append((suffix_vocab[s1], suffix_vocab[s2], 0.0, 'suffix'))
            weights.append(1.0)

    logger.info(f"Total pairs: {len(pairs)} (semantic={semantic_count}, cooccur={cooccurrence_count}, negative={len(pairs)-positive_count})")
    return pairs, weights


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
    parser.add_argument('--training-data', type=Path,
                        default=Path('data/training/combined_training.jsonl'),
                        help='Path to combined_training.jsonl with parsed ASTs')
    parser.add_argument('--root-embeddings', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'),
                        help='Path to trained root embeddings for bootstrapping')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/affix_embeddings'))
    parser.add_argument('--log-dir', type=Path, default=Path('logs/training'))
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-sentences', type=int, default=500000,
                        help='Max sentences to process from training data')
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

    # Check training data exists
    if not args.training_data.exists():
        logger.error(f"Training data not found: {args.training_data}")
        logger.error("Run corpus building first to generate combined_training.jsonl")
        return

    # Build pairs from parsed ASTs
    pairs, weights = build_affix_pairs(
        prefix_vocab, suffix_vocab,
        args.training_data,
        args.root_embeddings if args.root_embeddings.exists() else None,
        args.max_sentences
    )

    if args.dry_run:
        logger.info(f"\nDry run - would train on {len(pairs)} pairs")
        return

    if len(pairs) < 100:
        logger.error(f"Not enough training pairs: {len(pairs)}")
        return

    # Split with weights
    combined = list(zip(pairs, weights))
    random.shuffle(combined)
    split = int(len(combined) * 0.9)
    train_combined, val_combined = combined[:split], combined[split:]
    train_pairs = [c[0] for c in train_combined]
    val_pairs = [c[0] for c in val_combined]

    train_dataset = AffixPairDataset(train_pairs)
    val_dataset = AffixPairDataset(val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = AffixEmbeddings(len(prefix_vocab), len(suffix_vocab), args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({len(prefix_vocab)} prefixes + {len(suffix_vocab)} suffixes) × {args.embedding_dim}d")

    logger.info(f"\nStarting training on {len(train_pairs)} pairs...")
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
    logger.info(f"Model saved to: {args.output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
