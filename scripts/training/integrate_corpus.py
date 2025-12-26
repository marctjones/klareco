#!/usr/bin/env python3
"""
Integrate corpus data to refine embeddings with more co-occurrence patterns.

Phase 3 of Fundamento-Centered Training (Issue #70)

This phase:
1. Loads the trained root and affix embeddings
2. Processes the main corpus to extract more co-occurrence patterns
3. Fine-tunes embeddings with corpus signal (lower weight than Fundamento)

Output: Updates models/root_embeddings/ and models/affix_embeddings/
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


class RootEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def similarity(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        emb1 = F.normalize(self.embeddings(idx1), dim=-1)
        emb2 = F.normalize(self.embeddings(idx2), dim=-1)
        return (emb1 * emb2).sum(dim=-1)


class CorpusPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int, float, float]]):
        self.pairs = pairs  # (idx1, idx2, target, weight)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, target, weight = self.pairs[idx]
        return (
            torch.tensor(idx1, dtype=torch.long),
            torch.tensor(idx2, dtype=torch.long),
            torch.tensor(target, dtype=torch.float),
            torch.tensor(weight, dtype=torch.float),
        )


# Function words to exclude from semantic training.
# These are grammatical/structural words handled by the AST, not learned embeddings.
# Including them causes "embedding collapse" where all content words become similar
# because they all co-occur with the same high-frequency function words.
# See Wiki: "Function Word Exclusion Principle"
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns/correlatives
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Common verbs/copula
    'est', 'far', 'hav', 'pov', 'dev', 'vol', 'deb',
    # Articles/particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
    # Note: Numbers (unu, du, tri, etc.) are NOT included here.
    # Unlike true function words, numbers carry semantic content (quantity).
    # See Issue #83 for discussion.
}

# Semantic clusters for negative sampling - roots that should NOT be similar.
# Used to prevent embedding collapse by explicitly pushing apart unrelated categories.
SEMANTIC_CLUSTERS = {
    'family': ['patr', 'matr', 'fil', 'frat', 'edz', 'av', 'nev', 'onkl', 'nep'],
    'animals': ['hund', 'kat', 'bird', 'fiŝ', 'ĉeval', 'bov', 'ŝaf', 'kok', 'leon'],
    'body': ['kap', 'man', 'brak', 'okul', 'buŝ', 'nas', 'orel', 'kor', 'pied', 'fingr'],
    'time': ['tag', 'nokt', 'hor', 'jar', 'monat', 'semajn', 'minut', 'sekund'],
    'places': ['dom', 'urb', 'land', 'lok', 'ĉambr', 'strat', 'vilaĝ', 'mont', 'mar'],
    'food': ['pan', 'lakt', 'viand', 'frukt', 'legom', 'suk', 'vin', 'kaĉ'],
    'nature': ['arb', 'flor', 'herb', 'sun', 'lun', 'stel', 'nub', 'pluv', 'vent'],
}


def extract_roots_from_ast(ast: dict) -> List[str]:
    """Extract content word roots from a parsed AST, excluding function words."""
    roots = []

    def visit(node):
        if isinstance(node, dict):
            if 'radiko' in node:
                root = node['radiko']
                # Filter out function words - they don't carry semantic meaning
                if root not in FUNCTION_WORDS:
                    roots.append(root)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return roots


def build_corpus_pairs(corpus_path: Path, root_to_idx: Dict[str, int],
                       max_sentences: int = 50000) -> List[Tuple]:
    """Build training pairs from corpus co-occurrence.

    Key design decisions (see Wiki: "Function Word Exclusion Principle"):
    1. Function words are excluded - they're grammatical, not semantic
    2. Semantic cluster negatives prevent embedding collapse
    3. Graded targets based on co-occurrence frequency
    """
    pairs = []
    positive_pairs_set = set()
    cooccurrence_counts = defaultdict(int)  # Track frequency for graded targets

    logger.info(f"Processing corpus: {corpus_path}")
    logger.info(f"Excluding {len(FUNCTION_WORDS)} function words from training")

    # Build content-word-only index for negative sampling
    content_roots = [r for r in root_to_idx.keys() if r not in FUNCTION_WORDS]
    content_indices = [root_to_idx[r] for r in content_roots]
    logger.info(f"Content words for negative sampling: {len(content_indices)}")

    count = 0
    with open(corpus_path) as f:
        for line in f:
            if count >= max_sentences:
                break

            try:
                entry = json.loads(line)
                ast = entry.get('ast', {})
                # extract_roots_from_ast already filters function words
                roots = extract_roots_from_ast(ast)

                # Filter to known roots
                known_roots = [r for r in roots if r in root_to_idx]
                if len(known_roots) < 2:
                    continue

                # Co-occurring roots in same sentence → similar
                for i in range(len(known_roots)):
                    for j in range(i + 1, min(i + 5, len(known_roots))):
                        r1, r2 = known_roots[i], known_roots[j]
                        idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                        pair_key = (min(idx1, idx2), max(idx1, idx2))
                        cooccurrence_counts[pair_key] += 1
                        if pair_key not in positive_pairs_set:
                            positive_pairs_set.add(pair_key)

                count += 1
                if count % 10000 == 0:
                    logger.info(f"Processed {count} sentences, {len(positive_pairs_set)} unique pairs")

            except (json.JSONDecodeError, KeyError):
                continue

    # Convert to graded targets based on co-occurrence frequency
    # More co-occurrences → higher similarity target (but capped)
    for pair_key, count in cooccurrence_counts.items():
        idx1, idx2 = pair_key
        # Graded target: 0.5 base + 0.1 per co-occurrence, max 0.9
        target = min(0.5 + 0.1 * count, 0.9)
        pairs.append((idx1, idx2, target, 1.0))

    logger.info(f"Created {len(pairs)} corpus positive pairs (graded targets)")

    # Add semantic cluster negatives (high weight to prevent collapse)
    cluster_negative_count = 0
    cluster_names = list(SEMANTIC_CLUSTERS.keys())
    for i, name1 in enumerate(cluster_names):
        for name2 in cluster_names[i+1:]:
            roots1 = [r for r in SEMANTIC_CLUSTERS[name1] if r in root_to_idx]
            roots2 = [r for r in SEMANTIC_CLUSTERS[name2] if r in root_to_idx]
            for r1 in roots1:
                for r2 in roots2:
                    idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                    pair_key = (min(idx1, idx2), max(idx1, idx2))
                    if pair_key not in positive_pairs_set:
                        # Weight=3.0 to strongly push apart different categories
                        pairs.append((idx1, idx2, 0.0, 3.0))
                        cluster_negative_count += 1

    logger.info(f"Added {cluster_negative_count} semantic cluster negatives (weight=3.0)")

    # Add random negatives (content words only, no function words)
    random_negative_count = 0
    target_negatives = len(positive_pairs_set)  # Match positive count
    while random_negative_count < target_negatives:
        idx1, idx2 = random.sample(content_indices, 2)
        pair_key = (min(idx1, idx2), max(idx1, idx2))
        if pair_key not in positive_pairs_set:
            pairs.append((idx1, idx2, 0.0, 1.0))
            random_negative_count += 1

    logger.info(f"Added {random_negative_count} random negatives (content words only)")
    logger.info(f"Total corpus pairs: {len(pairs)}")
    return pairs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for idx1, idx2, targets, weights in dataloader:
        idx1, idx2, targets, weights = [x.to(device) for x in (idx1, idx2, targets, weights)]

        preds = model.similarity(idx1, idx2)
        loss = (weights * (preds - targets) ** 2).mean()

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
        for idx1, idx2, targets, weights in dataloader:
            idx1, idx2, targets, weights = [x.to(device) for x in (idx1, idx2, targets, weights)]

            preds = model.similarity(idx1, idx2)
            loss = (weights * (preds - targets) ** 2).mean()
            total_loss += loss.item()

            pred_class = (preds > 0.5).float()
            target_class = (targets > 0.5).float()
            correct += (pred_class == target_class).sum().item()
            total += len(targets)

    return total_loss / len(dataloader), correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Integrate corpus for embedding refinement')
    parser.add_argument('--corpus', type=Path,
                        default=Path('data/corpus_with_sources_v2.jsonl'))
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/root_embeddings'))
    parser.add_argument('--log-dir', type=Path, default=Path('logs/training'))
    parser.add_argument('--max-sentences', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'integrate_corpus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 3: Corpus Integration")
    logger.info("=" * 60)

    # Load pre-trained root embeddings
    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        logger.error("Run Phase 2 (root embedding training) first")
        sys.exit(1)

    checkpoint = torch.load(args.root_model)
    root_to_idx = checkpoint['root_to_idx']
    embedding_dim = checkpoint['embedding_dim']
    logger.info(f"Loaded root embeddings: {len(root_to_idx)} roots, {embedding_dim}d")

    # Check corpus
    if not args.corpus.exists():
        logger.warning(f"Corpus not found: {args.corpus}")
        logger.info("Skipping corpus integration (no corpus available)")
        return

    # Build corpus pairs
    pairs = build_corpus_pairs(args.corpus, root_to_idx, args.max_sentences)

    if args.dry_run:
        logger.info(f"\nDry run - would fine-tune on {len(pairs)} corpus pairs")
        return

    if len(pairs) < 100:
        logger.warning("Too few corpus pairs, skipping fine-tuning")
        return

    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RootEmbeddings(len(root_to_idx), embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Use lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Data
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    train_loader = DataLoader(CorpusPairDataset(train_pairs), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CorpusPairDataset(val_pairs), batch_size=args.batch_size)

    logger.info(f"\nFine-tuning on corpus...")
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, accuracy = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # Save refined model
            refined_checkpoint = {
                'epoch': checkpoint['epoch'] + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'correlation': accuracy,
                'embedding_dim': embedding_dim,
                'vocab_size': len(root_to_idx),
                'root_to_idx': root_to_idx,
                'idx_to_root': checkpoint['idx_to_root'],
                'corpus_refined': True,
            }
            torch.save(refined_checkpoint, args.output_dir / 'best_model.pt')
            logger.info(f"Saved refined model (accuracy: {accuracy:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping")
            break

    logger.info(f"\nCorpus integration complete! Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
