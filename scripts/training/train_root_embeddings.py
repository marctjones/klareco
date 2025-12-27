#!/usr/bin/env python3
"""
Train root embeddings using Fundamento-Centered approach.

Phases 0-1 of Fundamento-Centered Training (Issues #65-67, #73)

This script trains root embeddings using:
1. Fundamento UV as semantic anchors (highest weight)
2. PV definition similarity (definition roots → embedding similarity)
3. Ekzercaro co-occurrence (Zamenhof's curated examples)

Approach:
- Initialize embeddings for all Fundamento roots
- Use definition overlap to create similarity targets
- Train with contrastive loss weighted by source authority

Output: models/root_embeddings/
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
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
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


class RootEmbeddings(nn.Module):
    """
    Learnable embeddings for Esperanto roots.

    These encode semantic meaning only - grammatical features are added separately
    as frozen deterministic vectors in the full compositional system.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with larger variance to spread embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.5)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embeddings(indices)

    def get_normalized(self, indices: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized embeddings."""
        return F.normalize(self.embeddings(indices), dim=-1)

    def similarity(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two root embeddings."""
        emb1 = self.get_normalized(idx1)
        emb2 = self.get_normalized(idx2)
        return (emb1 * emb2).sum(dim=-1)


class DefinitionSimilarityDataset(Dataset):
    """
    Dataset of (root1, root2, target_similarity) pairs.

    Similarity targets are computed from:
    1. Shared definition roots (PV)
    2. Co-occurrence in Ekzercaro
    3. Translation alignment (Fundamento UV)
    """

    def __init__(self, pairs: List[Tuple[int, int, float]], weights: List[float]):
        self.pairs = pairs
        self.weights = weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        root1, root2, target = self.pairs[idx]
        weight = self.weights[idx]
        return (
            torch.tensor(root1, dtype=torch.long),
            torch.tensor(root2, dtype=torch.long),
            torch.tensor(target, dtype=torch.float),
            torch.tensor(weight, dtype=torch.float)
        )


def build_vocabulary(fundamento_roots: dict, revo_entries: dict,
                     ekzercaro: List[dict],
                     clean_vocab_path: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build root vocabulary from all sources.

    If clean_vocab_path is provided, use only roots from that file (recommended).
    Otherwise, build from sources (may include junk from Ekzercaro extraction).
    """
    # Use clean vocabulary if provided (RECOMMENDED)
    if clean_vocab_path and clean_vocab_path.exists():
        logger.info(f"Using clean vocabulary from {clean_vocab_path}")
        with open(clean_vocab_path) as f:
            clean_data = json.load(f)

        all_roots = set(clean_data['roots'].keys())
        logger.info(f"Clean vocabulary: {len(all_roots)} validated roots")
        logger.info(f"  Tier 1 (Fundamento): {clean_data['metadata']['tiers'].get('1', clean_data['metadata']['tiers'].get(1, 0))}")
        logger.info(f"  Tier 2 (Core): {clean_data['metadata']['tiers'].get('2', clean_data['metadata']['tiers'].get(2, 0))}")
        logger.info(f"  Tier 3 (Extended): {clean_data['metadata']['tiers'].get('3', clean_data['metadata']['tiers'].get(3, 0))}")
    else:
        # Legacy mode: build from sources (may include junk)
        logger.warning("No clean vocabulary provided - building from sources (may include junk)")
        all_roots = set()

        # Fundamento roots (highest priority)
        all_roots.update(fundamento_roots.keys())
        logger.info(f"Fundamento roots: {len(fundamento_roots)}")

        # ReVo headwords and definition roots
        for headword, data in revo_entries.items():
            all_roots.add(headword)
            all_roots.update(data.get('definition_roots', []))
        logger.info(f"After ReVo: {len(all_roots)} unique roots")

        # Ekzercaro roots - SKIP to avoid junk
        # for sent in ekzercaro:
        #     all_roots.update(sent.get('roots', []))
        logger.info(f"Skipping Ekzercaro roots (use --clean-vocab instead)")

    # Build vocab
    root_to_idx = {root: idx for idx, root in enumerate(sorted(all_roots))}
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    return root_to_idx, idx_to_root


def compute_jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def build_similarity_pairs(fundamento_roots: dict, revo_entries: dict,
                           ekzercaro: List[dict], root_to_idx: dict,
                           negative_ratio: int = 3,
                           use_revo: bool = True,
                           use_hard_negatives: bool = False,
                           revo_relations_path: Optional[Path] = None) -> Tuple[List, List]:
    """
    Build training pairs with GRADED similarity targets.

    Uses curated semantic relations from authoritative Esperanto sources:
    1. ReVo synonyms, antonyms, hypernyms (human-curated by lexicographers)
    2. Fundamento translation equivalence (Zamenhof's original definitions)
    3. Ekzercaro co-occurrence (Zamenhof's curated examples)

    Key design:
    - Curated relations > weak Jaccard overlap
    - Graded targets (0.0-1.0) based on relation type
    - Function words filtered to prevent collapse
    - Negative sampling for separation

    Returns: (pairs, weights) where pairs = [(idx1, idx2, target_sim), ...]
    """
    pairs = []
    weights = []
    pair_targets = {}  # (idx1, idx2) -> max target seen

    # Function words to filter out - they don't carry semantic meaning
    # These appear in almost every definition and cause embedding collapse
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

    # Track co-occurrence counts for graded scoring
    cooccurrence_count = defaultdict(int)  # pair -> count
    root_sentence_count = defaultdict(int)  # root -> sentence count
    root_freq = defaultdict(int)  # track overall frequency for weighting

    # =========================================================================
    # 1. Ekzercaro co-occurrence with GRADED targets
    # =========================================================================
    logger.info("Building Ekzercaro co-occurrence pairs (graded)...")

    for sent in ekzercaro:
        # Filter out function words from roots
        roots = [r for r in sent.get('roots', []) if r in root_to_idx and r not in FUNCTION_WORDS]
        for r in roots:
            root_sentence_count[r] += 1
            root_freq[r] += 1

        if len(roots) < 2:
            continue

        # Only pair adjacent or nearby roots (within window of 3)
        for i in range(len(roots)):
            for j in range(i + 1, min(i + 4, len(roots))):
                r1, r2 = roots[i], roots[j]
                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))
                cooccurrence_count[pair_key] += 1

    # Convert co-occurrence to similarity scores
    ekz_pair_count = 0
    for pair_key, count in cooccurrence_count.items():
        idx1, idx2 = pair_key
        # Graded target: more co-occurrences = higher similarity
        # Use log scaling to prevent very common pairs from dominating
        target = min(0.9, 0.3 + 0.2 * math.log1p(count))  # Range: 0.3-0.9

        if pair_key not in pair_targets or target > pair_targets[pair_key]:
            pair_targets[pair_key] = target
            pairs.append((idx1, idx2, target))
            weights.append(10.0)  # Ekzercaro weight
            ekz_pair_count += 1

    logger.info(f"Created {ekz_pair_count} Ekzercaro pairs (graded 0.3-0.9)")

    # =========================================================================
    # 2. ReVo definition Jaccard similarity (baseline - this worked well)
    # =========================================================================
    revo_jaccard_count = 0
    if use_revo:
        logger.info("Building ReVo definition similarity pairs (Jaccard-graded)...")

        # Build definition root sets for each headword, FILTERING function words
        headword_def_roots = {}
        for headword, data in revo_entries.items():
            if headword not in root_to_idx:
                continue
            if headword in FUNCTION_WORDS:
                continue
            # Filter function words from definition roots
            def_roots = set(r for r in data.get('definition_roots', []) if r not in FUNCTION_WORDS)
            if len(def_roots) >= 3:
                headword_def_roots[headword] = def_roots
                root_freq[headword] += 1
                for r in def_roots:
                    root_freq[r] += 1

        headwords = list(headword_def_roots.keys())
        logger.info(f"  Processing {len(headwords)} headwords with definition roots...")

        for i, h1 in enumerate(headwords):
            def1 = headword_def_roots[h1]
            for h2 in headwords[i+1:]:
                def2 = headword_def_roots[h2]

                # Compute Jaccard similarity of definition roots
                jaccard = compute_jaccard(def1, def2)

                # Only create pair if significant overlap
                if jaccard >= 0.20:  # At least 20% overlap
                    idx1, idx2 = root_to_idx[h1], root_to_idx[h2]
                    pair_key = (min(idx1, idx2), max(idx1, idx2))

                    # Scale Jaccard to target range 0.4-0.8
                    target = 0.4 + 0.4 * jaccard

                    if pair_key not in pair_targets or target > pair_targets[pair_key]:
                        pair_targets[pair_key] = target
                        pairs.append((idx1, idx2, target))
                        weights.append(2.0 + 3.0 * jaccard)
                        revo_jaccard_count += 1

        logger.info(f"Created {revo_jaccard_count} ReVo Jaccard pairs (graded 0.4-0.8)")

    # =========================================================================
    # 2b. ReVo curated semantic relations (BONUS - high-weight refinement)
    # =========================================================================
    revo_relation_count = 0
    revo_antonym_count = 0

    if use_revo and revo_relations_path and revo_relations_path.exists():
        logger.info("Adding ReVo curated semantic relations (bonus)...")

        with open(revo_relations_path) as f:
            revo_rels = json.load(f)

        # Curated relations get BONUS weight on top of Jaccard
        # Targets are moderate (not too high) to work with MSE loss
        RELATION_TARGETS = {
            'synonym': (0.75, 8.0),      # Similar - bonus weight
            'hypernym': (0.60, 5.0),     # X is-a Y (hundo→besto)
            'hyponym': (0.60, 5.0),      # Y is-a X (besto→hundo)
            'part_of': (0.55, 4.0),      # X is-part-of Y (fingro→mano)
            'has_part': (0.55, 4.0),     # Y has-part X (mano→fingro)
            'antonym': (0.10, 6.0),      # Opposites - low similarity
        }

        for rel_type, (target, weight) in RELATION_TARGETS.items():
            rel_pairs = revo_rels.get('relations', {}).get(rel_type, [])
            count = 0

            for r1, r2 in rel_pairs:
                if r1 not in root_to_idx or r2 not in root_to_idx:
                    continue
                if r1 in FUNCTION_WORDS or r2 in FUNCTION_WORDS:
                    continue

                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))

                # For antonyms, we want low similarity (don't override with higher)
                if rel_type == 'antonym':
                    if pair_key not in pair_targets:
                        pair_targets[pair_key] = target
                        pairs.append((idx1, idx2, target))
                        weights.append(weight)
                        count += 1
                        revo_antonym_count += 1
                else:
                    # For positive relations, take max target
                    if pair_key not in pair_targets or target > pair_targets[pair_key]:
                        pair_targets[pair_key] = target
                        pairs.append((idx1, idx2, target))
                        weights.append(weight)
                        count += 1
                        revo_relation_count += 1

            logger.info(f"  {rel_type}: {count} pairs (target={target:.2f})")

        logger.info(f"Created {revo_relation_count} ReVo curated pairs, {revo_antonym_count} antonym pairs")

    elif use_revo and revo_relations_path:
        logger.info("ReVo relations file not found - using Jaccard only")

    # =========================================================================
    # 3. Fundamento translation overlap (graded)
    # =========================================================================
    logger.info("Building Fundamento translation pairs (graded)...")

    fund_pair_count = 0
    fund_roots = list(fundamento_roots.keys())

    for i, r1 in enumerate(fund_roots):
        if r1 not in root_to_idx:
            continue
        trans1 = set(v.lower().strip() for v in fundamento_roots[r1].get('translations', {}).values() if v)

        for r2 in fund_roots[i+1:]:
            if r2 not in root_to_idx:
                continue
            trans2 = set(v.lower().strip() for v in fundamento_roots[r2].get('translations', {}).values() if v)

            # Compute translation overlap
            overlap = compute_jaccard(trans1, trans2)

            if overlap > 0:
                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))

                # Target based on overlap: 0.5-0.95
                target = 0.5 + 0.45 * overlap

                if pair_key not in pair_targets or target > pair_targets[pair_key]:
                    pair_targets[pair_key] = target
                    pairs.append((idx1, idx2, target))
                    weights.append(5.0)  # Fundamento weight
                    fund_pair_count += 1

    logger.info(f"Created {fund_pair_count} Fundamento translation pairs")

    total_positive = len(pairs)
    positive_pairs_set = set(pair_targets.keys())
    logger.info(f"Total positive pairs: {total_positive}")

    # =========================================================================
    # 4. HARD NEGATIVES - medium similarity targets (optional)
    # =========================================================================
    hard_neg_count = 0
    if use_hard_negatives:
        logger.info("Building hard negative pairs...")

        # Group roots by first letter for hard negatives
        roots_by_prefix = defaultdict(list)
        idx_to_root = {idx: root for root, idx in root_to_idx.items()}
        for root, idx in root_to_idx.items():
            if len(root) >= 2:
                prefix = root[:2]
                roots_by_prefix[prefix].append(idx)

        target_hard_neg = total_positive // 2  # Half as many hard negatives

        prefixes = list(roots_by_prefix.keys())
        attempts = 0
        max_attempts = target_hard_neg * 20

        while hard_neg_count < target_hard_neg and attempts < max_attempts:
            attempts += 1

            # Pick a random prefix with enough roots
            prefix = random.choice(prefixes)
            if len(roots_by_prefix[prefix]) < 2:
                continue

            idx1, idx2 = random.sample(roots_by_prefix[prefix], 2)
            pair_key = (min(idx1, idx2), max(idx1, idx2))

            if pair_key in positive_pairs_set:
                continue

            # Hard negative: same prefix suggests some relation, but target is low
            pairs.append((idx1, idx2, 0.15))  # Low but not zero
            weights.append(2.0)  # Medium weight
            positive_pairs_set.add(pair_key)
            hard_neg_count += 1

        logger.info(f"Created {hard_neg_count} hard negative pairs (target=0.15)")
    else:
        logger.info("Skipping hard negatives (use_hard_negatives=False)")

    # =========================================================================
    # 5. SEMANTIC CLUSTER PAIRS - both positive (within) and negative (between)
    # =========================================================================
    # Key insight: We need POSITIVE pairs within clusters to pull related words together
    # AND negative pairs between clusters to push unrelated words apart.
    # Without intra-cluster positives, animals won't cluster even if they're all
    # pushed away from non-animals - they have no attraction to each other.

    SEMANTIC_CLUSTERS = {
        'family': ['patr', 'matr', 'fil', 'frat', 'edz', 'av', 'nev', 'onkl', 'kuzo', 'nep'],
        'animals': ['hund', 'kat', 'bird', 'fiŝ', 'ĉeval', 'bov', 'ŝaf', 'kok', 'mus', 'leon', 'tigr', 'elefant'],
        'body': ['kap', 'man', 'brak', 'okul', 'buŝ', 'nas', 'orel', 'kor', 'pied', 'fingr', 'dent', 'har'],
        'time': ['tag', 'nokt', 'hor', 'jar', 'monat', 'semajn', 'minut', 'sekund', 'moment'],
        'places': ['dom', 'urb', 'land', 'lok', 'ĉambr', 'strat', 'vilaĝ', 'mont', 'mar', 'river', 'arb'],
        'actions': ['ir', 'ven', 'kur', 'paŝ', 'salt', 'naĝ', 'flug', 'sid', 'star', 'kuŝ'],
        'food': ['manĝ', 'trink', 'pan', 'akv', 'vand', 'lakt', 'viand', 'frukt', 'legom', 'suk'],
        'abstract': ['am', 'ide', 'pens', 'sci', 'sent', 'vol', 'kred', 'esper', 'tim', 'ĝoj'],
        'objects': ['tabl', 'seĝ', 'lit', 'libr', 'paper', 'krajpn', 'teler', 'glaso', 'kuler'],
        'qualities': ['bon', 'bel', 'grand', 'jun', 'nov', 'alt', 'larg', 'long', 'fort', 'rapid'],
        # Additional semantic groups for better coverage
        'communication': ['parol', 'dir', 'skrib', 'leg', 'aŭd', 'demand', 'respond', 'kri', 'kant'],
        'colors': ['blank', 'nigr', 'ruĝ', 'blu', 'verd', 'flav', 'brun', 'griz', 'oranĝ', 'violet'],
        'nature': ['sun', 'lun', 'stel', 'ĉiel', 'nub', 'pluv', 'neĝ', 'vent', 'ter', 'fajr'],
        'containers': ['sak', 'skatol', 'barel', 'botel', 'kruĉ', 'poŝ', 'kest', 'ujo'],
    }

    # -------------------------------------------------------------------------
    # 5a. INTRA-CLUSTER POSITIVES - pull same-category words together
    # -------------------------------------------------------------------------
    semantic_pos_count = 0
    logger.info("Building semantic cluster POSITIVE pairs (intra-cluster)...")

    for cluster_name, roots in SEMANTIC_CLUSTERS.items():
        valid_roots = [r for r in roots if r in root_to_idx and r not in FUNCTION_WORDS]
        if len(valid_roots) < 2:
            continue

        # Create positive pairs for all combinations within the cluster
        for i, r1 in enumerate(valid_roots):
            idx1 = root_to_idx[r1]
            for r2 in valid_roots[i+1:]:
                idx2 = root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))

                # Same-category words should have moderate-high similarity
                # Not 0.9 (they're not synonyms) but definitely related
                target = 0.45  # Moderate positive - "related but not identical"

                if pair_key not in pair_targets or target > pair_targets[pair_key]:
                    pair_targets[pair_key] = target
                    pairs.append((idx1, idx2, target))
                    weights.append(6.0)  # High weight to ensure clustering
                    positive_pairs_set.add(pair_key)
                    semantic_pos_count += 1

    logger.info(f"Created {semantic_pos_count} semantic cluster POSITIVE pairs (target=0.45)")

    # -------------------------------------------------------------------------
    # 5b. INTER-CLUSTER NEGATIVES - push different-category words apart
    # -------------------------------------------------------------------------
    semantic_neg_count = 0
    cluster_names = list(SEMANTIC_CLUSTERS.keys())
    logger.info("Building semantic cluster NEGATIVE pairs (inter-cluster)...")

    # Create negative pairs between different semantic clusters
    # This ensures semantically unrelated content words are pushed apart
    for i, name1 in enumerate(cluster_names):
        roots1 = [r for r in SEMANTIC_CLUSTERS[name1] if r in root_to_idx and r not in FUNCTION_WORDS]
        for name2 in cluster_names[i+1:]:
            roots2 = [r for r in SEMANTIC_CLUSTERS[name2] if r in root_to_idx and r not in FUNCTION_WORDS]

            # Create negative pairs between all roots in different clusters
            for r1 in roots1:
                idx1 = root_to_idx[r1]
                for r2 in roots2:
                    idx2 = root_to_idx[r2]
                    pair_key = (min(idx1, idx2), max(idx1, idx2))

                    if pair_key in positive_pairs_set:
                        continue

                    # Strong negative: different semantic domains should have low similarity
                    pairs.append((idx1, idx2, 0.0))
                    weights.append(5.0)  # High weight to counter-balance the collapse
                    positive_pairs_set.add(pair_key)
                    semantic_neg_count += 1

    logger.info(f"Created {semantic_neg_count} semantic cluster negative pairs (target=0.0)")

    # =========================================================================
    # 6. EASY NEGATIVES - truly random pairs (target = 0.0)
    # =========================================================================
    logger.info("Building easy negative pairs...")

    # Filter to content words only for easy negatives
    content_indices = [idx for root, idx in root_to_idx.items() if root not in FUNCTION_WORDS]
    easy_neg_count = 0
    target_easy_neg = total_positive * negative_ratio - hard_neg_count - semantic_neg_count

    attempts = 0
    max_attempts = target_easy_neg * 10

    while easy_neg_count < target_easy_neg and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = random.sample(content_indices, 2)
        pair_key = (min(idx1, idx2), max(idx1, idx2))

        if pair_key in positive_pairs_set:
            continue

        pairs.append((idx1, idx2, 0.0))  # True negative
        weights.append(1.0)
        positive_pairs_set.add(pair_key)
        easy_neg_count += 1

    total_negative = hard_neg_count + semantic_neg_count + easy_neg_count
    logger.info(f"Created {easy_neg_count} easy negative pairs (target=0.0)")
    logger.info(f"Total training pairs: {len(pairs)}")
    logger.info(f"Positive:Negative ratio = {total_positive}:{total_negative} (1:{total_negative/total_positive:.1f})")

    return pairs, weights


def graded_contrastive_loss(pred_sim: torch.Tensor, target: torch.Tensor,
                            margin: float = 0.2) -> torch.Tensor:
    """
    Contrastive loss for graded similarity targets.

    For each pair:
    - If target > 0.5 (positive): push pred toward target
    - If target < 0.2 (negative): push pred below (target + margin)
    - If 0.2 <= target <= 0.5 (hard negative): push pred toward target

    This creates a smooth loss landscape that respects graded targets.
    """
    # Direct regression toward target for all pairs
    regression_loss = (pred_sim - target) ** 2

    # Additional margin loss for negatives to ensure separation
    negative_mask = target < 0.2
    if negative_mask.any():
        neg_pred = pred_sim[negative_mask]
        neg_target = target[negative_mask]
        # Push predictions below (target + margin)
        margin_violation = F.relu(neg_pred - (neg_target + margin))
        margin_loss = margin_violation ** 2

        # Combine: regression + extra penalty for margin violation
        total_loss = regression_loss.mean() + 0.5 * margin_loss.mean()
    else:
        total_loss = regression_loss.mean()

    return total_loss


def train_epoch(model: RootEmbeddings, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                margin: float = 0.2) -> float:
    """
    Train for one epoch using direct regression + attraction/repulsion.

    Key insight: Use MSE to target values, but add explicit push/pull terms
    to ensure positive pairs are pulled together even when pred=0.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        idx1, idx2, target_sim, weight = [b.to(device) for b in batch]

        # Get embeddings (not just similarities)
        emb1 = model.get_normalized(idx1)  # (batch, dim)
        emb2 = model.get_normalized(idx2)  # (batch, dim)

        # Compute cosine similarity manually
        pred_sim = (emb1 * emb2).sum(dim=1)  # (batch,)

        # Split by target type
        pos_mask = target_sim >= 0.3
        neg_mask = target_sim < 0.2

        # Direct MSE loss toward target for all pairs
        mse_loss = ((pred_sim - target_sim) ** 2 * weight).mean()

        # Additional explicit loss terms
        pos_loss = torch.tensor(0.0, device=device)
        neg_loss = torch.tensor(0.0, device=device)

        # For positives: maximize (1 - distance) = maximize similarity
        if pos_mask.any():
            pos_pred = pred_sim[pos_mask]
            pos_target = target_sim[pos_mask]
            # Hinge loss: penalize if pred < target - margin
            pos_violation = F.relu(pos_target - margin - pos_pred)
            pos_loss = pos_violation.mean()

        # For negatives: ensure pred stays below threshold
        if neg_mask.any():
            neg_pred = pred_sim[neg_mask]
            neg_target = target_sim[neg_mask]
            # Hinge loss: penalize if pred > target + margin
            neg_violation = F.relu(neg_pred - neg_target - margin)
            neg_loss = neg_violation.mean()

        # Combined loss with strong emphasis on structure
        # MSE handles regression, hinge ensures separation
        loss = 0.3 * mse_loss + 0.35 * pos_loss + 0.35 * neg_loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model: RootEmbeddings, dataloader: DataLoader,
             device: torch.device) -> Tuple[float, float, float, float, float]:
    """
    Evaluate model on validation set.

    Returns: (loss, accuracy, correlation, pos_avg, neg_avg)
    - accuracy: classification accuracy (pred > 0.4 vs target > 0.4)
    - correlation: Pearson correlation between pred and target
    - pos_avg: average prediction for positive pairs (target >= 0.3)
    - neg_avg: average prediction for negative pairs (target < 0.3)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            idx1, idx2, target_sim, weight = [b.to(device) for b in batch]

            pred_sim = model.similarity(idx1, idx2)
            loss = (weight * (pred_sim - target_sim) ** 2).mean()

            total_loss += loss.item()
            all_preds.extend(pred_sim.cpu().tolist())
            all_targets.extend(target_sim.cpu().tolist())

    # Classification accuracy with threshold at 0.4
    # (positive if > 0.4, negative if <= 0.4)
    correct = 0
    total = len(all_preds)
    pos_preds = []
    neg_preds = []

    for pred, target in zip(all_preds, all_targets):
        pred_class = 1 if pred > 0.4 else 0
        target_class = 1 if target > 0.4 else 0
        if pred_class == target_class:
            correct += 1
        # Track positive vs negative predictions separately
        if target >= 0.3:
            pos_preds.append(pred)
        else:
            neg_preds.append(pred)

    accuracy = correct / total if total > 0 else 0.0

    # Average predictions for pos/neg
    pos_avg = sum(pos_preds) / len(pos_preds) if pos_preds else 0.0
    neg_avg = sum(neg_preds) / len(neg_preds) if neg_preds else 0.0

    # Pearson correlation
    if len(all_preds) > 1:
        preds_t = torch.tensor(all_preds)
        targets_t = torch.tensor(all_targets)
        preds_centered = preds_t - preds_t.mean()
        targets_centered = targets_t - targets_t.mean()
        correlation = (preds_centered * targets_centered).sum() / (
            (preds_centered.norm() * targets_centered.norm()) + 1e-8
        )
        correlation = correlation.item()
    else:
        correlation = 0.0

    return total_loss / len(dataloader), accuracy, correlation, pos_avg, neg_avg


def save_checkpoint(model: RootEmbeddings, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, correlation: float,
                    root_to_idx: dict, idx_to_root: dict,
                    output_dir: Path, is_best: bool = False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'correlation': correlation,
        'embedding_dim': model.embedding_dim,
        'vocab_size': len(root_to_idx),
        'root_to_idx': root_to_idx,
        'idx_to_root': idx_to_root
    }

    # Save latest
    temp_path = output_dir / 'checkpoint.pt.tmp'
    try:
        torch.save(checkpoint, temp_path)
        temp_path.rename(output_dir / 'checkpoint.pt')
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return

    # Save best
    if is_best:
        prev_best = output_dir / 'best_model.pt'
        if prev_best.exists():
            prev_best.rename(output_dir / 'best_model.prev.pt')

        temp_path = output_dir / 'best_model.pt.tmp'
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(output_dir / 'best_model.pt')
            logger.info(f"Saved new best model (correlation: {correlation:.4f})")
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            if temp_path.exists():
                temp_path.unlink()


def load_checkpoint(output_dir: Path, model: RootEmbeddings,
                    optimizer: torch.optim.Optimizer) -> Tuple[int, float]:
    """Load checkpoint if exists."""
    checkpoint_path = output_dir / 'checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0.0

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint.get('correlation', 0.0)


def main():
    parser = argparse.ArgumentParser(description='Train root embeddings')
    parser.add_argument('--fundamento-roots', type=Path,
                        default=Path('data/vocabularies/fundamento_roots.json'),
                        help='Fundamento roots JSON')
    parser.add_argument('--revo-definitions', type=Path,
                        default=Path('data/revo/revo_definitions_with_roots.json'),
                        help='ReVo definitions JSON (cleaner than old PV)')
    parser.add_argument('--ekzercaro', type=Path,
                        default=Path('data/training/ekzercaro_sentences.jsonl'),
                        help='Ekzercaro sentences JSONL')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/root_embeddings'),
                        help='Output directory')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Log directory')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (lower = more stable)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Contrastive loss margin')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore checkpoints)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build data but do not train')
    parser.add_argument('--no-pv', action='store_true',
                        help='Skip PV definition pairs (for debugging)')
    parser.add_argument('--hard-negatives', action='store_true',
                        help='Include hard negative pairs')
    parser.add_argument('--clean-vocab', type=Path,
                        default=Path('data/vocabularies/clean_roots.json'),
                        help='Clean vocabulary file (RECOMMENDED)')
    parser.add_argument('--revo-relations', type=Path,
                        default=Path('data/revo/revo_semantic_relations.json'),
                        help='ReVo curated semantic relations (synonyms, antonyms, hypernyms)')

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'train_root_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Root Embedding Training")
    logger.info("=" * 60)
    logger.info(f"Embedding dim: {args.embedding_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    # Load data
    logger.info("\nLoading data...")

    # Fundamento roots
    fundamento_roots = {}
    if args.fundamento_roots.exists():
        with open(args.fundamento_roots) as f:
            data = json.load(f)
            fundamento_roots = data.get('roots', data)
        logger.info(f"Loaded {len(fundamento_roots)} Fundamento roots")
    else:
        logger.warning(f"Fundamento roots not found: {args.fundamento_roots}")

    # ReVo definitions (cleaner than old PV)
    revo_entries = {}
    if args.revo_definitions.exists():
        with open(args.revo_definitions) as f:
            revo_entries = json.load(f)
        logger.info(f"Loaded {len(revo_entries)} ReVo entries")
    else:
        logger.warning(f"ReVo definitions not found: {args.revo_definitions}")

    # Ekzercaro
    ekzercaro = []
    if args.ekzercaro.exists():
        with open(args.ekzercaro) as f:
            ekzercaro = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(ekzercaro)} Ekzercaro sentences")
    else:
        logger.warning(f"Ekzercaro not found: {args.ekzercaro}")

    # Build vocabulary
    logger.info("\nBuilding vocabulary...")
    root_to_idx, idx_to_root = build_vocabulary(
        fundamento_roots, revo_entries, ekzercaro,
        clean_vocab_path=args.clean_vocab
    )
    logger.info(f"Total vocabulary: {len(root_to_idx)} roots")

    # Build training pairs
    logger.info("\nBuilding training pairs...")
    pairs, weights = build_similarity_pairs(
        fundamento_roots, revo_entries, ekzercaro, root_to_idx,
        use_revo=not args.no_pv,
        use_hard_negatives=args.hard_negatives,
        revo_relations_path=args.revo_relations
    )

    if args.dry_run:
        logger.info("\nDry run - not training")
        logger.info(f"Would train on {len(pairs)} pairs")
        return

    # Split into train/val (shuffle first)
    combined = list(zip(pairs, weights))
    random.shuffle(combined)
    pairs, weights = zip(*combined)
    pairs = list(pairs)
    weights = list(weights)

    split_idx = int(len(pairs) * 0.9)
    train_pairs, train_weights = pairs[:split_idx], weights[:split_idx]
    val_pairs, val_weights = pairs[split_idx:], weights[split_idx:]

    logger.info(f"Train pairs: {len(train_pairs)}")
    logger.info(f"Val pairs: {len(val_pairs)}")

    # Create datasets
    train_dataset = DefinitionSimilarityDataset(train_pairs, train_weights)
    val_dataset = DefinitionSimilarityDataset(val_pairs, val_weights)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model = RootEmbeddings(len(root_to_idx), args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    best_val_loss = float('inf')
    if not args.fresh:
        start_epoch, best_accuracy = load_checkpoint(args.output_dir, model, optimizer)
        if start_epoch > 0:
            logger.info(f"Resuming from epoch {start_epoch}, best accuracy: {best_accuracy:.4f}")

    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Margin: {args.margin}, LR: {args.learning_rate}")
    patience_counter = 0
    min_epochs = 30  # Train at least this many epochs
    best_correlation = -1.0

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, margin=args.margin)
        val_loss, accuracy, correlation, pos_avg, neg_avg = evaluate(model, val_loader, device)

        # Compute embedding spread (check for collapse)
        with torch.no_grad():
            sample_indices = torch.randint(0, len(root_to_idx), (100,), device=device)
            sample_embs = model.get_normalized(sample_indices)
            avg_sim = (sample_embs @ sample_embs.T).mean().item()

        logger.info(f"Epoch {epoch + 1}/{args.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"corr={correlation:.4f}, pos_sim={pos_avg:.3f}, neg_sim={neg_avg:.3f}")

        # Check for improvement (prioritize correlation for graded targets)
        is_best = False
        if correlation > best_correlation + 0.005:  # Meaningful improvement
            best_correlation = correlation
            best_accuracy = accuracy
            is_best = True
            patience_counter = 0
        elif accuracy > best_accuracy + 0.005:
            best_accuracy = accuracy
            is_best = True
            patience_counter = 0
        elif val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch + 1, val_loss, correlation,
                        root_to_idx, idx_to_root, args.output_dir, is_best)

        # Early stopping (only after minimum epochs)
        if epoch >= min_epochs and patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break

        # Warning if embeddings are collapsing
        if avg_sim > 0.7:
            logger.warning(f"High average similarity ({avg_sim:.3f}) - embeddings may be collapsing!")

    logger.info(f"\nTraining complete!")
    logger.info(f"Best correlation: {best_correlation:.4f}, Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Model saved to {args.output_dir}")
    logger.info(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()
