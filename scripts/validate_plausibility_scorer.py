#!/usr/bin/env python3
"""
Validate Plausibility Scorer - Corpus Analysis

VERSION: v2.0
COMPATIBLE WITH: v2.0 hybrid plausibility scorer
DEPENDENCIES: Trained plausibility model, filtered corpus
STAGE: Evaluation

Description:
    Score all corpus triples to determine if plausibility filtering is needed.
    Answers the critical question: How many corpus triples are implausible?

Usage:
    python scripts/validate_plausibility_scorer.py

Outputs:
    - Score distribution analysis
    - Samples of low/mid/high scoring triples
    - Recommendation on whether to use the scorer

Last Updated: 2026-03-23
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import jsonlines
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class HybridWordEncoder:
    """Simplified word encoder for inference."""
    
    def __init__(self, root_embedder, use_affix_features=True, use_lexicon_features=True):
        from klareco.morphology.affix_semantics import AFFIX_SEMANTICS
        from klareco.morphology.root_lexicon import ROOT_LEXICON
        
        self.root_embedder = root_embedder
        self.use_affix_features = use_affix_features
        self.use_lexicon_features = use_lexicon_features
        self.output_dim = 128 + (22 if use_affix_features else 0) + (22 if use_lexicon_features else 0)
        
        self.AFFIX_SEMANTICS = AFFIX_SEMANTICS
        self.ROOT_LEXICON = ROOT_LEXICON
        
        # Animacy encoding
        self.animacy_map = {'animate': 0, 'inanimate': 1, 'abstract': 2, 'collective': 3}
        
        # Type encoding (top 17)
        self.type_map = {
            'person': 0, 'agent': 1, 'animal': 2, 'tool': 3, 'thing': 4,
            'place': 5, 'quality': 6, 'action': 7, 'product': 8, 'collective': 9,
            'property': 10, 'offspring': 11, 'diminutive': 12, 'augmentative': 13,
            'abstraction': 14, 'passive': 15, 'object': 16
        }
    
    def encode(self, word_data):
        """Encode a word to 172D vector."""
        root = word_data['root']
        affixes = word_data.get('affixes', [])
        semantics = word_data.get('semantics', {})
        
        # Root embedding (128D)
        root_emb = self.root_embedder.embed_root(root)
        
        features = [root_emb]
        
        # Affix features (22D)
        if self.use_affix_features:
            affix_feat = self._encode_affixes(affixes)
            features.append(affix_feat)
        
        # Lexicon features (22D)
        if self.use_lexicon_features:
            lex_feat = self._encode_lexicon_semantics(semantics)
            features.append(lex_feat)
        
        return torch.cat(features, dim=-1)
    
    def _encode_affixes(self, affixes):
        """Encode affixes to 22D vector."""
        feat = torch.zeros(22)
        
        for i, affix in enumerate(affixes[:5]):  # Max 5 affixes
            if affix in self.AFFIX_SEMANTICS:
                sem = self.AFFIX_SEMANTICS[affix]
                
                # Animacy (4D one-hot)
                if sem.get('animacy') in self.animacy_map:
                    feat[i*4 + self.animacy_map[sem['animacy']]] = 1.0
        
        # Count features (2D)
        feat[20] = min(len(affixes) / 5.0, 1.0)  # Normalized count
        feat[21] = 1.0 if len(affixes) > 0 else 0.0  # Has affixes
        
        return feat
    
    def _encode_lexicon_semantics(self, semantics):
        """Encode lexicon semantics to 22D vector."""
        feat = torch.zeros(22)
        
        # Animacy (4D one-hot)
        animacy = semantics.get('animacy', 'unknown')
        if animacy in self.animacy_map:
            feat[self.animacy_map[animacy]] = 1.0
        
        # Type (17D one-hot)
        type_ = semantics.get('type', 'unknown')
        if type_ in self.type_map:
            feat[4 + self.type_map[type_]] = 1.0
        
        # Coverage flag (1D)
        feat[21] = 1.0 if animacy != 'unknown' else 0.0
        
        return feat


def load_model():
    """Load trained plausibility scorer."""
    logger.info("Loading plausibility scorer...")
    
    # Load checkpoint
    checkpoint = torch.load(
        "models/hybrid_plausibility_word_level_final/model_best.pt",
        map_location='cpu'
    )
    
    # Load root embedder
    from klareco.embeddings.unified_root_embedder import UnifiedRootEmbedder
    root_embedder = UnifiedRootEmbedder(
        production_model_path="models/root_embeddings_phase1_fast/root_embeddings_best.pt",
        ast_model_path="models/root_embeddings_fundamento_ast/root_embeddings_best.pt"
    )
    
    # Create word encoder
    word_encoder = HybridWordEncoder(
        root_embedder=root_embedder,
        use_affix_features=True,
        use_lexicon_features=True
    )
    
    # Create scorer
    import torch.nn as nn
    
    input_dim = word_encoder.output_dim * 3  # 172 * 3 = 516
    scorer = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Load weights (extract only scorer weights)
    state_dict = checkpoint['model_state_dict']
    scorer_state = {k.replace('scorer.', ''): v for k, v in state_dict.items() if k.startswith('scorer.')}
    scorer.load_state_dict(scorer_state)
    scorer.eval()
    
    logger.info(f"Model loaded. Training F1: {checkpoint.get('best_f1', 0):.2%}")
    
    return word_encoder, scorer


def score_corpus(word_encoder, scorer, corpus_path):
    """Score all triples in corpus."""
    logger.info(f"Loading corpus from {corpus_path}...")
    
    triples = []
    with jsonlines.open(corpus_path) as reader:
        for triple in reader:
            triples.append(triple)
    
    logger.info(f"Scoring {len(triples):,} triples...")
    
    scores = []
    scored_triples = []
    
    with torch.no_grad():
        for triple in tqdm(triples, desc="Scoring"):
            try:
                # Encode words
                subj_emb = word_encoder.encode(triple['subject'])
                verb_emb = word_encoder.encode(triple['verb'])
                obj_emb = word_encoder.encode(triple['object'])
                
                # Concatenate
                combined = torch.cat([subj_emb, verb_emb, obj_emb], dim=-1).unsqueeze(0)
                
                # Score
                score = scorer(combined).item()
                scores.append(score)
                
                scored_triples.append({
                    **triple,
                    'plausibility_score': score
                })
            except Exception as e:
                logger.warning(f"Failed to score triple: {e}")
                scores.append(0.5)  # Neutral score for errors
                scored_triples.append({
                    **triple,
                    'plausibility_score': 0.5
                })
    
    return np.array(scores), scored_triples


def analyze_distribution(scores):
    """Analyze score distribution."""
    logger.info("\n" + "="*70)
    logger.info("SCORE DISTRIBUTION ANALYSIS")
    logger.info("="*70)
    
    logger.info(f"Total triples: {len(scores):,}")
    logger.info(f"Mean score: {np.mean(scores):.3f}")
    logger.info(f"Median score: {np.median(scores):.3f}")
    logger.info(f"Std dev: {np.std(scores):.3f}")
    logger.info(f"Min score: {np.min(scores):.3f}")
    logger.info(f"Max score: {np.max(scores):.3f}")
    
    # Percentiles
    logger.info("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        logger.info(f"  {p}th: {np.percentile(scores, p):.3f}")
    
    # Thresholds
    logger.info("\nTriples by plausibility threshold:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh in thresholds:
        count = np.sum(scores < thresh)
        pct = 100 * count / len(scores)
        logger.info(f"  < {thresh:.1f}: {count:,} ({pct:.1f}%)")
    
    # Bins
    logger.info("\nScore bins:")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = np.sum((scores >= low) & (scores < high))
        pct = 100 * count / len(scores)
        logger.info(f"  [{low:.1f}-{high:.1f}): {count:,} ({pct:.1f}%)")


def sample_triples(scored_triples, n_per_bin=5):
    """Sample triples from different score ranges."""
    logger.info("\n" + "="*70)
    logger.info("SAMPLE TRIPLES FOR MANUAL REVIEW")
    logger.info("="*70)
    
    bins = [
        ("Very Low (0.0-0.2)", 0.0, 0.2),
        ("Low (0.2-0.4)", 0.2, 0.4),
        ("Medium (0.4-0.6)", 0.4, 0.6),
        ("High (0.6-0.8)", 0.6, 0.8),
        ("Very High (0.8-1.0)", 0.8, 1.0)
    ]
    
    for label, low, high in bins:
        # Filter triples in this range
        bin_triples = [t for t in scored_triples if low <= t['plausibility_score'] < high]
        
        if not bin_triples:
            logger.info(f"\n{label}: No triples in this range")
            continue
        
        # Sample
        import random
        random.seed(42)
        samples = random.sample(bin_triples, min(n_per_bin, len(bin_triples)))
        
        logger.info(f"\n{label} ({len(bin_triples):,} total):")
        for i, triple in enumerate(samples, 1):
            subj = triple['subject']
            verb = triple['verb']
            obj = triple['object']
            score = triple['plausibility_score']
            
            logger.info(f"\n  {i}. {subj['text']} → {verb['text']} → {obj['text']} (score: {score:.3f})")
            logger.info(f"     Subject: {subj.get('animacy', '?')}/{subj.get('type', '?')}")
            logger.info(f"     Object: {obj.get('animacy', '?')}/{obj.get('type', '?')}")


def generate_recommendation(scores):
    """Generate recommendation on whether to use the scorer."""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    
    low_score_pct = 100 * np.sum(scores < 0.4) / len(scores)
    high_score_pct = 100 * np.sum(scores >= 0.6) / len(scores)
    
    logger.info(f"\nLow plausibility (<0.4): {low_score_pct:.1f}% of corpus")
    logger.info(f"High plausibility (≥0.6): {high_score_pct:.1f}% of corpus")
    
    if low_score_pct < 5:
        logger.info("\n✓ RECOMMENDATION: Plausibility filtering NOT needed")
        logger.info("  Reason: <5% of corpus has low plausibility scores")
        logger.info("  Action: The parser is already producing high-quality triples")
        logger.info("  Impact: Adding filtering would add complexity for minimal benefit")
    elif low_score_pct < 15:
        logger.info("\n⚠ RECOMMENDATION: Plausibility filtering OPTIONAL")
        logger.info(f"  Reason: {low_score_pct:.1f}% of corpus has low plausibility")
        logger.info("  Action: Manually review low-scoring samples first")
        logger.info("  Decision: Filter only if manual review shows real errors")
    else:
        logger.info("\n✓ RECOMMENDATION: Plausibility filtering RECOMMENDED")
        logger.info(f"  Reason: {low_score_pct:.1f}% of corpus has low plausibility")
        logger.info("  Action: Filter triples with score < 0.5 before indexing")
        logger.info("  Impact: Should significantly improve knowledge base quality")
    
    logger.info("\n" + "="*70)


def main():
    # Load model
    word_encoder, scorer = load_model()
    
    # Score corpus
    corpus_path = Path("data/semantic_types/svo_triples_word_level_filtered.jsonl")
    scores, scored_triples = score_corpus(word_encoder, scorer, corpus_path)
    
    # Analyze
    analyze_distribution(scores)
    
    # Sample
    sample_triples(scored_triples, n_per_bin=5)
    
    # Recommend
    generate_recommendation(scores)
    
    # Save scored corpus
    output_path = Path("data/semantic_types/svo_triples_scored.jsonl")
    logger.info(f"\nSaving scored corpus to {output_path}...")
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(scored_triples)
    logger.info(f"Saved {len(scored_triples):,} scored triples")


if __name__ == '__main__':
    main()
