"""
Test Plausibility Scorer - Quick Evaluation

VERSION: v2.1
COMPATIBLE WITH: v2.1 plausibility scorer, hybrid root embeddings
DEPENDENCIES: Hybrid Root Embeddings (Production + AST)
STAGE: Evaluation

Description:
    Test the trained plausibility scorer on hand-crafted examples to verify
    it learned meaningful semantic patterns.

Usage:
    python scripts/test_plausibility_scorer.py
    python scripts/test_plausibility_scorer.py --interactive

Inputs:
    - Trained model: models/plausibility_scorer/model_best.pt
    - Hybrid embedder: models/root_embeddings_*/

Outputs:
    - Plausibility scores for test triples
    - Analysis of learned patterns

Last Updated: 2026-03-22
"""

import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.embeddings.hybrid import load_hybrid_embedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlausibilityScorer(nn.Module):
    """
    Simple concatenation-based plausibility scorer.

    Must match the architecture from train_plausibility_scorer.py
    """

    def __init__(self, embedder):
        super().__init__()

        # Frozen embeddings
        self.embedder = embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

        # MLP scorer (matches training architecture)
        self.scorer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, subject_roots, verb_roots, object_roots):
        """
        Score plausibility of (subject, verb, object) triples.

        Args:
            subject_roots: List of subject roots
            verb_roots: List of verb roots
            object_roots: List of object roots

        Returns:
            Plausibility scores (batch_size,)
        """
        # Get embeddings
        subj_embs = []
        verb_embs = []
        obj_embs = []

        for subj, verb, obj in zip(subject_roots, verb_roots, object_roots):
            subj_emb = self.embedder.get_embedding(subj)
            verb_emb = self.embedder.get_embedding(verb)
            obj_emb = self.embedder.get_embedding(obj)

            # Handle unknown roots
            if subj_emb is None or verb_emb is None or obj_emb is None:
                device = next(self.parameters()).device
                zero_emb = torch.zeros(128, device=device)
                if subj_emb is None:
                    subj_emb = zero_emb
                if verb_emb is None:
                    verb_emb = zero_emb
                if obj_emb is None:
                    obj_emb = zero_emb

            subj_embs.append(subj_emb)
            verb_embs.append(verb_emb)
            obj_embs.append(obj_emb)

        # Stack embeddings
        subj_embs = torch.stack(subj_embs)
        verb_embs = torch.stack(verb_embs)
        obj_embs = torch.stack(obj_embs)

        # Concatenate
        concat = torch.cat([subj_embs, verb_embs, obj_embs], dim=1)

        # Score
        scores = self.scorer(concat).squeeze(-1)

        return scores


def load_model(model_path: str, device: str = "cpu"):
    """Load trained plausibility scorer."""
    logger.info(f"Loading model from {model_path}")

    # Load hybrid embedder
    embedder = load_hybrid_embedder(device=device)

    # Create model
    model = PlausibilityScorer(embedder)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded (vocab: {len(embedder.root_to_idx)} roots)")

    return model


def score_triple(model, subject, verb, object_):
    """Score a single (subject, verb, object) triple."""
    with torch.no_grad():
        score = model([subject], [verb], [object_])
        return score.item()


def run_test_suite(model):
    """Run comprehensive test suite on hand-crafted examples."""
    print("\n" + "="*80)
    print("PLAUSIBILITY SCORER TEST SUITE")
    print("="*80 + "\n")

    # Test categories
    tests = {
        "✅ PLAUSIBLE - Common sense": [
            ("hom", "manĝ", "pom"),       # person eats apple
            ("hund", "vid", "kat"),        # dog sees cat
            ("infant", "plor", "plu"),    # baby cries more (adverb as object - corpus pattern)
            ("autor", "verk", "libr"),     # author writes book
            ("student", "lern", "lingv"),  # student learns language
            ("birb", "flug", "ĉiel"),      # bird flies sky
        ],

        "✅ PLAUSIBLE - Esperanto corpus patterns": [
            ("land", "hav", "loĝant"),     # country has inhabitants
            ("urb", "est", "grav"),        # city is important
            ("parol", "est", "bel"),       # speech is beautiful
        ],

        "❌ IMPLAUSIBLE - Type violations": [
            ("pom", "manĝ", "hom"),        # apple eats person (reversed)
            ("tabl", "flug", "ĉiel"),      # table flies sky
            ("libr", "plor", "plu"),       # book cries more
            ("ĉiel", "verk", "libr"),      # sky writes book
        ],

        "❌ IMPLAUSIBLE - Selectional restrictions": [
            ("hom", "manĝ", "sonĝ"),       # person eats dream
            ("hund", "verk", "poem"),      # dog writes poem
            ("tabl", "lern", "lingv"),     # table learns language
        ],

        "⚠️ EDGE CASES - Metaphorical/Abstract": [
            ("sci", "est", "fort"),        # knowledge is power (abstract)
            ("vort", "vund", "kord"),      # words wound heart (metaphor)
            ("esper", "viv", "kor"),       # hope lives in heart
        ],
    }

    for category, triples in tests.items():
        print(f"\n{category}")
        print("-" * 80)

        for subj, verb, obj in triples:
            score = score_triple(model, subj, verb, obj)
            # Color code by score
            if score >= 0.7:
                indicator = "🟢"
            elif score >= 0.5:
                indicator = "🟡"
            else:
                indicator = "🔴"

            print(f"{indicator} ({subj}, {verb}, {obj}) → {score:.3f}")

    print("\n" + "="*80)


def interactive_mode(model):
    """Interactive testing mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nEnter triples as: subject verb object")
    print("Example: hom manĝ pom")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Invalid format. Use: subject verb object")
                continue

            subj, verb, obj = parts
            score = score_triple(model, subj, verb, obj)

            # Interpret score
            if score >= 0.7:
                verdict = "🟢 PLAUSIBLE"
            elif score >= 0.5:
                verdict = "🟡 UNCERTAIN"
            else:
                verdict = "🔴 IMPLAUSIBLE"

            print(f"{verdict} - Score: {score:.3f}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Test plausibility scorer")
    parser.add_argument(
        '--model-path',
        default='models/plausibility_scorer/model_best.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Interactive testing mode'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device (cpu/cuda)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, device=args.device)

    if args.interactive:
        interactive_mode(model)
    else:
        run_test_suite(model)


if __name__ == '__main__':
    main()
