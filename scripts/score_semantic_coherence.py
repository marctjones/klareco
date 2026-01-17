#!/usr/bin/env python3
"""
Score semantic coherence of Esperanto sentences.

Computes average pairwise similarity of content roots to measure
how semantically related the words in a sentence are.

High score = semantically coherent sentence
Low score = unrelated words / nonsense

Usage:
    python scripts/score_semantic_coherence.py "La hundo manĝas viandon"
    python scripts/score_semantic_coherence.py --batch corpus.jsonl
    python scripts/score_semantic_coherence.py -i  # Interactive
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


class SemanticCoherenceScorer:
    """Score semantic coherence using root embeddings."""

    # Function words to exclude from coherence scoring
    FUNCTION_WORDS = {
        'la', 'kaj', 'sed', 'aŭ', 'nek', 'de', 'da', 'al', 'el', 'en',
        'sur', 'sub', 'inter', 'antaŭ', 'post', 'ĉe', 'ekster',
        'sen', 'kun', 'per', 'pro', 'dum', 'ĝis', 'je',
        'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si',
        'tiu', 'ĉi', 'kiu', 'kio', 'kie', 'kiam', 'kiel', 'kial', 'kiom',
        'tio', 'tie', 'tiam', 'tiel', 'tial', 'tiom',
        'ĉio', 'ĉie', 'ĉiam', 'ĉiel', 'ĉial', 'ĉiom',
        'nenio', 'nenie', 'neniam', 'neniel', 'nenial', 'neniom',
        'io', 'ie', 'iam', 'iel', 'ial', 'iom',
        'ĉiu', 'neniu',  'iu',
        'ke', 'se', 'ĉar', 'kvankam', 'dum', 'krom',
        'ne', 'jes', 'nu',
    }

    def __init__(self, embeddings_path: Path):
        """Initialize with root embeddings."""
        # Load embeddings
        checkpoint = torch.load(embeddings_path, map_location='cpu')
        self.root_to_idx = checkpoint['root_to_idx']
        self.idx_to_root = checkpoint['idx_to_root']

        # Create embedding layer
        self.embedding = torch.nn.Embedding(
            checkpoint['vocab_size'],
            checkpoint['embedding_dim']
        )
        self.embedding.weight.data = checkpoint['model_state_dict']['embeddings.weight']

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity."""
        return (a @ b) / (torch.norm(a) * torch.norm(b) + 1e-8)

    def extract_content_roots(self, ast: dict) -> List[str]:
        """Extract content roots (excluding function words) from AST."""
        roots = []

        def extract(node):
            if node is None or not isinstance(node, dict):
                return

            if 'radiko' in node:
                root = node['radiko']
                if root and root not in self.FUNCTION_WORDS:
                    roots.append(root)

            # Recurse
            if node.get('tipo') == 'vortgrupo':
                extract(node.get('kerno'))
                for priskr in node.get('priskriboj', []):
                    extract(priskr)

            if node.get('tipo') == 'frazo':
                extract(node.get('subjekto'))
                extract(node.get('verbo'))
                extract(node.get('objekto'))
                for alian in node.get('aliaj', []):
                    extract(alian)

        extract(ast)
        return roots

    def score_sentence(self, sentence: str) -> Optional[float]:
        """
        Score semantic coherence of sentence.

        Returns:
            Coherence score (0-1), or None if < 2 content roots
        """
        try:
            ast = parse(sentence)
        except Exception:
            return None

        # Extract content roots
        roots = self.extract_content_roots(ast)

        # Filter roots that are in embeddings
        valid_roots = [r for r in roots if r in self.root_to_idx]

        if len(valid_roots) < 2:
            return None  # Need at least 2 roots for pairwise similarity

        # Compute pairwise similarities
        similarities = []
        for i in range(len(valid_roots)):
            idx_i = self.root_to_idx[valid_roots[i]]
            emb_i = self.embedding(torch.tensor(idx_i))

            for j in range(i+1, len(valid_roots)):
                idx_j = self.root_to_idx[valid_roots[j]]
                emb_j = self.embedding(torch.tensor(idx_j))

                sim = self.cosine_similarity(emb_i, emb_j).item()
                similarities.append(sim)

        if not similarities:
            return None

        # Average similarity
        return sum(similarities) / len(similarities)


def main():
    parser = argparse.ArgumentParser(
        description='Score semantic coherence of Esperanto sentences'
    )
    parser.add_argument('sentence', nargs='?', help='Sentence to score')
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to root embeddings')
    parser.add_argument('--batch', type=Path,
                       help='Score sentences from JSONL file')
    parser.add_argument('--output', type=Path,
                       help='Output file for batch results')
    parser.add_argument('--threshold', type=float,
                       help='Filter: only show scores above threshold')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--pipe-friendly', action='store_true',
                       help='Output only Esperanto sentence for piping (no labels)')

    args = parser.parse_args()

    # Load scorer
    print(f"Loading embeddings from {args.embeddings}...", file=sys.stderr)
    scorer = SemanticCoherenceScorer(args.embeddings)
    print(f"Loaded {len(scorer.root_to_idx)} root embeddings\n", file=sys.stderr)

    def score_and_display(sentence: str):
        """Score and display results for single sentence."""
        score = scorer.score_sentence(sentence)

        if args.pipe_friendly:
            # Pipe-friendly: just output the sentence
            print(sentence)
        else:
            # Human-readable output
            if score is None:
                print(f"❌ Unable to score (< 2 content roots)")
            else:
                # Determine assessment
                if score < 0.2:
                    assessment = "❌ Very low (likely nonsense)"
                elif score < 0.35:
                    assessment = "⚠️  Low (possibly incoherent)"
                elif score < 0.5:
                    assessment = "✓ Moderate"
                elif score < 0.65:
                    assessment = "✓✓ Good"
                else:
                    assessment = "✓✓✓ High (very coherent)"

                print(f"Coherence: {score:.3f} {assessment}")

            print(f"Sentence: {sentence}\n")

    # Batch mode
    if args.batch:
        print(f"Processing {args.batch}...", file=sys.stderr)
        output_file = args.output or args.batch.with_suffix('.coherence.jsonl')

        with open(args.batch) as fin, open(output_file, 'w') as fout:
            for line_num, line in enumerate(fin, 1):
                entry = json.loads(line)
                sentence = entry.get('text', '')

                score = scorer.score_sentence(sentence)

                if score is not None:
                    if args.threshold is None or score >= args.threshold:
                        entry['coherence_score'] = score
                        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')

                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} sentences...", file=sys.stderr)

        print(f"\nResults written to {output_file}", file=sys.stderr)

    # Interactive mode
    elif args.interactive:
        print("Interactive mode (type 'quit' to exit)\n", file=sys.stderr)
        while True:
            try:
                sentence = input("Sentence: ").strip()
                if sentence.lower() in ['quit', 'exit', 'q']:
                    break
                if sentence:
                    score_and_display(sentence)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    # Single sentence
    elif args.sentence:
        score_and_display(args.sentence)

    else:
        # Examples
        examples = [
            ("La hundo manĝas viandon", "coherent - dog eats meat"),
            ("La hundo manĝas libron", "incoherent - dog eats book"),
            ("La kato vidas la hundon", "moderate - cat sees dog"),
            ("Mi pensas pri tio", "coherent - I think about that"),
            ("La bela tago", "coherent - beautiful day"),
        ]

        print("Example sentences:\n", file=sys.stderr)
        for sentence, description in examples:
            print(f"[{description}]", file=sys.stderr)
            score_and_display(sentence)


if __name__ == '__main__':
    main()
