#!/usr/bin/env python3
"""
Klareco Pipeline Demo
=====================

Demonstrates each stage of the Esperanto AI pipeline:

Stage 0: Deterministic Parser
  - Parses Esperanto text into structured ASTs
  - Decomposes words into roots, prefixes, suffixes
  - 100% rule-based, no learned parameters

Stage 1a: Root Embeddings
  - 11,121 roots × 64d = ~712K params
  - Semantic similarity between roots
  - Trained on ReVo definitions + Ekzercaro co-occurrence

Stage 1b: Affix Transformations
  - 41 affixes × 512 params = ~21K params
  - Low-rank transforms that modify root embeddings
  - mal- flips polarity, -ej clusters places, etc.

Combined: Compositional Word Embeddings
  - Parse word → get root → apply prefix/suffix transforms
  - Handles novel words never seen in training!

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --interactive
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Optional


def get_model_info(model_path: Path) -> dict:
    """Get model file info including modification time and version detection."""
    info = {
        'path': str(model_path),
        'name': model_path.name,
        'exists': model_path.exists(),
        'modified': None,
        'version': None,
        'size_kb': None,
    }
    if model_path.exists():
        stat = model_path.stat()
        info['modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        info['size_kb'] = stat.st_size // 1024
        # Detect version from path
        if '_v2' in str(model_path) or 'v2' in model_path.parent.name:
            info['version'] = 'V2'
        elif '_v3' in str(model_path) or 'v3' in model_path.parent.name:
            info['version'] = 'V3'
        else:
            info['version'] = 'V1'
    return info

from klareco.parser import parse


class LowRankTransform(nn.Module):
    """Low-rank transformation for affixes."""
    def __init__(self, dim: int = 64, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class KlarecoPipeline:
    """Complete Klareco pipeline with all trained models."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.root_embeddings = None
        self.root_to_idx = None
        self.prefix_transforms = {}
        self.suffix_transforms = {}
        self.root_model_path = None
        self.affix_model_path = None
        self._load_models()

    def _load_models(self):
        """Load all trained models."""
        # Load root embeddings
        root_path = self.models_dir / "root_embeddings" / "best_model.pt"
        self.root_model_path = root_path
        if root_path.exists():
            ckpt = torch.load(root_path, map_location='cpu', weights_only=False)
            self.root_to_idx = ckpt['root_to_idx']
            self.root_embeddings = nn.Embedding(ckpt['vocab_size'], ckpt['embedding_dim'])
            self.root_embeddings.load_state_dict({
                'weight': ckpt['model_state_dict']['embeddings.weight']
            })
            self.embedding_dim = ckpt['embedding_dim']
            print(f"✓ Loaded root embeddings: {ckpt['vocab_size']:,} roots × {ckpt['embedding_dim']}d")
        else:
            print(f"✗ Root embeddings not found at {root_path}")
            return

        # Load affix transforms (V2 model - uses low-rank transforms with anti-collapse)
        affix_path = self.models_dir / "affix_transforms_v2" / "best_model.pt"
        self.affix_model_path = affix_path
        if affix_path.exists():
            ckpt = torch.load(affix_path, map_location='cpu', weights_only=False)

            for p in ckpt['prefixes']:
                t = LowRankTransform(self.embedding_dim, ckpt['rank'])
                t.down.weight.data = ckpt['model_state_dict'][f'prefix_transforms.{p}.down.weight']
                t.up.weight.data = ckpt['model_state_dict'][f'prefix_transforms.{p}.up.weight']
                self.prefix_transforms[p] = t

            for s in ckpt['suffixes']:
                t = LowRankTransform(self.embedding_dim, ckpt['rank'])
                t.down.weight.data = ckpt['model_state_dict'][f'suffix_transforms.{s}.down.weight']
                t.up.weight.data = ckpt['model_state_dict'][f'suffix_transforms.{s}.up.weight']
                self.suffix_transforms[s] = t

            print(f"✓ Loaded affix transforms: {len(self.prefix_transforms)} prefixes, {len(self.suffix_transforms)} suffixes")
        else:
            print(f"✗ Affix transforms not found at {affix_path}")

    def get_root_embedding(self, root: str) -> Optional[torch.Tensor]:
        """Get embedding for a root word."""
        if root in self.root_to_idx:
            idx = self.root_to_idx[root]
            return self.root_embeddings(torch.tensor([idx])).squeeze()
        return None

    def get_word_embedding(self, word_ast: dict) -> Optional[torch.Tensor]:
        """
        Get compositional embedding for a parsed word.

        Applies: prefixes → root → suffixes
        """
        root = word_ast.get('radiko')
        if not root:
            return None

        # Start with root embedding
        emb = self.get_root_embedding(root)
        if emb is None:
            return None

        # Apply prefix transforms (in order)
        prefixes = word_ast.get('prefiksoj', [])
        for p in prefixes:
            if p in self.prefix_transforms:
                emb = self.prefix_transforms[p](emb)

        # Apply suffix transforms (in order)
        suffixes = word_ast.get('sufiksoj', [])
        for s in suffixes:
            if s in self.suffix_transforms:
                emb = self.suffix_transforms[s](emb)

        return emb

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Cosine similarity between embeddings."""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def print_model_info(self):
        """Print detailed model information with timestamps and versions."""
        print("\n" + "-" * 60)
        print("MODEL INFORMATION")
        print("-" * 60)
        print(f"Session timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Root model info
        root_info = get_model_info(self.root_model_path)
        print(f"Root Embeddings:")
        print(f"  Path: {root_info['path']}")
        if root_info['exists']:
            print(f"  Version: {root_info['version']}")
            print(f"  Modified: {root_info['modified']}")
            print(f"  Size: {root_info['size_kb']:,} KB")
            if self.root_to_idx:
                print(f"  Roots: {len(self.root_to_idx):,}")
        else:
            print(f"  Status: NOT FOUND")

        print()

        # Affix model info
        affix_info = get_model_info(self.affix_model_path)
        print(f"Affix Transforms:")
        print(f"  Path: {affix_info['path']}")
        if affix_info['exists']:
            print(f"  Version: {affix_info['version']}")
            print(f"  Modified: {affix_info['modified']}")
            print(f"  Size: {affix_info['size_kb']:,} KB")
            print(f"  Prefixes: {len(self.prefix_transforms)}")
            print(f"  Suffixes: {len(self.suffix_transforms)}")
        else:
            print(f"  Status: NOT FOUND")

        print("-" * 60)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_ast(ast: dict, indent: int = 0):
    """Pretty print an AST."""
    prefix = "  " * indent
    if ast.get('tipo') == 'frazo':
        stats = ast.get('parse_statistics', {})
        print(f"{prefix}Sentence (parse rate: {stats.get('success_rate', 0)*100:.0f}%)")
        for role in ['subjekto', 'verbo', 'objekto', 'aliaj']:
            if role in ast and ast[role]:
                print(f"{prefix}  {role}:")
                content = ast[role]
                if isinstance(content, list):
                    for item in content:
                        print_ast(item, indent + 2)
                else:
                    print_ast(content, indent + 2)
    elif ast.get('tipo') == 'vortgrupo':
        kerno = ast.get('kerno', {})
        print_ast(kerno, indent)
    elif ast.get('tipo') == 'vorto':
        parts = []
        if ast.get('prefiksoj'):
            parts.append(f"[{'+'.join(ast['prefiksoj'])}]")
        parts.append(ast.get('radiko', '?'))
        if ast.get('sufiksoj'):
            parts.append(f"[{'+'.join(ast['sufiksoj'])}]")
        parts.append(f"-{ast.get('finaĵo', '?')}")
        word_str = ''.join(parts)
        print(f"{prefix}{word_str} ({ast.get('vortspeco', '?')})")


def demo_stage0():
    """Demonstrate the deterministic parser."""
    print_header("Stage 0: Deterministic Parser")

    test_sentences = [
        "La hundo manĝas la katon.",
        "Mi lernas Esperanton.",
        "La malbona homo malrapide iris al la lernejo.",
        "Ĉu vi parolas Esperanton?",
    ]

    for sentence in test_sentences:
        print(f"\n>>> {sentence}")
        ast = parse(sentence)
        print_ast(ast)


def demo_stage1a(pipeline: KlarecoPipeline):
    """Demonstrate root embeddings."""
    print_header("Stage 1a: Root Embeddings")

    print("\n--- Semantic Clusters ---")
    clusters = {
        "Animals": ["hund", "kat", "bird", "fiŝ", "ĉeval"],
        "Family": ["patr", "frat", "fil", "edz"],
        "Qualities": ["bon", "bel", "grand", "fort"],
    }

    for cluster_name, roots in clusters.items():
        print(f"\n{cluster_name}:")
        embeddings = []
        for root in roots:
            emb = pipeline.get_root_embedding(root)
            if emb is not None:
                embeddings.append((root, emb))

        # Show pairwise similarities
        for i, (r1, e1) in enumerate(embeddings):
            for r2, e2 in embeddings[i+1:]:
                sim = pipeline.similarity(e1, e2)
                print(f"  {r1} ↔ {r2}: {sim:.3f}")

    print("\n--- Cross-cluster (should be lower) ---")
    hund = pipeline.get_root_embedding("hund")
    bon = pipeline.get_root_embedding("bon")
    patr = pipeline.get_root_embedding("patr")

    if hund is not None and bon is not None:
        print(f"  hund ↔ bon: {pipeline.similarity(hund, bon):.3f}")
    if hund is not None and patr is not None:
        print(f"  hund ↔ patr: {pipeline.similarity(hund, patr):.3f}")


def demo_stage1b(pipeline: KlarecoPipeline):
    """Demonstrate affix transformations."""
    print_header("Stage 1b: Affix Transformations")

    print("\n--- mal- prefix (polarity flip) ---")
    test_roots = ["bon", "grand", "jun", "bel"]
    for root in test_roots:
        emb = pipeline.get_root_embedding(root)
        if emb is not None and 'mal' in pipeline.prefix_transforms:
            mal_emb = pipeline.prefix_transforms['mal'](emb)
            dist = 1 - pipeline.similarity(emb, mal_emb)
            print(f"  {root} → mal{root}: distance = {dist:.3f}")

    print("\n--- -ej suffix (place clustering) ---")
    place_roots = ["lern", "labor", "kuir", "preĝ", "vend"]
    place_embs = []
    for root in place_roots:
        emb = pipeline.get_root_embedding(root)
        if emb is not None and 'ej' in pipeline.suffix_transforms:
            ej_emb = pipeline.suffix_transforms['ej'](emb)
            place_embs.append((root, ej_emb))

    print("  Pairwise similarities of -ejo words:")
    for i, (r1, e1) in enumerate(place_embs):
        for r2, e2 in place_embs[i+1:]:
            sim = pipeline.similarity(e1, e2)
            print(f"    {r1}ejo ↔ {r2}ejo: {sim:.3f}")

    print("\n--- -ist suffix (person clustering) ---")
    person_roots = ["labor", "art", "muzik", "scien"]
    person_embs = []
    for root in person_roots:
        emb = pipeline.get_root_embedding(root)
        if emb is not None and 'ist' in pipeline.suffix_transforms:
            ist_emb = pipeline.suffix_transforms['ist'](emb)
            person_embs.append((root, ist_emb))

    print("  Pairwise similarities of -isto words:")
    for i, (r1, e1) in enumerate(person_embs):
        for r2, e2 in person_embs[i+1:]:
            sim = pipeline.similarity(e1, e2)
            print(f"    {r1}isto ↔ {r2}isto: {sim:.3f}")


def demo_compositional(pipeline: KlarecoPipeline):
    """Demonstrate full compositional embeddings."""
    print_header("Compositional Word Embeddings")

    print("\n--- Parse + Embed ---")
    test_words = [
        "lernejo",      # lern + ej + o
        "malbona",      # mal + bon + a
        "laboristo",    # labor + ist + o
        "relerni",      # re + lern + i
        "malgrandega",  # mal + grand + eg + a
    ]

    for word in test_words:
        ast = parse(word)
        # Find the word node
        def find_word(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    return node
                for v in node.values():
                    result = find_word(v)
                    if result:
                        return result
            elif isinstance(node, list):
                for item in node:
                    result = find_word(item)
                    if result:
                        return result
            return None

        word_ast = find_word(ast)
        if word_ast:
            parts = []
            if word_ast.get('prefiksoj'):
                parts.extend(word_ast['prefiksoj'])
            parts.append(word_ast.get('radiko', '?'))
            if word_ast.get('sufiksoj'):
                parts.extend(word_ast['sufiksoj'])
            parts.append(word_ast.get('finaĵo', '?'))

            emb = pipeline.get_word_embedding(word_ast)
            status = "✓" if emb is not None else "✗"
            print(f"  {word}: {' + '.join(parts)} {status}")

    print("\n--- Novel Word Generalization ---")
    print("  Testing words that may not be in training data...")

    # Create embeddings for comparison
    lernejo = parse("lernejo")
    laborejo = parse("laborejo")

    def get_word_emb(text):
        ast = parse(text)
        def find_word(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    return node
                for v in node.values():
                    result = find_word(v)
                    if result:
                        return result
            elif isinstance(node, list):
                for item in node:
                    result = find_word(item)
                    if result:
                        return result
            return None
        word_ast = find_word(ast)
        if word_ast:
            return pipeline.get_word_embedding(word_ast)
        return None

    lernejo_emb = get_word_emb("lernejo")
    laborejo_emb = get_word_emb("laborejo")
    laboristo_emb = get_word_emb("laboristo")

    if lernejo_emb is not None and laborejo_emb is not None:
        sim = pipeline.similarity(lernejo_emb, laborejo_emb)
        print(f"  lernejo ↔ laborejo (both places): {sim:.3f}")

    if lernejo_emb is not None and laboristo_emb is not None:
        sim = pipeline.similarity(lernejo_emb, laboristo_emb)
        print(f"  lernejo ↔ laboristo (place vs person): {sim:.3f}")


def demo_interactive(pipeline: KlarecoPipeline):
    """Interactive demo mode."""
    print_header("Interactive Mode")
    print("Enter Esperanto words or sentences to analyze.")
    print("Commands: 'quit' to exit, 'sim word1 word2' for similarity")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĜis revido!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Ĝis revido!")
            break

        if user_input.lower().startswith('sim '):
            # Similarity comparison
            parts = user_input[4:].split()
            if len(parts) >= 2:
                word1, word2 = parts[0], parts[1]

                def get_word_emb(text):
                    ast = parse(text)
                    def find_word(node):
                        if isinstance(node, dict):
                            if node.get('tipo') == 'vorto':
                                return node
                            for v in node.values():
                                result = find_word(v)
                                if result:
                                    return result
                        elif isinstance(node, list):
                            for item in node:
                                result = find_word(item)
                                if result:
                                    return result
                        return None
                    word_ast = find_word(ast)
                    if word_ast:
                        return pipeline.get_word_embedding(word_ast), word_ast
                    return None, None

                emb1, ast1 = get_word_emb(word1)
                emb2, ast2 = get_word_emb(word2)

                if emb1 is not None and emb2 is not None:
                    sim = pipeline.similarity(emb1, emb2)
                    print(f"  Similarity: {sim:.3f}")
                else:
                    if emb1 is None:
                        print(f"  Could not embed: {word1}")
                    if emb2 is None:
                        print(f"  Could not embed: {word2}")
            continue

        # Parse and analyze
        ast = parse(user_input)
        print("\nParsed AST:")
        print_ast(ast)

        # Show word embeddings
        def find_words(node, words=None):
            if words is None:
                words = []
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    words.append(node)
                for v in node.values():
                    find_words(v, words)
            elif isinstance(node, list):
                for item in node:
                    find_words(item, words)
            return words

        words = find_words(ast)
        if words:
            print("\nWord embeddings:")
            for word_ast in words:
                emb = pipeline.get_word_embedding(word_ast)
                radiko = word_ast.get('radiko', '?')
                status = "✓ embedded" if emb is not None else "✗ unknown root"
                print(f"  {radiko}: {status}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Klareco Pipeline Demo")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Run in interactive mode")
    args = parser.parse_args()

    print("=" * 60)
    print(" KLARECO PIPELINE DEMO")
    print(" Esperanto-First AI with Minimal Learned Parameters")
    print("=" * 60)

    # Load pipeline
    print("\nLoading models...")
    pipeline = KlarecoPipeline()

    # Print model information
    pipeline.print_model_info()

    if args.interactive:
        demo_interactive(pipeline)
    else:
        # Run all demos
        demo_stage0()
        demo_stage1a(pipeline)
        demo_stage1b(pipeline)
        demo_compositional(pipeline)

        print_header("Summary")
        print("""
Pipeline Components:
  Stage 0: Parser (deterministic)     - 0 params
  Stage 1a: Root Embeddings           - ~712K params
  Stage 1b: Affix Transforms          - ~21K params
  ─────────────────────────────────────────────
  Total Learned Parameters:           - ~733K params

Key Insight:
  By making grammar 100% deterministic and only learning
  semantics, we achieve compositional generalization to
  novel words while keeping parameters minimal.

Next Stages (not yet implemented):
  Stage 2: Grammatical Model (~52K params)
  Stage 3: Discourse Model (~100K params)
  Stage 4: Reasoning Core (20-100M params)
""")

        print("\nRun with --interactive for live testing!")


if __name__ == '__main__':
    main()
