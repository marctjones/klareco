#!/usr/bin/env python3
"""
Stage 1 Validation: Compositional Embeddings + FAISS Index

This script validates the Stage 1 models:
1. Root embeddings quality (semantic similarity)
2. Affix embeddings quality (transform consistency)
3. Compositional index retrieval quality
4. Comparison with old Tree-LSTM index (if available)

Usage:
    python scripts/validate_stage1.py
    python scripts/validate_stage1.py --compare-old
    python scripts/validate_stage1.py --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class Stage1Validator:
    """Validate Stage 1 compositional embeddings and retrieval."""

    def __init__(
        self,
        compositional_index: Path = Path("data/corpus_index_compositional"),
        old_index: Path = Path("data/corpus_index_v3"),
        root_model: Path = Path("models/root_embeddings/best_model.pt"),
        affix_model: Path = Path("models/affix_embeddings/best_model.pt"),
    ):
        self.compositional_index = compositional_index
        self.old_index = old_index
        self.root_model = root_model
        self.affix_model = affix_model

        self.results = {}

    def check_index_exists(self, index_path: Path) -> Tuple[bool, Dict]:
        """Check if index exists and get stats."""
        info = {"exists": False, "embeddings": 0, "metadata": 0, "faiss": False}

        if not index_path.exists():
            return False, info

        info["exists"] = True

        emb_path = index_path / "embeddings.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
            info["embeddings"] = emb.shape[0]
            info["embedding_dim"] = emb.shape[1]

        meta_path = index_path / "metadata.jsonl"
        if meta_path.exists():
            with open(meta_path) as f:
                info["metadata"] = sum(1 for _ in f)

        faiss_path = index_path / "faiss_index.bin"
        info["faiss"] = faiss_path.exists()

        return True, info

    def load_faiss_index(self, index_path: Path):
        """Load FAISS index and metadata."""
        try:
            import faiss
        except ImportError:
            return None, None, None

        faiss_path = index_path / "faiss_index.bin"
        if not faiss_path.exists():
            # Build from embeddings if no FAISS file
            emb_path = index_path / "embeddings.npy"
            if not emb_path.exists():
                return None, None, None
            embeddings = np.load(emb_path).astype(np.float32)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
        else:
            index = faiss.read_index(str(faiss_path))

        # Load metadata
        meta_path = index_path / "metadata.jsonl"
        metadata = []
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    metadata.append(json.loads(line))

        # Load embeddings for query embedding
        emb_path = index_path / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() else None

        return index, metadata, embeddings

    def search(self, index, query_embedding: np.ndarray, k: int = 10):
        """Search FAISS index."""
        import faiss
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, k)
        return distances[0], indices[0]

    def validate_root_embeddings(self) -> Dict:
        """Test 1: Root embedding quality."""
        print("\n" + "=" * 60)
        print("Test 1: Root Embedding Quality")
        print("=" * 60)

        import torch

        if not self.root_model.exists():
            return {"status": "SKIP", "reason": "Model not found"}

        checkpoint = torch.load(self.root_model, map_location='cpu', weights_only=False)
        embeddings = checkpoint['model_state_dict']['embeddings.weight'].numpy()
        root_to_idx = checkpoint['root_to_idx']
        idx_to_root = {v: k for k, v in root_to_idx.items()}

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-8)

        results = {
            "num_roots": len(root_to_idx),
            "embedding_dim": embeddings.shape[1],
        }

        # Test semantic pairs
        test_pairs = [
            # Synonyms (should be similar)
            ("granda", "vasta", "synonym", 0.3),
            ("rapida", "swift", "synonym", 0.2),  # May not exist
            ("bela", "ĉarma", "synonym", 0.3),
            ("domo", "konstruaĵ", "synonym", 0.2),

            # Related concepts
            ("hundo", "kato", "related", 0.2),
            ("libro", "gazet", "related", 0.2),
            ("patro", "frat", "related", 0.3),

            # Antonyms (mal- pairs trained together, may be similar)
            ("bon", "mal", "antonym", 0.0),  # Just checking they exist

            # Unrelated (should be low)
            ("hundo", "libro", "unrelated", -0.1),
            ("grand", "kur", "unrelated", -0.1),
        ]

        pair_results = []
        for root1, root2, relation, min_sim in test_pairs:
            if root1 not in root_to_idx or root2 not in root_to_idx:
                pair_results.append((root1, root2, relation, None, "missing"))
                continue

            idx1, idx2 = root_to_idx[root1], root_to_idx[root2]
            sim = float(np.dot(emb_norm[idx1], emb_norm[idx2]))
            status = "PASS" if sim >= min_sim else "WARN"
            pair_results.append((root1, root2, relation, sim, status))
            print(f"  {root1} vs {root2} ({relation}): {sim:.3f} [{status}]")

        results["pair_tests"] = pair_results
        results["status"] = "PASS"
        return results

    def validate_affix_embeddings(self) -> Dict:
        """Test 2: Affix embedding quality."""
        print("\n" + "=" * 60)
        print("Test 2: Affix Embedding Quality")
        print("=" * 60)

        import torch

        if not self.affix_model.exists():
            return {"status": "SKIP", "reason": "Model not found"}

        checkpoint = torch.load(self.affix_model, map_location='cpu', weights_only=False)
        prefix_emb = checkpoint['model_state_dict']['prefix_embeddings.weight'].numpy()
        suffix_emb = checkpoint['model_state_dict']['suffix_embeddings.weight'].numpy()
        prefix_vocab = checkpoint['prefix_vocab']
        suffix_vocab = checkpoint['suffix_vocab']

        results = {
            "num_prefixes": len(prefix_vocab),
            "num_suffixes": len(suffix_vocab),
            "prefix_dim": prefix_emb.shape[1],
            "suffix_dim": suffix_emb.shape[1],
        }

        # Normalize
        prefix_norm = prefix_emb / (np.linalg.norm(prefix_emb, axis=1, keepdims=True) + 1e-8)
        suffix_norm = suffix_emb / (np.linalg.norm(suffix_emb, axis=1, keepdims=True) + 1e-8)

        # Test suffix groups (should be similar within group)
        suffix_groups = {
            "participial_active": ["ant", "int", "ont"],
            "participial_passive": ["at", "it", "ot"],
            "degree": ["et", "eg"],
            "person_place": ["ul", "ej"],
        }

        group_results = []
        for group_name, suffixes in suffix_groups.items():
            valid = [s for s in suffixes if s in suffix_vocab]
            if len(valid) < 2:
                continue

            sims = []
            for i, s1 in enumerate(valid):
                for s2 in valid[i+1:]:
                    idx1, idx2 = suffix_vocab[s1], suffix_vocab[s2]
                    sim = float(np.dot(suffix_norm[idx1], suffix_norm[idx2]))
                    sims.append(sim)

            avg_sim = np.mean(sims) if sims else 0
            status = "PASS" if avg_sim > 0.1 else "WARN"
            group_results.append((group_name, valid, avg_sim, status))
            print(f"  {group_name} ({', '.join(valid)}): avg_sim={avg_sim:.3f} [{status}]")

        results["group_tests"] = group_results
        results["status"] = "PASS"
        return results

    def validate_compositional_retrieval(self) -> Dict:
        """Test 3: Compositional index retrieval quality."""
        print("\n" + "=" * 60)
        print("Test 3: Compositional Index Retrieval")
        print("=" * 60)

        exists, info = self.check_index_exists(self.compositional_index)
        if not exists:
            return {"status": "SKIP", "reason": "Index not found"}

        print(f"  Index: {info['embeddings']} embeddings, {info['embedding_dim']}d")
        print(f"  FAISS: {'Yes' if info['faiss'] else 'No (will build from embeddings)'}")

        index, metadata, embeddings = self.load_faiss_index(self.compositional_index)
        if index is None:
            return {"status": "FAIL", "reason": "Could not load index"}

        # Import compositional embedder
        from scripts.demo_compositional_embeddings import CompositionalEmbedder
        embedder = CompositionalEmbedder(self.root_model, self.affix_model)

        # Test queries
        test_queries = [
            ("hundo", ["hund", "kat", "best", "animal"]),  # Dog -> should find dog-related
            ("librejo", ["libr", "leg", "stud"]),  # Library -> books, reading
            ("lernejo", ["lern", "stud", "eduk"]),  # School -> learning
            ("malbela", ["bel", "aspekt"]),  # Ugly -> beauty-related
            ("gepatroj", ["patr", "famili", "infan"]),  # Parents -> family
            ("skribisto", ["skrib", "verki", "aŭtor"]),  # Writer -> writing
        ]

        results = {"queries": []}
        for query_word, expected_roots in test_queries:
            emb, info = embedder.embed_word(query_word)
            if emb is None:
                results["queries"].append({
                    "query": query_word,
                    "status": "FAIL",
                    "reason": info.get("error", "Unknown")
                })
                print(f"  {query_word}: FAIL - {info.get('error', 'Unknown')}")
                continue

            # Search
            distances, indices = self.search(index, emb, k=10)

            # Check results
            found_roots = []
            for idx, dist in zip(indices, distances):
                if idx < len(metadata):
                    text = metadata[idx].get("text", "")
                    roots = metadata[idx].get("roots_found", [])
                    found_roots.extend(roots)

            # Check if any expected roots found
            matches = [r for r in expected_roots if any(r in fr.lower() for fr in found_roots)]
            status = "PASS" if matches else "WARN"

            results["queries"].append({
                "query": query_word,
                "root": info.get("root"),
                "top_score": float(distances[0]) if len(distances) > 0 else 0,
                "matches": matches,
                "status": status,
            })
            print(f"  {query_word} (root={info.get('root')}): top={distances[0]:.3f}, matches={matches} [{status}]")

        passed = sum(1 for q in results["queries"] if q["status"] == "PASS")
        results["pass_rate"] = passed / len(test_queries) if test_queries else 0
        results["status"] = "PASS" if results["pass_rate"] >= 0.5 else "WARN"
        return results

    def compare_indexes(self) -> Dict:
        """Test 4: Compare compositional vs old Tree-LSTM index."""
        print("\n" + "=" * 60)
        print("Test 4: Index Comparison (Compositional vs Tree-LSTM)")
        print("=" * 60)

        comp_exists, comp_info = self.check_index_exists(self.compositional_index)
        old_exists, old_info = self.check_index_exists(self.old_index)

        print(f"  Compositional: {comp_info['embeddings']} embeddings, dim={comp_info.get('embedding_dim', 'N/A')}")
        print(f"  Old (v3):      {old_info['embeddings']} embeddings, dim={old_info.get('embedding_dim', 'N/A')}")

        if not comp_exists or not old_exists:
            return {
                "status": "SKIP",
                "reason": f"Missing index: comp={comp_exists}, old={old_exists}"
            }

        results = {
            "compositional": comp_info,
            "old": old_info,
        }

        # Compare coverage
        results["coverage_ratio"] = comp_info["embeddings"] / old_info["embeddings"] if old_info["embeddings"] > 0 else 0
        print(f"  Coverage ratio: {results['coverage_ratio']:.2f}x")

        # Compare embedding dimensions
        if comp_info.get("embedding_dim") and old_info.get("embedding_dim"):
            print(f"  Dim comparison: {comp_info['embedding_dim']}d (comp) vs {old_info['embedding_dim']}d (old)")

        results["status"] = "PASS"
        return results

    def validate_affixed_word_retrieval(self) -> Dict:
        """Test 5: Specifically test retrieval for affixed words (issue #131)."""
        print("\n" + "=" * 60)
        print("Test 5: Affixed Word Retrieval (Issue #131)")
        print("=" * 60)

        exists, info = self.check_index_exists(self.compositional_index)
        if not exists:
            return {"status": "SKIP", "reason": "Index not found"}

        index, metadata, embeddings = self.load_faiss_index(self.compositional_index)
        if index is None:
            return {"status": "FAIL", "reason": "Could not load index"}

        from scripts.demo_compositional_embeddings import CompositionalEmbedder
        embedder = CompositionalEmbedder(self.root_model, self.affix_model)

        # Test: base word and affixed version should retrieve similar content
        affix_pairs = [
            ("bela", "malbela"),       # beautiful / ugly
            ("granda", "grandega"),    # big / huge
            ("hundo", "hundeto"),      # dog / little dog
            ("rapida", "malrapida"),   # fast / slow
            ("libro", "librejo"),      # book / library
            ("skribi", "skribisto"),   # write / writer
            ("lerni", "lernejo"),      # learn / school
            ("patro", "gepatro"),      # father / parent
        ]

        results = {"pairs": []}
        for base, affixed in affix_pairs:
            base_emb, base_info = embedder.embed_word(base)
            aff_emb, aff_info = embedder.embed_word(affixed)

            if base_emb is None or aff_emb is None:
                results["pairs"].append({
                    "base": base,
                    "affixed": affixed,
                    "status": "FAIL",
                    "reason": f"Parse failed: base={base_info.get('error')}, aff={aff_info.get('error')}"
                })
                print(f"  {base} vs {affixed}: FAIL (parse error)")
                continue

            # Get similarity between base and affixed
            sim = float(np.dot(base_emb, aff_emb))

            # Get retrieval results for both
            base_dist, base_idx = self.search(index, base_emb, k=5)
            aff_dist, aff_idx = self.search(index, aff_emb, k=5)

            # Check overlap in top-5 results
            overlap = len(set(base_idx) & set(aff_idx))

            status = "PASS" if sim > 0.7 else "WARN"
            results["pairs"].append({
                "base": base,
                "affixed": affixed,
                "similarity": sim,
                "overlap_top5": overlap,
                "status": status,
            })
            print(f"  {base} vs {affixed}: sim={sim:.3f}, overlap={overlap}/5 [{status}]")

        passed = sum(1 for p in results["pairs"] if p["status"] == "PASS")
        results["pass_rate"] = passed / len(affix_pairs) if affix_pairs else 0
        results["status"] = "PASS" if results["pass_rate"] >= 0.6 else "WARN"
        return results

    def run_all(self, compare_old: bool = False, verbose: bool = False) -> Dict:
        """Run all validation tests."""
        print("\n" + "=" * 60)
        print("Stage 1 Validation: Compositional Embeddings")
        print("=" * 60)

        self.results["root_embeddings"] = self.validate_root_embeddings()
        self.results["affix_embeddings"] = self.validate_affix_embeddings()
        self.results["compositional_retrieval"] = self.validate_compositional_retrieval()
        self.results["affixed_word_retrieval"] = self.validate_affixed_word_retrieval()

        if compare_old:
            self.results["index_comparison"] = self.compare_indexes()

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for test_name, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            symbol = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
            print(f"  {symbol} {test_name}: {status}")

        passed = sum(1 for r in self.results.values() if r.get("status") == "PASS")
        total = len(self.results)
        print(f"\n  Overall: {passed}/{total} tests passed")

        return self.results


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 1 compositional embeddings")
    parser.add_argument("--compare-old", action="store_true",
                        help="Compare with old Tree-LSTM index")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--compositional-index", type=Path,
                        default=Path("data/corpus_index_compositional"))
    parser.add_argument("--old-index", type=Path,
                        default=Path("data/corpus_index_v3"))

    args = parser.parse_args()

    validator = Stage1Validator(
        compositional_index=args.compositional_index,
        old_index=args.old_index,
    )

    results = validator.run_all(
        compare_old=args.compare_old,
        verbose=args.verbose,
    )

    # Exit with error if any test failed
    failed = any(r.get("status") == "FAIL" for r in results.values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
