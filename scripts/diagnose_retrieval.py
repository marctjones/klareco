#!/usr/bin/env python3
"""
Comprehensive Retrieval Pipeline Diagnostic Script
===================================================

This script diagnoses where the retrieval pipeline is breaking down by:
1. Query Embedding Verification - Are query embeddings generated correctly?
2. HNSW Search Transparency - What does HNSW return before reranking?
3. Ground Truth Comparison - How does HNSW compare to keyword search?
4. Index Integrity Verification - Are documents indexed with correct embeddings?
5. Slot Embedding Sanity Check - Are slots assigned correctly?
6. Semantic Pipeline Alignment - Do 64d and 128d embeddings align?

Usage:
    python scripts/diagnose_retrieval.py
    python scripts/diagnose_retrieval.py --query "Kiu kreis Esperanton?"
    python scripts/diagnose_retrieval.py --detailed
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from klareco.parser import parse as parse_sentence
from klareco.rag.ast_aware_retriever import ASTAwareRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST QUERIES - Diverse set with known ground truth
# ============================================================================

TEST_QUERIES = [
    {
        "query": "Kiu kreis Esperanton?",
        "type": "WHO",
        "keywords": ["zamenhof", "esperant", "kre", "fond"],
        "expected_answer": "Zamenhof",
        "must_contain": ["zamenhof"],
    },
    {
        "query": "Kio estas la Fundamento?",
        "type": "WHAT",
        "keywords": ["fundamento", "libro", "baz"],
        "expected_answer": "libro/bazo de Esperanto",
        "must_contain": ["fundamento"],
    },
    {
        "query": "Kie naskiĝis Zamenhof?",
        "type": "WHERE",
        "keywords": ["zamenhof", "naskiĝ", "bialystok", "bjalistok"],
        "expected_answer": "Bialystok",
        "must_contain": ["zamenhof", "naskiĝ"],
    },
    {
        "query": "Kio estas la akuzativo?",
        "type": "WHAT",
        "keywords": ["akuzativ", "kazo", "-n"],
        "expected_answer": "kazo markita per -n",
        "must_contain": ["akuzativ"],
    },
    {
        "query": "Kio estas hundo?",
        "type": "WHAT",
        "keywords": ["hundo", "besto", "animal"],
        "expected_answer": "besto/animalo",
        "must_contain": ["hundo"],
    },
]


class RetrievalDiagnostics:
    """Comprehensive retrieval pipeline diagnostics."""

    def __init__(self, index_path: Path, detailed: bool = False):
        self.index_path = index_path
        self.detailed = detailed
        self.parse = parse_sentence  # Use module-level function
        self.retriever = None
        self.results = {}

        # Load retriever
        self._load_retriever()

    def _load_retriever(self):
        """Load the ASTAwareRetriever."""
        logger.info(f"Loading retriever from {self.index_path}")
        try:
            self.retriever = ASTAwareRetriever(
                index_path=self.index_path,
                use_prefilter=True,
            )
            logger.info("Retriever loaded successfully")

            # Log configuration
            logger.info(f"  HNSW index loaded: {self.retriever.hnsw_index is not None}")
            logger.info(f"  Semantic pipeline: {self.retriever.semantic_pipeline is not None}")
            logger.info(f"  Document count: {len(self.retriever.doc_offsets) if self.retriever.doc_offsets is not None else 0}")

            if self.retriever.hnsw_index:
                logger.info(f"  HNSW index size: {self.retriever.hnsw_index.get_current_count()}")

        except Exception as e:
            logger.error(f"Failed to load retriever: {e}")
            raise

    def run_all_diagnostics(self, queries: List[Dict] = None):
        """Run all diagnostic phases."""
        queries = queries or TEST_QUERIES

        print("\n" + "=" * 80)
        print("RETRIEVAL PIPELINE DIAGNOSTICS")
        print("=" * 80)

        for i, query_spec in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"Query {i}/{len(queries)}: {query_spec['query']}")
            print(f"Type: {query_spec['type']} | Expected: {query_spec.get('expected_answer', 'N/A')}")
            print(f"{'─' * 80}")

            self._diagnose_single_query(query_spec)

        # Print summary
        self._print_summary()

    def _diagnose_single_query(self, query_spec: Dict):
        """Run all diagnostics for a single query."""
        query = query_spec["query"]
        keywords = query_spec["keywords"]
        must_contain = query_spec.get("must_contain", [])

        # Phase 1: Query Embedding Verification
        print("\n[PHASE 1] Query Embedding Verification")
        query_embedding_result = self._phase1_query_embedding(query)

        # Phase 2: HNSW Search Transparency
        print("\n[PHASE 2] HNSW Search Transparency")
        hnsw_result = self._phase2_hnsw_search(query, query_embedding_result)

        # Phase 3: Ground Truth Comparison
        print("\n[PHASE 3] Ground Truth Comparison (Keyword Baseline)")
        ground_truth_result = self._phase3_ground_truth(query, keywords, must_contain, hnsw_result)

        # Phase 4: Index Integrity (for found docs)
        print("\n[PHASE 4] Index Integrity Check")
        self._phase4_index_integrity(ground_truth_result)

        # Phase 5: Slot Embedding Analysis
        print("\n[PHASE 5] Slot Embedding Analysis")
        self._phase5_slot_analysis(query)

        # Phase 6: Semantic Pipeline Check
        print("\n[PHASE 6] Semantic Pipeline Alignment")
        self._phase6_semantic_alignment(query)

        # Store results
        self.results[query] = {
            "query_embedding": query_embedding_result,
            "hnsw": hnsw_result,
            "ground_truth": ground_truth_result,
        }

    def _phase1_query_embedding(self, query: str) -> Dict:
        """Phase 1: Verify query embedding generation."""
        result = {
            "success": False,
            "parse_success": False,
            "embedding_dim": None,
            "embedding_norm": None,
            "slot_embeddings": {},
        }

        try:
            # Parse query
            ast = self.parse(query)
            parse_stats = ast.get("parse_statistics", {})
            result["parse_success"] = parse_stats.get("success_rate", 0) > 0.5

            print(f"  Parse success rate: {parse_stats.get('success_rate', 0):.1%}")
            print(f"  Words parsed: {parse_stats.get('parsed_words', 0)}/{parse_stats.get('total_words', 0)}")

            # Get slot embeddings
            if hasattr(self.retriever, '_extract_slot_embeddings'):
                slots = self.retriever._extract_slot_embeddings(ast)
                for slot_name, slot_emb in slots.items():
                    if slot_emb is not None:
                        if isinstance(slot_emb, torch.Tensor):
                            emb_np = slot_emb.cpu().numpy()
                        else:
                            emb_np = np.array(slot_emb)
                        norm = np.linalg.norm(emb_np)
                        result["slot_embeddings"][slot_name] = {
                            "dim": len(emb_np),
                            "norm": float(norm),
                            "first_5": emb_np[:5].tolist(),
                            "is_zero": norm < 1e-6,
                        }
                        status = "✓" if norm > 1e-6 else "✗ ZERO"
                        print(f"  Slot {slot_name}: dim={len(emb_np)}, norm={norm:.4f} {status}")
                    else:
                        print(f"  Slot {slot_name}: None")
                        result["slot_embeddings"][slot_name] = None

            # Get full query embedding (if applicable)
            if hasattr(self.retriever, '_build_query_embedding'):
                query_emb = self.retriever._build_query_embedding(ast)
                if query_emb is not None:
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy()
                    result["embedding_dim"] = len(query_emb)
                    result["embedding_norm"] = float(np.linalg.norm(query_emb))
                    print(f"  Full query embedding: dim={result['embedding_dim']}, norm={result['embedding_norm']:.4f}")

            result["success"] = True

        except Exception as e:
            print(f"  ERROR: {e}")
            result["error"] = str(e)

        return result

    def _phase2_hnsw_search(self, query: str, query_embedding_result: Dict) -> Dict:
        """Phase 2: Examine raw HNSW search results."""
        result = {
            "success": False,
            "raw_results": [],
            "distances": [],
            "texts": [],
        }

        try:
            if not self.retriever.hnsw_index:
                print("  HNSW index not available - using keyword fallback")
                result["fallback"] = True
                return result

            # Parse and get query embedding for HNSW
            ast = self.parse(query)

            # Get slot embeddings and combine for HNSW query
            slots = self.retriever._extract_slot_embeddings(ast)

            # Build combined query vector (same as retriever does internally)
            slot_vectors = []
            for slot_name in ['SUBJ', 'VERB', 'OBJ']:
                slot_emb = slots.get(slot_name)
                if slot_emb is not None:
                    if isinstance(slot_emb, torch.Tensor):
                        slot_vectors.append(slot_emb.cpu().numpy())
                    else:
                        slot_vectors.append(np.array(slot_emb))

            if slot_vectors:
                # Average slot vectors
                query_vector = np.mean(slot_vectors, axis=0)
            else:
                print("  No slot vectors available")
                return result

            # Ensure correct dimension
            expected_dim = self.retriever.hnsw_index.get_max_elements()
            actual_dim = len(query_vector)

            # Get HNSW index dimension
            if hasattr(self.retriever.hnsw_index, 'dim'):
                index_dim = self.retriever.hnsw_index.dim
            else:
                # Try to infer from a sample
                index_dim = 128  # Default assumption

            print(f"  Query vector dim: {actual_dim}, Index dim: {index_dim}")

            # Perform raw HNSW search
            k = 50  # Get top 50 for analysis

            # Normalize query vector
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            query_vector = query_vector.astype(np.float32).reshape(1, -1)

            labels, distances = self.retriever.hnsw_index.knn_query(query_vector, k=k)

            print(f"  HNSW returned {len(labels[0])} results")
            print(f"  Distance range: {distances[0].min():.4f} to {distances[0].max():.4f}")

            # Get document texts for top results
            for i, (label, dist) in enumerate(zip(labels[0][:10], distances[0][:10])):
                try:
                    doc = self.retriever._get_document(int(label))
                    if doc:
                        text = doc.get("text", doc.get("teksto", ""))[:100]
                        result["raw_results"].append({
                            "rank": i + 1,
                            "label": int(label),
                            "distance": float(dist),
                            "text": text,
                        })
                        print(f"    #{i+1}: dist={dist:.4f} | {text[:60]}...")
                except Exception as e:
                    print(f"    #{i+1}: dist={dist:.4f} | ERROR loading doc: {e}")

            result["success"] = True
            result["distances"] = distances[0].tolist()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)

        return result

    def _phase3_ground_truth(self, query: str, keywords: List[str],
                              must_contain: List[str], hnsw_result: Dict) -> Dict:
        """Phase 3: Compare HNSW results to keyword ground truth."""
        result = {
            "keyword_matches": [],
            "overlap_with_hnsw": 0,
            "first_relevant_rank": None,
            "recall_at_10": 0.0,
            "recall_at_50": 0.0,
        }

        try:
            index_file = self.index_path / "slot_index.jsonl"

            # Find documents containing must_contain keywords
            relevant_doc_indices = set()

            for kw in must_contain:
                cmd = ["grep", "-n", "-i", kw, str(index_file)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    for line in proc.stdout.strip().split("\n")[:100]:  # Limit
                        line_num = int(line.split(":")[0]) - 1  # 0-indexed
                        relevant_doc_indices.add(line_num)

            print(f"  Keyword search found {len(relevant_doc_indices)} relevant documents")

            # Check overlap with HNSW results
            if hnsw_result.get("raw_results"):
                hnsw_labels = {r["label"] for r in hnsw_result["raw_results"]}
                overlap = relevant_doc_indices & hnsw_labels
                result["overlap_with_hnsw"] = len(overlap)

                # Find first relevant rank
                for r in hnsw_result["raw_results"]:
                    if r["label"] in relevant_doc_indices:
                        result["first_relevant_rank"] = r["rank"]
                        print(f"  First relevant doc at rank {r['rank']}: {r['text'][:50]}...")
                        break

                if result["first_relevant_rank"] is None:
                    print(f"  ⚠ No relevant docs found in HNSW top-{len(hnsw_result['raw_results'])}")

                # Calculate recall
                hnsw_top_10 = {r["label"] for r in hnsw_result["raw_results"][:10]}
                hnsw_top_50 = {r["label"] for r in hnsw_result["raw_results"][:50]}

                if relevant_doc_indices:
                    result["recall_at_10"] = len(hnsw_top_10 & relevant_doc_indices) / len(relevant_doc_indices)
                    result["recall_at_50"] = len(hnsw_top_50 & relevant_doc_indices) / len(relevant_doc_indices)
                    print(f"  Recall@10: {result['recall_at_10']:.1%}")
                    print(f"  Recall@50: {result['recall_at_50']:.1%}")

            # Sample some relevant docs for inspection
            if relevant_doc_indices:
                sample_indices = list(relevant_doc_indices)[:3]
                for idx in sample_indices:
                    if idx < len(self.retriever.doc_offsets):
                        try:
                            doc = self.retriever._get_document(idx)
                            text = doc.get("text", doc.get("teksto", ""))[:100]
                            result["keyword_matches"].append({
                                "index": idx,
                                "text": text,
                            })
                            if self.detailed:
                                print(f"    Sample relevant doc [{idx}]: {text[:60]}...")
                        except Exception as e:
                            print(f"    Sample doc [{idx}]: ERROR: {e}")

        except Exception as e:
            print(f"  ERROR: {e}")
            result["error"] = str(e)

        return result

    def _phase4_index_integrity(self, ground_truth_result: Dict):
        """Phase 4: Verify index integrity for known documents."""
        try:
            # Check embedding integrity for sample documents
            sample_docs = ground_truth_result.get("keyword_matches", [])[:3]

            for doc_info in sample_docs:
                idx = doc_info["index"]
                if idx < len(self.retriever.doc_offsets):
                    doc = self.retriever._get_document(idx)

                    # Check slots
                    slots = doc.get("slots", {})
                    print(f"  Doc [{idx}] slot analysis:")

                    for slot_name in ["SUBJ", "VERB", "OBJ"]:
                        slot_data = slots.get(slot_name)
                        if slot_data is not None:
                            if isinstance(slot_data, list):
                                arr = np.array(slot_data)
                                norm = np.linalg.norm(arr)
                                is_zero = norm < 1e-6
                                status = "✗ ZERO" if is_zero else "✓"
                                print(f"    {slot_name}: dim={len(arr)}, norm={norm:.4f} {status}")
                            else:
                                print(f"    {slot_name}: unexpected type {type(slot_data)}")
                        else:
                            print(f"    {slot_name}: None")

                    # Check full embedding
                    full_emb = doc.get("full_embedding")
                    if full_emb:
                        arr = np.array(full_emb)
                        norm = np.linalg.norm(arr)
                        print(f"    full_embedding: dim={len(arr)}, norm={norm:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    def _phase5_slot_analysis(self, query: str):
        """Phase 5: Analyze slot assignment and embedding quality."""
        try:
            ast = self.parse(query)

            print("  Query AST slots:")

            # Check subjekto
            subj = ast.get("subjekto")
            if subj:
                kerno = subj.get("kerno", {})
                radiko = kerno.get("radiko", "")
                print(f"    SUBJ: radiko='{radiko}'")
            else:
                print(f"    SUBJ: None")

            # Check verbo
            verbo = ast.get("verbo")
            if verbo:
                radiko = verbo.get("radiko", "")
                tempo = verbo.get("tempo", "")
                print(f"    VERB: radiko='{radiko}', tempo='{tempo}'")
            else:
                print(f"    VERB: None")

            # Check objekto
            obj = ast.get("objekto")
            if obj:
                kerno = obj.get("kerno", {})
                radiko = kerno.get("radiko", "")
                print(f"    OBJ: radiko='{radiko}'")
            else:
                print(f"    OBJ: None")

            # Check aliaj (other elements)
            aliaj = ast.get("aliaj", [])
            if aliaj:
                print(f"    aliaj: {len(aliaj)} elements")
                for a in aliaj[:3]:
                    if isinstance(a, dict):
                        radiko = a.get("radiko", a.get("kerno", {}).get("radiko", "?"))
                        print(f"      - {radiko}")

        except Exception as e:
            print(f"  ERROR: {e}")

    def _phase6_semantic_alignment(self, query: str):
        """Phase 6: Check alignment between SemanticPipeline and HybridEmbeddings."""
        try:
            if not self.retriever.semantic_pipeline:
                print("  SemanticPipeline not available")
                return

            # Get SemanticPipeline embeddings (64d)
            enriched = self.retriever.semantic_pipeline.for_retrieval(query)

            if hasattr(enriched, 'content_embeddings') and enriched.content_embeddings:
                print(f"  SemanticPipeline content_embeddings: {len(enriched.content_embeddings)} words")

                for word_id, emb in list(enriched.content_embeddings.items())[:3]:
                    if isinstance(emb, torch.Tensor):
                        emb_np = emb.cpu().numpy()
                    else:
                        emb_np = np.array(emb)
                    norm = np.linalg.norm(emb_np)
                    print(f"    '{word_id}': dim={len(emb_np)}, norm={norm:.4f}")
            else:
                print("  No content_embeddings available")

            # Check if slot embeddings use same dimension
            if hasattr(self.retriever, '_extract_slot_embeddings'):
                ast = self.parse(query)
                slots = self.retriever._extract_slot_embeddings(ast)

                for slot_name, slot_emb in slots.items():
                    if slot_emb is not None:
                        if isinstance(slot_emb, torch.Tensor):
                            dim = slot_emb.shape[-1]
                        else:
                            dim = len(slot_emb)
                        print(f"    Slot {slot_name} embedding dim: {dim}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _print_summary(self):
        """Print summary of all diagnostics."""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        total_queries = len(self.results)
        found_relevant = 0
        avg_first_rank = []

        for query, result in self.results.items():
            gt = result.get("ground_truth", {})
            first_rank = gt.get("first_relevant_rank")
            if first_rank:
                found_relevant += 1
                avg_first_rank.append(first_rank)

        print(f"\nQueries with relevant docs in HNSW results: {found_relevant}/{total_queries}")
        if avg_first_rank:
            print(f"Average first relevant rank: {np.mean(avg_first_rank):.1f}")
        else:
            print("⚠ No queries found relevant documents in HNSW results!")

        # Identify bottleneck
        print("\n" + "-" * 40)
        print("BOTTLENECK ANALYSIS:")
        print("-" * 40)

        # Check query embedding issues
        zero_slots = 0
        for query, result in self.results.items():
            qe = result.get("query_embedding", {})
            slots = qe.get("slot_embeddings", {})
            for slot_name, slot_info in slots.items():
                if slot_info and slot_info.get("is_zero"):
                    zero_slots += 1

        if zero_slots > 0:
            print(f"⚠ Found {zero_slots} zero-valued slot embeddings")

        # Check if keyword search finds docs but HNSW doesn't
        hnsw_miss = 0
        for query, result in self.results.items():
            gt = result.get("ground_truth", {})
            kw_matches = gt.get("keyword_matches", [])
            overlap = gt.get("overlap_with_hnsw", 0)
            if kw_matches and overlap == 0:
                hnsw_miss += 1
                print(f"  Query '{query[:40]}...': Keyword found docs, HNSW missed all")

        if hnsw_miss > 0:
            print(f"\n⚠ BOTTLENECK IDENTIFIED: HNSW search is not finding keyword-matched documents")
            print("  Possible causes:")
            print("    1. Query embeddings don't match document embeddings semantically")
            print("    2. Document embeddings in index don't capture content well")
            print("    3. Dimension mismatch between query and index")
            print("    4. Slot assignment differs between query and documents")


def main():
    parser = argparse.ArgumentParser(description="Diagnose retrieval pipeline")
    parser.add_argument("--index-path", type=Path,
                        default=Path("data/indexes/slot_hybrid"),
                        help="Path to slot index")
    parser.add_argument("--query", type=str, help="Single query to diagnose")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    diagnostics = RetrievalDiagnostics(args.index_path, detailed=args.detailed)

    if args.query:
        # Single query mode
        query_spec = {
            "query": args.query,
            "type": "CUSTOM",
            "keywords": args.query.lower().split()[:3],
            "must_contain": [w for w in args.query.split() if len(w) > 3][:2],
        }
        diagnostics.run_all_diagnostics([query_spec])
    else:
        # Full diagnostic suite
        diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    main()
