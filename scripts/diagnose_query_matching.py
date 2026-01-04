#!/usr/bin/env python3
"""
Query Matching Diagnostic Tool (Issue #224)

Diagnoses why specific queries fail to retrieve expected answers.
Shows step-by-step breakdown of:
1. Query parsing and slot extraction
2. Stage 1 prefiltering (slot-based matching)
3. Stage 2 reranking (full embedding similarity)
4. Where the correct answer gets lost

Usage:
    python scripts/diagnose_query_matching.py \
        --query "Kiu fondis Esperanton?" \
        --expected-answer "Zamenhof" \
        --index data/indexes/slot_full \
        --top-k 20

Author: Claude Code
Date: 2026-01-04
Related: GitHub issues #223, #224, #225, #226
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class QueryDiagnosticTool:
    """Diagnostic tool for query matching failures."""

    def __init__(self, index_dir: Path):
        """Initialize diagnostic tool with index directory."""
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / 'slot_index.jsonl'

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        logger.info(f"Loading index from {self.index_dir}")
        self._load_index()

    def _load_index(self):
        """Load slot index for diagnostic analysis."""
        logger.info("Loading slot index (this may take a moment)...")
        self.documents = []

        with open(self.index_path) as f:
            for i, line in enumerate(f):
                self.documents.append(json.loads(line))
                if (i + 1) % 100000 == 0:
                    logger.info(f"  Loaded {i+1:,} documents...")

        logger.info(f"Loaded {len(self.documents):,} documents")

    def diagnose_query(
        self,
        query: str,
        expected_answer: Optional[str] = None,
        top_k: int = 20,
        rerank_top_n: int = 100,
    ) -> Dict:
        """
        Diagnose why a query fails to retrieve expected answer.

        Args:
            query: Question to diagnose
            expected_answer: Expected text in answer (optional)
            top_k: Number of final results to return
            rerank_top_n: Number of candidates for Stage 2

        Returns:
            Diagnostic report dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"DIAGNOSING QUERY: {query}")
        logger.info(f"{'='*80}\n")

        # Step 1: Parse query
        logger.info("STEP 1: Parsing query...")
        query_ast = parse(query)

        logger.info(f"Query AST:")
        self._print_ast_summary(query_ast)

        # Extract slots from AST
        query_slots = self._extract_slots_from_ast(query_ast)
        logger.info(f"\nQuery slots:")
        for slot, value in query_slots.items():
            logger.info(f"  {slot}: {value}")

        # Step 2: Stage 1 filtering (slot-based)
        logger.info(f"\nSTEP 2: Stage 1 filtering (slot-based matching)...")
        logger.info(f"Scanning {len(self.documents):,} documents...")

        stage1_candidates = []
        for i, doc in enumerate(self.documents):
            doc_slots = doc.get('slot_roots', {})
            slot_score = self._compute_slot_similarity(query_slots, doc_slots)

            if slot_score > 0:
                stage1_candidates.append({
                    'doc_id': i,
                    'slot_score': slot_score,
                    'text': doc.get('text', ''),
                    'slots': doc_slots,
                    'source': doc.get('source_metadata', {})
                })

        # Sort by slot score
        stage1_candidates.sort(key=lambda x: x['slot_score'], reverse=True)
        top_stage1 = stage1_candidates[:rerank_top_n]

        logger.info(f"Stage 1 results:")
        logger.info(f"  Candidates with slot_score > 0: {len(stage1_candidates):,}")
        logger.info(f"  Top-{rerank_top_n} for reranking: {len(top_stage1)}")
        logger.info(f"  Score range: {top_stage1[0]['slot_score']:.3f} to {top_stage1[-1]['slot_score']:.3f}")

        # Show top 5 Stage 1 results
        logger.info(f"\n  Top 5 Stage 1 candidates:")
        for i, cand in enumerate(top_stage1[:5], 1):
            logger.info(f"    #{i} [score={cand['slot_score']:.3f}] {cand['text'][:80]}...")

        # Check if expected answer is in top-N
        if expected_answer:
            self._check_expected_answer_presence(
                expected_answer,
                top_stage1,
                "Stage 1 (top-100)"
            )

        # Step 3: Would do Stage 2 reranking here (requires loading embeddings)
        logger.info(f"\nSTEP 3: Stage 2 reranking (SKIPPED - requires embeddings)")
        logger.info(f"  In production: Would rerank top-{rerank_top_n} with full embeddings")
        logger.info(f"  For now, using Stage 1 scores as final ranking")

        final_results = top_stage1[:top_k]

        # Step 4: Diagnosis
        logger.info(f"\n{'='*80}")
        logger.info(f"DIAGNOSIS")
        logger.info(f"{'='*80}\n")

        diagnosis = self._generate_diagnosis(
            query, query_ast, query_slots,
            stage1_candidates, final_results,
            expected_answer
        )

        for line in diagnosis:
            logger.info(line)

        # Return diagnostic report
        return {
            'query': query,
            'query_ast': query_ast,
            'query_slots': query_slots,
            'stage1_count': len(stage1_candidates),
            'top_stage1': top_stage1[:top_k],
            'diagnosis': diagnosis,
        }

    def _print_ast_summary(self, ast: Dict):
        """Print AST summary."""
        if ast.get('tipo') != 'frazo':
            logger.info(f"  Type: {ast.get('tipo')} (not a sentence)")
            return

        subj = ast.get('subjekto')
        verb = ast.get('verbo')
        obj = ast.get('objekto')

        logger.info(f"  Subject: {self._format_ast_node(subj)}")
        logger.info(f"  Verb: {self._format_ast_node(verb)}")
        logger.info(f"  Object: {self._format_ast_node(obj)}")

    def _format_ast_node(self, node: Optional[Dict]) -> str:
        """Format AST node for display."""
        if node is None:
            return "None"

        if isinstance(node, dict):
            if 'kerno' in node:
                # vortgrupo
                kerno = node['kerno']
                return f"{kerno.get('radiko', '?')} ({kerno.get('vortspeco', '?')})"
            elif 'radiko' in node:
                # vorto
                return f"{node.get('radiko', '?')} ({node.get('vortspeco', '?')})"
            else:
                return str(node.get('tipo', 'unknown'))

        return str(node)

    def _extract_slots_from_ast(self, ast: Dict) -> Dict[str, Optional[str]]:
        """Extract SUBJ/VERB/OBJ roots from AST."""
        slots = {
            'SUBJ': None,
            'VERB': None,
            'OBJ': None,
        }

        if ast.get('tipo') != 'frazo':
            return slots

        # Extract subject
        subj = ast.get('subjekto')
        if subj:
            if isinstance(subj, dict):
                if 'kerno' in subj:
                    slots['SUBJ'] = subj['kerno'].get('radiko')
                elif 'radiko' in subj:
                    slots['SUBJ'] = subj.get('radiko')

        # Extract verb
        verb = ast.get('verbo')
        if verb and isinstance(verb, dict):
            slots['VERB'] = verb.get('radiko')

        # Extract object
        obj = ast.get('objekto')
        if obj:
            if isinstance(obj, dict):
                if 'kerno' in obj:
                    slots['OBJ'] = obj['kerno'].get('radiko')
                elif 'radiko' in obj:
                    slots['OBJ'] = obj.get('radiko')

        return slots

    def _compute_slot_similarity(
        self,
        query_slots: Dict[str, Optional[str]],
        doc_slots: Dict[str, Optional[str]],
    ) -> float:
        """
        Compute slot similarity score (mimics slot_retriever.py logic).

        This is a simplified version for diagnostic purposes.
        """
        slot_weights = {'SUBJ': 0.3, 'VERB': 0.4, 'OBJ': 0.3}
        score = 0.0
        matched_slots = 0

        for slot, weight in slot_weights.items():
            query_val = query_slots.get(slot)
            doc_val = doc_slots.get(slot)

            if query_val is not None and doc_val is not None:
                # Both have slot: exact match (simplified - no embeddings)
                if query_val == doc_val:
                    score += weight * 1.0
                    matched_slots += 1
                else:
                    score += weight * 0.3  # Partial credit for different roots
                    matched_slots += 1

            elif query_val is None and doc_val is not None:
                # Query missing slot: partial match bonus
                score += weight * 0.5  # BUG #2: Should be 0.8 for questions
                matched_slots += 1

        # Normalize
        if matched_slots > 0:
            return score / matched_slots
        else:
            return 0.0

    def _check_expected_answer_presence(
        self,
        expected_answer: str,
        candidates: List[Dict],
        stage_name: str
    ):
        """Check if expected answer is in candidates."""
        expected_lower = expected_answer.lower()

        for i, cand in enumerate(candidates):
            if expected_lower in cand['text'].lower():
                logger.info(f"\n  ✓ FOUND expected answer '{expected_answer}' at rank #{i+1} in {stage_name}")
                logger.info(f"    Score: {cand.get('slot_score', 0):.3f}")
                logger.info(f"    Text: {cand['text'][:120]}...")
                return i + 1

        logger.info(f"\n  ✗ Expected answer '{expected_answer}' NOT FOUND in {stage_name}")
        return None

    def _generate_diagnosis(
        self,
        query: str,
        query_ast: Dict,
        query_slots: Dict,
        stage1_candidates: List[Dict],
        final_results: List[Dict],
        expected_answer: Optional[str]
    ) -> List[str]:
        """Generate diagnostic report."""
        diagnosis = []

        # Check for common issues
        has_subject = query_slots.get('SUBJ') is not None
        has_verb = query_slots.get('VERB') is not None
        has_object = query_slots.get('OBJ') is not None

        diagnosis.append(f"Query structure:")
        diagnosis.append(f"  Has subject: {has_subject} ({'✓' if has_subject else '✗'})")
        diagnosis.append(f"  Has verb: {has_verb} ({'✓' if has_verb else '✗'})")
        diagnosis.append(f"  Has object: {has_object} ({'✓' if has_object else '✗'})")
        diagnosis.append("")

        # Identify issues
        issues = []

        if not has_subject:
            issues.append("⚠ BUG #1: Query has no subject (parser failed to extract)")
            issues.append("  → May be a question word (Kiu, Kio) or proper noun")
            issues.append("  → This loses critical matching information")

        if not has_subject and not has_object:
            issues.append("⚠ Only verb extracted - very weak matching signal")
            issues.append("  → Slot similarity will be low (~0.3-0.4)")
            issues.append("  → May not rank in top-100 for Stage 2")

        # Check partial match bonus
        none_count = sum(1 for v in query_slots.values() if v is None)
        if none_count > 0:
            issues.append(f"⚠ BUG #2: {none_count} slots are None → using partial bonus (0.5)")
            issues.append(f"  → For questions, partial bonus should be 0.8+")
            issues.append(f"  → This artificially lowers slot similarity scores")

        # Check for question words
        query_lower = query.lower()
        question_words = ['kiu', 'kio', 'kiam', 'kie', 'kiom', 'kial', 'kiel']
        found_qwords = [qw for qw in question_words if qw in query_lower]

        if found_qwords:
            issues.append(f"⚠ BUG #3: Question word(s) detected: {', '.join(found_qwords)}")
            issues.append(f"  → Should map to entity types (e.g., Kiu → PERSON)")
            issues.append(f"  → Should boost sentences with matching entity types")
            issues.append(f"  → Currently ignored - no semantic boosting")

        diagnosis.append("Identified issues:")
        if issues:
            for issue in issues:
                diagnosis.append(f"  {issue}")
        else:
            diagnosis.append("  No obvious issues detected")

        diagnosis.append("")

        # Expected answer analysis
        if expected_answer:
            diagnosis.append(f"Expected answer analysis:")
            found_in_stage1 = any(
                expected_answer.lower() in c['text'].lower()
                for c in stage1_candidates
            )
            if found_in_stage1:
                # Find rank
                for i, c in enumerate(stage1_candidates):
                    if expected_answer.lower() in c['text'].lower():
                        rank = i + 1
                        score = c['slot_score']
                        diagnosis.append(f"  ✓ Found in Stage 1 at rank #{rank} (score={score:.3f})")
                        if rank > 100:
                            diagnosis.append(f"  ✗ But rank > 100 → filtered out before Stage 2")
                            diagnosis.append(f"  → Need to improve slot matching to get into top-100")
                        break
            else:
                diagnosis.append(f"  ✗ NOT found in any Stage 1 candidates")
                diagnosis.append(f"  → Either:")
                diagnosis.append(f"     1. Document not in index")
                diagnosis.append(f"     2. Slot similarity = 0 (no overlap)")

        return diagnosis


def main():
    parser = argparse.ArgumentParser(description="Diagnose query matching failures")
    parser.add_argument('--query', '-q', required=True, help="Query to diagnose")
    parser.add_argument('--expected-answer', '-e', help="Expected text in answer")
    parser.add_argument('--index', '-i', required=True, help="Path to slot index directory")
    parser.add_argument('--top-k', '-k', type=int, default=20, help="Number of results to show")
    parser.add_argument('--rerank-top-n', type=int, default=100, help="Stage 1 candidates for Stage 2")

    args = parser.parse_args()

    # Run diagnostic
    tool = QueryDiagnosticTool(Path(args.index))
    result = tool.diagnose_query(
        query=args.query,
        expected_answer=args.expected_answer,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"Diagnostic complete. Check output above for details.")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
