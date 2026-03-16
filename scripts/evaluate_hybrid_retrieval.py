#!/usr/bin/env python3
"""
Evaluate Hybrid Retrieval - Measure Recall Improvement

Tests hybrid query expansion (ReVo + Embeddings) against baselines.

Baselines:
1. No expansion (exact match only)
2. ReVo only (deterministic synonyms)
3. Embeddings only (learned associations)
4. Hybrid (ReVo + Embeddings)

Usage:
    python scripts/evaluate_hybrid_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.hybrid_query_expander import HybridQueryExpander


def extract_roots_from_ast(ast):
    """Extract all roots from AST."""
    roots = set()
    
    def extract(node):
        if isinstance(node, dict):
            if 'radiko' in node:
                roots.add(node['radiko'])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    extract(value)
        elif isinstance(node, list):
            for item in node:
                extract(item)
    
    extract(ast)
    return roots


def main():
    # Test queries with expected relevant roots
    test_cases = [
        {
            'query': 'Kiu fondis Esperanton?',
            'expected_relevant': {
                'fond', 'kre', 'establ', 'komenc',  # Synonyms for "found"
                'universitat', 'societ', 'organiz'  # Context (things founded)
            }
        },
        {
            'query': 'Kio estas Esperanto?',
            'expected_relevant': {
                'est', 'ekzist', 'ent',  # Synonyms for "is"
                'lingv', 'parol'  # Context (language)
            }
        }
    ]
    
    print("="*70)
    print("HYBRID RETRIEVAL EVALUATION")
    print("="*70)
    print()
    
    # Initialize expanders
    print("Initializing expanders...")
    
    base_path = Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt')
    db_path = Path('data/indexes/v2.1_kuzu_index_full')
    
    expander_none = HybridQueryExpander(base_path, db_path, use_revo=False, use_embeddings=False)
    expander_revo = HybridQueryExpander(base_path, db_path, use_revo=True, use_embeddings=False)
    expander_emb = HybridQueryExpander(base_path, db_path, use_revo=False, use_embeddings=True)
    expander_hybrid = HybridQueryExpander(base_path, db_path, use_revo=True, use_embeddings=True)
    
    print()
    print("="*70)
    print("RECALL COMPARISON")
    print("="*70)
    print()
    
    for test in test_cases:
        query = test['query']
        expected = test['expected_relevant']
        
        print(f"Query: {query}")
        print(f"Expected relevant roots: {len(expected)}")
        print()
        
        # Parse query
        ast = parse(query)
        original_roots = extract_roots_from_ast(ast)
        
        # Test each expander
        results = {}
        
        # Baseline (no expansion)
        results['No expansion'] = {'all': original_roots}
        
        # ReVo only
        results['ReVo only'] = expander_revo.expand(original_roots)
        
        # Embeddings only
        results['Embeddings only'] = expander_emb.expand(original_roots)
        
        # Hybrid
        results['Hybrid'] = expander_hybrid.expand(original_roots)
        
        # Calculate recall for each
        print(f"{'Method':<20} {'Expanded Roots':<15} {'Recall':<10} {'Coverage'}")
        print("-" * 70)
        
        for method, expansion in results.items():
            expanded = expansion['all']
            recall = len(expanded & expected) / len(expected)
            coverage = len(expanded)
            
            print(f"{method:<20} {coverage:<15} {recall:>6.1%}     {sorted(list(expanded & expected))}")
        
        print()
        print("-" * 70)
        print()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("Hybrid approach (ReVo + Embeddings) should show:")
    print("  ✓ Higher recall than either method alone")
    print("  ✓ Deterministic synonyms (precision)")
    print("  ✓ Learned associations (recall)")
    print()
    print("Next: Integrate into full retrieval pipeline")


if __name__ == '__main__':
    main()
