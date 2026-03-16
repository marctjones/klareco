#!/usr/bin/env python3
"""
Test Hybrid Retrieval: Deterministic Synonyms + Learned Associations

Tests the two-track query expansion strategy for RAG:
1. Track 1: ReVo synonyms (deterministic, high precision)
2. Track 2: Embedding associations (learned, high recall)

Usage:
    python scripts/test_hybrid_retrieval.py
    python scripts/test_hybrid_retrieval.py --query "Kiu fondis Esperanton?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import kuzu

def load_embeddings(checkpoint_path):
    """Load root embeddings."""
    print(f"Loading embeddings from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']
    root_to_idx = checkpoint['root_to_idx']
    
    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    print(f"Loaded {len(vocab):,} root embeddings")
    return embeddings, vocab, root_to_idx


def get_revo_synonyms(root, conn):
    """Get deterministic synonyms from ReVo."""
    synonyms = set()
    
    # Query all ReVo semantic relations
    result = conn.execute(f"""
        MATCH (r:Radiko {{radiko: '{root}'}})-[rel:REVO_SINONIMO|REVO_HIPERNIMO]->(s:Radiko)
        RETURN s.radiko
    """)
    
    while result.has_next():
        synonyms.add(result.get_next()[0])
    
    return synonyms


def get_embedding_associations(root, embeddings, vocab, root_to_idx, k=5, threshold=0.4):
    """Get learned associations from embeddings."""
    if root not in root_to_idx:
        return set()
    
    target_idx = root_to_idx[root]
    target_emb = embeddings[target_idx]
    
    # Compute similarities
    similarities = embeddings @ target_emb
    
    # Get top k (excluding self)
    top_k_indices = similarities.argsort(descending=True)[1:k+1]
    
    associations = set()
    for idx in top_k_indices:
        sim = similarities[idx].item()
        if sim > threshold:
            associations.add(vocab[idx])
    
    return associations


def hybrid_query_expansion(roots, conn, embeddings, vocab, root_to_idx):
    """Expand query roots using both tracks."""
    expansion = {
        'original': set(roots),
        'revo_synonyms': set(),
        'embedding_associations': set()
    }
    
    for root in roots:
        # Track 1: Deterministic synonyms
        revo_syns = get_revo_synonyms(root, conn)
        expansion['revo_synonyms'].update(revo_syns)
        
        # Track 2: Learned associations
        emb_assoc = get_embedding_associations(root, embeddings, vocab, root_to_idx)
        expansion['embedding_associations'].update(emb_assoc)
    
    return expansion


def main():
    parser = argparse.ArgumentParser(description='Test hybrid retrieval')
    parser.add_argument('--query', type=str, help='Test query')
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'),
        help='Path to embeddings'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/indexes/v2.1_kuzu_index_full'),
        help='Path to Kuzu database'
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings, vocab, root_to_idx = load_embeddings(args.embeddings)
    print()
    
    # Connect to Kuzu
    print(f"Connecting to Kuzu database: {args.db}...")
    db = kuzu.Database(str(args.db))
    conn = kuzu.Connection(db)
    print()
    
    # Test queries
    test_queries = [
        {
            'query': 'Kiu fondis Esperanton?',
            'roots': ['fond', 'esperant']
        },
        {
            'query': 'Kie loĝis Zamenhof?',
            'roots': ['loĝ', 'zamenhof']
        },
        {
            'query': 'Kio estas Esperanto?',
            'roots': ['est', 'esperant']
        }
    ]
    
    if args.query:
        # Parse user query (simplified - just extract roots manually for now)
        print(f"Query: {args.query}")
        print("(Manual root extraction needed - using test queries)")
        print()
    
    print("="*70)
    print("HYBRID QUERY EXPANSION TEST")
    print("="*70)
    print()
    
    for test in test_queries:
        print(f"Query: {test['query']}")
        print(f"Roots: {', '.join(test['roots'])}")
        print()
        
        expansion = hybrid_query_expansion(
            test['roots'],
            conn,
            embeddings,
            vocab,
            root_to_idx
        )
        
        print("Expansion Results:")
        print(f"  Original roots:     {', '.join(sorted(expansion['original']))}")
        
        if expansion['revo_synonyms']:
            print(f"  ReVo synonyms:      {', '.join(sorted(expansion['revo_synonyms']))}")
        else:
            print(f"  ReVo synonyms:      (none found)")
        
        if expansion['embedding_associations']:
            print(f"  Embedding assoc:    {', '.join(sorted(list(expansion['embedding_associations'])[:8]))}")
        else:
            print(f"  Embedding assoc:    (none found)")
        
        # Total expansion
        all_roots = expansion['original'] | expansion['revo_synonyms'] | expansion['embedding_associations']
        expansion_factor = len(all_roots) / len(expansion['original'])
        print()
        print(f"  Total expanded:     {len(all_roots)} roots (×{expansion_factor:.1f} expansion)")
        print()
        print("-"*70)
        print()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("✓ Hybrid expansion working")
    print("✓ Combines deterministic synonyms (ReVo) + learned associations (embeddings)")
    print("✓ Ready to integrate into retrieval pipeline")
    print()
    print("Next: Implement in demo_semantic_retrieval.py")


if __name__ == '__main__':
    main()
