#!/usr/bin/env python3
"""
Demo Full Hybrid Retrieval - Show Actual Document Results

Performs complete retrieval pipeline:
1. Parse query
2. Expand with hybrid approach (ReVo + Embeddings)
3. Retrieve documents from Kuzu
4. Show top N results with content

Usage:
    python scripts/demo_full_hybrid_retrieval.py "Kio estas Esperanto?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.hybrid_query_expander import HybridQueryExpander
import kuzu


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


def retrieve_documents(expanded_roots, conn, top_k=10):
    """Retrieve documents containing any of the expanded roots."""
    # Build query to find documents containing any of the roots
    roots_list = list(expanded_roots)
    
    # Query: Find sentences containing any of the expanded roots
    query = f"""
        MATCH (d:Dokumento)-[:ENHAVAS_FRAZON]->(f:Frazo)
        MATCH (f)-[:HAVAS_VORTON]->(v:Vorto)
        WHERE v.radiko IN {roots_list}
        RETURN DISTINCT d.dokumento_id AS doc_id, 
                        d.source AS source,
                        f.teksto AS sentence_text,
                        count(DISTINCT v.radiko) AS matching_roots
        ORDER BY matching_roots DESC
        LIMIT {top_k}
    """
    
    try:
        result = conn.execute(query)
        
        documents = []
        while result.has_next():
            row = result.get_next()
            documents.append({
                'doc_id': row[0],
                'source': row[1],
                'text': row[2],
                'matching_roots': row[3]
            })
        
        return documents
    except Exception as e:
        print(f"Query error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Demo full hybrid retrieval')
    parser.add_argument('query', help='Esperanto query')
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
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL HYBRID RETRIEVAL DEMO")
    print("="*70)
    print()
    
    # Initialize expander
    print("Loading hybrid query expander...")
    expander = HybridQueryExpander(
        embedding_path=args.embeddings,
        db_path=args.db,
        use_revo=True,
        use_embeddings=True
    )
    print()
    
    # Parse query
    print(f"Query: {args.query}")
    print("-"*70)
    ast = parse(args.query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")
    print()
    
    # Expand
    expansion = expander.expand(original_roots)
    
    print("Expansion:")
    if expansion['revo_synonyms']:
        print(f"  ReVo synonyms: {', '.join(sorted(expansion['revo_synonyms']))}")
    else:
        print(f"  ReVo synonyms: (none)")
    
    if expansion['embedding_associations']:
        emb_list = sorted(list(expansion['embedding_associations']))[:8]
        print(f"  Embedding assoc: {', '.join(emb_list)}")
    else:
        print(f"  Embedding assoc: (none)")
    
    print(f"  Total expanded: {len(expansion['all'])} roots")
    print()
    
    # Retrieve
    print(f"Retrieving top {args.top_k} documents...")
    conn = kuzu.Connection(kuzu.Database(str(args.db)))
    
    documents = retrieve_documents(expansion['all'], conn, args.top_k)
    
    print()
    print("="*70)
    print(f"TOP {args.top_k} RESULTS")
    print("="*70)
    print()
    
    if not documents:
        print("No documents found!")
    else:
        for i, doc in enumerate(documents, 1):
            text = doc['text'][:200] if doc['text'] else 'NO TEXT'
            source = doc.get('source', 'unknown')
            matching = doc.get('matching_roots', 0)
            
            print(f"{i}. [Matching roots: {matching}] [{source}]")
            print(f"   {text}...")
            print()


if __name__ == '__main__':
    main()
