#!/usr/bin/env python3
"""
Query Kuzu Database Directly for Retrieval Results

Bypasses the retriever infrastructure to directly query for documents
containing expanded query roots.

Usage:
    python scripts/query_kuzu_direct.py "Kio estas Esperanto?"
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import kuzu

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def extract_roots_from_ast(ast):
    """Extract all content word roots from AST."""
    roots = set()
    skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

    def extract(node):
        if not node or not isinstance(node, dict):
            return
        if node.get('tipo') == 'vorto':
            vortspeco = node.get('vortspeco', '')
            if vortspeco not in skip_vortspeco:
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    roots.add(root.lower())
        elif node.get('tipo') == 'vortgrupo':
            extract(node.get('kerno'))
            for p in node.get('priskriboj', []):
                extract(p)
        elif node.get('tipo') == 'frazo':
            extract(node.get('subjekto'))
            extract(node.get('verbo'))
            extract(node.get('objekto'))
            for a in node.get('aliaj', []):
                extract(a)

    extract(ast)
    return roots


def expand_with_embeddings(roots, embeddings_path, k=5, threshold=0.4):
    """Expand roots using embeddings."""
    checkpoint = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']
    root_to_idx = {root: idx for idx, root in enumerate(vocab)}

    expanded = set(roots)

    for root in roots:
        if root not in root_to_idx:
            continue

        root_idx = root_to_idx[root]
        root_emb = embeddings[root_idx].unsqueeze(0)
        similarities = F.cosine_similarity(root_emb, embeddings)
        top_k_sims, top_k_indices = torch.topk(similarities, k=k + 1)

        for sim, idx in zip(top_k_sims.tolist(), top_k_indices.tolist()):
            if idx == root_idx:
                continue
            if sim >= threshold:
                expanded.add(vocab[idx])

    return expanded


def query_kuzu(db_path, roots, top_k=10):
    """Query Kuzu database for documents containing roots."""
    print(f"Connecting to Kuzu: {db_path}")
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    # First check what node types exist
    print("Checking database schema...")
    result = conn.execute("CALL SHOW_TABLES() RETURN *")
    tables = []
    while result.has_next():
        row = result.get_next()
        tables.append(str(row[0]))
    print(f"Available tables: {', '.join(tables[:10])}")

    # Try a simple query for Frazoteksto nodes containing any of the roots
    roots_list = list(roots)

    # Build query - try to find Frazoteksto nodes with matching text
    query = f"""
        MATCH (ft:Frazoteksto)
        WHERE ft.teksto IS NOT NULL
        RETURN ft.teksto AS text, ft.id AS id
        LIMIT {top_k * 10}
    """

    print(f"\nExecuting query (first {top_k * 10} sentences)...")
    result = conn.execute(query)

    # Fetch results and score by root matches
    documents = []
    while result.has_next():
        row = result.get_next()
        text = row[0]
        doc_id = row[1]

        if not text:
            continue

        # Count matching roots in text
        text_lower = text.lower()
        matching_roots = [r for r in roots_list if r in text_lower]

        if matching_roots:
            documents.append({
                'text': text,
                'id': doc_id,
                'matching_roots': matching_roots,
                'num_matches': len(matching_roots)
            })

    # Sort by number of matches
    documents.sort(key=lambda d: d['num_matches'], reverse=True)

    return documents[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='Esperanto query')
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'))
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--expand', action='store_true', help='Use embedding expansion')

    args = parser.parse_args()

    print("="*70)
    print("KUZU DIRECT QUERY")
    print("="*70)
    print()

    # Parse query
    print(f"Query: {args.query}")
    print("-"*70)
    ast = parse(args.query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")

    # Expand if requested
    if args.expand:
        print("\nExpanding with embeddings...")
        expanded_roots = expand_with_embeddings(original_roots, args.embeddings)
        print(f"Expanded to: {len(expanded_roots)} roots")
        print(f"Added: {', '.join(sorted(expanded_roots - original_roots))}")
        query_roots = expanded_roots
    else:
        print("\nNo expansion (use --expand to enable)")
        query_roots = original_roots

    print()

    # Query Kuzu
    documents = query_kuzu(args.db, query_roots, args.top_k)

    print()
    print("="*70)
    print(f"TOP {args.top_k} RESULTS")
    print("="*70)
    print()

    if not documents:
        print("No documents found!")
    else:
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            if len(text) > 200:
                text = text[:200] + "..."

            roots_str = ', '.join(sorted(doc['matching_roots']))

            print(f"{i}. [{doc['num_matches']} matching roots: {roots_str}]")
            print(f"   {text}")
            print()


if __name__ == '__main__':
    main()
