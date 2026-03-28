#!/usr/bin/env python3
"""
Debug script to analyze retrieval quality for failing questions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.knowledge import get_synonyms

def extract_roots_from_ast(ast):
    """Extract content roots from AST."""
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

# Test question
query = 'Kiu fondis Esperanton?'
expected = ['zamenhof', 'ludovic', 'lazaro']

print(f'Query: {query}')
print(f'Expected answer: {expected}')
print('=' * 80)

# Parse and extract roots
ast = parse(query)
roots = extract_roots_from_ast(ast)
print(f'\nExtracted roots: {sorted(roots)}')

# Expand (current behavior - ALL synonyms)
expanded = list(roots)
for root in roots:
    if root != 'esperant':  # Skip entity
        syns = get_synonyms(root)
        expanded.extend(syns)

print(f'Expanded roots ({len(expanded)}): {sorted(set(expanded))}')

# Initialize retriever
retriever = WhooshRetriever(
    whoosh_index_dir=Path('data/indexes/whoosh_fts'),
    kuzu_db_path=Path('data/indexes/v2.1_kuzu_index_full')
)

# Retrieve with CURRENT expansion
print('\n' + '=' * 80)
print('RETRIEVING WITH CURRENT EXPANSION (16 roots)...')
print('=' * 80)

documents = retriever.retrieve(
    query_roots=expanded,
    top_k=10,
    retrieval_limit=200
)

print(f'\nRetrieved {len(documents)} documents')
print('\n--- TOP 10 DOCUMENTS ---')

for i, doc in enumerate(documents[:10]):
    text = doc['text']
    score = doc['score']

    # Check if contains expected answer
    contains_zamenhof = any(exp in text.lower() for exp in expected)

    print(f'\n[{i+1}] Score: {score:.4f} | Contains Zamenhof: {contains_zamenhof}')
    print(f'Text: {text[:200]}...')

    # Show which roots matched
    matched = [r for r in expanded if r in text.lower()]
    print(f'Matched roots: {matched[:5]}{"..." if len(matched) > 5 else ""}')

# Now try with LIMITED expansion (only core synonyms)
print('\n' + '=' * 80)
print('TESTING WITH LIMITED EXPANSION (top 3 synonyms only)...')
print('=' * 80)

limited_expanded = list(roots)
for root in roots:
    if root != 'esperant':
        syns = get_synonyms(root)
        # Take only top 3 most common/core synonyms
        # Sort alphabetically for consistency, take top 3
        top_syns = sorted(syns)[:3]
        limited_expanded.extend(top_syns)

print(f'Limited expansion ({len(set(limited_expanded))}): {sorted(set(limited_expanded))}')

documents_limited = retriever.retrieve(
    query_roots=list(set(limited_expanded)),
    top_k=10,
    retrieval_limit=200
)

print(f'\nRetrieved {len(documents_limited)} documents')
print('\n--- TOP 10 DOCUMENTS (LIMITED) ---')

for i, doc in enumerate(documents_limited[:10]):
    text = doc['text']
    score = doc['score']

    contains_zamenhof = any(exp in text.lower() for exp in expected)

    print(f'\n[{i+1}] Score: {score:.4f} | Contains Zamenhof: {contains_zamenhof}')
    print(f'Text: {text[:200]}...')

# Compare
print('\n' + '=' * 80)
print('COMPARISON')
print('=' * 80)

count_current = sum(1 for doc in documents[:10] if any(exp in doc['text'].lower() for exp in expected))
count_limited = sum(1 for doc in documents_limited[:10] if any(exp in doc['text'].lower() for exp in expected))

print(f'Current expansion: {count_current}/10 top docs contain answer')
print(f'Limited expansion: {count_limited}/10 top docs contain answer')
