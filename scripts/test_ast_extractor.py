#!/usr/bin/env python3
"""
Test ASTAnswerExtractor directly with retrieved documents.

Debug why answer extraction isn't using the correct top-ranked document.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.answer_extractor import ASTAnswerExtractor
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.knowledge import get_synonyms

# Test query
query = 'Kiu fondis Esperanton?'
print(f'Query: {query}')
print('=' * 80)

# Parse query
query_ast = parse(query)
print(f'\nQuery AST: {query_ast}')

# Extract query entity (the thing being asked about)
def extract_query_entity(ast):
    """Extract entity from objekto (accusative object) or aliaj."""
    # First try objekto
    obj = ast.get('objekto')
    if obj:
        if obj.get('tipo') == 'vortgrupo':
            kerno = obj.get('kerno', {})
        else:
            kerno = obj

        if kerno.get('vortspeco') == 'substantivo':
            entity = kerno.get('radiko', '')
            if entity:
                return entity

    # If not in objekto, check aliaj for substantivo in accusative case
    aliaj = ast.get('aliaj', [])
    for alia in aliaj:
        if isinstance(alia, dict):
            if (alia.get('vortspeco') == 'substantivo' and
                alia.get('kazo') == 'akuzativo'):
                entity = alia.get('radiko', '')
                if entity:
                    return entity

    return None

query_entity = extract_query_entity(query_ast)
print(f'\nQuery entity: {query_entity}')

# Extract roots from query
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

roots = extract_roots_from_ast(query_ast)
print(f'\nExtracted roots: {sorted(roots)}')

# Expand roots (limited to top 3 synonyms)
expanded = list(roots)
for root in roots:
    if root != query_entity:  # Don't expand entity
        syns = get_synonyms(root, max_count=3)
        expanded.extend(syns)

expanded_unique = sorted(set(expanded))
print(f'\nExpanded roots ({len(expanded_unique)}): {expanded_unique}')

# Initialize retriever
retriever = WhooshRetriever(
    whoosh_index_dir=Path('data/indexes/whoosh_fts'),
    kuzu_db_path=Path('data/indexes/v2.1_kuzu_index_full')
)

# Retrieve documents
print('\n' + '=' * 80)
print('RETRIEVING DOCUMENTS...')
print('=' * 80)

documents = retriever.retrieve(
    query_roots=expanded_unique,
    top_k=10,
    retrieval_limit=200,
    question_type='who',
    query_entity=query_entity,  # Pass entity for entity-specific expansion
    query_ast=query_ast  # NEW: Enable AST role retrieval
)

print(f'\nRetrieved {len(documents)} documents')

# Show top 5 documents
print('\n--- TOP 5 RETRIEVED DOCUMENTS ---')
for i, doc in enumerate(documents[:5]):
    text = doc['text']
    score = doc['score']

    # Check if contains expected terms
    contains_zamenhof = 'zamenhof' in text.lower()
    contains_kre = any(kre_form in text.lower() for kre_form in ['kre', 'krei', 'kreis', 'kreinta', 'kreinto'])
    contains_esperant = 'esperant' in text.lower()

    print(f'\n[{i+1}] Score: {score:.4f}')
    print(f'    Zamenhof: {contains_zamenhof} | Kre*: {contains_kre} | Esperant*: {contains_esperant}')
    print(f'    Text: {text[:250]}...')

# Now test ASTAnswerExtractor
print('\n' + '=' * 80)
print('TESTING ASTAnswerExtractor...')
print('=' * 80)

extractor = ASTAnswerExtractor()

# Prepare documents in format expected by extract_answer_from_multiple_docs
# Format: List[Tuple[float, Dict, Dict]] = (score, doc, stats)
ranked_docs = []
for i, doc in enumerate(documents[:10]):
    score = doc.get('score', 1.0 / (i + 1))
    ranked_docs.append((score, doc, {}))

# Extract answer
print(f'\nCalling extract_answer_from_multiple_docs with top 10 docs...')
result = extractor.extract_answer_from_multiple_docs(
    query_ast,
    ranked_docs,
    top_n=5  # Try top 5 documents
)

if result:
    print(f'\n✓ EXTRACTED ANSWER:')
    print(f'  Text: {result["text"]}')
    print(f'  Confidence: {result["confidence"]:.3f}')
    print(f'  Method: {result["method"]}')
    print(f'  Explanation: {result["explanation"]}')

    if 'aggregation_stats' in result:
        stats = result['aggregation_stats']
        print(f'\n  Aggregation Stats:')
        print(f'    Docs extracted from: {stats["num_docs_extracted"]}/{len(ranked_docs[:10])}')
        print(f'    Unique entities found: {stats["num_unique_entities"]}')
        print(f'    Occurrence count: {stats["occurrence_count"]}')
        print(f'    Document ranks: {stats["doc_ranks"]}')
        print(f'    Avg confidence: {stats["avg_confidence"]:.3f}')
else:
    print(f'\n✗ NO ANSWER EXTRACTED')
    print('\nTrying single document extraction...')

    # Try extracting from top document directly
    top_doc = documents[0]
    top_doc_text = top_doc['text']
    top_doc_ast = parse(top_doc_text)

    single_result = extractor.extract_answer(query_ast, top_doc_ast, top_doc_text)

    if single_result:
        print(f'\n✓ SINGLE DOC EXTRACTION WORKED:')
        print(f'  Text: {single_result["text"]}')
        print(f'  Confidence: {single_result["confidence"]:.3f}')
        print(f'  Explanation: {single_result["explanation"]}')
    else:
        print(f'\n✗ SINGLE DOC EXTRACTION ALSO FAILED')
        print(f'\nTop document text: {top_doc_text}')
        print(f'\nTop document AST: {top_doc_ast}')

print('\n' + '=' * 80)
