#!/usr/bin/env python3
"""
Demo Extractive QA - Complete RAG Pipeline with Answer Generation

Demonstrates the full RAG pipeline:
1. Parse query → extract roots
2. Expand query with embeddings
3. Retrieve relevant sentences from v2.1 database
4. Extract facts from ASTs
5. Rank facts by importance
6. Plan discourse structure
7. Generate coherent answer

Usage:
    python scripts/demo_extractive_qa.py "Kio estas Esperanto?"
    python scripts/demo_extractive_qa.py --interactive
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import kuzu

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.extractive_answering import (
    ExtractiveAnswerGenerator, QuestionType, classify_question_type
)


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


def extract_query_entity(ast, question_type):
    """
    Extract the entity being asked about from query AST.

    Works for all question types, searching in priority order:
    1. objekto - e.g., "Kiu kreis Esperanton?" → "esperant"
    2. aliaj (proper nouns) - e.g., "Kiu estis Benjamin Franklin?" → "Benjamin Franklin"
    3. aliaj (substantivo) - e.g., "Kiam... pri Esperanto?" → "esperant"
    4. subjekto (fallback) - e.g., "Kion inventis Benjamin Franklin?" → "Benjamin Franklin"
    """
    # 1. Check objekto for substantivo (all question types)
    # Handles: "Kiu kreis Esperanton?", "Kio estas Fundamento?"
    obj = ast.get('objekto')
    if obj:
        if obj.get('tipo') == 'vortgrupo':
            kerno = obj.get('kerno', {})
        else:
            kerno = obj

        # Get substantivo from objekto
        if kerno.get('vortspeco') == 'substantivo':
            entity = kerno.get('radiko', '')
            if entity:
                return entity

    # 2. Check aliaj for proper nouns (capitalized words)
    # Handles: "Kiu estis Benjamin Franklin?", "Kiam naskiĝis Benjamin Franklin?"
    aliaj = ast.get('aliaj', [])
    proper_nouns = []

    for alia in aliaj:
        if isinstance(alia, dict):
            plena_vorto = alia.get('plena_vorto', '')
            if plena_vorto and plena_vorto[0].isupper():
                proper_nouns.append(plena_vorto)

    # If we found proper nouns, combine them (e.g., "Benjamin Franklin")
    if proper_nouns:
        return ' '.join(proper_nouns)

    # 3. Check aliaj for substantivo
    # Handles: "Kiam... pri Esperanto?", "Kie okazis... Esperanto-Kongreso?"
    for alia in aliaj:
        if isinstance(alia, dict) and alia.get('vortspeco') == 'substantivo':
            entity = alia.get('radiko', '')
            if entity:
                return entity

    # 4. Check subjekto as final fallback (for inverted questions)
    # Handles: "Kion inventis Benjamin Franklin?" (if Franklin is subject)
    subj = ast.get('subjekto')
    if subj and subj.get('tipo') == 'vortgrupo':
        # Check for proper nouns in subjekto
        subj_proper_nouns = []
        priskriboj = subj.get('priskriboj', [])
        for priskribo in priskriboj:
            if isinstance(priskribo, dict):
                plena_vorto = priskribo.get('plena_vorto', '')
                if plena_vorto and plena_vorto[0].isupper():
                    subj_proper_nouns.append(plena_vorto)

        if subj_proper_nouns:
            return ' '.join(subj_proper_nouns)

    return None


def retrieve_sentences(db_path, roots, question_type, query_entity=None, top_k=10):
    """Retrieve sentences from Kuzu database with entity-aware prioritization."""
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    roots_list = list(roots)

    # If we have a query entity (proper noun or main entity), prioritize Wikipedia article
    if query_entity:
        # For single-word Esperanto roots like "esperant", add -o ending
        if ' ' not in query_entity and query_entity[0].islower():
            query_title = query_entity[0].upper() + query_entity[1:] + 'o'
        else:
            # For proper nouns like "Benjamin Franklin", use as-is
            query_title = query_entity

        # Try to get from Wikipedia article about the entity
        query = f"""
            MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
            WHERE ft.teksto IS NOT NULL
              AND d.metadatenoj CONTAINS 'wikipedia'
              AND d.titolo = '{query_title}'
            RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
            LIMIT {top_k * 10}
        """

        result = conn.execute(query)

        # Check if we got results
        temp_docs = []
        while result.has_next():
            row = result.get_next()
            temp_docs.append(row)

        # If we got results from the Wikipedia article, use them
        if temp_docs:
            print(f"Found {len(temp_docs)} sentences from Wikipedia article: '{query_title}'")
            # Process these results
            documents = []
            for row in temp_docs:
                text = row[0]
                if not text:
                    continue

                # Parse sentence
                ast = parse(text)

                # Count matching roots
                text_lower = text.lower()
                matching_roots = [r for r in roots_list if r in text_lower]

                if matching_roots or len(documents) < 3:  # Keep at least 3 sentences
                    documents.append({
                        'text': text,
                        'ast': ast,
                        'id': row[1],
                        'doc_title': row[2],
                        'metadata': row[3],
                        'matching_roots': matching_roots,
                        'num_matches': len(matching_roots)
                    })

            if documents:
                documents.sort(key=lambda d: d['num_matches'], reverse=True)
                return documents[:top_k]

    # Fallback: broad search if no entity or no Wikipedia article found
    query = f"""
        MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
        WHERE ft.teksto IS NOT NULL
        RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
        LIMIT {top_k * 50}
    """

    result = conn.execute(query)

    # Score documents by matching roots
    documents = []
    while result.has_next():
        row = result.get_next()
        text = row[0]
        doc_id = row[1]
        doc_title = row[2]
        metadata = row[3]

        if not text:
            continue

        # Count matching roots
        text_lower = text.lower()
        matching_roots = [r for r in roots_list if r in text_lower]

        if not matching_roots:
            continue

        # Parse sentence to get AST
        ast = parse(text)

        documents.append({
            'text': text,
            'ast': ast,
            'id': doc_id,
            'doc_title': doc_title,
            'metadata': metadata,
            'matching_roots': matching_roots,
            'num_matches': len(matching_roots)
        })

    # Sort by match count
    documents.sort(key=lambda d: d['num_matches'], reverse=True)

    return documents[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='?', help='Esperanto query')
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'))
    parser.add_argument('--top-k', type=int, default=10, help='Sentences to retrieve')
    parser.add_argument('--max-facts', type=int, default=4, help='Facts to include in answer')
    parser.add_argument('--expand', action='store_true', help='Use embedding expansion')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    generator = ExtractiveAnswerGenerator()

    # Interactive mode
    if args.interactive:
        print("=" * 70)
        print("EXTRACTIVE QA - INTERACTIVE MODE")
        print("=" * 70)
        print("Type your Esperanto questions (or 'quit' to exit)")
        print()

        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue

                process_query(query, args, generator)
                print()

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

    # Single query mode
    elif args.query:
        process_query(args.query, args, generator)

    else:
        parser.print_help()


def process_query(query, args, generator):
    """Process a single query."""
    print("-" * 70)
    print(f"Query: {query}")
    print("-" * 70)

    # Parse query
    ast = parse(query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")

    # Classify question type
    question_type = classify_question_type(query)
    print(f"Question type: {question_type.value}")

    # Extract query entity
    query_entity = extract_query_entity(ast, question_type)
    if query_entity:
        print(f"Query entity: {query_entity}")

    # Expand if requested
    if args.expand:
        expanded_roots = expand_with_embeddings(original_roots, args.embeddings)
        print(f"Expanded to: {len(expanded_roots)} roots")
        added = expanded_roots - original_roots
        if added:
            print(f"Added: {', '.join(sorted(added))}")
        query_roots = expanded_roots
    else:
        query_roots = original_roots

    print()

    # Retrieve sentences
    print(f"Retrieving top {args.top_k} sentences...")
    sentences = retrieve_sentences(args.db, query_roots, question_type,
                                   query_entity, args.top_k)

    if not sentences:
        print("No sentences found!")
        return

    print(f"Retrieved {len(sentences)} sentences")
    print()

    # Generate answer
    print("Generating answer...")
    answer = generator.generate(
        sentences=sentences,
        query=query,
        question_type=question_type,
        query_entity=query_entity,
        max_facts=args.max_facts
    )

    # Display results
    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print()
    print(answer.text)
    print()

    # Citations section (Issue #674)
    if answer.citations:
        print("=" * 70)
        print("CITATIONS")
        print("=" * 70)
        for citation in answer.citations:
            print(f"[{citation.id}] {citation.doc_title} ({citation.doc_source})")
            # Truncate sentence if too long
            sent_text = citation.sentence_text
            if len(sent_text) > 100:
                sent_text = sent_text[:97] + "..."
            print(f"    {sent_text}")
            if citation.sentence_id and citation.sentence_id != "unknown":
                print(f"    ID: {citation.sentence_id}")
            print()

    print("=" * 70)
    print("METADATA")
    print("=" * 70)
    print(f"Facts extracted: {answer.num_facts_extracted}")
    print(f"Facts selected: {answer.num_facts_selected}")
    print()

    if answer.score_breakdowns:
        print("FACT SCORES")
        print("-" * 70)
        for i, (fact, score) in enumerate(zip(answer.facts_used, answer.score_breakdowns), 1):
            print(f"{i}. {fact}")
            print(f"   {score}")
            print()


if __name__ == '__main__':
    main()
