#!/usr/bin/env python3
"""
Smart Retrieval Demo - Question-Aware Retrieval with v2.1 Database

Uses question classification and entity-aware boosting from ASTAwareRetriever
but queries the v2.1 Kuzu schema directly.

Usage:
    python scripts/demo_smart_retrieval.py "Kio estas Esperanto?"
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import kuzu

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.question_classifier import QuestionClassifier, QuestionType


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


def score_document_for_question(doc_text, question_type, query_entity=None):
    """
    Apply question-aware scoring boost.

    WHAT questions: Boost definitional language about the query entity
    WHO questions: Boost person names
    WHERE questions: Boost place names
    WHEN questions: Boost dates/times
    """
    boost = 1.0
    text_lower = doc_text.lower()

    if question_type == QuestionType.WHAT:
        # For "What is X?" questions, boost sentences that define X
        if query_entity:
            # Check if entity appears near "estas" (definitional pattern)
            # "Esperanto estas..." or "estas ... Esperanto"
            entity_lower = query_entity.lower()
            if entity_lower in text_lower:
                # Check if entity is near "estas" (within 20 chars)
                entity_pos = text_lower.find(entity_lower)
                estas_pos = text_lower.find(' estas ')
                if estas_pos >= 0:
                    distance = abs(entity_pos - estas_pos)
                    if distance < 30:  # Entity near "estas"
                        # Check if this is actually definitional
                        # (not just grammar example with "estas")
                        if entity_pos < 50:  # Entity at start of sentence
                            boost *= 5.0  # Strong definitional boost
                        else:
                            boost *= 2.0

        # Also boost sentences with definitional vocabulary
        if 'signifas' in text_lower or 'difino' in text_lower:
            boost *= 2.0

    elif question_type == QuestionType.WHO:
        # Boost sentences with person names (capitalized words)
        words = doc_text.split()
        capitalized = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
        if capitalized >= 2:  # Likely contains names
            boost *= 2.0

    elif question_type == QuestionType.WHERE:
        # Boost location markers
        if any(loc in text_lower for loc in ['urbo', 'lando', 'loko', 'regiono']):
            boost *= 2.0

    elif question_type == QuestionType.WHEN:
        # Boost dates/times
        if any(char.isdigit() for char in doc_text):  # Contains numbers (likely dates)
            boost *= 2.0

    return boost


def query_with_smart_scoring(db_path, roots, question_type, query_entity=None, top_k=10):
    """Query with question-aware scoring."""
    print(f"Connecting to Kuzu: {db_path}")
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    roots_list = list(roots)

    # Query for sentences WITH document title information
    # For WHAT questions with a query entity, prioritize Wikipedia articles about that entity
    if question_type == QuestionType.WHAT and query_entity:
        # Create proper title (capitalize + add -o ending if it's a root)
        # e.g., "esperant" -> "Esperanto"
        query_title = query_entity[0].upper() + query_entity[1:] + 'o'

        # First get from Wikipedia article about the entity
        query = f"""
            MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
            WHERE ft.teksto IS NOT NULL
              AND d.metadatenoj CONTAINS 'wikipedia'
              AND d.titolo = '{query_title}'
            RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
            LIMIT {top_k * 10}
        """

        print(f"Querying Wikipedia article titled '{query_title}'...")
        result_wiki = conn.execute(query)

        # Also get general results
        query_general = f"""
            MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
            WHERE ft.teksto IS NOT NULL
            RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
            LIMIT {top_k * 20}
        """

        print(f"Also querying general corpus...")
        result_general = conn.execute(query_general)

        # Merge results (Wikipedia first)
        result = result_wiki
    else:
        query = f"""
            MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
            WHERE ft.teksto IS NOT NULL
            RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
            LIMIT {top_k * 50}
        """
        print(f"Fetching candidate documents...")
        result = conn.execute(query)

    # Score documents
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

        # Base score: number of matching roots
        base_score = len(matching_roots)

        # Apply question-aware boost
        question_boost = score_document_for_question(text, question_type, query_entity)

        # Apply document title boost (HUGE for matching Wikipedia articles)
        doc_title_boost = 1.0
        if query_entity and doc_title:
            # If document title matches query entity (e.g., "Esperanto" article for "Esperanto" query)
            if query_entity.lower() in doc_title.lower():
                doc_title_boost = 10.0  # Massive boost!
                # Extra boost if sentence STARTS with the entity
                if text.lower().strip().startswith(query_entity.lower()):
                    doc_title_boost = 50.0  # Extreme boost for definitional sentences

        # Final score
        final_score = base_score * question_boost * doc_title_boost

        documents.append({
            'text': text,
            'id': doc_id,
            'doc_title': doc_title,
            'matching_roots': matching_roots,
            'num_matches': len(matching_roots),
            'base_score': base_score,
            'question_boost': question_boost,
            'doc_title_boost': doc_title_boost,
            'final_score': final_score
        })

    # Sort by final score
    documents.sort(key=lambda d: d['final_score'], reverse=True)

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
    print("SMART RETRIEVAL DEMO (Question-Aware)")
    print("="*70)
    print()

    # Parse query
    print(f"Query: {args.query}")
    print("-"*70)
    ast = parse(args.query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")

    # Classify question type
    classifier = QuestionClassifier()
    classification = classifier.classify(args.query, ast)
    question_type = classification['question_type']
    print(f"Question type: {question_type.value}")

    # Extract query entity (for WHAT questions like "What is X?")
    query_entity = None
    if question_type == QuestionType.WHAT:
        # Look for the entity being asked about
        # Check object first
        obj = ast.get('objekto')
        if obj:
            if obj.get('tipo') == 'vortgrupo':
                kerno = obj.get('kerno', {})
            else:
                kerno = obj
            query_entity = kerno.get('radiko', '')

        # If no object, check aliaj (modifiers) for substantivo
        if not query_entity:
            aliaj = ast.get('aliaj', [])
            for alia in aliaj:
                if alia.get('vortspeco') == 'substantivo':
                    query_entity = alia.get('radiko', '')
                    break

        if query_entity:
            print(f"Query entity: {query_entity}")

    print()

    # Expand if requested
    if args.expand:
        print("Expanding with embeddings...")
        expanded_roots = expand_with_embeddings(original_roots, args.embeddings)
        print(f"Expanded to: {len(expanded_roots)} roots")
        added = expanded_roots - original_roots
        if added:
            print(f"Added: {', '.join(sorted(added))}")
        query_roots = expanded_roots
    else:
        query_roots = original_roots

    print()

    # Query with smart scoring
    documents = query_with_smart_scoring(args.db, query_roots, question_type, query_entity, args.top_k)

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
            doc_title = doc.get('doc_title', 'unknown')

            score_parts = f"{doc['base_score']} roots × {doc['question_boost']:.1f}x Q"
            if doc.get('doc_title_boost', 1.0) > 1.0:
                score_parts += f" × {doc['doc_title_boost']:.1f}x Title"

            print(f"{i}. [Score: {doc['final_score']:.1f} = {score_parts}]")
            print(f"   Article: {doc_title}")
            print(f"   Matching: {roots_str}")
            print(f"   {text}")
            print()


if __name__ == '__main__':
    main()
