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
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import (
    ExtractiveAnswerGenerator, QuestionType, classify_question_type
)
from klareco.knowledge import expand_with_morphology, expand_by_question_type, get_synonyms


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


# Manual synonym expansion - ULTRA-CONSERVATIVE (tuned to reduce retrieval noise)
#
# STRATEGY: Only add synonyms when semantically very close AND low ambiguity.
# REMOVED: Broad cross-mappings (kre/fond/establ) - caused retrieval of meta-content
# KEPT: High-precision pairs only
#
# NOTE: Morphological normalization (reflexive ↔ transitive) handles most verb variants now.
MANUAL_SYNONYMS = {
    # Create/found verbs - ONLY for Esperanto-specific vocabulary
    # CRITICAL: "iniciati" is the ONLY synonym that matters for Zamenhof queries
    # (corpus has "iniciatinto de Esperanto" but queries use "kre/fond")
    'kre': ['iniciati'],       # ONLY iniciati (not fond/establ - too noisy)
    'fond': ['iniciati'],      # ONLY iniciati
    'establ': ['iniciati'],    # ONLY iniciati
    'iniciati': ['kre'],       # ONLY kre (most common in queries)

    # Language-related - keep (low ambiguity)
    'ling': ['lingv'],         # Language only (removed parol/idiom)
    'lingv': ['ling'],

    # Know/understand - keep (high precision)
    'sci': ['kon'],
    'kon': ['sci'],

    # REMOVED ALL OTHER SYNONYMS:
    # - vid/rimark/observ/konsider (too broad, retrieves meta-content)
    # - parol/idiom (ambiguous)
    # - verk/skrib/publik (causes book/writing noise)
    # - libr/dokument (causes document metadata noise)
    # - person/hom (too generic)
    # - lern/stud (too broad)
}


def is_entity_root(root):
    """
    Check if a root represents an entity (proper name) that should not be expanded.

    Entities include:
    - Proper names (Esperanto, Zamenhof, etc.)
    - Place names from knowledge module
    - Common Esperanto-specific terms

    Returns True if root should NOT be expanded with synonyms.
    """
    from klareco.knowledge import place_names

    # Common entities that should not be expanded
    known_entities = {
        'esperant',  # Esperanto (the language) - don't expand to "kre/establ"
        'zamenhof',  # Zamenhof (the person) - don't expand
        'fundament', # Fundamento (specific document) - don't expand
        'bjalistok', # Bjalistoko (city)
        'varsov',    # Varsovio (city)
        'pol',       # Pollando (country)
    }

    # Check if in known entities
    if root.lower() in known_entities:
        return True

    # Check if in place names gazetteer (case-insensitive)
    if any(root.lower() == place.lower() for place in place_names):
        return True

    return False


def expand_with_manual_synonyms(roots):
    """
    Expand query roots with synonyms from semantic ontology + manual fallback.

    Uses semantic ontology (v2.2+) for verb class synonyms, with fallback to
    manually curated synonyms for backwards compatibility.

    Returns expanded set of roots.

    IMPORTANT:
    - Does NOT expand entity roots (proper names) to avoid retrieval noise
    - DOES expand verbs using semantic ontology verb classes
    - Limits synonym expansion to prevent retrieval noise
    """
    expanded = set(roots)

    for root in roots:
        # Skip entities - don't expand proper names
        if is_entity_root(root):
            continue

        # Get synonyms from semantic ontology
        synonyms = get_synonyms(root)

        # Limit to top 5 synonyms to prevent retrieval noise
        if synonyms:
            # Sort for consistency, take top 5
            top_synonyms = sorted(synonyms)[:5]
            expanded.update(top_synonyms)

        # Also include manual synonyms as fallback (for entities not in ontology)
        if root in MANUAL_SYNONYMS:
            expanded.update(MANUAL_SYNONYMS[root])

    return expanded


def expand_with_embeddings(roots, embeddings_path, k=5, threshold=0.70):
    """
    Expand roots using embeddings.

    IMPORTANT: Does NOT expand entity roots (proper names) to avoid retrieval noise.
    """
    checkpoint = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']
    root_to_idx = {root: idx for idx, root in enumerate(vocab)}

    expanded = set(roots)

    for root in roots:
        # Skip entities - don't expand proper names
        if is_entity_root(root):
            continue

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


def retrieve_sentences(retriever, roots, question_type, query_entity=None, top_k=10):
    """Retrieve sentences using Whoosh FTS index with AST-aware filtering."""
    # Convert roots to list
    roots_list = list(roots)

    # Convert question_type enum to string value
    question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)

    # Extract root from query_entity (strip case endings like -n, -j, -jn AND word endings like -o, -a)
    entity_root = None
    if query_entity:
        # Strip common Esperanto endings: -n (accusative), -j (plural), -jn (plural accusative)
        entity_root = query_entity.lower()
        if entity_root.endswith('jn'):
            entity_root = entity_root[:-2]
        elif entity_root.endswith('n') or entity_root.endswith('j'):
            entity_root = entity_root[:-1]

        # CRITICAL: Strip word ending (-o, -a, -e) to get ROOT for AST matching
        # ASTs contain ROOTS (esperant), not full words (esperanto)
        if entity_root.endswith('o') or entity_root.endswith('a') or entity_root.endswith('e'):
            entity_root = entity_root[:-1]

    # Retrieve using Whoosh BM25 search + AST filtering
    # Get more candidates for reranking (top_k * 10 ensures good recall)
    documents = retriever.retrieve(
        query_roots=roots_list,
        top_k=top_k * 10,
        retrieval_limit=200,  # Reduced for speed with wildcard queries
        question_type=question_type_str,
        query_entity=entity_root
    )

    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='?', help='Esperanto query')
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'))
    parser.add_argument('--top-k', type=int, default=10, help='Sentences to retrieve')
    parser.add_argument('--max-facts', type=int, default=4, help='Facts to include in answer')
    parser.add_argument('--no-expand', action='store_true', help='Disable neural embedding expansion')
    parser.add_argument('--no-rerank', action='store_true', help='Disable neural reranker')
    parser.add_argument('--no-m1', action='store_true', help='Disable M1 selectional filtering')
    parser.add_argument('--m1-threshold', type=float, default=0.3, help='M1 plausibility threshold (0-1)')
    parser.add_argument('--reranker-path', type=Path,
                       default=Path('models/reranker/best_model.pt'), help='Path to reranker model')
    parser.add_argument('--m1-path', type=Path,
                       default=Path('models/m1_selectional/best_model.pt'), help='Path to M1 model')
    parser.add_argument('--single-span-types', type=str, nargs='+',
                       choices=['KIU', 'KIO', 'KIE', 'KIAM', 'KIAL', 'KIEL'],
                       help='Question types that should return single-span answers (Esperanto: kiu/kio/kie/kiam/kial/kiel). Default: none, all use multi-sentence')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show retrieved sentences and extraction attempts')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    # Initialize Whoosh retriever (load once, reuse for all queries)
    print("Loading Whoosh FTS index...")
    retriever = WhooshRetriever(
        whoosh_index_dir=Path('data/indexes/whoosh_fts'),
        kuzu_db_path=args.db
    )
    print("Whoosh index loaded.")

    # Build multi_sentence_question_types dict from --single-span-types flag
    # Default: all question types use multi-sentence answers (True)
    # If --single-span-types is provided, those types use single-span (False)
    multi_sentence_config = None
    if args.single_span_types:
        from klareco.rag.importance_scorer import QuestionType

        # Map Esperanto question words to QuestionType enum
        eo_to_enum = {
            'KIU': QuestionType.WHO,      # kiu = who/which
            'KIO': QuestionType.WHAT,     # kio = what
            'KIE': QuestionType.WHERE,    # kie = where
            'KIAM': QuestionType.WHEN,    # kiam = when
            'KIAL': QuestionType.WHY,     # kial = why
            'KIEL': QuestionType.HOW,     # kiel = how
        }

        # Convert Esperanto words to enum values
        single_span_enums = {eo_to_enum[word] for word in args.single_span_types}

        multi_sentence_config = {
            QuestionType.WHO: QuestionType.WHO not in single_span_enums,
            QuestionType.WHAT: QuestionType.WHAT not in single_span_enums,
            QuestionType.WHERE: QuestionType.WHERE not in single_span_enums,
            QuestionType.WHEN: QuestionType.WHEN not in single_span_enums,
            QuestionType.WHY: QuestionType.WHY not in single_span_enums,
            QuestionType.HOW: QuestionType.HOW not in single_span_enums,
            QuestionType.OTHER: True,  # OTHER always uses multi-sentence
        }
        print(f"Single-span types: {args.single_span_types} (Esperanto question words)")

    # Initialize answer generator with neural models
    # Note: Reranker and M1 use 64D embeddings (models/root_embeddings/best_model.pt)
    # while query expansion uses 128D embeddings (args.embeddings)
    generator = ExtractiveAnswerGenerator(
        reranker_path=args.reranker_path,
        m1_model_path=args.m1_path,
        # Don't pass embedding_path - let it use default 64D embeddings for models
        use_reranker=not args.no_rerank,
        use_m1=not args.no_m1,
        m1_threshold=args.m1_threshold,
        multi_sentence_question_types=multi_sentence_config
    )

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

                process_query(query, args, generator, retriever)
                print()

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

    # Single query mode
    elif args.query:
        process_query(args.query, args, generator, retriever)

    else:
        parser.print_help()


def process_query(query, args, generator, retriever):
    """Process a single query."""
    print("-" * 70)
    print(f"Query: {query}")
    print("-" * 70)

    # Parse query
    ast = parse(query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")

    # Check for entity roots (proper names that won't be expanded)
    entity_roots = {r for r in original_roots if is_entity_root(r)}
    if entity_roots:
        print(f"Entity roots (protected from expansion): {', '.join(sorted(entity_roots))}")

    # Classify question type
    question_type = classify_question_type(query)
    print(f"Question type: {question_type.value}")

    # Extract query entity
    query_entity = extract_query_entity(ast, question_type)
    if query_entity:
        print(f"Query entity: {query_entity}")

    # STAGE 1: Morphological normalization (reflexive ↔ transitive mapping)
    # This is the highest-priority fix for vocabulary mismatch (70% of failures)
    # Example: "naskiĝis" (reflexive) → {naskiĝ, nask} → matches corpus "naskita"
    morph_expanded = expand_with_morphology(original_roots)
    if morph_expanded != original_roots:
        added_morph = morph_expanded - original_roots
        print(f"Morphological variants added: {', '.join(sorted(added_morph))}")

    # STAGE 2A: Quick Win #682: Apply manual synonym expansion (always on)
    # BUT: Skip entity roots to avoid retrieval noise
    synonym_expanded = expand_with_manual_synonyms(morph_expanded)
    if synonym_expanded != morph_expanded:
        added_synonyms = synonym_expanded - morph_expanded
        print(f"Manual synonyms added: {', '.join(sorted(added_synonyms))}")

    # STAGE 2B: Question-type specific expansion (NEW - addresses 20% of failures)
    # Add semantic category terms based on question type:
    # - WHEN: jaro, dato, aper, komenc (temporal vocabulary)
    # - WHAT: tipo, specio, mamul (category indicators)
    # - WHO "estis X?": kuracist, doktor, profesor (professions)
    question_expanded = expand_by_question_type(
        synonym_expanded,
        question_type.value if hasattr(question_type, 'value') else str(question_type),
        query
    )
    if question_expanded != synonym_expanded:
        added_question = question_expanded - synonym_expanded
        print(f"Question-type expansion added: {', '.join(sorted(added_question))}")

    # STAGE 3: Neural expansion: Always use root embeddings for semantic query expansion
    # (unless --no-expand flag is set)
    # NOTE: Threshold raised to 0.70 (from 0.65) to reduce noise
    if not args.no_expand:
        expanded_roots = expand_with_embeddings(question_expanded, args.embeddings)
        print(f"Expanded to: {len(expanded_roots)} roots")
        added = expanded_roots - question_expanded
        if added:
            print(f"Embedding expansion added: {', '.join(sorted(added))}")
        query_roots = expanded_roots
    else:
        query_roots = question_expanded

    print()

    # Retrieve sentences
    print(f"Retrieving top {args.top_k} sentences...")
    sentences = retrieve_sentences(retriever, query_roots, question_type,
                                   query_entity, args.top_k)

    if not sentences:
        print("No sentences found!")
        return

    print(f"Retrieved {len(sentences)} sentences")

    # Show retrieved sentences if verbose
    if args.verbose:
        print("\n" + "=" * 70)
        print("TOP RETRIEVED SENTENCES (before extraction)")
        print("=" * 70)
        for i, sent in enumerate(sentences[:10], 1):  # Show top 10
            score = sent.get('score', 0.0)
            text = sent.get('text', '')
            # Truncate long sentences
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    {text}")
        print()

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
