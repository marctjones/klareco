#!/usr/bin/env python3
"""
End-to-End RAG Demo with M1 Plausibility Filtering

Combines:
1. ASTAwareRetriever for deterministic retrieval (Kuzu backend)
2. M1 Selectional Preferences for plausibility filtering
3. Answer extraction from retrieved sentences

⚠️  KLARECO PURITY GUARANTEE:
   - The Klareco AI model is 100% Pure Esperanto
   - ALL processing happens in Esperanto (parsing, retrieval, scoring)
   - English translations in this demo are DISPLAY ONLY (like subtitles)
   - No English contaminates training data, ASTs, embeddings, or models

Usage:
    python scripts/demo_rag_with_m1.py                          # Run example queries (with EN translations)
    python scripts/demo_rag_with_m1.py -i                       # Interactive mode
    python scripts/demo_rag_with_m1.py "Kiu fondis Esperanton?" # Single query
    python scripts/demo_rag_with_m1.py -i --no-translate        # Pure Esperanto (no translations)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.models.m1_inference import M1Inference
from klareco.parser import parse


class RAGWithM1:
    """RAG system with M1 plausibility filtering."""

    def __init__(
        self,
        index_path: Path = None,
        m1_model_path: Path = None,
        stage1_path: Path = None,
    ):
        """
        Initialize RAG system.

        Args:
            index_path: Path to Kuzu index (default: data/indexes/kuzu_index)
            m1_model_path: Path to M1 model (default: models/m1_semantic_full/best_model.pt)
            stage1_path: Path to Stage 1 embeddings (default: models/root_embeddings_tier0/best_model.pt)
        """
        print("Initializing RAG system with M1 filtering...")

        # Initialize retriever
        if index_path is None:
            index_path = Path('data/indexes/kuzu_index')
        self.retriever = ASTAwareRetriever(index_path=index_path)
        print(f"  ✓ Retriever loaded")

        # Initialize M1 inference
        self.m1 = M1Inference(
            model_path=m1_model_path,
            comp_model_path=stage1_path,  # Name kept as stage1_path for backward compat in RAGWithM1
            device='cpu'
        )
        print(f"  ✓ M1 model loaded")
        print()

    def extract_svo_triple(self, ast: Dict) -> Tuple[str, str, str]:
        """
        Extract subject-verb-object triple from AST.

        Returns:
            (subject_root, verb_root, object_root) or (None, None, None) if incomplete
        """
        def get_root(node):
            if node is None:
                return None
            if isinstance(node, dict):
                if node.get('tipo') == 'vortgrupo':
                    kerno = node.get('kerno', {})
                    return kerno.get('radiko')
                elif node.get('tipo') == 'vorto':
                    return node.get('radiko')
            return None

        subj = get_root(ast.get('subjekto'))
        verb = get_root(ast.get('verbo'))
        obj = get_root(ast.get('objekto'))

        return (subj, verb, obj)

    def answer(
        self,
        query: str,
        top_k: int = 10,
        m1_threshold: float = 0.5,
        rerank_with_m1: bool = True,
    ) -> List[Dict]:
        """
        Answer a query using retrieval + M1 filtering.

        Args:
            query: Query string (Esperanto)
            top_k: Number of results to retrieve initially
            m1_threshold: Minimum M1 plausibility score (0.5 = default)
            rerank_with_m1: Whether to rerank by M1 score

        Returns:
            List of answer dicts with: text, source, score, m1_score, plausible
        """
        # Step 1: Retrieve candidates using AST-aware retrieval
        results = self.retriever.search(query, top_k=top_k)

        if not results:
            return []

        # Step 2: Score with M1 and filter
        answers = []
        for score, doc, stats in results:
            text = doc.get('text', '')

            # Parse retrieved sentence
            try:
                doc_ast = parse(text)
            except Exception:
                # If parse fails, include with neutral M1 score
                answers.append({
                    'text': text,
                    'source': doc.get('source', {}).get('name', 'unknown'),
                    'retrieval_score': score,
                    'm1_score': 0.5,
                    'plausible': False,
                    'parse_failed': True,
                })
                continue

            # Extract S-V-O triple
            subj, verb, obj = self.extract_svo_triple(doc_ast)

            if subj and verb and obj:
                # Score with M1
                m1_score = self.m1.score_triple(subj, verb, obj)
                plausible = m1_score >= m1_threshold

                answers.append({
                    'text': text,
                    'source': doc.get('source', {}).get('name', 'unknown'),
                    'retrieval_score': score,
                    'm1_score': m1_score,
                    'plausible': plausible,
                    'triple': (subj, verb, obj),
                })
            else:
                # No complete triple, include with neutral score
                answers.append({
                    'text': text,
                    'source': doc.get('source', {}).get('name', 'unknown'),
                    'retrieval_score': score,
                    'm1_score': 0.5,
                    'plausible': False,
                    'incomplete_triple': True,
                })

        # Step 3: Rerank by M1 score if requested
        if rerank_with_m1:
            # Sort by M1 score descending (plausible answers first)
            answers.sort(key=lambda x: x['m1_score'], reverse=True)

        # Step 4: Filter to only plausible
        plausible_answers = [a for a in answers if a['plausible']]

        return plausible_answers if plausible_answers else answers[:3]  # Fallback: top 3 by retrieval


def format_answer(rank: int, answer: Dict, show_translations: bool = True) -> str:
    """Format a single answer for display."""
    lines = []

    # Header
    retrieval = answer['retrieval_score']
    m1 = answer['m1_score']
    status = "✓ PLAUSIBLE" if answer.get('plausible') else "✗ IMPLAUSIBLE"

    lines.append(f"  {rank}. {status}")
    lines.append(f"     {answer['text']}")

    # Add English translation if available
    if show_translations:
        translation = translate_to_english(answer['text'])
        if translation:
            lines.append(f"     → {translation}")

    lines.append(f"     Source: {answer['source']}")
    lines.append(f"     Retrieval: {retrieval:.3f} | M1: {m1:.3f}")

    # Show triple if available
    if 'triple' in answer:
        subj, verb, obj = answer['triple']
        lines.append(f"     Triple: ({subj}, {verb}, {obj})")

    return '\n'.join(lines)


# Lazy-loaded translator (None until first use)
_TRANSLATOR = None
_TRANSLATOR_FAILED = False


def _get_translator():
    """
    Lazy load Helsinki-NLP MarianMT translator (Esperanto → English).

    ⚠️  DISPLAY ONLY - NOT PART OF KLARECO MODEL
    Returns None if translation unavailable (will fallback to simple dictionary).
    """
    global _TRANSLATOR, _TRANSLATOR_FAILED

    if _TRANSLATOR_FAILED:
        return None

    if _TRANSLATOR is not None:
        return _TRANSLATOR

    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        model_name = "Helsinki-NLP/opus-mt-eo-en"
        print(f"Loading translator: {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        _TRANSLATOR = (model, tokenizer)
        print("  ✓ Translator loaded")
        return _TRANSLATOR
    except Exception as e:
        print(f"  ⚠ Translator unavailable (will use fallback): {e}")
        _TRANSLATOR_FAILED = True
        return None


def translate_to_english(eo_text: str) -> str:
    """
    Translate Esperanto to English using Helsinki-NLP MarianMT.

    ⚠️  IMPORTANT: This is ONLY for demo UI display purposes.
    ⚠️  This function has ZERO contact with the Klareco AI model.
    ⚠️  The Klareco pipeline remains Pure Esperanto (no English in processing).

    This is a display helper for demo purposes - not part of Klareco.
    Think of it like subtitles: the movie (Klareco) is in Esperanto,
    subtitles are just for the viewer.

    Uses:
    1. Helsinki-NLP MarianMT (if available)
    2. Fallback to simple dictionary (if MarianMT fails)
    """
    # Try MarianMT first
    translator = _get_translator()
    if translator is not None:
        model, tokenizer = translator
        try:
            import torch
            inputs = tokenizer(eo_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=100, num_beams=4)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            # Fall through to dictionary fallback
            pass

    # Fallback: Simple dictionary (if MarianMT unavailable)
    word_map = {
        'kiu': 'who', 'kio': 'what', 'kie': 'where', 'kiam': 'when',
        'estas': 'is', 'fondis': 'founded', 'naskiĝis': 'was born',
        'esperanto': 'Esperanto', 'esperanton': 'Esperanto',
        'zamenhof': 'Zamenhof', 'la': 'the', 'kaj': 'and',
    }

    words = eo_text.lower().replace('?', ' ?').replace('.', ' .').split()
    translated = []
    for word in words:
        root = word.rstrip('ojnaes')
        if word in word_map:
            translated.append(word_map[word])
        elif root in word_map:
            translated.append(word_map[root])
        else:
            translated.append(f"[{word}]")

    result = ' '.join(translated)
    result = result.replace(' ?', '?').replace(' .', '.')
    return result


def run_query(rag: RAGWithM1, query: str, top_k: int = 10, show_translations: bool = True):
    """Run a single query and display results."""
    print(f"\nQuery: {query}")

    # Show English translation of query
    if show_translations:
        query_translation = translate_to_english(query)
        if query_translation:
            print(f"  → {query_translation}")

    print("-" * 70)

    start = time.time()
    try:
        answers = rag.answer(query, top_k=top_k, rerank_with_m1=True)
        elapsed = time.time() - start

        if not answers:
            print("  No results found.")
            return

        print(f"  Found {len(answers)} plausible answers in {elapsed:.2f}s\n")

        for i, answer in enumerate(answers[:5], 1):
            print(format_answer(i, answer, show_translations=show_translations))
            print()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(rag: RAGWithM1, show_translations: bool = True):
    """Run interactive query loop."""
    print("\n" + "=" * 70)
    print("RAG with M1 Filtering - Interactive Mode")
    print("=" * 70)
    print("Enter questions in Esperanto. Type 'quit' to exit.")
    print(f"English translations: {'ON' if show_translations else 'OFF'}")
    print()

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', ':q'):
            print("Goodbye!")
            break

        run_query(rag, query, show_translations=show_translations)


def main():
    parser = argparse.ArgumentParser(
        description="RAG demo with M1 plausibility filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('query', nargs='?', help='Query to search (or use -i for interactive)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--index', type=str, default='data/indexes/kuzu_index',
                        help='Index directory')
    parser.add_argument('--m1-model', type=str, default='models/m1_semantic_full/best_model.pt',
                        help='M1 model path')
    parser.add_argument('--stage1', type=str, default='models/root_embeddings_tier0/best_model.pt',
                        help='Stage 1 embeddings path (deprecated, use --comp-model)')
    parser.add_argument('--comp-model', type=str, default=None,
                        help='CompositionalEmbedding path (replaces --stage1)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to retrieve')
    parser.add_argument('--no-translate', action='store_true',
                        help='Disable English translations (show only Esperanto)')

    args = parser.parse_args()

    index_path = Path(args.index)
    m1_model_path = Path(args.m1_model)

    # Use comp-model if specified, otherwise fall back to stage1
    comp_model_path = Path(args.comp_model) if args.comp_model else Path(args.stage1)

    # Check paths exist
    if not (index_path / "kuzu.db").exists():
        print(f"Error: Kuzu index not found at {index_path}/kuzu.db")
        print("Build index: python scripts/index_kuzu.py")
        sys.exit(1)

    if not m1_model_path.exists():
        print(f"Error: M1 model not found at {m1_model_path}")
        print("Train M1: ./scripts/train_m1_semantic.sh")
        sys.exit(1)

    if not comp_model_path.exists():
        print(f"Error: CompositionalEmbedding not found at {comp_model_path}")
        print("Train embeddings: ./scripts/train_roots.sh")
        sys.exit(1)

    # Initialize RAG system
    try:
        rag = RAGWithM1(
            index_path=index_path,
            m1_model_path=m1_model_path,
            stage1_path=comp_model_path,
        )
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Ready!\n")

    # Determine translation setting
    show_translations = not args.no_translate

    # Interactive or single query
    if args.interactive:
        interactive_mode(rag, show_translations=show_translations)
    elif args.query:
        run_query(rag, args.query, top_k=args.top_k, show_translations=show_translations)
    else:
        # Default: run example queries
        example_queries = [
            "Kiu fondis Esperanton?",
            "Kio estas Esperanto?",
            "Kie naskiĝis Zamenhof?",
        ]
        print("Running example queries (use -i for interactive mode):\n")
        for query in example_queries:
            run_query(rag, query, top_k=10, show_translations=show_translations)


if __name__ == '__main__':
    main()
