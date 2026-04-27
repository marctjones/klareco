"""
Simplified Command-Line Interface for Klareco POC.

Focused on essential POC functionality:
- Parsing Esperanto text
- Querying corpus with retrieval
- Corpus management
"""
import sys
import os
import argparse
import json
from pathlib import Path


def cmd_parse(args):
    """Parse Esperanto text into AST."""
    from klareco.parser import parse

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        print("Enter Esperanto text:")
        text = input().strip()

    # Parse
    try:
        ast = parse(text)
        if args.format == 'json':
            print(json.dumps(ast, indent=2, ensure_ascii=False))
        else:
            # Simple readable format
            print(f"Sentence type: {ast.get('tipo', 'unknown')}")
            if 'subjekto' in ast:
                print(f"Subject: {ast['subjekto']}")
            if 'verbo' in ast:
                print(f"Verb: {ast['verbo']}")
            if 'objekto' in ast:
                print(f"Object: {ast['objekto']}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_query(args):
    """Answer an Esperanto question using the orchestration pipeline."""
    from klareco.orchestrator import build_default_pipeline

    whoosh_dir = args.whoosh_dir or "data/indexes/whoosh"
    kuzu_path = args.kuzu_path or "data/indexes/v2.1_kuzu_index_full"

    try:
        pipeline = build_default_pipeline(
            whoosh_index_dir=whoosh_dir,
            kuzu_db_path=kuzu_path,
            top_k=args.top_k,
        )
    except Exception as e:
        print(f"ERROR initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    if args.query:
        query_text = args.query
    else:
        print("Enter question in Esperanto:")
        query_text = input().strip()

    try:
        result = pipeline.answer(query_text)
        print(f"\n=== {query_text} ===\n")
        print(result.text or "(no answer)")
        if args.verbose:
            print()
            result.print_trace()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_translate(args):
    """Translate text to/from Esperanto."""
    from klareco.translator import TranslationService
    from klareco.lang_id import identify_language

    translator = TranslationService()

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        print("Enter text:")
        text = input().strip()

    # Determine direction
    if args.from_lang and args.to_lang:
        source_lang = args.from_lang
        target_lang = args.to_lang
    else:
        # Auto-detect
        detected = identify_language(text)
        if detected == 'eo':
            source_lang = 'eo'
            target_lang = args.to_lang or 'en'
        else:
            source_lang = detected
            target_lang = 'eo'

    # Translate
    result = translator.translate(text, source_lang, target_lang)
    print(result)


def cmd_corpus_validate(args):
    """Validate Esperanto corpus file."""
    from klareco.corpus_manager import CorpusManager

    manager = CorpusManager()
    result = manager.validate_file(args.file)

    print(f"File: {args.file}")
    print(f"Valid: {result['valid']}")
    print(f"Total sentences: {result['total_sentences']}")
    print(f"Parseable: {result['parseable_count']} ({result['parse_rate']:.1%})")

    if not result['valid']:
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")


def cmd_corpus_add(args):
    """Add corpus file to registry."""
    from klareco.corpus_manager import CorpusManager

    manager = CorpusManager()
    manager.add_corpus(
        file_path=args.file,
        title=args.title,
        corpus_type=args.type,
        language='eo'
    )
    print(f"✓ Added {args.file} to corpus registry")


def cmd_corpus_list(args):
    """List registered corpus files."""
    from klareco.corpus_manager import CorpusManager

    manager = CorpusManager()
    corpora = manager.list_corpora()

    if not corpora:
        print("No corpus files registered")
        return

    print(f"Registered corpus files ({len(corpora)}):\n")
    for corpus in corpora:
        print(f"  {corpus['title']}")
        print(f"    File: {corpus['file_path']}")
        print(f"    Type: {corpus['type']}")
        print(f"    Sentences: {corpus.get('sentence_count', 'unknown')}")
        print()


def cmd_info(args):
    """Display system information."""
    import torch
    from klareco.parser import KNOWN_PREFIXES, KNOWN_SUFFIXES

    print("=== Klareco POC System Information ===\n")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"\nVocabulary (deterministic):")
    print(f"  Prefixes: {len(KNOWN_PREFIXES)}")
    print(f"  Suffixes: {len(KNOWN_SUFFIXES)}")

    # Check for corpus index
    index_path = Path('data/corpus_index_v3')
    if index_path.exists():
        metadata_path = index_path / 'metadata.jsonl'
        if metadata_path.exists():
            with open(metadata_path) as f:
                count = sum(1 for _ in f)
            print(f"\nCorpus index: {count:,} sentences")
        else:
            print(f"\nCorpus index: Found (metadata missing)")
    else:
        print(f"\nCorpus index: Not found at {index_path}")

    # Check for model
    model_path = Path('models/tree_lstm/best_model.pt')
    if model_path.exists():
        print(f"Tree-LSTM model: Found at {model_path}")
    else:
        print(f"Tree-LSTM model: Not found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='klareco',
        description='Klareco: Pure Esperanto AI - POC with deterministic processing + minimal learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse Esperanto text
  klareco parse "La hundo vidas la katon"
  klareco parse --file input.txt --format json

  # Query corpus with hybrid retrieval
  klareco query "Kio estas Esperanto?"
  klareco query --top-k 5 --verbose

  # Translate text
  klareco translate "The dog sees the cat" --to eo
  klareco translate "Mi amas vin" --to en

  # Corpus management
  klareco corpus validate data/raw/book.txt
  klareco corpus add data/cleaned/book.txt --title "My Book" --type literature
  klareco corpus list

  # System info
  klareco info

POC Goals:
  Month 1-2: Answer 50 questions using ONLY deterministic + retrieval (zero learned reasoning)
  Month 3-4: Add 20M param reasoning core, measure improvement
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- parse command ---
    parser_parse = subparsers.add_parser('parse', help='Parse Esperanto text into AST')
    parser_parse.add_argument('text', nargs='?', help='Esperanto text to parse')
    parser_parse.add_argument('-f', '--file', help='Read input from file')
    parser_parse.add_argument('--format', choices=['text', 'json'], default='text',
                             help='Output format (default: text)')
    parser_parse.set_defaults(func=cmd_parse)

    # --- query command ---
    parser_query = subparsers.add_parser('query', help='Answer an Esperanto question')
    parser_query.add_argument('query', nargs='?', help='Question in Esperanto')
    parser_query.add_argument('--whoosh-dir', help='Path to Whoosh index (default: data/indexes/whoosh)')
    parser_query.add_argument('--kuzu-path', help='Path to Kuzu DB (default: data/indexes/v2.1_kuzu_index_full)')
    parser_query.add_argument('--top-k', type=int, default=10, help='Passages to retrieve (default: 10)')
    parser_query.add_argument('-v', '--verbose', action='store_true', help='Show pipeline trace')
    parser_query.set_defaults(func=cmd_query)

    # --- translate command ---
    parser_translate = subparsers.add_parser('translate', help='Translate text to/from Esperanto')
    parser_translate.add_argument('text', nargs='?', help='Text to translate')
    parser_translate.add_argument('-f', '--file', help='Read input from file')
    parser_translate.add_argument('--from', dest='from_lang',
                                 help='Source language code (auto-detect if not specified)')
    parser_translate.add_argument('--to', dest='to_lang', help='Target language code')
    parser_translate.set_defaults(func=cmd_translate)

    # --- corpus command ---
    parser_corpus = subparsers.add_parser('corpus', help='Corpus management')
    corpus_subparsers = parser_corpus.add_subparsers(dest='corpus_command', help='Corpus commands')

    corpus_validate = corpus_subparsers.add_parser('validate', help='Validate Esperanto corpus file')
    corpus_validate.add_argument('file', help='Path to corpus file')
    corpus_validate.set_defaults(func=cmd_corpus_validate)

    corpus_add = corpus_subparsers.add_parser('add', help='Add corpus file to registry')
    corpus_add.add_argument('file', help='Path to corpus file')
    corpus_add.add_argument('--title', required=True, help='Corpus title')
    corpus_add.add_argument('--type', required=True,
                           choices=['literature', 'dictionary', 'wikipedia', 'other'],
                           help='Corpus type')
    corpus_add.set_defaults(func=cmd_corpus_add)

    corpus_list = corpus_subparsers.add_parser('list', help='List registered corpus files')
    corpus_list.set_defaults(func=cmd_corpus_list)

    # --- info command ---
    parser_info = subparsers.add_parser('info', help='Display system information')
    parser_info.set_defaults(func=cmd_info)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
