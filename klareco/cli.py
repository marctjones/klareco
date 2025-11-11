"""
Unified Command-Line Interface for Klareco.

Provides a coherent interface for all Klareco functionality:
- Running the pipeline on input text
- Running tests
- Managing corpus data
- Setup and configuration
- Utility commands (parse, translate)
"""
import sys
import os
import argparse
import json
from pathlib import Path


def cmd_run(args):
    """Run the Klareco pipeline on input text."""
    from klareco.pipeline import KlarecoPipeline
    from klareco.logging_config import setup_logging

    # Setup logging
    setup_logging(log_file=args.log_file, debug=args.debug)

    # Get input text
    if args.text:
        query = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            query = f.read().strip()
    else:
        # Read from stdin
        print("Enter text (Ctrl+D when done):")
        query = sys.stdin.read().strip()

    # Run pipeline
    pipeline = KlarecoPipeline()
    trace = pipeline.run(query, stop_after=args.stop_after)

    # Output result
    if args.output_format == 'json':
        print(trace.to_json())
    elif args.output_format == 'trace':
        print(trace.to_json(indent=2))
    else:  # 'text'
        if trace.error:
            print(f"ERROR: {trace.error}", file=sys.stderr)
            sys.exit(1)
        else:
            print(trace.final_response or "[No response generated]")


def cmd_test(args):
    """Run integration tests."""
    # Import here to avoid loading test dependencies if not needed
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if args.pytest:
        # Use pytest
        import pytest
        pytest_args = ['tests/']
        if args.verbose:
            pytest_args.append('-v')
        if args.pattern:
            pytest_args.append('-k')
            pytest_args.append(args.pattern)
        sys.exit(pytest.main(pytest_args))
    else:
        # Use integration test script
        from scripts.run_integration_test import run_integration_test
        run_integration_test(
            corpus_path=args.corpus,
            num_sentences=args.num_sentences,
            stop_after=args.stop_after,
            debug=args.debug
        )


def cmd_parse(args):
    """Parse Esperanto text into AST (debugging utility)."""
    from klareco.parser import parse
    import json

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
        print(json.dumps(ast, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_translate(args):
    """Translate text to/from Esperanto."""
    from klareco.translator import EsperantoTranslator
    from klareco.lang_id import identify_language

    translator = EsperantoTranslator()

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
    if source_lang == 'eo':
        result = translator.translate_from_esperanto(text, target_lang)
    else:
        result = translator.translate_to_esperanto(text, source_lang)

    print(result)


def cmd_corpus_clean(args):
    """Clean corpus data."""
    from scripts.clean_corpus import main as clean_main
    clean_main()


def cmd_corpus_verify(args):
    """Verify corpus integrity."""
    from scripts.verify_corpus import main as verify_main
    verify_main()


def cmd_corpus_create_test(args):
    """Create test corpus."""
    from scripts.create_test_corpus import main as create_test_main
    create_test_main()


def cmd_setup(args):
    """Setup Klareco (download models, build vocabularies)."""
    print("Setting up Klareco...")

    # Download FastText model if needed
    if args.download_models or args.all:
        print("\n1. Downloading FastText language identification model...")
        from scripts.download_fasttext_model import main as download_main
        download_main()

    # Build morpheme vocabulary if needed
    if args.build_vocab or args.all:
        print("\n2. Building morpheme vocabulary...")
        from scripts.build_morpheme_vocab import main as vocab_main
        vocab_main()

    print("\n✓ Setup complete!")


def cmd_info(args):
    """Display system information."""
    import torch
    from klareco.parser import KNOWN_ROOTS, KNOWN_PREFIXES, KNOWN_SUFFIXES

    print("=== Klareco System Information ===\n")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"\nVocabulary:")
    print(f"  Roots: {len(KNOWN_ROOTS)}")
    print(f"  Prefixes: {len(KNOWN_PREFIXES)}")
    print(f"  Suffixes: {len(KNOWN_SUFFIXES)}")

    # Check for test corpus
    corpus_path = Path(__file__).parent.parent / 'data' / 'test_corpus.json'
    if corpus_path.exists():
        with open(corpus_path) as f:
            corpus = json.load(f)
        print(f"\nTest corpus: {len(corpus)} sentences")
    else:
        print(f"\nTest corpus: Not found")

    # Check for models
    model_dir = Path(__file__).parent.parent / 'models'
    if model_dir.exists():
        models = list(model_dir.glob('*'))
        print(f"Models: {len(models)} files in models/")
    else:
        print(f"Models: Directory not found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='klareco',
        description='Klareco: A neuro-symbolic AI agent using Esperanto as an intermediate representation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline on text
  klareco run "La hundo vidas la katon."
  klareco run --file input.txt

  # Run tests
  klareco test
  klareco test --num-sentences 10

  # Parse Esperanto text
  klareco parse "mi amas la hundon"

  # Translate text
  klareco translate "The dog sees the cat."

  # Setup and info
  klareco setup --all
  klareco info

For more information, visit: https://github.com/yourusername/klareco
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- run command ---
    parser_run = subparsers.add_parser('run', help='Run the Klareco pipeline on input text')
    parser_run.add_argument('text', nargs='?', help='Input text to process')
    parser_run.add_argument('-f', '--file', help='Read input from file')
    parser_run.add_argument('-o', '--output-format', choices=['text', 'json', 'trace'],
                           default='text', help='Output format (default: text)')
    parser_run.add_argument('--stop-after', choices=['SafetyMonitor', 'FrontDoor', 'Parser',
                           'SafetyMonitor_AST', 'IntentClassifier', 'Responder'],
                           help='Stop pipeline after this step')
    parser_run.add_argument('--log-file', default='klareco.log', help='Log file path')
    parser_run.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser_run.set_defaults(func=cmd_run)

    # --- test command ---
    parser_test = subparsers.add_parser('test', help='Run integration tests')
    parser_test.add_argument('--corpus', default='data/test_corpus.json',
                            help='Path to test corpus (default: data/test_corpus.json)')
    parser_test.add_argument('--num-sentences', type=int, help='Limit number of test sentences')
    parser_test.add_argument('--stop-after', help='Stop pipeline after this step')
    parser_test.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser_test.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser_test.add_argument('--pytest', action='store_true', help='Use pytest instead of integration script')
    parser_test.add_argument('--pattern', help='Test name pattern (for pytest -k)')
    parser_test.set_defaults(func=cmd_test)

    # --- parse command ---
    parser_parse = subparsers.add_parser('parse', help='Parse Esperanto text into AST (debugging)')
    parser_parse.add_argument('text', nargs='?', help='Esperanto text to parse')
    parser_parse.add_argument('-f', '--file', help='Read input from file')
    parser_parse.set_defaults(func=cmd_parse)

    # --- translate command ---
    parser_translate = subparsers.add_parser('translate', help='Translate text to/from Esperanto')
    parser_translate.add_argument('text', nargs='?', help='Text to translate')
    parser_translate.add_argument('-f', '--file', help='Read input from file')
    parser_translate.add_argument('--from', dest='from_lang', help='Source language code (auto-detect if not specified)')
    parser_translate.add_argument('--to', dest='to_lang', help='Target language code')
    parser_translate.set_defaults(func=cmd_translate)

    # --- corpus command ---
    parser_corpus = subparsers.add_parser('corpus', help='Corpus management')
    corpus_subparsers = parser_corpus.add_subparsers(dest='corpus_command', help='Corpus commands')

    corpus_clean = corpus_subparsers.add_parser('clean', help='Clean corpus data')
    corpus_clean.set_defaults(func=cmd_corpus_clean)

    corpus_verify = corpus_subparsers.add_parser('verify', help='Verify corpus integrity')
    corpus_verify.set_defaults(func=cmd_corpus_verify)

    corpus_test = corpus_subparsers.add_parser('create-test', help='Create test corpus')
    corpus_test.set_defaults(func=cmd_corpus_create_test)

    # --- setup command ---
    parser_setup = subparsers.add_parser('setup', help='Setup Klareco (download models, build vocab)')
    parser_setup.add_argument('--all', action='store_true', help='Run all setup steps')
    parser_setup.add_argument('--download-models', action='store_true', help='Download required models')
    parser_setup.add_argument('--build-vocab', action='store_true', help='Build morpheme vocabulary')
    parser_setup.set_defaults(func=cmd_setup)

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
