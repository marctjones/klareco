"""
CLI commands for corpus management.

Commands:
- add: Add a new text to the corpus
- remove: Remove a text from the corpus
- list: List all texts
- validate: Validate a text file
- stats: Show corpus statistics
- rebuild: Rebuild index from database
"""

import argparse
from pathlib import Path

from ..corpus_manager import CorpusManager


def cmd_add(args):
    """Add a text file to the corpus."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return 1

    with CorpusManager(Path(args.data_dir)) as manager:
        success, message, text_id = manager.add_text_from_file(
            file_path,
            title=args.title,
            source_type=args.type,
            auto_clean=not args.no_clean,
            auto_index=args.index
        )

        if success:
            print(f"✅ {message}")
            return 0
        else:
            print(f"❌ {message}")
            return 1


def cmd_remove(args):
    """Remove a text from the corpus."""
    with CorpusManager(Path(args.data_dir)) as manager:
        if args.id:
            text_id = args.id
        elif args.name:
            # Look up by filename
            text = manager.db.get_text_by_filename(args.name)
            if not text:
                print(f"❌ Error: Text '{args.name}' not found")
                return 1
            text_id = text['id']
        else:
            print("❌ Error: Must specify --id or --name")
            return 1

        # Confirm if not forced
        if not args.force:
            text = manager.db.get_text(text_id)
            if not text:
                print(f"❌ Error: Text ID {text_id} not found")
                return 1

            confirm = input(f"Remove '{text['title']}' (ID: {text_id})? [y/N] ")
            if confirm.lower() != 'y':
                print("Cancelled")
                return 0

        success, message = manager.remove_text(text_id)

        if success:
            print(f"✅ {message}")
            return 0
        else:
            print(f"❌ {message}")
            return 1


def cmd_list(args):
    """List all texts in corpus."""
    with CorpusManager(Path(args.data_dir)) as manager:
        texts = manager.list_texts(indexed_only=args.indexed)

        if not texts:
            print("No texts found")
            return 0

        # Format for display
        print(f"{'ID':<4} {'Title':<42} {'Type':<12} {'Idx':<4} {'Valid':<8} {'Score':<7} {'Sentences':<10}")
        print("-" * 100)

        for text in texts:
            indexed = "✅" if text['is_indexed'] else "❌"
            validation = text['validation_status']
            score = f"{text['validation_score']:.1%}" if text['validation_score'] else "N/A"
            title = text['title'][:40] if len(text['title']) <= 40 else text['title'][:37] + "..."

            print(f"{text['id']:<4} {title:<42} {text['source_type']:<12} {indexed:<4} {validation:<8} {score:<7} {text['sentence_count'] or 0:<10}")

        # Summary
        stats = manager.get_stats()
        print()
        print(f"Total: {stats['total_texts']} texts, {stats['indexed_texts']} indexed, {stats['total_sentences']} sentences")

        return 0


def cmd_validate(args):
    """Validate a text file."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return 1

    print(f"🔍 Validating: {file_path.name}")
    print()

    from ..corpus_manager import TextValidator
    validator = TextValidator()

    is_valid, score, message = validator.validate_file(file_path)

    if is_valid:
        print(f"✅ {message}")
        print(f"   Validation score: {score:.1%}")
        return 0
    else:
        print(f"❌ {message}")
        print(f"   Validation score: {score:.1%}")
        return 1


def cmd_stats(args):
    """Show corpus statistics."""
    with CorpusManager(Path(args.data_dir)) as manager:
        stats = manager.get_stats()

        print("📊 Corpus Statistics")
        print()
        print(f"  Total texts: {stats['total_texts']}")
        print(f"  Indexed texts: {stats['indexed_texts']}")
        print(f"  Total sentences: {stats['total_sentences'] or 0:,}")

        if stats['total_size']:
            size_mb = stats['total_size'] / (1024 * 1024)
            print(f"  Total size: {size_mb:.1f} MB")

        print()

        # List indexed texts
        texts = manager.list_texts(indexed_only=True)
        if texts:
            print("Indexed texts:")
            for text in texts:
                print(f"  - {text['title']} ({text['sentence_count']} sentences)")

        return 0


def cmd_rebuild(args):
    """Rebuild index from database."""
    print("🔧 Rebuilding index from database...")
    print("⚠️  This feature is not yet implemented")
    print()
    print("For now, use:")
    print("  python scripts/build_corpus_with_sources.py")
    print("  python scripts/index_corpus.py --corpus data/corpus_with_sources.jsonl ...")
    return 1


def add_corpus_parser(subparsers):
    """Add corpus management commands to CLI."""
    corpus_parser = subparsers.add_parser(
        'corpus',
        help='Manage corpus texts and indexing'
    )

    corpus_subs = corpus_parser.add_subparsers(dest='corpus_cmd', required=True)

    # Add command
    add_parser = corpus_subs.add_parser('add', help='Add a text file to corpus')
    add_parser.add_argument('file', help='Path to text file')
    add_parser.add_argument('--title', help='Title (default: filename)')
    add_parser.add_argument('--type', default='literature',
                           help='Source type (literature, wikipedia, dictionary)')
    add_parser.add_argument('--no-clean', action='store_true',
                           help='Skip automatic cleaning')
    add_parser.add_argument('--index', action='store_true',
                           help='Auto-index after adding')
    add_parser.add_argument('--data-dir', default='data',
                           help='Data directory (default: data)')
    add_parser.set_defaults(func=cmd_add)

    # Remove command
    remove_parser = corpus_subs.add_parser('remove', help='Remove a text from corpus')
    remove_parser.add_argument('--id', type=int, help='Text ID to remove')
    remove_parser.add_argument('--name', help='Text filename to remove')
    remove_parser.add_argument('--force', action='store_true',
                              help='Skip confirmation prompt')
    remove_parser.add_argument('--data-dir', default='data',
                              help='Data directory (default: data)')
    remove_parser.set_defaults(func=cmd_remove)

    # List command
    list_parser = corpus_subs.add_parser('list', help='List all texts')
    list_parser.add_argument('--indexed', action='store_true',
                            help='Show only indexed texts')
    list_parser.add_argument('--data-dir', default='data',
                            help='Data directory (default: data)')
    list_parser.set_defaults(func=cmd_list)

    # Validate command
    validate_parser = corpus_subs.add_parser('validate', help='Validate a text file')
    validate_parser.add_argument('file', help='Path to text file to validate')
    validate_parser.set_defaults(func=cmd_validate)

    # Stats command
    stats_parser = corpus_subs.add_parser('stats', help='Show corpus statistics')
    stats_parser.add_argument('--data-dir', default='data',
                             help='Data directory (default: data)')
    stats_parser.set_defaults(func=cmd_stats)

    # Rebuild command
    rebuild_parser = corpus_subs.add_parser('rebuild', help='Rebuild index from database')
    rebuild_parser.add_argument('--data-dir', default='data',
                               help='Data directory (default: data)')
    rebuild_parser.set_defaults(func=cmd_rebuild)

    return corpus_parser
