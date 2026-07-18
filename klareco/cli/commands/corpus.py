"""Corpus registry management: validate / add / list."""
from __future__ import annotations

from klareco.cli._base import EXIT_OK, add_common, emit


def _mgr():
    from klareco.corpus_manager import CorpusManager
    return CorpusManager()


def cmd_validate(args) -> int:
    r = _mgr().validate_file(args.file)
    emit(args, text=(
        f"File: {args.file}\nValid: {r['valid']}\n"
        f"Total sentences: {r['total_sentences']}\n"
        f"Parseable: {r['parseable_count']} ({r['parse_rate']:.1%})" +
        ("".join(f"\n  - {e}" for e in r['errors']) if not r['valid'] else "")),
        data=r)
    return EXIT_OK


def cmd_add(args) -> int:
    _mgr().add_corpus(file_path=args.file, title=args.title,
                      corpus_type=args.type, language='eo')
    emit(args, text=f"✓ Added {args.file}", data={'added': args.file})
    return EXIT_OK


def cmd_list(args) -> int:
    corpora = _mgr().list_corpora()
    emit(args,
         text=("No corpus files registered" if not corpora else
               f"Registered ({len(corpora)}):\n" +
               "\n".join(f"  {c['title']} — {c['file_path']} [{c['type']}]"
                         for c in corpora)),
         data={'corpora': corpora})
    return EXIT_OK


def register(sub) -> None:
    c = sub.add_parser('corpus', help='Corpus registry management')
    cs = c.add_subparsers(dest='corpus_command')
    v = cs.add_parser('validate', help='Validate a corpus file')
    v.add_argument('file'); add_common(v); v.set_defaults(func=cmd_validate)
    a = cs.add_parser('add', help='Register a corpus file')
    a.add_argument('file'); a.add_argument('--title', required=True)
    a.add_argument('--type', required=True,
                   choices=['literature', 'dictionary', 'wikipedia', 'other'])
    add_common(a); a.set_defaults(func=cmd_add)
    ls = cs.add_parser('list', help='List registered corpus files')
    add_common(ls); ls.set_defaults(func=cmd_list)
