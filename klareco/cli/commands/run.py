"""
Run the AI system — the orchestration engine.

  query    answer an Esperanto question (cited)
  explain  answer AND decode the thought at every stage (#882)
"""
from __future__ import annotations

import sys

from klareco.cli._base import EXIT_OK, add_common, emit, err, read_text_input

DEFAULT_WHOOSH = "data/indexes/whoosh_v2"


def _resolve_question(args) -> str:
    if getattr(args, 'text', None):
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Enter question in Esperanto:", file=sys.stderr)
    return input().strip()


def _build(args):
    from klareco.orchestrator import build_default_pipeline
    return build_default_pipeline(
        whoosh_index_dir=args.whoosh_dir or DEFAULT_WHOOSH,
        top_k=args.top_k,
    )


def cmd_query(args) -> int:
    try:
        pipeline = _build(args)
    except Exception as e:
        return err(f"initializing pipeline: {e}")
    result = pipeline.answer(_resolve_question(args))
    emit(args, text=(result.text or "(neniu respondo)"),
         data={'question': result.question, 'text': result.text,
               'confidence': result.confidence,
               'citations': [c.sentence_id for c in result.citations]})
    if args.verbose and not args.json:
        print(); result.print_trace()
    return EXIT_OK


def cmd_explain(args) -> int:
    from klareco.orchestrator.decoder import decode_result
    try:
        pipeline = _build(args)
    except Exception as e:
        return err(f"initializing pipeline: {e}")
    result = pipeline.answer(_resolve_question(args))
    emit(args,
         text=decode_result(result, per_stage=not args.final_only,
                            max_passages=args.max_passages),
         data={'question': result.question, 'text': result.text,
               'stages': [e.stage_name for e in result.trace]})
    return EXIT_OK


def register(sub) -> None:
    q = sub.add_parser('query', help='Answer an Esperanto question (cited)')
    q.add_argument('text', nargs='?', help='Question in Esperanto')
    q.add_argument('--whoosh-dir', help=f'Whoosh index (default: {DEFAULT_WHOOSH})')
    q.add_argument('--top-k', type=int, default=20, help='Passages to retrieve (default: 20)')
    q.add_argument('-v', '--verbose', action='store_true', help='Show the pipeline trace')
    add_common(q)
    q.set_defaults(func=cmd_query)

    e = sub.add_parser('explain',
                       help='Answer AND decode the thought at every stage (#882)')
    e.add_argument('text', nargs='?', help='Question in Esperanto')
    e.add_argument('--whoosh-dir', help=f'Whoosh index (default: {DEFAULT_WHOOSH})')
    e.add_argument('--top-k', type=int, default=20, help='Passages to retrieve (default: 20)')
    e.add_argument('--max-passages', type=int, default=5, help='Passages rendered per thought')
    e.add_argument('--final-only', action='store_true',
                   help='Only the final thought (skip per-stage evolution)')
    add_common(e)
    e.set_defaults(func=cmd_explain)
