"""
Language primitives (no retrieval):

  parse      Esperanto text -> role-annotated AST
  translate  deterministic translation to/from Esperanto
"""
from __future__ import annotations

import json as _json

from klareco.cli._base import (
    EXIT_OK, add_common, emit, err, read_text_input,
)


def cmd_parse(args) -> int:
    from klareco.parser import parse
    try:
        ast = parse(read_text_input(args))
    except Exception as e:
        return err(str(e))
    if getattr(args, 'json', False) or args.format == 'json':
        print(_json.dumps(ast, indent=2, ensure_ascii=False))
    else:
        emit(args, text=(
            f"Sentence type: {ast.get('tipo', 'unknown')}\n"
            f"Subject: {ast.get('subjekto')}\n"
            f"Verb: {ast.get('verbo')}\n"
            f"Object: {ast.get('objekto')}"))
    return EXIT_OK


def cmd_translate(args) -> int:
    from klareco.translator import TranslationService
    from klareco.lang_id import identify_language
    text = read_text_input(args)
    if args.from_lang and args.to_lang:
        src, tgt = args.from_lang, args.to_lang
    else:
        detected = identify_language(text)
        src = detected
        tgt = (args.to_lang or 'en') if detected == 'eo' else 'eo'
    result = TranslationService().translate(text, src, tgt)
    emit(args, text=result, data={'source': src, 'target': tgt, 'text': result})
    return EXIT_OK


def register(sub) -> None:
    p = sub.add_parser('parse', help='Parse Esperanto text into an AST')
    p.add_argument('text', nargs='?', help='Esperanto text to parse')
    p.add_argument('-f', '--file', help='Read input from a file')
    p.add_argument('--format', choices=['text', 'json'], default='text')
    add_common(p)
    p.set_defaults(func=cmd_parse)

    t = sub.add_parser('translate', help='Translate text to/from Esperanto')
    t.add_argument('text', nargs='?', help='Text to translate')
    t.add_argument('-f', '--file', help='Read input from a file')
    t.add_argument('--from', dest='from_lang', help='Source language (auto if omitted)')
    t.add_argument('--to', dest='to_lang', help='Target language')
    add_common(t)
    t.set_defaults(func=cmd_translate)
