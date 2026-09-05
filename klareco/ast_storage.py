"""Versioned AST storage: store tokens once, retain every structural field.

Version 2 uses a token table plus a recursively encoded structure. Only complete
token dictionaries equal to their table entry become references. Clause/phrase
nodes and token overrides remain explicit; a head ID is not a whole phrase.
"""
from __future__ import annotations

from copy import deepcopy

FORMAT_VERSION = 2
_REFERENCE = '$token'


def is_compact_ast(ast: dict) -> bool:
    return '_ast_format' in ast or any(
        key in ast for key in ('subjekto_id', 'verbo_id', 'objekto_id'))


def _token_table(tokens: list) -> dict:
    if not isinstance(tokens, list):
        raise ValueError('AST token table must be a list')
    table = {}
    for token in tokens:
        token_id = token.get('id') if isinstance(token, dict) else None
        if type(token_id) is not int or token_id <= 0 or token_id in table:
            raise ValueError(f'Invalid or duplicate AST token id: {token_id!r}')
        table[token_id] = token
    return table


def compact_ast(ast: dict) -> dict:
    """Encode a JSON-compatible expanded AST without discarding annotations."""
    if is_compact_ast(ast):
        raise ValueError('AST is already compact; expand it before encoding')
    tokens = ast['vortoj']
    table = _token_table(tokens)

    def encode(value):
        if isinstance(value, dict):
            if _REFERENCE in value:
                raise ValueError(f'Reserved AST storage key: {_REFERENCE}')
            token_id = value.get('id')
            if type(token_id) is int and token_id in table and value == table[token_id]:
                return {_REFERENCE: token_id}
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return deepcopy(value)

    structure = encode({key: value for key, value in ast.items() if key != 'vortoj'})
    return {'_ast_format': FORMAT_VERSION, 'vortoj': deepcopy(tokens),
            'structure': structure}


def expand_ast(ast: dict) -> dict:
    """Decode v2, annotate lossy legacy input, or copy an expanded AST.

    Invalid references and unknown versions raise. Legacy v1 never stored phrase
    wrappers or embedded clauses: `_legacy_compact_lossy` records that permanent
    limitation and survives any later re-encoding.
    """
    if not is_compact_ast(ast):
        return deepcopy(ast)
    version = ast.get('_ast_format', 1)
    if type(version) is not int or version not in (1, FORMAT_VERSION):
        raise ValueError(f'Unsupported AST storage version: {version!r}')
    tokens = deepcopy(ast['vortoj'])
    table = _token_table(tokens)

    def resolve(token_id):
        if type(token_id) is not int or token_id not in table:
            raise ValueError(f'Dangling or invalid AST token reference: {token_id!r}')
        return table[token_id]

    if version == FORMAT_VERSION:
        if set(ast) != {'_ast_format', 'vortoj', 'structure'}:
            raise ValueError('Unexpected AST storage envelope fields')
        def decode(value):
            if isinstance(value, dict):
                if _REFERENCE in value:
                    if len(value) != 1:
                        raise ValueError('Malformed AST token reference')
                    return resolve(value[_REFERENCE])
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return deepcopy(value)

        if (not isinstance(ast.get('structure'), dict)
                or 'vortoj' in ast['structure'] or _REFERENCE in ast['structure']):
            raise ValueError('Invalid AST structure payload')
        result = decode(ast['structure'])
        result['vortoj'] = tokens
        return result

    def frame(value):
        reference_keys = {'subjekto_id', 'verbo_id', 'objekto_id', 'aliaj_idj'}
        result = {k: deepcopy(v) for k, v in value.items()
                  if k not in reference_keys | {'propozicioj', 'vortoj', '_ast_format'}}
        for slot in ('subjekto', 'verbo', 'objekto'):
            token_id = value.get(slot + '_id')
            result[slot] = None if token_id is None else resolve(token_id)
        result['aliaj'] = [resolve(i) for i in value.get('aliaj_idj', [])]
        return result

    result = frame(ast)
    result['vortoj'] = tokens
    result['propozicioj'] = [dict(tipo='propozicio', **frame(c))
                            if 'tipo' not in c else frame(c)
                            for c in ast.get('propozicioj', [])]
    result['_legacy_compact_lossy'] = True
    return result
