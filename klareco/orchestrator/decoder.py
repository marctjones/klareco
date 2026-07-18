"""
Universal thought decoder (#882): render the pipeline's state — the "thought" —
as readable Esperanto at any stage boundary.

The thought is the dual-layer QueryContext (SymbolicLayer + LatentLayer).
Because Esperanto's grammar is regular and the core root vocabulary is small,
every *symbolic* enrichment can be rendered back into readable text
deterministically: the deparser for sentence ASTs, glossers for facts,
candidates, citations, and answer segments. The latent layer has, by
definition, no clean Esperanto encoding — it is described (shapes, presence),
never pretended.

Contract role (DESIGN.md → "The orchestration contract", rule 4): this module
is a TEST ORACLE, not just a debugging tool. If the decoder cannot render an
enrichment, that enrichment does not merge. Accordingly the decoder itself
NEVER raises while rendering — a value it cannot decode is rendered as a loud
`⟂ nedekodebla (...)` marker so the failure is visible instead of fatal.

Provenance: each rendered item is tagged `[regulo]` (rule) or `[modelo]`
(model). Until #883 makes attribution a required field, provenance is read
from a `deveno`/`provenance`/`fonto_tipo` key when present and defaults to
`[regulo]` — everything the current deterministic pipeline produces.

Usage:
    from klareco.orchestrator.decoder import decode_context, decode_result
    result = pipeline.answer("Kiu fondis Esperanton?")
    print(decode_result(result))            # per-stage thought evolution
    # or from the shell:  python -m klareco explain "Kiu fondis Esperanton?"
"""
from __future__ import annotations

from typing import Any, Optional

from klareco.orchestrator.context import (
    QueryContext, ContextDelta, FactFragment, ParsedPassage,
)

_RULE = '[regulo]'
_MODEL = '[modelo]'
_SEP = '─' * 62
_HEAVY = '═' * 62


# ---------------------------------------------------------------------------
# Small, never-raising helpers
# ---------------------------------------------------------------------------

def _undecodable(what: str, exc: Exception) -> str:
    """A value the decoder could not render — loud, visible, non-fatal."""
    return f'⟂ nedekodebla {what} ({type(exc).__name__}: {exc})'


def _provenance(obj: Any) -> str:
    """Read provenance off an object/dict; default [regulo] (TODO #883)."""
    for key in ('deveno', 'provenance', 'fonto_tipo'):
        val = None
        if isinstance(obj, dict):
            val = obj.get(key)
        else:
            val = getattr(obj, key, None)
        if val:
            return _MODEL if str(val).lower() in ('modelo', 'model',
                                                  'learned') else _RULE
    return _RULE


def _safe_deparse(ast: Optional[dict]) -> str:
    """Deparse an AST to Esperanto text; failures are visible, not fatal."""
    if ast is None:
        return '—'
    try:
        from klareco.deparser import deparse
        text = deparse(ast)
        return text if text.strip() else '⟂ nedekodebla (malplena deparse)'
    except Exception as exc:                       # noqa: BLE001 — see module doc
        return _undecodable('AST', exc)


def _clip(text: str, limit: int = 100) -> str:
    text = (text or '').replace('\n', ' ')
    return text if len(text) <= limit else text[:limit - 1] + '…'


# ---------------------------------------------------------------------------
# Leaf renderers — one per SymbolicLayer item type (decodability registry)
# ---------------------------------------------------------------------------

def _fmt_passage(p: ParsedPassage, rank: int) -> str:
    try:
        text = p.text or _safe_deparse(p.ast)
        return (f'  {rank:>2}. [sid {p.sentence_id}] '
                f'{p.score:.3f}  “{_clip(text, 90)}”')
    except Exception as exc:                       # noqa: BLE001
        return '  ' + _undecodable('trafo', exc)


def _fmt_fact(f: FactFragment) -> str:
    try:
        args = ', '.join(f'{role}={val}' for role, val in (f.arguments or ()))
        src = f', el sid {f.source_passage_id}' if f.source_passage_id else ''
        return (f'  • {f.entity} —{f.relation}→ {args or "?"}  '
                f'(konf {f.confidence:.2f}{src}) {_provenance(f)}')
    except Exception as exc:                       # noqa: BLE001
        return '  ' + _undecodable('fakto', exc)


def _fmt_segment(seg) -> str:
    try:
        cites = ''.join(f'[{c}]' for c in (seg.citation_ids or ()))
        return f'  ▸ {_clip(seg.text, 110)} {cites}'
    except Exception as exc:                       # noqa: BLE001
        return '  ' + _undecodable('segmento', exc)


def _fmt_citation(c) -> str:
    try:
        return (f'  [{c.id}] sid {c.sentence_id} ({c.doc_title or c.doc_source}) '
                f'“{_clip(c.snippet, 80)}”')
    except Exception as exc:                       # noqa: BLE001
        return '  ' + _undecodable('citaĵo', exc)


def _fmt_latent_value(name: str, val: Any) -> str:
    if val is None:
        return f'  {name}: —'
    shape = getattr(val, 'shape', None)
    if shape is not None:
        return f'  {name}: denso {tuple(shape)} {_MODEL}'
    if isinstance(val, tuple):
        return f'  {name}: {len(val)} ero(j) {_MODEL}'
    return f'  {name}: {type(val).__name__} {_MODEL}'


def _fmt_flags(flags) -> list[str]:
    """Render pipeline flags; recovered stage failures are ⚠-prominent."""
    lines: list[str] = []
    for key in sorted(flags):
        if key.startswith('stage_failed:'):
            lines.append(f'  ⚠ FALO {key.split(":", 1)[1]}: {flags[key]}')
        else:
            lines.append(f'  {key} = {flags[key]}')
    return lines


# ---------------------------------------------------------------------------
# The full thought
# ---------------------------------------------------------------------------

def decode_context(ctx: QueryContext, *, max_passages: int = 5,
                   max_facts: int = 12) -> str:
    """Render one QueryContext — the complete thought — as readable text."""
    sym, lat = ctx.symbolic, ctx.latent
    out: list[str] = []
    out.append(_HEAVY)
    out.append(f'PENSO — “{ctx.question}”   (konfidenco {ctx.confidence:.2f})')
    out.append(_HEAVY)

    out.append(f'demando {_provenance(sym.question_ast)}  '
               f'tipo={sym.question_type}')
    out.append(f'  {_safe_deparse(sym.question_ast)}')

    n = len(sym.passage_asts)
    out.append(_SEP)
    out.append(f'trafoj ({n}{f"; montrataj {max_passages}" if n > max_passages else ""})')
    for i, p in enumerate(sym.passage_asts[:max_passages], 1):
        out.append(_fmt_passage(p, i))

    out.append(_SEP)
    out.append(f'faktoj ({len(sym.fact_fragments)})')
    for f in sym.fact_fragments[:max_facts]:
        out.append(_fmt_fact(f))

    if sym.answer_segments:
        out.append(_SEP)
        out.append(f'respondaj segmentoj ({len(sym.answer_segments)})')
        for seg in sym.answer_segments:
            out.append(_fmt_segment(seg))

    if sym.citations:
        out.append(_SEP)
        out.append(f'citaĵoj ({len(sym.citations)})')
        for c in sym.citations:
            out.append(_fmt_citation(c))

    out.append(_SEP)
    out.append(f'fina teksto: {sym.final_text or "—"}')

    out.append(_SEP)
    out.append('latenta tavolo (sen AST-kodigo — nur priskribo)')
    out.append(_fmt_latent_value('question_embedding', lat.question_embedding))
    out.append(_fmt_latent_value('passage_embeddings', lat.passage_embeddings or None))
    out.append(_fmt_latent_value('relevance_matrix', lat.relevance_matrix))

    if ctx.flags:
        out.append(_SEP)
        out.append('flagoj')
        out.extend(_fmt_flags(ctx.flags))
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Per-stage evolution (from the orchestrator trace)
# ---------------------------------------------------------------------------

def decode_delta(entry) -> str:
    """Render one StageTrace entry: what this stage did to the thought."""
    if entry.skipped:
        return f'[{entry.stage_name}] (preterlasita)'

    delta: ContextDelta = entry.delta
    m = entry.metrics
    head = f'[{entry.stage_name}]'
    if m is not None:
        head += (f' {m.timing_ms:.1f}ms  '
                 f'konf {m.confidence_before:.2f}→{m.confidence_after:.2f}')

    lines = [head]
    for key, val in (delta.symbolic or {}).items():
        if isinstance(val, tuple):
            lines.append(f'  + {key}: {len(val)} ero(j)')
            for i, item in enumerate(val[:2], 1):
                if isinstance(item, ParsedPassage):
                    lines.append(_fmt_passage(item, i))
                elif isinstance(item, FactFragment):
                    lines.append(_fmt_fact(item))
        elif isinstance(val, dict):       # an AST
            lines.append(f'  + {key}: {_clip(_safe_deparse(val), 90)}')
        else:
            lines.append(f'  + {key} = {_clip(str(val), 90)!r}')
    for key in (delta.latent or {}):
        lines.append(f'  + latenta.{key} {_MODEL}')
    for flag_line in _fmt_flags(delta.flags or {}):
        lines.append(flag_line)
    if len(lines) == 1:
        lines.append('  (nenio ŝanĝita)')
    return '\n'.join(lines)


def decode_trace(trace: list) -> str:
    """Render the whole per-stage evolution of the thought."""
    return '\n'.join(decode_delta(entry) for entry in trace)


def decode_result(result, *, per_stage: bool = True,
                  max_passages: int = 5) -> str:
    """Render an AnswerResult: per-stage evolution + the final thought."""
    parts: list[str] = []
    if per_stage and result.trace:
        parts.append('EVOLUO DE LA PENSO (po-etapa)')
        parts.append(decode_trace(result.trace))
        parts.append('')
        final_ctx = result.trace[-1].ctx_after
        parts.append(decode_context(final_ctx, max_passages=max_passages))
    else:
        parts.append(f'demando: {result.question}')
        parts.append(f'respondo: {result.text or "—"}')
    return '\n'.join(parts)


def explain(question: str, pipeline) -> str:
    """Answer a question and return the fully decoded thought evolution."""
    return decode_result(pipeline.answer(question))
