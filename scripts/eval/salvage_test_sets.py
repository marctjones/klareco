#!/usr/bin/env python3
"""
Salvage the non-trivial pairs from the synthetic WHO test sets (#793)

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (data/indexes/duckdb_store.db), Whoosh index v2
                 (data/indexes/whoosh_v2), R16/R17 of QA_TEST_SET_QUALITY_STANDARD
DEPENDENCIES: whoosh, klareco.eval.answer_scoring (normalize),
              scripts/eval/audit_discriminability.py (gold_rank — imported, not
              reimplemented)
STAGE: Evaluation

Description:
    Triage the two synthetic WHO sets into ONE usable capability set.

    Both sets are saturated: BM25 already puts the gold passage at rank 1 for
    46% (the 50) / 58.8% (the 17) of pairs. A pair whose gold passage BM25
    already ranks first carries ZERO information about reranking — a perfect
    reranker could not move it. That is R16. This script keeps only the pairs
    inside the measurable band (BM25 gold rank 2..K) and drops the rest.

    It also backfills R17's `gold_answer_span` from `expected_answer` — and
    VERIFIES it rather than trusting it. Two tiers:
      * hard drop (`bad-span`): the span is not a substring of the source
        sentence (R9 fail), or it IS the source sentence (R17 degenerate).
        Unscoreable, so it leaves.
      * soft flag (kept, tagged `gold_answer_span_suspect`, listed in the
        report): a probable-but-unproven defect. These are NOT auto-repaired:
        guessing the right span would put a silently-wrong gold answer in the
        file, which is precisely the R17 trap this issue exists to close.

    WHY THE SPAN CHECK CANNOT USE THE SOURCE AST (a finding, not an omission)
    -------------------------------------------------------------------------
    The obvious check — read the source sentence's `ast_json` from the store and
    ask whether the span is a `propra_nomo` sitting in the `subjekto` — is
    CIRCULAR and always passes. The generator BUILT these spans by lifting
    `subjekto.kerno` out of that very AST. Asking the AST to confirm its own
    output validates nothing.

    Worse, the AST is wrong in a way that produced these pairs. With the
    proper-noun artifacts lost in the June migration (see CLAUDE.md), the parser
    tags EVERY sentence-initial capitalised word as `propra_nomo`. Verified in
    the store:

        "Nuntempe" (adverb)  -> vortspeco=propra_nomo, subjekto.kerno
        "Drame"    (adverb)  -> vortspeco=propra_nomo, subjekto.kerno
        "Britaj"   (adjective) -> vortspeco=propra_nomo, subjekto.kerno
        "Teorio"   (common noun) -> vortspeco=propra_nomo, subjekto.kerno

    The generator saw "the subject is a proper noun" and emitted a WHO question
    whose gold answer is an ADVERB. That is the root cause of the junk spans in
    these sets, and it is upstream of this script.

    So verification uses an INDEPENDENT signal: Esperanto's regular morphology
    of the surface span, which needs no lexicon and no parse.
      - `-e`            -> adverb        -> cannot answer `Kiu` (R1)
      - `-a` / `-aj`    -> adjective     -> cannot answer `Kiu` (R1)
      - `-is/-as/-os`   -> verb          -> cannot answer `Kiu` (R1)
      - lone `-o`/`-oj` -> noun-SHAPED; may be a common noun (`Teorio`,
                           `Kompanio`, `Adeptoj`) or a genuine name (`Siruŝo`).
                           Ambiguous by construction, so it is FLAGGED for a
                           human, not decided here.
    Plus a surface check for truncated multiword names in BOTH directions
    (`Maksim` → `Maksim Gorkij`; `Company` ← `London Company`) — the generator
    took the subject `kerno`, which is ONE word, so truncation is systematic.

    Over-flagging is the correct error direction: a false flag costs a reviewer
    seconds; a missed one puts a silently-wrong gold answer inside the
    measuring instrument.

    The two input sets OVERLAP — the 17 is an id-subset of the 50 — so the
    output is a UNION DEDUPLICATED BY id. Concatenating would inflate the
    headline count.

Pipeline Position:
    synthetic_who_rebuild_{50,17_cleanish}.jsonl + Whoosh BM25
      → [THIS SCRIPT] → data/test_sets/salvaged_who_nontrivial.jsonl

Usage:
    python scripts/eval/salvage_test_sets.py                 # defaults below
    python scripts/eval/salvage_test_sets.py --top-k 50 --dry-run

Inputs:
    - data/test_sets/synthetic_who_rebuild_50.jsonl (JSONL)
    - data/test_sets/synthetic_who_rebuild_17_cleanish.jsonl (JSONL)
    - data/indexes/whoosh_v2 (Whoosh BM25 index)

Outputs:
    - data/test_sets/salvaged_who_nontrivial.jsonl — survivors, each carrying
      `gold_answer_span` (R17), `bm25_gold_rank` (R16), `salvaged_from`.
    - stdout: before/after report — n in, n dropped by reason
      (rank-1 / not-found / bad-span), n out (clean vs suspect), gold-rank
      histogram, rank-1 share (must be 0%).

Quality Checks:
    - R16: rank-1 share of the OUTPUT must be 0% (asserted; non-zero → exit 1).
    - R16: every kept pair has 2 <= bm25_gold_rank <= K.
    - R17/R9: gold_answer_span is a case/diacritic-folded substring of
      source_sentence_text and strictly shorter than it.
    - Suspect-span heuristics reported, never silently accepted.

Last Updated: 2026-07-13
Author: Claude Code
Related Issues: #793 (triage), #778 (R16), #783 (R17)
See Also: docs/QA_TEST_SET_QUALITY_STANDARD.md (R7, R9, R16, R17),
          docs/QA_TEST_SET_PIPELINE.md, scripts/eval/audit_discriminability.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'eval'))

import duckdb  # noqa: E402
from whoosh.index import open_dir  # noqa: E402
from whoosh.qparser import OrGroup, QueryParser  # noqa: E402

# Reuse, do not reimplement: the R16 rank probe and its histogram/gate.
from audit_discriminability import gold_rank, r16_report  # noqa: E402
from klareco.eval.answer_scoring import normalize  # noqa: E402

DEFAULT_INPUTS = [
    'data/test_sets/synthetic_who_rebuild_50.jsonl',
    'data/test_sets/synthetic_who_rebuild_17_cleanish.jsonl',
]
DEFAULT_OUTPUT = 'data/test_sets/salvaged_who_nontrivial.jsonl'
WHOOSH_DIR = 'data/indexes/whoosh_v2'
DUCKDB_PATH = 'data/indexes/duckdb_store.db'

# R7 sets the floor at BM25 top-50; R16 sets the ceiling at "not rank 1".
# The measurable band is therefore 2..50. A pair at rank 75 is outside the
# instrument, not inside it — hence K=50, not the auditor's default of 200.
DEFAULT_TOP_K = 50

_WORD_RE = re.compile(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ'’-]+")

# Esperanto's endings are regular and lexicon-free — which is exactly why this
# project exists. They give POS without the parser's (currently broken) proper-
# noun data. A word must have >= 3 chars for the ending to be informative.
MORPH_ADVERB = re.compile(r'^.{2,}e$')
MORPH_ADJECTIVE = re.compile(r'^.{2,}(a|aj|an|ajn)$')
MORPH_VERB = re.compile(r'^.{2,}(is|as|os|us|i)$')
MORPH_NOUN = re.compile(r'^.{2,}(o|oj|on|ojn)$')
# A noun built with a person-denoting SUFFIX is a title/appositive, not a name:
# `matematik-IST-o`, `prezid-ANT-o`, `direktor-o`. `Georgo` has no such suffix,
# so this does not fire on genuine -o names.
MORPH_TITLE = re.compile(r'^.{2,}(ist|ul|ant|int|ont|estr|an)oj?n?$')
# Non-Latin script inside a span answering an Esperanto `Kiu` — usually a
# transcribed title lifted whole (a prize, a book), not the agent.
NON_LATIN = re.compile(r'[Ͱ-ϿЀ-ӿ֐-׿؀-ۿ'
                       r'一-鿿぀-ヿ]')
# R6 — corpus noise. A Wikipedia TALK-PAGE signature is not a sentence about the
# world; its "subject" is a username. e.g.
#   "ThomasPusch (diskuto) 21:45, 29 Dec. 2013 (UTC) : Mi redaktis la paĝojn…"
WIKI_TALK_NOISE = re.compile(r'\(diskuto\)|\(UTC\)|\d{2}:\d{2},\s*\d{1,2}\s')


# --- R17/R9 span verification ---------------------------------------------

def _fold(s: str) -> str:
    """Case- and diacritic-folded form, using the SAME normalizer the scorer uses."""
    return normalize(s or '')


def _words(node) -> list[dict]:
    """Flatten every `vorto` node under an AST node."""
    out: list[dict] = []
    if isinstance(node, dict):
        if node.get('tipo') == 'vorto':
            out.append(node)
        for v in node.values():
            out.extend(_words(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_words(v))
    return out


def load_source_ast(con, sid) -> dict | None:
    """Read the source sentence's AST from the store. NEVER re-parse (CLAUDE.md)."""
    if con is None or sid is None:
        return None
    try:
        row = con.execute(
            'SELECT ast_json FROM sentences WHERE sid = ?', [sid]).fetchone()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def verify_span(span: str, source: str, ast: dict | None) -> tuple[str, list[str]]:
    """Verify a candidate gold_answer_span against its source sentence + AST.

    Returns (verdict, reasons) where verdict is one of:
      'ok'      — span is well-formed
      'bad'     — HARD failure; the pair cannot be scored (drop it)
      'suspect' — probable defect; keep, but tag and report for human review
    """
    if not span or not span.strip():
        return 'bad', ['empty expected_answer / gold_answer_span']
    if not source or not source.strip():
        return 'bad', ['no source_sentence_text to verify the span against']

    f_span, f_src = _fold(span), _fold(source)

    # R9: the answer must appear verbatim in the source.
    if f_span not in f_src:
        return 'bad', [f'span not a substring of source_sentence_text (R9)']

    # R17 degenerate case: the "span" is the sentence.
    if f_span == f_src:
        return 'bad', ['span IS the whole source sentence (R17 degenerate)']

    span_toks = _WORD_RE.findall(span)
    src_toks = _WORD_RE.findall(source)
    if len(span_toks) >= 12 or (src_toks and len(span_toks) / len(src_toks) > 0.6):
        return 'bad', [f'span is sentence-length ({len(span_toks)} of '
                       f'{len(src_toks)} source tokens) — not an answer span (R17)']

    reasons: list[str] = []

    # --- surface signal: truncated multiword name, in BOTH directions --------
    # The generator lifted the parser's subject `kerno`, which is ONE word, so a
    # multiword name loses everything but one token. "Maksim" -> "Maksim Gorkij".
    m = re.search(re.escape(span) + r"\s+([A-ZĈĜĤĴŜŬ][\wĉĝĥĵŝŭ'’-]+)", source)
    if span_toks and span_toks[-1][:1].isupper() and m:
        reasons.append(f'possible truncated name (right): source continues '
                       f'{span!r} -> {span + " " + m.group(1)!r}')
    m2 = re.search(r"([A-ZĈĜĤĴŜŬ][\wĉĝĥĵŝŭ'’-]+)\s+" + re.escape(span), source)
    # Only meaningful when the preceding capitalised token is NOT sentence-initial
    # (a sentence-initial capital is just orthography, not part of a name).
    if (span_toks and span_toks[0][:1].isupper() and m2
            and not source.lstrip().startswith(m2.group(1))):
        reasons.append(f'possible truncated name (left): source reads '
                       f'{m2.group(1) + " " + span!r}')

    # --- morphological signal: an INDEPENDENT check (see module docstring on
    #     why the source AST cannot be used — the span was derived from it).
    #     Esperanto's endings are regular, so POS needs no lexicon.
    if len(span_toks) == 1:
        tok = span_toks[0]
        low = tok.lower()
        if MORPH_ADVERB.search(low):
            reasons.append(
                f'{tok!r} has the Esperanto ADVERB ending -e — an adverb cannot '
                f'answer `Kiu` (R1). Almost certainly a sentence-initial word '
                f'the parser mislabelled as a proper noun.')
        elif MORPH_ADJECTIVE.search(low):
            reasons.append(
                f'{tok!r} has an Esperanto ADJECTIVE ending (-a/-aj/-an/-ajn) — '
                f'an adjective cannot answer `Kiu` (R1); the answer may be the '
                f'noun it modifies.')
        elif MORPH_VERB.search(low):
            reasons.append(
                f'{tok!r} has an Esperanto VERB ending — not an entity (R1).')
        elif MORPH_NOUN.search(low):
            reasons.append(
                f'{tok!r} is noun-SHAPED (-o/-oj/-on/-ojn): may be a common noun '
                f'(not a rigid designator, R1) rather than a name. Ambiguous by '
                f'construction — a human must decide.')
    else:
        # Multiword span: it should START at the name. A leading noun built with
        # a person-denoting suffix is an appositive title that crept in.
        first = span_toks[0].lower()
        if MORPH_TITLE.search(first):
            reasons.append(
                f'span begins with the common-noun title {span_toks[0]!r} '
                f'(person-denoting suffix) — an answer span should start at the '
                f'name itself, not its appositive (R17).')

    # R6 — the SOURCE is corpus noise, so no span drawn from it can be gold.
    if WIKI_TALK_NOISE.search(source):
        reasons.append(
            'source sentence is Wikipedia TALK-PAGE noise (signature/timestamp) '
            '— not a statement about the world; its "subject" is a username (R6).')

    # A non-Latin-script span answering an Esperanto `Kiu` is usually a title
    # lifted whole (a prize, a work), i.e. the wrong role — not the agent.
    if NON_LATIN.search(span):
        reasons.append(
            f'span {span!r} is in a non-Latin script — typically a transcribed '
            f'TITLE (prize/work) rather than the agent who answers `Kiu`. '
            f'Verify the role.')

    # --- AST: reported for context only. It CANNOT validate the span (circular:
    #     the generator lifted the span out of this AST), but a mismatch between
    #     what the AST believes and what the morphology says is worth surfacing.
    if ast is not None and reasons:
        span_fold_toks = {_fold(t) for t in span_toks}
        for w in _words(ast):
            if _fold(w.get('plena_vorto', '')) in span_fold_toks:
                if w.get('vortspeco') == 'propra_nomo':
                    reasons.append(
                        f'(the source AST calls it `propra_nomo` — that is the '
                        f'known sentence-initial-capital mislabel, not evidence)')
                break

    return ('suspect' if reasons else 'ok'), reasons


# --- salvage ---------------------------------------------------------------

def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--inputs', nargs='+', default=DEFAULT_INPUTS)
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    ap.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                    help=f'BM25 band ceiling; R7 floor is 50 (default {DEFAULT_TOP_K})')
    ap.add_argument('--dry-run', action='store_true',
                    help='report only; do not write the output file')
    args = ap.parse_args()

    idx_dir = REPO_ROOT / WHOOSH_DIR
    if not idx_dir.exists():
        print(f'ERROR: Whoosh index not found at {idx_dir}', file=sys.stderr)
        return 1
    ix = open_dir(str(idx_dir))
    searcher = ix.searcher()
    qp = QueryParser('text', ix.schema, group=OrGroup)

    db_path = REPO_ROOT / DUCKDB_PATH
    if not db_path.exists():
        # Fail loudly. A silently-degrading dependency is a bug (CLAUDE.md):
        # without the store there is no AST, and span verification silently
        # weakens to a substring check — which is exactly what R17 forbids.
        print(f'ERROR: DuckDB store not found at {db_path}. Span verification '
              f'needs the source AST; refusing to run degraded.', file=sys.stderr)
        return 1
    con = duckdb.connect(str(db_path), read_only=True)

    # --- 1. Union the inputs, DEDUPED BY id. The 17-set is a subset of the 50;
    #        concatenating would double-count every one of its pairs.
    merged: dict[str, dict] = {}
    per_file_n: dict[str, int] = {}
    dup_ids: list[str] = []
    conflicts: list[str] = []
    for p in args.inputs:
        fp = REPO_ROOT / p if not Path(p).is_absolute() else Path(p)
        if not fp.exists():
            print(f'ERROR: input not found: {fp}', file=sys.stderr)
            return 1
        rows = load(fp)
        per_file_n[fp.name] = len(rows)
        for r in rows:
            pid = r.get('id') or f'{fp.name}:{len(merged)}'
            if pid in merged:
                dup_ids.append(pid)
                prev = merged[pid]
                if (prev.get('question') != r.get('question')
                        or prev.get('expected_answer') != r.get('expected_answer')
                        or prev.get('source_sentence_id') != r.get('source_sentence_id')):
                    conflicts.append(pid)
                merged[pid]['salvaged_from'].append(fp.name)
                continue
            r = dict(r)
            r['salvaged_from'] = [fp.name]
            merged[pid] = r

    n_in_lines = sum(per_file_n.values())
    n_in = len(merged)

    print('=' * 74)
    print('SALVAGE — synthetic WHO sets → one non-trivial capability set (#793)')
    print('=' * 74)
    print('BEFORE')
    for name, n in per_file_n.items():
        print(f'  {name:<45s} {n:>4d} pairs')
    print(f'  {"lines read (with overlap)":<45s} {n_in_lines:>4d}')
    print(f'  {"unique ids after dedup":<45s} {n_in:>4d}   '
          f'({len(set(dup_ids))} id(s) present in both files)')
    if conflicts:
        print(f'  ⚠️  {len(conflicts)} duplicated id(s) DISAGREE across files: '
              f'{conflicts[:5]}')
    print()

    # --- 2. Rank + verify every pair.
    kept: list[dict] = []
    dropped = {'rank-1': [], 'not-found': [], 'bad-span': []}
    suspects: list[tuple[str, str, str, str]] = []  # id, question, span, why
    all_ranks: list[tuple[str, int | None]] = []

    for pid, r in merged.items():
        q = r.get('question') or ''
        sid = r.get('source_sentence_id')
        src = r.get('source_sentence_text') or ''
        span = r.get('gold_answer_span') or r.get('expected_answer') or ''

        rank = gold_rank(searcher, qp, q, sid, args.top_k)
        all_ranks.append((pid, rank))

        # R17/R9 first — a pair with an unscoreable span is dead regardless of rank.
        ast = load_source_ast(con, sid)
        verdict, reasons = verify_span(span, src, ast)
        why = '; '.join(reasons)
        if ast is None:
            why = ('; '.join(reasons + ['(no AST in store for this sid — '
                                        'grammatical checks skipped)'])).lstrip('; ')
        if verdict == 'bad':
            dropped['bad-span'].append((pid, q, span, why))
            continue

        # R16 — the ceiling.
        if rank is None:
            dropped['not-found'].append(
                (pid, q, span, f'gold sid={sid} not in BM25 top-{args.top_k} (R7 fail)'))
            continue
        if rank == 1:
            dropped['rank-1'].append(
                (pid, q, span, 'BM25 already ranks the gold passage first — '
                               'reranking cannot show up here (R16)'))
            continue

        out = dict(r)
        out['gold_answer_span'] = span          # R17
        out['bm25_gold_rank'] = rank            # R16 provenance
        out['bm25_top_k'] = args.top_k
        if verdict == 'suspect':
            out['gold_answer_span_suspect'] = why
            out['review_status'] = 'needs_review'
            suspects.append((pid, q, span, why))
        kept.append(out)

    # --- 3. Report.
    print('DROPPED')
    total_dropped = sum(len(v) for v in dropped.values())
    for reason in ('rank-1', 'not-found', 'bad-span'):
        rows = dropped[reason]
        share = (len(rows) / n_in * 100) if n_in else 0.0
        print(f'  {reason:<12s} {len(rows):>4d}  ({share:5.1f}% of {n_in})')
    print(f'  {"TOTAL":<12s} {total_dropped:>4d}')
    print()

    for reason in ('bad-span', 'not-found', 'rank-1'):
        rows = dropped[reason]
        if not rows:
            continue
        print(f'  --- dropped: {reason} ---')
        for pid, q, span, why in rows:
            print(f'    {pid}  {q}')
            print(f'        span={span!r}  |  {why}')
        print()

    print('SUSPECT SPANS (kept, but tagged `gold_answer_span_suspect` + '
          '`review_status: needs_review`)')
    if suspects:
        print('  These were NOT auto-repaired — guessing the boundary would put a')
        print('  silently-wrong gold answer in the file, which is the exact R17 trap.')
        for pid, q, span, why in suspects:
            print(f'    {pid}  {q}')
            print(f'        span={span!r}  |  {why}')
    else:
        print('  none')
    print()

    n_out = len(kept)
    n_suspect = len(suspects)
    print('AFTER')
    print(f'  pairs out (total):        {n_out}')
    print(f'  ├─ clean:                 {n_out - n_suspect}')
    print(f'  └─ suspect span (review): {n_suspect}')
    print(f'  survival rate:            {(n_out / n_in * 100) if n_in else 0:.1f}%')

    # --- 4. Histograms: the inputs (union) and the output.
    print()
    print('### INPUT (deduped union) — gold-rank distribution')
    r16_report(all_ranks, args.top_k, gate=True)

    print()
    print('### OUTPUT — gold-rank distribution (R16 must be 0% rank-1)')
    out_ranks = [(r['id'], r['bm25_gold_rank']) for r in kept]
    out_stats = r16_report(out_ranks, args.top_k, gate=True)

    ok = True
    if out_stats['rank_buckets']['1'] != 0:
        print('\n❌ R16 VIOLATION: output still contains rank-1 pairs.')
        ok = False
    if any(not (2 <= r['bm25_gold_rank'] <= args.top_k) for r in kept):
        print(f'\n❌ R16 VIOLATION: a kept pair is outside the band 2..{args.top_k}.')
        ok = False
    if ok:
        print(f'\n✅ R16: rank-1 share of the salvaged set = 0.0%; every kept pair '
              f'is inside the measurable band 2..{args.top_k}.')

    # --- 5. Write.
    if args.dry_run:
        print('\n[dry-run] not writing output')
        return 0 if ok else 1

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() \
        else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.jsonl.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.rename(out_path)
    print(f'\nWROTE {n_out} pairs → {out_path}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
