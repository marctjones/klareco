#!/usr/bin/env python3
"""
Strip obvious Esperanto-morphology pollution from the proper-noun dictionary.

VERSION: v2.1
COMPATIBLE WITH: v2.1 parser (reads klareco.parser._FUNDAMENTO_ROOTS,
                 _is_genuine_esperanto_compound, KNOWN_SUFFIXES)
DEPENDENCIES: klareco.parser
STAGE: Data / cleanup

Description:
    The proper-noun dictionary at data/proper_nouns_dynamic.json was
    extracted from corpus by tagging capitalized non-Esperanto-looking
    words. The extraction wasn't strict, so plural substantivos
    (Hejmpaĝoj, Membroŝtatoj), adjectives (Tablotenisa, Hungaraj),
    Fundamento-rooted words, and verified compounds (Hejmpaĝo,
    Sonfilmo) leaked in. These are NOT proper nouns.

    This script applies a CONSERVATIVE filter — it strips only entries
    with strong Esperanto-morphology evidence. Ambiguous single-letter-
    ending forms (Maria with -a, Goethe with -e) are KEPT, since some
    real names have the same surface shape; the parser's agreement
    validation handles those at parse time.

    Filter rules (entry is STRIPPED if any matches):

      R1  surface ends in -aj or -ajn (plural / accusative-plural
          adjective surfaces — names virtually never take these)
      R2  surface ends in -oj or -ojn (plural / accusative-plural noun
          surfaces — names virtually never take these)
      R3  base (after stripping a single Esperanto noun ending if any)
          is itself in _FUNDAMENTO_ROOTS — i.e. the entry is a
          well-known Esperanto root, not a name
      R4  base is a verified Esperanto compound via
          _is_genuine_esperanto_compound (e.g., Hejmpaĝ → hejm+paĝ)
      R5  base ends in a known Esperanto suffix AND the prefix-of-suffix
          is in DICTIONARY_ROOTS (catches Atomist, Realisto, Sciencisto
          patterns where the surface looks like a derived Esp common noun)
      R6  surface length ≤3 chars (rarely a real name; high false-positive
          rate — e.g. AB, FB, etc.)

    Entries containing foreign letters (q/w/x/y, sh/ch/th/ph) are
    PRESERVED unconditionally — strong genuine-name signal.

Pipeline Position:
    data/proper_nouns_dynamic.json
        → [THIS SCRIPT] (filter)
        → data/proper_nouns_dynamic_v2.json (cleaned)

Usage:
    # Dry-run (default, prints counts, no writes):
    python scripts/filter_proper_noun_dictionary.py

    # Apply (writes cleaned dict + audit log):
    python scripts/filter_proper_noun_dictionary.py --apply

    # Sample of stripped entries (sanity check before --apply):
    python scripts/filter_proper_noun_dictionary.py --sample 50

Inputs:
    data/proper_nouns_dynamic.json (source dictionary, ~190K entries)

Outputs:
    data/proper_nouns_dynamic_v2.json (cleaned dictionary; written only with --apply)
    logs/dict_cleanup/strip_<timestamp>.jsonl (audit log per stripped entry)
    Console summary of strip counts by rule.

Quality Checks:
    - Idempotent: running twice on the cleaned output strips 0 entries
    - Reversible: audit log enables full reconstruction
    - Foreign-letter entries always preserved (sanity check assertion)

Last Updated: 2026-05-07
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klareco.parser import (
    DICTIONARY_ROOTS,
    EXTENDED_PREFIXES,
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
    _FUNDAMENTO_ROOTS,
    _has_foreign_orthography,
)


_NOUN_ENDINGS = ('ojn', 'oj', 'on', 'o')   # longest-first
_ADJ_PLURAL    = ('ajn', 'aj')              # plural / accusative-plural adj
_PLURAL_NOUN   = ('ojn', 'oj')              # plural noun endings
_MIN_BASE_LEN_R3 = 4   # base must be ≥4 chars for Fundamento-base rule
                       # (otherwise spurious matches: Lion → 'li', Amon → 'am')
_MIN_HALF_LEN_R4 = 3   # BOTH halves of a compound must be ≥3 chars
                       # (otherwise spurious matches: sak+si, kas+ti+li,
                       # bonaer → bon+aer (3+3 OK actual real compound))
_RULES = ('R1_adj_plural', 'R2_noun_plural', 'R3_fundamento_base',
          'R4_compound_base')


def _strict_esperanto_compound(stem: str) -> bool:
    """STRICT compound check for dict-cleanup. Conservative compared to
    the parser's _is_genuine_esperanto_compound:

      - BOTH halves must be ≥_MIN_HALF_LEN_R4 chars (so we don't accept
        2+3 splits like Sak+si — Sak is in F but Saksi is really a
        single-morpheme name for Saxony).
      - Second-half must be in _FUNDAMENTO_ROOTS (no DICTIONARY_ROOTS
        which is itself polluted).
      - First-half must be in _FUNDAMENTO_ROOTS / KNOWN_PREFIXES /
        EXTENDED_PREFIXES (or Fundamento root with linking-o stripped).

    No productive paths (Path B suffix-base, Path C extended-prefix +
    DICT) — false positives here cost us real names.

    Examples:
      Hejmpaĝ   → hejm(F) + paĝ(F)        → True
      Membroŝtat→ membr(F-linkingo) + ŝtat(F) → True
      Sonfilm   → son(F) + film(NOT F)    → False
      Hobit     → hob + it (no F path)    → False
      Mordor    → mor(F) + dor(NOT F)     → False
      Saksi     → sak+si (both <3 chars)  → False (KEEP; Saxony is a name)
      Kastili   → no valid 3+3+ split     → False (KEEP; Castile is a name)
    """
    if not stem or len(stem) < (2 * _MIN_HALF_LEN_R4):
        return False
    s = stem.lower()
    n = len(s)
    for i in range(_MIN_HALF_LEN_R4, n - _MIN_HALF_LEN_R4 + 1):
        first, second = s[:i], s[i:]
        if second not in _FUNDAMENTO_ROOTS:
            continue
        if (first in _FUNDAMENTO_ROOTS
                or first in KNOWN_PREFIXES
                or first in EXTENDED_PREFIXES):
            return True
        if (first.endswith('o') and len(first) > _MIN_HALF_LEN_R4
                and first[:-1] in _FUNDAMENTO_ROOTS):
            return True
    return False


def strip_noun_ending(lower: str) -> str:
    for e in _NOUN_ENDINGS:
        if lower.endswith(e) and len(lower) > len(e) + 1:
            return lower[: -len(e)]
    return lower


def classify(surface: str) -> tuple[bool, str | None]:
    """Return (should_strip, rule_name | None)."""
    s = surface.lower()

    # Foreign-letter signal anywhere in the surface form → genuine-name
    # evidence; KEEP unconditionally. Esperanto has no q/w/x/y or
    # sh/ch/th/ph digraphs, so any of these is a strong foreign signal.
    if _has_foreign_orthography(s):
        return False, None

    # R6 (too-short) was deliberately removed: stripping all ≤3-char
    # entries killed real foreign-name fragments like San (San Diego),
    # Le (Le Monde), Jan, Don, Ott. The lower bound is now: any entry
    # ≥1 char with no other Esperanto signal is kept.

    # R1: ends in -aj/-ajn (plural / accusative-plural adjective)
    # AND the base after stripping is a recognized Esperanto root.
    # Without the base check, foreign names like Nikolaj / Hokusaj would
    # be stripped (they coincidentally end in -aj but the base isn't
    # Esperanto). With the base check we catch true adjective pollution
    # like Profundaj / Indoneziaj / Romiaj.
    for ending in _ADJ_PLURAL:
        if s.endswith(ending) and len(s) > len(ending) + 2:
            base_aj = s[: -len(ending)]
            if (base_aj in _FUNDAMENTO_ROOTS
                    or _strict_esperanto_compound(base_aj)
                    or _has_attested_esperanto_root(base_aj)):
                return True, 'R1_adj_plural'
            break  # only one match possible
    # R2: ends in -oj/-ojn (plural noun) — same base-validated rule.
    for ending in _PLURAL_NOUN:
        if s.endswith(ending) and len(s) > len(ending) + 2:
            base_oj = s[: -len(ending)]
            if (base_oj in _FUNDAMENTO_ROOTS
                    or _strict_esperanto_compound(base_oj)
                    or _has_attested_esperanto_root(base_oj)):
                return True, 'R2_noun_plural'
            break

    # Strip ONE noun ending if present, look at the base.
    base = strip_noun_ending(s)

    # R3: base is a Fundamento root (and ≥4 chars to avoid spurious
    # matches like 'li' for Lion or 'am' for Amon).
    if len(base) >= _MIN_BASE_LEN_R3 and base in _FUNDAMENTO_ROOTS:
        return True, 'R3_fundamento_base'

    # R4: base is a STRICT Esperanto compound (Fundamento+Fundamento
    # only; both halves ≥3 chars). Catches Hejmpaĝ / Membroŝtat without
    # over-firing on Saksi / Bonaer (single-morpheme foreign names).
    if _strict_esperanto_compound(base):
        return True, 'R4_compound_base'

    # No strong Esperanto-morphology signal — KEEP. May still be
    # pollution, but we can't tell from morphology alone. The parser
    # fixes already gate dict-overrides to nekonata-only, so residual
    # pollution is mostly inert at parse time.
    return False, None


def _has_attested_esperanto_root(base: str) -> bool:
    """True if `base` ends with a known Esperanto suffix and the prefix
    before that suffix is in _FUNDAMENTO_ROOTS or KNOWN_PREFIXES.
    Catches derivational adjectives like 'profund' (root), 'indonezi'
    (national-derivation suffix -ie? no), 'esperantigit' (...ig+it).
    Only Fundamento, no DICTIONARY_ROOTS, to avoid pollution loops."""
    if len(base) < 4:
        return False
    for suffix in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
        if not base.endswith(suffix):
            continue
        prefix = base[: -len(suffix)]
        if len(prefix) < 3:
            continue
        if prefix in _FUNDAMENTO_ROOTS or prefix in KNOWN_PREFIXES:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',  default='data/proper_nouns_dynamic.json')
    ap.add_argument('--output', default='data/proper_nouns_dynamic_v2.json')
    ap.add_argument('--apply',  action='store_true',
                    help='Write the filtered dict + audit log (default: dry-run)')
    ap.add_argument('--rules', default='R1,R2',
                    help='Comma-separated rule prefixes to enable (default: R1,R2 — '
                         'high-confidence only). Use "all" for R1,R2,R3,R4 (lossy — '
                         'includes Tolkien-name and Esperantized place-name false '
                         'positives).')
    ap.add_argument('--sample', type=int, default=15,
                    help='Number of stripped entries to print per rule (default 15)')
    ap.add_argument('--audit-dir', default='logs/dict_cleanup')
    args = ap.parse_args()

    if args.rules.strip().lower() == 'all':
        enabled_prefixes = {'R1', 'R2', 'R3', 'R4'}
    else:
        enabled_prefixes = {p.strip() for p in args.rules.split(',') if p.strip()}
    print(f'Enabled rule prefixes: {sorted(enabled_prefixes)}')

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f'ERROR: input dict not found at {in_path}')

    print(f'Mode: {"APPLY" if args.apply else "DRY-RUN"}')
    print(f'Reading  {in_path}')
    t0 = time.time()
    with open(in_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'  {len(d):,} entries  ({time.time() - t0:.1f}s)')

    rule_counts = Counter()
    rule_samples: dict[str, list[tuple[str, dict]]] = {r: [] for r in _RULES}
    foreign_letter_kept = 0
    kept = {}
    audit_lines: list[str] = []

    for surface, meta in d.items():
        # Quick check for foreign-letter assertion
        if _has_foreign_orthography(surface):
            foreign_letter_kept += 1
        should_strip, rule = classify(surface)
        # Filter by enabled rule prefix (R1 / R2 / R3 / R4).
        if should_strip and rule and rule.split('_', 1)[0] not in enabled_prefixes:
            should_strip = False
            rule = None
        if should_strip:
            rule_counts[rule] += 1
            if len(rule_samples[rule]) < args.sample:
                rule_samples[rule].append((surface, meta))
            audit_lines.append(json.dumps(
                {'op': 'strip', 'rule': rule, 'surface': surface, 'meta': meta},
                ensure_ascii=False))
            # Sanity: foreign-letter entries should NEVER be stripped
            if _has_foreign_orthography(surface):
                raise AssertionError(
                    f'BUG: foreign-letter entry stripped: {surface!r} via {rule}')
        else:
            kept[surface] = meta

    n_total   = len(d)
    n_strip   = sum(rule_counts.values())
    n_kept    = len(kept)

    print(f'\n=== Filter result ===')
    print(f'  Total entries:               {n_total:>8,}')
    print(f'  Stripped:                    {n_strip:>8,}  ({100 * n_strip / n_total:5.1f}%)')
    print(f'  Kept:                        {n_kept:>8,}  ({100 * n_kept / n_total:5.1f}%)')
    print(f'  Foreign-letter (preserved):  {foreign_letter_kept:>8,}')
    print(f'\n=== By rule ===')
    for rule in _RULES:
        n = rule_counts[rule]
        pct = 100 * n / n_total
        print(f'  {rule:25s} {n:>8,}  ({pct:5.2f}%)')

    print(f'\n=== Samples ===')
    for rule in _RULES:
        if not rule_samples[rule]:
            continue
        print(f'  [{rule}]')
        for surface, meta in rule_samples[rule]:
            cat  = meta.get('category', '?') if isinstance(meta, dict) else '?'
            freq = meta.get('frequency', '?') if isinstance(meta, dict) else '?'
            print(f'    {surface:28s}  cat={cat:10s}  freq={freq}')

    if not args.apply:
        print('\n[DRY-RUN] No files written. Re-run with --apply to commit.')
        return

    # Write filtered dict
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f'\nWrote {n_kept:,} kept entries → {out_path}')

    # Write audit log
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f'strip_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    with open(audit_path, 'w', encoding='utf-8') as f:
        for line in audit_lines:
            f.write(line + '\n')
    print(f'Wrote {len(audit_lines):,} audit lines → {audit_path}')


if __name__ == '__main__':
    main()
