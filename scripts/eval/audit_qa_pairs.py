#!/usr/bin/env python3
"""
Per-pair quality audit for synthetic Q&A test sets.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: synthetic_*_active.jsonl / synthetic_who_trivia_v2.jsonl
                 produced by build_synthetic_qa_test_set.py and
                 build_synthetic_who_test_set.py
DEPENDENCIES: duckdb
STAGE: Evaluation

Description:
    Runs three quality checks on every Q&A pair in the supplied test sets
    and emits a per-pair verdict (PASS / FAIL) with the failing check names.

    Checks:
      1. Question coherence (shape) — starts with a valid Esperanto question
         word, ends with `?`, plausible token count, references the source
         sentence's anchor (proper noun or quoted work).
      2. Retrievability — source_sentence_id must resolve to a row in the
         current DuckDB store, and the stored text must match the JSONL
         snapshot (drift detector).
      3. Specific trivia — the question must carry a rigid designator
         («quoted work» or a capitalised proper-noun token), not a generic
         common noun. Generic ones have no unique answer and aren't trivia.

    Reports aggregate + per-set + per-failure-type breakdowns to stdout
    and writes a per-pair JSONL audit log via --output.

Pipeline Position:
    synthetic_*_active.jsonl → [THIS SCRIPT] → audit report
                            → (regenerate / filter / accept the set)

Usage:
    python scripts/eval/audit_qa_pairs.py \\
        --test-sets data/test_sets/synthetic_who_trivia_v2.jsonl \\
                    data/test_sets/synthetic_kie_active.jsonl \\
        --output data/test_sets/audit_2026-05-20.jsonl

Inputs:
    --test-sets   one or more JSONL files with {id, question, expected_answer,
                  source_sentence_id, source_sentence_text, question_type, …}
    --duckdb-path data/indexes/duckdb_store.db

Outputs:
    Per-pair JSONL audit log (one JSON per pair) with verdict + reasons.
    Aggregate summary printed to stdout.

Quality Checks:
    See the function docstrings of `check_question_shape`,
    `check_retrievable`, `check_trivia_specificity` for the exact rules.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.parser import parse


_QUESTION_WORDS = ('Kiu', 'Kion', 'Kiun', 'Kio', 'Kie', 'Kien', 'Kiam',
                   'Kial', 'Kiel', 'Kiom', 'Kies', 'Kia', 'Kiaj', 'Kiajn')
# Prepositions that can front a ki- correlative ("En kiu jaro...", "Por kio...")
_FRONTED_PP_PREPS = ('En', 'De', 'Al', 'Por', 'Per', 'Pri', 'Pro',
                     'Kun', 'Sur', 'Sub', 'Ĉe', 'Tra', 'Antaŭ',
                     'Post', 'Inter', 'Kontraŭ', 'Ekde', 'Dum')
# Match an initial uppercase letter from the Latin alphabet + Esperanto
# supersigned letters + common accented Latin (é, á, ñ, …) used in proper
# names borrowed from other languages.
_PROPER_NOUN_TOKEN_RE = re.compile(
    r'^[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ-]{2,}$'
)
_QUOTED_WORK_RE = re.compile(r'[«"„]\s*[^\s«»"][^«»"]*[»"]')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

# Esperanto common words that are sentence-initial-capitalised in some
# corpora and may be mis-tagged as propra_nomo by the parser model. If one
# of these is the "proper-noun designator" in a question, the pair is
# almost certainly a parser false-positive, not real trivia.
_COMMON_WORDS_AS_PROPER = {
    # Conjunctions / discourse
    'Kaj', 'Sed', 'Aŭ', 'Do', 'Tamen', 'Tial', 'Ke', 'Ankaŭ',
    # Function words / prepositions
    'Anstataŭ', 'Krom', 'Malgraŭ', 'Sen', 'Por', 'Antaŭ', 'Post',
    'Dum', 'Tra', 'Trans', 'Apud', 'Pri', 'Pro',
    # Adverbs sometimes flagged
    'Tiam', 'Tiun', 'Tiu', 'Ĉi', 'Jen', 'Nun', 'Hodiaŭ',
    # Common nouns frequently capitalised in headings
    'Estro', 'Estis', 'Konsilio', 'Organizaĵo',
}


def check_question_shape(question: str) -> tuple[bool, str]:
    """Question starts with an Esperanto interrogative (or a fronted PP
    + interrogative within first 3 tokens), ends with `?`, 3-20 tokens,
    no junk markers."""
    if not question:
        return False, 'empty'
    toks = question.split()
    first = toks[0] if toks else ''
    # Direct interrogative
    if first in _QUESTION_WORDS:
        ok_lead = True
    # Fronted PP: preposition followed by interrogative within first 4 tokens.
    # Case-insensitive against _QUESTION_WORDS since "En kiu jaro" has
    # lowercase "kiu" mid-sentence.
    elif first in _FRONTED_PP_PREPS and any(
        t.strip(',.;:?').capitalize() in _QUESTION_WORDS
        for t in toks[1:4]
    ):
        ok_lead = True
    else:
        ok_lead = False
    if not ok_lead:
        return False, f'bad question word: {first!r}'
    if not question.rstrip().endswith('?'):
        return False, 'no `?` terminator'
    n_tok = len(toks)
    if n_tok < 3:
        return False, f'too short ({n_tok} tokens)'
    if n_tok > 20:
        return False, f'too long ({n_tok} tokens)'
    if any(m in question for m in ('[', ']', 'REDIRECT', 'ALIDIREKTU')):
        return False, 'junk marker in question'
    return True, 'ok'


def check_answer_shape(answer: str, question_type: str) -> tuple[bool, str]:
    """Answer is non-empty and roughly the right shape for the question type."""
    if not answer:
        return False, 'empty answer'
    if len(answer) > 250:
        return False, f'answer too long ({len(answer)} chars)'
    qt = (question_type or '').upper()
    if qt == 'KIAM':
        if not _YEAR_RE.search(answer) and not re.search(r'\d', answer):
            return False, 'KIAM answer has no year/digit'
    if qt in ('WHO', 'KIU', 'KIE'):
        if not answer[:1].isupper():
            return False, f'expected proper-noun answer, got {answer!r}'
    return True, 'ok'


def check_retrievable(conn, sid, expected_text: str) -> tuple[bool, str]:
    """Source sentence ID resolves in DuckDB and text agrees with JSONL."""
    if sid is None:
        return False, 'no source_sentence_id'
    try:
        sid_i = int(sid)
    except (TypeError, ValueError):
        return False, f'sid not int: {sid!r}'
    try:
        row = conn.execute(
            'SELECT text FROM sentences WHERE sid = ?', [sid_i]
        ).fetchone()
    except Exception as e:
        return False, f'duckdb error: {e}'
    if not row:
        return False, f'sid {sid_i} not in sentences table'
    db_text = (row[0] or '').strip()
    snap = (expected_text or '').strip()
    if not db_text:
        return False, 'DB text empty'
    if db_text == snap:
        return True, 'exact match'
    # Tolerate whitespace/quote normalization drift; require first 60 chars to align.
    if db_text[:60] == snap[:60]:
        return True, 'prefix match'
    return False, 'text drift between JSONL and DB'


# Definite description with a proper-noun complement
# ("la ĉefurbo de Aŭstralio", "la valuto de Japanio")
_DEFINITE_DESC_PROPER_RE = re.compile(
    r'\bla\s+\w+(?:\s+\w+)*\s+(?:de|en|por|al|kun|el)\s+'
    r'([A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]+(?:\s+[A-ZÀ-ÞĈĜĤĴŜŬ][\w-]+)?)'
)
# Definite description with technical/scientific complement
# ("la teorion de relativeco", "la kemian simbolon de oro")
_DEFINITE_DESC_TECH_RE = re.compile(
    r'\bla\s+(?:\w+\s+)?\w{5,}\s+de\s+\w{2,}',
    re.UNICODE,
)
# Single-noun definite description with adjectival modifier
# ("la elektran ampolon", "la unuan libron")
_DEFINITE_DESC_MODIFIED_RE = re.compile(
    r'\bla\s+\w{4,}(?:an|on)\s+\w{6,}',
    re.UNICODE,
)
# Bare definite description with a content noun ≥ 7 chars
# ("la penicilinon", "la radioaktivecon", "la telefonon", "la kosmon").
# The length bound + the generic-reject list together filter out
# "la libron" / "la firmaon" / "la homon".
_DEFINITE_DESC_BARE_RE = re.compile(
    r'\bla\s+(\w{6,})\b',
    re.UNICODE,
)
_SUPERLATIVE_RE = re.compile(
    r'\bla\s+plej\s+\w+\s+(?:\w+(?:\s+(?:de|en|sur|sub)\s+\w+)?)'
)
# Generic-noun-only patterns we still want to reject:
# "Kiu fondis firmaon?", "Kiu kreis libron?"  (single bare common noun)
_GENERIC_REJECT_TARGETS = {
    'firmaon', 'firmao', 'kompanio', 'kompanion',
    'libron', 'libro', 'rakonton', 'rakonto',
    'lando', 'landon', 'urbo', 'urbon',
    'homo', 'homon', 'aferon', 'afero',
}


def check_trivia_specificity(question: str) -> tuple[bool, str]:
    """The question must carry a RIGID DESIGNATOR — one of:
      1. A quoted work «...»
      2. A proper-noun token outside a function-word denylist
      3. A definite description with a proper-noun complement
         («la ĉefurbo de Aŭstralio», «la valuto de Japanio»)
      4. A superlative definite description («la plej alta monto sur la
         Tero», «la plej granda kontinento»)
      5. A natural-class question like «la kemia simbolo de oro»

    Generic-common-noun questions («Kiu fondis firmaon?») have no unique
    answer and still fail.

    Rejects the documented parser failure class where an Esperanto common
    word (Anstataŭ, Kaj, Tamen, …) is mis-tagged as propra_nomo and ends
    up as the question's designator.
    """
    if not question:
        return False, 'empty question'

    # 1. Quoted work
    if _QUOTED_WORK_RE.search(question):
        return True, 'quoted-work designator'

    # 2. Proper-noun token outside the question word
    tokens = [t.strip('?,.;:') for t in question.split()[1:]]
    proper = [t for t in tokens if _PROPER_NOUN_TOKEN_RE.match(t)]
    real_proper = [t for t in proper if t not in _COMMON_WORDS_AS_PROPER]
    if real_proper:
        return True, f'proper-noun designator: {real_proper[0]}'

    # 3. Definite description with proper-noun complement
    m = _DEFINITE_DESC_PROPER_RE.search(question)
    if m and m.group(1).strip() not in _COMMON_WORDS_AS_PROPER:
        return True, f'definite-description (proper-noun anchor: {m.group(1).strip()!r})'

    # 4. Superlative definite description
    if _SUPERLATIVE_RE.search(question):
        return True, 'superlative definite description'

    # 5. Definite description with technical complement
    if _DEFINITE_DESC_TECH_RE.search(question):
        # Only accept if the question doesn't ALSO just have a bare
        # generic noun target.
        q_tokens = set(t.strip('?,.;:').lower() for t in question.split())
        if not q_tokens & _GENERIC_REJECT_TARGETS:
            return True, 'definite-description (technical complement)'

    # 6. Definite description with adjectival modifier
    if _DEFINITE_DESC_MODIFIED_RE.search(question):
        q_tokens = set(t.strip('?,.;:').lower() for t in question.split())
        if not q_tokens & _GENERIC_REJECT_TARGETS:
            return True, 'definite-description (modified noun)'

    # 7. Bare definite description with a long content noun
    bare = _DEFINITE_DESC_BARE_RE.findall(question)
    if bare:
        q_tokens = set(t.strip('?,.;:').lower() for t in question.split())
        if (not q_tokens & _GENERIC_REJECT_TARGETS
                and not all(b.lower() in _GENERIC_REJECT_TARGETS for b in bare)):
            return True, f'definite-description (bare long noun: {bare[0]!r})'

    # All proper-noun tokens are function-word false-positives
    if proper and not real_proper:
        return False, f'designator is an Esperanto common word: {proper[0]!r}'

    return False, 'no rigid designator (no quoted work, no proper-noun, no definite description)'


_LOCATIVE_PREPS_RE = re.compile(
    r'(?:^|[\s,;:.\(])(en|ĉe|sur|apud|sub|super|tra|trans|kun|de|al|el|por|antaŭ|post|pri|pro)\s+',
    re.IGNORECASE,
)


def check_answer_role_alignment(answer: str, source_text: str,
                                question_type: str) -> tuple[bool, str]:
    """Re-parse the source and verify the answer is in the correct semantic
    role for the question type.

    Catches the dominant remaining failure class: the parser picked a name
    that happens to be in a prepositional phrase (e.g. `En "Głos" publikigis…`
    yields `Głos` as the "agent" of `publikigis`, but it's actually the
    venue). For WHO we require the answer to NOT be immediately preceded by
    a preposition in the surface text. For KIE we require the answer TO be
    preceded by a locative preposition. KIAM answers (years) are
    surface-position-agnostic.
    """
    if not answer or not source_text:
        return True, 'skipped (empty)'
    qt = (question_type or '').upper()
    a = answer.strip()
    # Find the answer span in the source.
    pos = source_text.find(a)
    if pos < 0:
        return True, 'skipped (answer not in source)'
    # Wider window so we can skip past intervening quotes/punctuation.
    pre = source_text[max(0, pos - 40):pos]
    # Last alphabetic-only token preceding the answer, regardless of
    # intervening `"`, `«`, parentheses, commas, etc.
    m = re.search(r'([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)[^\wĉĝĥĵŝŭĈĜĤĴŜŬ-]*$', pre)
    last_word = m.group(1).lower() if m else ''
    locative = {'en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans'}
    governing_preps = {
        'en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans',
        'kun', 'de', 'al', 'el', 'por', 'antaŭ', 'post', 'pri', 'pro',
        'kontraŭ', 'inter', 'sub',
    }
    if qt in ('WHO', 'KIU'):
        # `de` is genitive/possessive in EO ("armeo de Mehmed" — Mehmed
        # is the agent, expressed via a possessive NP). Don't reject it.
        # Reject only true agent-suppressing PP markers.
        agent_blocking_preps = {'al', 'por', 'kun', 'pri', 'kontraŭ',
                                 'antaŭ', 'post', 'super', 'sub'}
        if last_word in agent_blocking_preps:
            return False, f'WHO answer is governed by preposition {last_word!r}'
        return True, f'WHO answer ok (prev word: {last_word!r})'
    if qt == 'KIE':
        if last_word in locative:
            return True, f'KIE answer governed by locative prep {last_word!r}'
        # Coordination case: "ekzilita al Elbo kaj poste Sankta Heleno" —
        # the second conjunct inherits the first's locative. Look further
        # back for a locative in the recent window.
        wider = re.findall(r'\b([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)\b',
                            source_text[max(0, pos - 80):pos])
        if any(w.lower() in locative for w in wider[-6:]):
            return True, f'KIE answer in coordinated locative NP'
        # Compound-name case: the answer is part of a proper-noun phrase
        # like "Traktato de Versajlo" where `de` is genitive-of-name.
        # Accept if the answer is capitalized and immediately preceded
        # by `de` AND the wider context starts with a capitalized head.
        if last_word == 'de':
            head_window = re.findall(r'([A-ZÀ-ÞĈĜĤĴŜŬ]\w+\s+de\s+)',
                                      source_text[max(0, pos - 30):pos + 1])
            if head_window:
                return True, 'KIE answer is part of proper-noun phrase'
        return False, f'KIE answer not governed by locative prep (got {last_word!r})'
    return True, f'no role check for {qt!r}'


def check_answer_in_source(answer: str, source_text: str,
                           question_type: str) -> tuple[bool, str]:
    """The expected answer must appear verbatim (or as a clear prefix) inside
    the source sentence. This is the strongest signal that the answer is
    actually anchored in the source — a generator that hallucinated outside
    the source would fail here.

    For KIAL/KIEL the answer is a multi-token span starting with the marker
    (`ĉar …`, `per …`, or an -e adverb) — substring-match is appropriate.
    For KIU/KIO/KIE/KIAM the answer is a name or year — same.
    """
    if not answer or not source_text:
        return False, 'empty'
    a = answer.strip()
    s = source_text.strip()
    if a in s:
        return True, 'verbatim'
    # Try a fold that strips diacritics for a fuzzy fallback on noisy text.
    import unicodedata
    def _fold(x: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', x)
            if not unicodedata.combining(c)
        ).lower()
    if _fold(a) in _fold(s):
        return True, 'diacritic-fold match'
    return False, 'answer not found in source sentence'


# ---------------------------------------------------------------------------
# R12: Trivia-caliber checks (notability, depth, non-tautology)
# ---------------------------------------------------------------------------

_NOTABLE_PEOPLE: set[str] | None = None
_FOLD_NOTABLE_PEOPLE: set[str] | None = None


def _load_notable_people(
    path: Path = Path('data/eo_wikipedia_notable_people.json'),
) -> tuple[set[str], set[str]]:
    """Cached loader for the Wikidata-notable-people list."""
    global _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE
    if _NOTABLE_PEOPLE is not None and _FOLD_NOTABLE_PEOPLE is not None:
        return _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE

    if not path.exists():
        _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE = set(), set()
        return _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE

    try:
        data = json.loads(path.read_text())
    except Exception:
        _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE = set(), set()
        return _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE

    names = data.get('names') if isinstance(data, dict) else data
    if not isinstance(names, list):
        _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE = set(), set()
        return _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE

    import unicodedata

    def _fold(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s or '')
            if not unicodedata.combining(c)
        ).lower()

    _NOTABLE_PEOPLE = set(n for n in names if isinstance(n, str))
    _FOLD_NOTABLE_PEOPLE = set(_fold(n) for n in _NOTABLE_PEOPLE)
    return _NOTABLE_PEOPLE, _FOLD_NOTABLE_PEOPLE


def _extract_anchor(question: str) -> str | None:
    """Pull the anchor span from a question.

    Priority:
      1. Quoted work «...» (when present, it's always the anchor — even
         if a proper-noun token also appears outside the quotes).
      2. First proper-noun token sequence after the question word
         (and after any fronted PP), excluding tokens that appear
         inside a quoted work span.
    """
    # 1. Quoted work has top priority.
    m = _QUOTED_WORK_RE.search(question)
    if m:
        return m.group(0)

    # 2. Walk tokens skipping the leading question word + any fronted PP.
    toks = question.rstrip('?').split()
    skip_leading = {'En', 'De', 'Al', 'Por', 'Pri', 'Pro', 'Kun', 'Sur',
                    'Sub', 'Ĉe', 'Tra', 'Antaŭ', 'Post'}
    start = 1
    while start < len(toks) and toks[start - 1] in skip_leading:
        start += 1
    span = []
    for t in toks[start:]:
        tt = t.strip(',.;:?')
        if _PROPER_NOUN_TOKEN_RE.match(tt) and tt not in _COMMON_WORDS_AS_PROPER:
            span.append(tt)
            continue
        if span:
            break
    if not span:
        return None
    return ' '.join(span)


def check_trivia_notability(question: str) -> tuple[bool, str]:
    """R12: the anchor must be Wikidata-notable.

    Operational definition for now:
      - Anchor is in `data/eo_wikipedia_notable_people.json`, OR
      - Anchor is a quoted work «...» (we accept all quoted works as
        notable by construction — they're titled cultural artifacts), OR
      - Anchor is a multi-token proper-noun span (≥ 2 tokens) — this
        is a soft signal that the entity has a full Wikipedia article
        with a multi-word title rather than a stub.

    A single-token first-name anchor with no `notable_people` hit
    fails. This catches the "obscure first-name-only person" pattern
    that previously dominated the legacy sets.
    """
    anchor = _extract_anchor(question)
    if not anchor:
        return False, 'no anchor extractable'

    if anchor.startswith(('«', '"', '„')):
        return True, f'quoted-work anchor {anchor[:30]!r}'

    if ' ' in anchor:
        return True, f'multi-token anchor {anchor!r}'

    notable, fold_notable = _load_notable_people()
    if anchor in notable:
        return True, f'in notable-people list ({anchor!r})'
    import unicodedata
    folded = ''.join(
        c for c in unicodedata.normalize('NFKD', anchor)
        if not unicodedata.combining(c)
    ).lower()
    if folded in fold_notable:
        return True, f'in notable-people list (diacritic-fold, {anchor!r})'
    return False, f'single-token anchor {anchor!r} not in notable-people list'


def check_source_depth(sentence_text: str) -> tuple[bool, str]:
    """R12: source sentence must have ≥ 8 content words (not just `est`
    + entity + entity-type). The 5-word `X estas komunumo en Y` pattern
    fails here even though it's mechanically valid."""
    if not sentence_text:
        return False, 'no source text'
    # Strip punctuation and count alphabetic tokens with > 2 chars
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", sentence_text)
    content = [t for t in toks if len(t) > 2]
    if len(content) < 8:
        return False, f'source has only {len(content)} content words (< 8)'
    return True, f'source has {len(content)} content words'


def check_non_tautological(question: str, answer: str) -> tuple[bool, str]:
    """R12: the answer must not be a substring of the question's content
    portion (excluding the question word and `estas`). Catches
    `Kio estas komunumo? → komunumo` and similar definitional
    circularity.
    """
    if not question or not answer:
        return True, 'skipped (empty)'
    # Strip question word + estas/estis + trailing ?
    stripped = re.sub(r'^[Kk]i\S*\s+(?:estas|estis|estos|est\S*)?\s*',
                      '', question.rstrip('?'))
    stripped_low = stripped.lower().strip()
    answer_low = answer.lower().strip()
    if not stripped_low or not answer_low:
        return True, 'nothing to compare'
    # If the answer text appears in the question's content portion,
    # it's tautological.
    if answer_low in stripped_low or stripped_low in answer_low:
        return False, f'answer {answer!r} appears in question content'
    return True, 'non-tautological'


def audit_pair(pair: dict, conn, trivia_caliber: bool = False) -> dict:
    question = pair.get('question') or ''
    answer = pair.get('expected_answer') or ''
    qt = pair.get('question_type') or ''
    sid = pair.get('source_sentence_id')
    sentence_text = pair.get('source_sentence_text') or ''

    checks = {
        'question_shape':    check_question_shape(question),
        'answer_shape':      check_answer_shape(answer, qt),
        'retrievable':       check_retrievable(conn, sid, sentence_text),
        'trivia_specific':   check_trivia_specificity(question),
        'answer_in_source':  check_answer_in_source(answer, sentence_text, qt),
        'role_alignment':    check_answer_role_alignment(answer, sentence_text, qt),
    }

    if trivia_caliber:
        checks['trivia_notability'] = check_trivia_notability(question)
        checks['source_depth'] = check_source_depth(sentence_text)
        checks['non_tautological'] = check_non_tautological(question, answer)

    failed = [k for k, (ok, _) in checks.items() if not ok]
    return {
        'id':                  pair.get('id'),
        'question':            question,
        'expected_answer':     answer,
        'question_type':       qt,
        'source_sentence_id':  sid,
        'source_sentence_text': sentence_text,
        'verdict':             'PASS' if not failed else 'FAIL',
        'failed_checks':       failed,
        'reasons':             {k: r for k, (_, r) in checks.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-sets', nargs='+', required=True,
                    help='One or more JSONL test-set files.')
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--output', default=None,
                    help='Per-pair JSONL audit log path.')
    ap.add_argument('--show-failures', type=int, default=10,
                    help='Print at most N failure examples per failure-type.')
    ap.add_argument('--trivia-caliber', action='store_true',
                    help='Enable R12 checks: anchor notability, source '
                         'depth, non-tautology. Recommended for capability '
                         'sets; skip for honest-ceiling (real trivia) sets '
                         'where some R12 checks are intentionally relaxed.')
    args = ap.parse_args()

    conn = duckdb.connect(args.duckdb_path, read_only=True)

    all_results: list[dict] = []
    per_set_counts: dict[str, dict[str, int]] = {}

    for ts_path in args.test_sets:
        path = Path(ts_path)
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        pairs: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        per_set_counts[path.name] = {'pass': 0, 'fail': 0}
        for p in pairs:
            r = audit_pair(p, conn, trivia_caliber=args.trivia_caliber)
            r['source_set'] = path.name
            all_results.append(r)
            per_set_counts[path.name][r['verdict'].lower()] += 1

    n = len(all_results)
    n_pass = sum(1 for r in all_results if r['verdict'] == 'PASS')
    n_fail = n - n_pass

    print(f"\nAudited {n} pairs across {len(args.test_sets)} set(s).")
    print(f"  PASS: {n_pass} ({100*n_pass/n:.1f}%)" if n else "  PASS: 0")
    print(f"  FAIL: {n_fail} ({100*n_fail/n:.1f}%)" if n else "  FAIL: 0")

    print("\nPer-set breakdown:")
    for s, c in per_set_counts.items():
        tot = c['pass'] + c['fail']
        pct = (c['pass'] / tot * 100) if tot else 0
        print(f"  {s:<48s}  {c['pass']:>3}/{tot:<3}  {pct:5.1f}% pass")

    # Failure-type counts
    fail_counts: dict[str, int] = {}
    for r in all_results:
        for chk in r['failed_checks']:
            fail_counts[chk] = fail_counts.get(chk, 0) + 1
    if fail_counts:
        print("\nFailure breakdown by check:")
        for chk, k in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {chk:<22s}  {k:>3}")

        # Examples per failure type
        per_chk_examples: dict[str, list[dict]] = {chk: [] for chk in fail_counts}
        for r in all_results:
            for chk in r['failed_checks']:
                if len(per_chk_examples[chk]) < args.show_failures:
                    per_chk_examples[chk].append(r)
        print("\nExamples:")
        for chk, examples in per_chk_examples.items():
            print(f"\n  [{chk}]")
            for r in examples:
                reason = r['reasons'].get(chk, '?')
                print(f"    - {r['id']:<22s} {r['question'][:70]}")
                print(f"      reason: {reason}")
                if chk in ('answer_shape', 'retrievable'):
                    print(f"      A: {r['expected_answer'][:80]}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nFull audit written to {out_path}")


if __name__ == '__main__':
    main()
