"""
AST-aware reranker: per-question-type structured scoring over shredded
AST columns from DuckDB.

VERSION: v2.2
COMPATIBLE WITH: post-rebuild DuckDB v2.2 store (sentences with shredded
                 AST columns: subj_*, verb_*, obj_*, aliaj_json)
DEPENDENCIES: duckdb, klareco.parser (for question parsing in callers)
STAGE: Reranking (operates on a BM25 candidate pool)

Description:
    Implements the scoring framework described in #741. For each
    candidate sentence the reranker computes:

      score = BOOST * (Σ_k weight_k(qtype) * soft_component_k)

    subject to hard filters (binary) that zero out the score when the
    candidate is structurally incompatible with the question.

    Stage 1 (this file): use only data that already exists in the
    `sentences` table's shredded columns. No new indexes built — the
    `aliaj_*` flags for KIE/KIAM are derived on-the-fly by parsing
    aliaj_json, which is acceptable for a 100-candidate-pool bench
    but will be materialized into boolean columns in Stage 2.

Hard filters:
    - For KIU: candidate.subj_propranoma_kat IN {persono, ...person-like}
    - For KIE: candidate.aliaj_json has a loko-typed item
    - For KIAM: candidate.aliaj_json has a jaro/dato-typed item
    - Negation match between question.verb.negita and candidate.verb_negated

Soft components (each in [0, 1]):
    - anchor_verb_match: 1.0 exact radiko, 0.6 same verb_klaso, 0.0 else
    - anchor_object_match: 1.0 exact obj_radiko match, 0.3 root anywhere
    - bm25_normalized: sigmoid of incoming BM25 score
    - tense_compat: 1.0 same tempo, 0.5 different but not contradictory
    - answer_slot_type_score: 1.0 strong type match, 0.5 weak (pronoun)

Boosts:
    - x2.0 if candidate sid hits a pattern_kb table relevant to the qtype
    - x1.5 for definitional 'X estas Y' shape on KIO_DEF questions

Last Updated: 2026-05-22
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Question type detection
# ---------------------------------------------------------------------------

# Esperanto correlatives that introduce questions, with the answer-slot
# semantics they imply.
_KI_WORD_TO_QTYPE = {
    'kiu':  'KIU',       # who (subject) — but can also be "which" (adjective)
    'kiun': 'KIU_OBJ',   # whom (accusative) — answer is object
    'kio':  'KIO',       # what (subject or definitional)
    'kion': 'KIO_OBJ',   # what (accusative) — answer is object
    'kie':  'KIE',       # where — answer is location modifier
    'kien': 'KIE',       # whereto — location with motion
    'kiam': 'KIAM',      # when — answer is time modifier
    'kial': 'KIAL',      # why — answer is causal sub-clause
    'kiel': 'KIEL',      # how — manner adverbial
    'kiom': 'KIOM',      # how much/many — numeric quantifier
    'kies': 'KIES',      # whose — possessive
    'kia':  'KIA',       # what kind — adjective on noun
    'kiaj': 'KIA',
    'kian': 'KIA',
}

# Person-like vortspecoj that can satisfy a KIU answer slot.
_PERSON_KATEGORIOJ = {'persono', 'personaj_pronomoj'}
# Place-like
_LOKO_KATEGORIOJ = {'loko', 'lando', 'urbo', 'regiono'}


def detect_question_type(question_ast: dict) -> str:
    """Return one of KIU, KIU_OBJ, KIO, KIO_OBJ, KIO_DEF, KIE, KIAM,
    KIAL, KIEL, KIOM, KIES, KIA, or UNKNOWN.

    Walks the question AST for the first ki-correlative. Special-cases
    'Kio estas X?' → KIO_DEF (definitional, asks for the description of X).
    """
    if not isinstance(question_ast, dict):
        return 'UNKNOWN'

    # Find the ki-word and its role
    role_with_ki = _find_ki_in_ast(question_ast)
    if not role_with_ki:
        return 'UNKNOWN'
    role, ki_word = role_with_ki
    base_qtype = _KI_WORD_TO_QTYPE.get(ki_word.lower(), 'UNKNOWN')

    # Definitional override: "Kio estas X?" / "Kio estis X?"
    if base_qtype in ('KIO', 'KIU') and role == 'subjekto':
        verb = question_ast.get('verbo') or {}
        verb_radiko = (verb.get('radiko') or '').lower()
        if verb_radiko == 'est':
            return 'KIO_DEF' if base_qtype == 'KIO' else 'KIU_DEF'

    # KIU with subject role — standard "who did X" question
    return base_qtype


def _find_ki_in_ast(ast: dict) -> Optional[tuple[str, str]]:
    """Return (role_name, ki_word) for the first ki-correlative in the AST.

    Roles searched (in order): subjekto, objekto, aliaj.
    """
    for role in ('subjekto', 'objekto'):
        n = ast.get(role)
        if isinstance(n, dict):
            kerno = n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
            if isinstance(kerno, dict):
                radiko = (kerno.get('radiko') or '').lower()
                plena = (kerno.get('plena_vorto') or '').lower()
                if radiko == 'ki' or plena in _KI_WORD_TO_QTYPE:
                    return (role, plena or radiko)
    for item in ast.get('aliaj') or []:
        if isinstance(item, dict):
            kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
            if isinstance(kerno, dict):
                radiko = (kerno.get('radiko') or '').lower()
                plena = (kerno.get('plena_vorto') or '').lower()
                if radiko == 'ki' or plena in _KI_WORD_TO_QTYPE:
                    return ('aliaj', plena or radiko)
    return None


# ---------------------------------------------------------------------------
# aliaj_json walking
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2}|2100)\b')

# Esperanto prepositions that introduce a place. Empirically: `en` carries
# the bulk of place mentions in the corpus; the other directional prepositions
# (al/el/ĝis/ekde) plus locative prepositions (ĉe/sur/sub/tra/super/apud)
# round out the set. Time-only prepositions (dum, antaŭ-ol) are excluded.
_PLACE_PREPOSITIONS = {
    'en', 'ĉe', 'al', 'el', 'ekde', 'ĝis', 'tra', 'super', 'sur', 'sub',
    'apud', 'trans', 'kontraŭ', 'antaŭ', 'malantaŭ', 'inter',
}


def aliaj_has_loko(aliaj_json: Optional[str]) -> bool:
    """True iff the candidate's aliaj contain a place-like modifier.

    Detection (in priority order):

    1. **Explicit type tag**: kerno.propranoma_kat or kerno.enteca_tipo
       in {loko, lando, urbo, regiono}. Reliable but sparse (these are
       set by tier-2 enrichment, which hasn't been run for most rows).

    2. **Place preposition + propra_nomo**: a flat-list aliaj like
       `[{en}, {Bjalistok}]` where the first item is a place-prep and
       the next is a propra_nomo. Propra_nomos that match a year regex
       (e.g. `1887` showing up as a "propra_nomo") are rejected.

    3. **`en` + substantivo**: weakest signal but still place-y in
       practice (`en urbo`, `en lando`). Other place-prepositions are
       too ambiguous with a common noun (e.g. `al homo` — to a person).
    """
    if not aliaj_json:
        return False
    try:
        aliaj = json.loads(aliaj_json) if isinstance(aliaj_json, str) else aliaj_json
    except Exception:
        return False
    if not isinstance(aliaj, list):
        return False
    for i, item in enumerate(aliaj):
        if not isinstance(item, dict):
            continue
        kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
        if not isinstance(kerno, dict):
            continue

        # 1. Explicit type tag (rare but high-precision)
        kat = (kerno.get('propranoma_kat') or '').lower()
        if kat in _LOKO_KATEGORIOJ:
            return True
        type_id = (kerno.get('enteca_tipo') or '').lower()
        if type_id in _LOKO_KATEGORIOJ:
            return True

        # 2. Place preposition followed by propra_nomo
        radiko = (kerno.get('radiko') or '').lower()
        vortspeco = (kerno.get('vortspeco') or '').lower()
        if vortspeco == 'prepozicio' and radiko in _PLACE_PREPOSITIONS:
            # Scan up to 2 positions ahead for a place-y object
            for j in range(i + 1, min(i + 3, len(aliaj))):
                nxt = aliaj[j]
                if not isinstance(nxt, dict):
                    continue
                nxt_k = nxt.get('kerno') if nxt.get('tipo') == 'vortgrupo' else nxt
                if not isinstance(nxt_k, dict):
                    continue
                nxt_vs = (nxt_k.get('vortspeco') or '').lower()
                nxt_pv = str(nxt_k.get('plena_vorto') or '')
                if nxt_vs == 'propra_nomo' and not _YEAR_RE.match(nxt_pv):
                    return True
                # 3. `en` + substantivo fallback
                if nxt_vs == 'substantivo' and radiko == 'en':
                    return True
    return False


def aliaj_has_jaro(aliaj_json: Optional[str]) -> bool:
    """True iff the candidate has a 4-digit year or year-like modifier."""
    if not aliaj_json:
        return False
    # Cheap: text-level year regex on the serialized aliaj_json
    if _YEAR_RE.search(aliaj_json):
        return True
    try:
        aliaj = json.loads(aliaj_json) if isinstance(aliaj_json, str) else aliaj_json
    except Exception:
        return False
    if not isinstance(aliaj, list):
        return False
    for item in aliaj:
        if not isinstance(item, dict):
            continue
        kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
        if not isinstance(kerno, dict):
            continue
        kat = (kerno.get('propranoma_kat') or '').lower()
        if kat in ('jaro', 'tempo', 'dato'):
            return True
    return False


def aliaj_has_numeral(aliaj_json: Optional[str]) -> bool:
    """True iff the candidate has a numeral modifier (for KIOM)."""
    if not aliaj_json:
        return False
    try:
        aliaj = json.loads(aliaj_json) if isinstance(aliaj_json, str) else aliaj_json
    except Exception:
        return False
    if not isinstance(aliaj, list):
        return False
    for item in aliaj:
        if not isinstance(item, dict):
            continue
        kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
        if not isinstance(kerno, dict):
            continue
        if (kerno.get('vortspeco') or '').lower() == 'numeralo':
            return True
        if re.search(r'\d', str(kerno.get('plena_vorto') or '')):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-question-type configuration
# ---------------------------------------------------------------------------

@dataclass
class QTypeConfig:
    """Configuration for scoring a question type."""
    answer_slot:        str               # 'subjekto', 'objekto', 'aliaj_loko', etc.
    anchor_roles:       tuple[str, ...]   # which roles to match between question and candidate
    expected_kats:      tuple[str, ...]   # expected propranoma_kat at answer slot
    has_aliaj_check:    Optional[str]     # 'loko' / 'jaro' / 'numeral' / None
    weights:            dict[str, float]  # tuned per question type
    require_answer_slot: bool             # whether missing answer slot is a hard filter


_DEFAULT_WEIGHTS = {
    'anchor_verb':   0.25,
    'anchor_object': 0.20,
    'phrase_adj':    0.20,
    'bm25':          0.20,
    'tense':         0.05,
    'answer_type':   0.10,
}


_QTYPE_CONFIG: dict[str, QTypeConfig] = {
    'KIU': QTypeConfig(
        answer_slot='subjekto',
        anchor_roles=('verbo', 'objekto'),
        expected_kats=('persono', 'personaj_pronomoj'),
        has_aliaj_check=None,
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.30, 'anchor_object': 0.25,
                 'answer_type': 0.15},
        require_answer_slot=True,
    ),
    'KIU_OBJ': QTypeConfig(
        answer_slot='objekto',
        anchor_roles=('subjekto', 'verbo'),
        expected_kats=('persono', 'personaj_pronomoj'),
        has_aliaj_check=None,
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.30,
                 'answer_type': 0.15},
        require_answer_slot=True,
    ),
    'KIO': QTypeConfig(
        answer_slot='objekto',
        anchor_roles=('subjekto', 'verbo'),
        expected_kats=(),
        has_aliaj_check=None,
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.30, 'phrase_adj': 0.25},
        require_answer_slot=False,
    ),
    'KIO_OBJ': QTypeConfig(
        answer_slot='objekto',
        anchor_roles=('subjekto', 'verbo'),
        expected_kats=(),
        has_aliaj_check=None,
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.30, 'phrase_adj': 0.25},
        require_answer_slot=True,
    ),
    'KIO_DEF': QTypeConfig(
        # "Kio estas X?" — definitional, answer is the description in
        # the rest of the sentence
        answer_slot='objekto',  # 'estas X' Y → Y is in the post-verb material
        anchor_roles=('subjekto',),  # the X being defined
        expected_kats=(),
        has_aliaj_check=None,
        weights={**_DEFAULT_WEIGHTS,
                 'phrase_adj': 0.40, 'bm25': 0.30,
                 'anchor_verb': 0.0, 'anchor_object': 0.10,
                 'tense': 0.10, 'answer_type': 0.10},
        require_answer_slot=False,
    ),
    'KIE': QTypeConfig(
        answer_slot='aliaj_loko',
        anchor_roles=('subjekto', 'verbo'),
        expected_kats=(),
        has_aliaj_check='loko',
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.25, 'phrase_adj': 0.25, 'bm25': 0.20,
                 'answer_type': 0.15},
        require_answer_slot=True,
    ),
    'KIAM': QTypeConfig(
        answer_slot='aliaj_jaro',
        anchor_roles=('subjekto', 'verbo'),
        expected_kats=(),
        has_aliaj_check='jaro',
        weights={**_DEFAULT_WEIGHTS,
                 'anchor_verb': 0.25, 'phrase_adj': 0.25, 'bm25': 0.25,
                 'answer_type': 0.15},
        require_answer_slot=True,
    ),
    'KIOM': QTypeConfig(
        answer_slot='aliaj_numeral',
        anchor_roles=('subjekto', 'verbo', 'objekto'),
        expected_kats=(),
        has_aliaj_check='numeral',
        weights={**_DEFAULT_WEIGHTS,
                 'phrase_adj': 0.25},
        require_answer_slot=True,
    ),
    'KIES': QTypeConfig(
        answer_slot='aliaj_de_persono',
        anchor_roles=('objekto',),
        expected_kats=('persono',),
        has_aliaj_check=None,
        weights=_DEFAULT_WEIGHTS,
        require_answer_slot=False,
    ),
    'KIA': QTypeConfig(
        answer_slot='aliaj_adjective',
        anchor_roles=('subjekto',),
        expected_kats=(),
        has_aliaj_check=None,
        weights=_DEFAULT_WEIGHTS,
        require_answer_slot=False,
    ),
    'KIAL': QTypeConfig(
        answer_slot='aliaj_causal',
        anchor_roles=('subjekto', 'verbo', 'objekto'),
        expected_kats=(),
        has_aliaj_check=None,
        weights=_DEFAULT_WEIGHTS,
        require_answer_slot=False,
    ),
    'KIEL': QTypeConfig(
        answer_slot='aliaj_manner',
        anchor_roles=('subjekto', 'verbo', 'objekto'),
        expected_kats=(),
        has_aliaj_check=None,
        weights=_DEFAULT_WEIGHTS,
        require_answer_slot=False,
    ),
    'UNKNOWN': QTypeConfig(
        answer_slot='',
        anchor_roles=(),
        expected_kats=(),
        has_aliaj_check=None,
        weights={'bm25': 1.0},
        require_answer_slot=False,
    ),
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _q_radiko(question_ast: dict, role: str) -> Optional[str]:
    n = question_ast.get(role) if role in ('subjekto', 'objekto', 'verbo') else None
    if not isinstance(n, dict):
        return None
    if role == 'verbo':
        return n.get('radiko')
    kerno = n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
    if isinstance(kerno, dict):
        return kerno.get('radiko')
    return None


def _verb_klaso_lookup(conn, radiko: str) -> Optional[str]:
    """Return the VerbaKlaso id for a verb radiko, or None."""
    if not radiko:
        return None
    try:
        row = conn.execute(
            "SELECT class_id FROM ontology_edges "
            "WHERE rel = 'APARTENAS_AL_VERBA_KLASO' AND radiko = ?",
            [radiko]
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    """Normalize BM25 score into [0,1]. Whoosh scores typically 0..20."""
    if x is None:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(x - 5.0) / 3.0))


# ---------------------------------------------------------------------------
# The reranker
# ---------------------------------------------------------------------------

class ASTAwareScorer:
    """Pure scoring function — separable from the bench harness so unit
    tests can construct fake candidates and validate the layered score.

    Construct once per process; pass to the harness wrapper below."""

    def __init__(self, conn):
        self.conn = conn
        # Cache the question's verb_klaso once per call to .score_batch
        self._cached_q_klaso: Optional[str] = None

    # --- public API ---

    def score_batch(self, question_ast: dict, candidates: list,
                    bm25_scores: dict[int, float]) -> list[tuple[float, dict]]:
        """Score every candidate. Returns [(score, candidate_row)] sorted
        by score descending.

        `candidates` is a list of dicts with keys: sid, text, ast,
        and the shredded columns (subj_*, verb_*, obj_*, aliaj_json).
        `bm25_scores` maps sid → original BM25 score.

        Filter policy (Stage 1.5):
        - Negation mismatch is the ONLY hard filter (zero score).
        - All other structural filters are SOFT penalties that
          distinguish 'known mismatch' (heavier penalty) from
          'unknown — column is NULL' (lighter penalty).
        - Soft penalties multiply the base score, so a candidate
          with strong lexical match can still survive a single
          structural miss but a candidate weak on every front gets
          demoted into the long tail.
        """
        qtype = detect_question_type(question_ast)
        cfg = _QTYPE_CONFIG.get(qtype, _QTYPE_CONFIG['UNKNOWN'])

        q_verb_radiko = _q_radiko(question_ast, 'verbo')
        q_obj_radiko = _q_radiko(question_ast, 'objekto')
        q_subj_radiko = _q_radiko(question_ast, 'subjekto')
        q_verb_negita = bool((question_ast.get('verbo') or {}).get('negita'))
        q_verb_tempo = (question_ast.get('verbo') or {}).get('tempo')
        self._cached_q_klaso = _verb_klaso_lookup(self.conn, q_verb_radiko)

        # Anchor phrase: the surface text of question minus the ki-correlative
        anchor_phrase = self._build_anchor_phrase(question_ast)

        scored: list[tuple[float, dict]] = []
        for c in candidates:
            sid = int(c['sid'])
            bm25 = bm25_scores.get(sid, 0.0)

            # ---- Hard filter: negation mismatch (genuinely fatal) ----
            if c.get('verb_negated') is not None and \
               bool(c.get('verb_negated')) != q_verb_negita:
                scored.append((0.0, c))
                continue

            # ---- Soft components ----
            comp = {
                'anchor_verb':   self._anchor_verb_match(q_verb_radiko, c),
                'anchor_object': self._anchor_object_match(q_obj_radiko,
                                                           q_subj_radiko, c),
                'phrase_adj':    self._phrase_adj(anchor_phrase, c),
                'bm25':          _sigmoid(bm25),
                'tense':         self._tense_compat(q_verb_tempo, c),
                'answer_type':   self._answer_type_score(c, cfg),
            }
            base = sum(cfg.weights.get(k, 0.0) * comp[k] for k in comp)

            # ---- Structural filters (FINAL POLICY after Stage 1.5/1.6) ----
            # Hard-filter on EITHER 'missing' or 'unknown' for slot/aliaj
            # checks, and on 'mismatch' or 'unknown' for the type check.
            #
            # Counterintuitive but empirically validated on
            # capability_candidates_v1 (n=120):
            #
            #                       R@1  R@5  R@10  MRR   Ans%
            #   Stage 1   hard all   48   87   97  .531   53.3
            #   Stage 1.5 soft all   36   61   76  .399   48.3   (regression)
            #   Stage 1.6 keep NULL  37   55   66  .376   40.0   (regression)
            #
            # NULL kat does NOT mean 'unknown — give the benefit of the
            # doubt'. In practice it strongly correlates with 'subject is
            # a common noun or unclassified pronoun' — i.e., the sentence
            # is not about a person-as-subject. So for KIU questions,
            # filtering out NULL gives better results than letting them
            # compete.
            filter_failed = False
            if cfg.require_answer_slot:
                status = self._answer_slot_status(c, cfg)
                if status != 'present':
                    filter_failed = True
            if not filter_failed and cfg.expected_kats:
                status = self._answer_slot_type_status(c, cfg)
                if status != 'match':
                    filter_failed = True
            if not filter_failed and cfg.has_aliaj_check:
                status = self._aliaj_check_status(c, cfg)
                if status != 'present':
                    filter_failed = True
            if filter_failed:
                scored.append((0.0, c))
                continue

            # ---- Boost ----
            boost = self._boost(question_ast, qtype, c)
            scored.append((base * boost, c))

        scored.sort(key=lambda kv: -kv[0])
        return scored

    # --- internals ---

    def _build_anchor_phrase(self, question_ast: dict) -> str:
        """Reconstruct a 2-3 token anchor phrase from the question's
        non-ki tokens. For 'Kiu fondis Esperanton?' → 'fondis Esperanton'."""
        tokens = []
        for role in ('subjekto', 'verbo', 'objekto'):
            n = question_ast.get(role)
            if not isinstance(n, dict):
                continue
            kerno = n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
            if not isinstance(kerno, dict):
                if role == 'verbo' and 'plena_vorto' in n:
                    pv = n.get('plena_vorto')
                    if pv:
                        tokens.append(pv)
                continue
            pv = kerno.get('plena_vorto')
            radiko = (kerno.get('radiko') or '').lower()
            if pv and radiko != 'ki':  # skip ki-correlatives
                tokens.append(pv)
        return ' '.join(tokens)

    def _answer_slot_status(self, c: dict, cfg: QTypeConfig) -> str:
        """Return 'present', 'missing', or 'unknown'.

        'missing' = we have positive evidence the slot is absent
                    (e.g. parser succeeded but produced no subjekto).
        'unknown' = the shredded column is NULL, so we can't tell;
                    treat as soft signal rather than a hard fail.
        """
        slot = cfg.answer_slot
        if slot == 'subjekto':
            if c.get('subj_radiko'):
                return 'present'
            if c.get('verb_radiko'):
                # parse succeeded (we have a verb), so a missing subject
                # IS a real absence rather than a parse failure
                return 'missing'
            return 'unknown'
        if slot == 'objekto':
            if c.get('obj_radiko'):
                return 'present'
            if c.get('verb_radiko'):
                return 'missing'
            return 'unknown'
        # aliaj-based slots: prefer the materialized boolean columns
        # (Stage 2, #741) when present; fall back to aliaj_json parsing.
        if slot == 'aliaj_loko':
            return self._aliaj_status_for('loko', c)
        if slot == 'aliaj_jaro':
            return self._aliaj_status_for('jaro', c)
        if slot == 'aliaj_numeral':
            return self._aliaj_status_for('kvant', c)
        return 'present'

    def _aliaj_status_for(self, kind: str, c: dict) -> str:
        """Tri-state aliaj check, preferring the boolean columns when
        the index has been augmented with them (Stage 2)."""
        col = f'aliaj_has_{kind}'
        col_val = c.get(col)
        if col_val is True:
            return 'present'
        if col_val is False:
            return 'missing'
        # Boolean column NULL (index not yet augmented) — derive from JSON.
        aliaj_json = c.get('aliaj_json')
        check_fn = {'loko': aliaj_has_loko,
                    'jaro': aliaj_has_jaro,
                    'kvant': aliaj_has_numeral}[kind]
        if check_fn(aliaj_json):
            return 'present'
        return 'missing' if aliaj_json else 'unknown'

    def _answer_slot_type_status(self, c: dict, cfg: QTypeConfig) -> str:
        """Return 'match', 'mismatch', or 'unknown' for the answer-slot
        type check (e.g., is the subject a persono?)."""
        if not cfg.expected_kats:
            return 'match'
        if cfg.answer_slot == 'subjekto':
            kat_raw = c.get('subj_propranoma_kat')
            if not kat_raw:
                return 'unknown'  # column NULL — very common
            kat = kat_raw.lower()
            return 'match' if kat in cfg.expected_kats else 'mismatch'
        if cfg.answer_slot == 'objekto':
            # No obj_propranoma_kat column — can't enforce
            return 'unknown'
        return 'unknown'

    def _aliaj_check_status(self, c: dict, cfg: QTypeConfig) -> str:
        """Tri-state aliaj check for KIE/KIAM/KIOM. Prefers boolean columns
        when present; falls back to aliaj_json parsing."""
        kind = cfg.has_aliaj_check
        if not kind:
            return 'present'
        # Normalize 'numeral' (config term) to 'kvant' (column name)
        col_kind = 'kvant' if kind == 'numeral' else kind
        return self._aliaj_status_for(col_kind, c)

    def _anchor_verb_match(self, q_verb_radiko: Optional[str], c: dict) -> float:
        if not q_verb_radiko:
            return 0.0
        c_radiko = c.get('verb_radiko')
        if c_radiko and c_radiko == q_verb_radiko:
            return 1.0
        if self._cached_q_klaso and c.get('verb_klaso') == self._cached_q_klaso:
            return 0.6
        return 0.0

    def _anchor_object_match(self, q_obj_radiko: Optional[str],
                             q_subj_radiko: Optional[str], c: dict) -> float:
        if not q_obj_radiko and not q_subj_radiko:
            return 0.0
        c_obj = c.get('obj_radiko')
        c_subj = c.get('subj_radiko')
        text = (c.get('text') or '').lower()
        # Best match: the question's object root is in the candidate's object role
        if q_obj_radiko:
            if c_obj == q_obj_radiko:
                return 1.0
            if c_subj == q_obj_radiko:
                return 0.7  # object got fronted/passivized — still relevant
            if q_obj_radiko and q_obj_radiko in text:
                return 0.3
        # Otherwise try the subject anchor (for KIO_OBJ shapes where question
        # has a subject anchor and answer is in object)
        if q_subj_radiko:
            if c_subj == q_subj_radiko:
                return 1.0
            if q_subj_radiko in text:
                return 0.3
        return 0.0

    def _phrase_adj(self, anchor_phrase: str, c: dict) -> float:
        if not anchor_phrase:
            return 0.0
        text = (c.get('text') or '').lower()
        # Approximate adjacency: full anchor as substring (case-insensitive)
        if anchor_phrase.lower() in text:
            return 1.0
        # Partial: each token present individually contributes 0.3
        tokens = anchor_phrase.lower().split()
        if not tokens:
            return 0.0
        present = sum(1 for t in tokens if t in text)
        return 0.3 * present / max(1, len(tokens))

    def _tense_compat(self, q_tempo: Optional[str], c: dict) -> float:
        if not q_tempo:
            return 1.0  # not a constraint
        c_tempo = c.get('verb_tempo')
        if not c_tempo:
            return 0.7  # unknown — neutral
        if c_tempo == q_tempo:
            return 1.0
        # Past/present can both answer past questions in narrative
        if {q_tempo, c_tempo} == {'is', 'as'}:
            return 0.5
        return 0.3

    def _answer_type_score(self, c: dict, cfg: QTypeConfig) -> float:
        if cfg.answer_slot == 'subjekto':
            kat = (c.get('subj_propranoma_kat') or '').lower()
            if not cfg.expected_kats:
                return 0.8 if kat else 0.5  # any subject works absent constraint
            if kat in cfg.expected_kats:
                return 1.0
            if c.get('subj_vortspeco') in ('propra_nomo',):
                return 0.7  # proper noun but wrong type — still informative
            return 0.3
        if cfg.has_aliaj_check == 'loko':
            return 1.0 if aliaj_has_loko(c.get('aliaj_json')) else 0.0
        if cfg.has_aliaj_check == 'jaro':
            return 1.0 if aliaj_has_jaro(c.get('aliaj_json')) else 0.0
        if cfg.has_aliaj_check == 'numeral':
            return 1.0 if aliaj_has_numeral(c.get('aliaj_json')) else 0.0
        return 0.7

    def _boost(self, question_ast: dict, qtype: str, c: dict) -> float:
        # Definitional boost
        if qtype == 'KIO_DEF':
            verb_radiko = c.get('verb_radiko')
            if verb_radiko == 'est':
                return 1.5
        return 1.0
