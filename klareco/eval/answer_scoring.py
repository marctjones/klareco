"""
Answer scoring — deterministic, attributable, no LLM judge.

VERSION: v2.x
COMPATIBLE WITH: test sets carrying `gold_answer_span` (R17)
DEPENDENCIES: stdlib only
STAGE: Evaluation

Description:
    Scores extraction against a labeled gold answer span, SEPARATELY from
    retrieval. See issue #783 and R17 in docs/QA_TEST_SET_QUALITY_STANDARD.md.

    The defect this replaces
    ------------------------
    Answer correctness used to be substring containment:

        correct = any(kw.lower() in final.lower() for kw in expected)

    The pipeline's output is a whole retrieved sentence, so "correct" meant
    only: *a passage containing the expected keyword survived to the output.*
    It could not distinguish an extractor that returned `Zamenhof` from one
    that returned the entire paragraph, and it applied no verbosity penalty —
    more text could only ever help the score.

    Why that is disqualifying for THIS project
    ------------------------------------------
    It conflates retrieval and extraction into a single number, so no
    improvement can ever be attributed to a stage. **Decomposable attribution
    is the thesis** (VISION.md). A metric that cannot separate one stage's
    contribution from another's cannot test the thesis, however many stages we
    build.

    The contract
    ------------
    Three numbers, reported separately, never collapsed:

      retrieval_recall@k                    did the right passage rank well?
      extraction_exact_match | retrieved    GIVEN the right passage, did we
                                            pull out the right span?
      passage_selection                     from a noisy pool, did we pick it?

    The conditional in the middle row is the whole point. It is the number
    DESIGN.md has always *claimed* we measure and never actually computed.

    Scoring is deterministic: exact match after case- and diacritic-folding,
    plus token-level F1 for partial credit. **No LLM judge** — an unauditable
    instrument is not an instrument.

Last Updated: 2026-07-13
Related Issues: #783
See Also: docs/QA_TEST_SET_QUALITY_STANDARD.md (R17, §3b)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Esperanto's supersigned letters. We fold them for *matching* only — the
# corpus and the test sets are required to use them correctly (R13), so this
# is robustness against incidental encoding drift, not a licence to accept
# x-system input.
_EO_FOLD = str.maketrans({
    "ĉ": "c", "Ĉ": "c",
    "ĝ": "g", "Ĝ": "g",
    "ĥ": "h", "Ĥ": "h",
    "ĵ": "j", "Ĵ": "j",
    "ŝ": "s", "Ŝ": "s",
    "ŭ": "u", "Ŭ": "u",
})

# Punctuation to strip. Includes the Esperanto quotation marks «» and the
# typographic dashes that show up in Wikipedia-derived text.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

# --- Esperanto number-word folding (#899) --------------------------------
# The gold answer and the source often express a number differently — gold
# "16" vs source "dek ses", "6" vs "ses". Fold recognized number-word runs to
# their digit value so they match. Diacritics are already stripped upstream
# (naŭ -> nau), so the maps use the folded forms. Covers 0-999, the range that
# accounts for essentially all trivia numeric answers.
_ONES = {"nul": 0, "unu": 1, "du": 2, "tri": 3, "kvar": 4, "kvin": 5,
         "ses": 6, "sep": 7, "ok": 8, "nau": 9}


def _word_value(tok: str):
    """Value of a single Esperanto number token (cardinal or ordinal), or None."""
    t = tok[:-1] if tok.endswith("a") and tok not in _ONES else tok  # ordinal -a
    if t in _ONES:
        return _ONES[t]
    if t == "dek":
        return 10
    if t == "cent":
        return 100
    if t == "mil":
        return 1000
    # compound tens: dudek..naudek ; compound hundreds: ducent..naucent
    if t.endswith("dek") and t[:-3] in _ONES:
        return _ONES[t[:-3]] * 10
    if t.endswith("cent") and t[:-4] in _ONES:
        return _ONES[t[:-4]] * 100
    return None


def _fold_number_run(vals: list) -> int:
    """Sum an Esperanto number run: e.g. [100,20,3]->123, [10,6]->16."""
    total = h = 0
    for v in vals:
        if v >= 100:
            h += v; total += v
        elif v >= 10:
            total += v
        else:
            total += v
    return total


def _fold_numbers(text: str) -> str:
    """Replace maximal runs of number-words with their digit value."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        v = _word_value(words[i])
        if v is None:
            out.append(words[i]); i += 1
            continue
        run = []
        while i < len(words) and _word_value(words[i]) is not None:
            run.append(_word_value(words[i])); i += 1
        out.append(str(_fold_number_run(run)))
    return " ".join(out)


def normalize(text: str) -> str:
    """Fold to the canonical form used for all answer comparison.

    Lowercase, strip diacritics (Esperanto supersigns and any residual
    combining marks), drop punctuation, collapse whitespace.
    """
    if not text:
        return ""
    t = text.translate(_EO_FOLD).lower()
    # Strip any remaining combining marks (e.g. accented Latin from foreign
    # names: "Kálmán" -> "kalman"), so a citation-form mismatch on an imported
    # name doesn't read as an extraction failure.
    t = "".join(c for c in unicodedata.normalize("NFD", t)
                if not unicodedata.combining(c))
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    # #899: fold Esperanto number-words to digits so "16" and "dek ses" match.
    return _fold_numbers(t)


def tokens(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def exact_match(predicted: str, gold: str) -> bool:
    """True iff the prediction *is* the answer — not merely contains it.

    This is the check that makes verbosity cost something. An extractor that
    returns the whole source sentence fails here, as it should.
    """
    return bool(gold) and normalize(predicted) == normalize(gold)


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 — partial credit, with a real precision penalty.

    Recall alone would reward dumping the entire passage (it contains every
    gold token). The precision term is what stops that: a 20-token answer for
    a 1-token gold span scores F1 ~= 0.10, not 1.0.
    """
    p_toks, g_toks = tokens(predicted), tokens(gold)
    if not g_toks or not p_toks:
        return 0.0

    # Multiset intersection — a token repeated in the prediction only counts
    # as often as it appears in the gold span.
    overlap = 0
    remaining = list(g_toks)
    for t in p_toks:
        if t in remaining:
            remaining.remove(t)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def contains_gold(predicted: str, gold: str) -> bool:
    """The LEGACY criterion — substring containment.

    Retained only so we can report the old number alongside the new one and
    show the gap. **Do not use it as a headline metric.** It is the defect
    #783 exists to fix; if it is ever the only number in a report, the report
    is not measuring extraction.
    """
    if not gold:
        return False
    return normalize(gold) in normalize(predicted)


def score_extraction(predicted: str,
                     gold_span: Optional[str],
                     gold_retrieved: bool) -> dict:
    """Score one answer.

    Parameters
    ----------
    predicted      : the pipeline's answer text
    gold_span      : the labeled answer span (R17). None for legacy test sets
                     that predate R17 — in that case extraction is *not
                     scorable* and the metrics come back None rather than 0.0,
                     so a missing label can never be silently read as a failure.
    gold_retrieved : whether the gold passage was retrieved at all. This is the
                     conditioning variable: extraction quality is only
                     meaningful GIVEN that the extractor had the right passage
                     to work from.

    Returns a dict whose keys are stable across the eval stack.
    """
    if gold_span is None:
        return {
            "scorable":              False,
            "exact_match":           None,
            "token_f1":              None,
            "legacy_contains":       None,
            "em_given_retrieved":    None,
            "f1_given_retrieved":    None,
        }

    em = exact_match(predicted, gold_span)
    f1 = token_f1(predicted, gold_span)

    return {
        "scorable":           True,
        "exact_match":        em,
        "token_f1":           round(f1, 4),
        # The old substring number, kept visible so the gap between "a passage
        # containing the answer survived" and "we extracted the answer" is
        # legible in every report.
        "legacy_contains":    contains_gold(predicted, gold_span),
        # Conditional on retrieval — None (not False) when the gold passage was
        # never retrieved, because in that case extraction was never given the
        # chance to succeed and scoring it as a failure would blame the wrong
        # stage. This is the attribution the whole module exists for.
        "em_given_retrieved": em if gold_retrieved else None,
        "f1_given_retrieved": round(f1, 4) if gold_retrieved else None,
    }


def aggregate_extraction(per_question: list[dict]) -> dict:
    """Aggregate the per-question extraction scores.

    Reports the conditional metrics over *only* the questions where the gold
    passage was actually retrieved — the denominator matters, so it is
    reported too. A conditional metric with an unstated denominator is a
    number you can talk yourself into believing.
    """
    scorable = [q for q in per_question if q.get("scorable")]
    if not scorable:
        return {
            "scorable_questions": 0,
            "note": ("No question carried a gold_answer_span — extraction is "
                     "UNSCORABLE on this test set. See R17."),
        }

    conditioned = [q for q in scorable if q.get("em_given_retrieved") is not None]

    out = {
        "scorable_questions":   len(scorable),
        # Unconditional: over every scorable question, retrieved or not.
        "exact_match":          sum(bool(q["exact_match"]) for q in scorable) / len(scorable),
        "token_f1":             sum(q["token_f1"] for q in scorable) / len(scorable),
        "legacy_contains":      sum(bool(q["legacy_contains"]) for q in scorable) / len(scorable),
        # Conditional: the number that attributes to the extractor.
        "gold_retrieved_n":     len(conditioned),
        "gold_retrieved_frac":  len(conditioned) / len(scorable),
    }

    if conditioned:
        out["em_given_retrieved"] = (
            sum(bool(q["em_given_retrieved"]) for q in conditioned) / len(conditioned))
        out["f1_given_retrieved"] = (
            sum(q["f1_given_retrieved"] for q in conditioned) / len(conditioned))
    else:
        out["em_given_retrieved"] = None
        out["f1_given_retrieved"] = None
        out["note"] = ("The gold passage was never retrieved, so extraction "
                       "quality is unmeasurable on this run — the failure is "
                       "upstream, in retrieval.")

    return out
