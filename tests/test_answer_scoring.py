"""Tests for klareco.eval.answer_scoring — attributable extraction scoring (#783).

The central case, and the reason the module exists:

    An extractor that returns the whole source sentence must FAIL.

The old substring metric passed it, because the sentence contains the answer.
That is how retrieval and extraction got conflated into a single number, and
why no improvement could ever be attributed to a stage.
"""

import pytest

from klareco.eval.answer_scoring import (
    aggregate_extraction,
    contains_gold,
    exact_match,
    normalize,
    score_extraction,
    token_f1,
)


class TestNormalize:
    def test_folds_esperanto_supersigns(self):
        assert normalize("Ĉeĥio") == "cehio"
        assert normalize("ŝanĝo") == "sango"

    def test_folds_foreign_diacritics(self):
        """A citation-form mismatch on an imported name is not an extraction
        failure — `Kálmán` and `Kalman` are the same answer."""
        assert normalize("Kálmán") == normalize("Kalman")

    def test_strips_punctuation_and_case(self):
        assert normalize("«Faŭsto»,") == "fausto"
        assert normalize("  ZAMENHOF  ") == "zamenhof"


class TestExactMatch:
    def test_matches_the_span_itself(self):
        assert exact_match("Zamenhof", "Zamenhof")
        assert exact_match("zamenhof.", "Zamenhof")

    def test_whole_sentence_is_NOT_a_match(self):
        """THE case this module exists for.

        The old metric scored this correct — the sentence contains the answer.
        Exact match does not, and that is the entire point: verbosity must
        cost something, or the extractor is never held to account.
        """
        sentence = "En 1859 naskiĝis Ludoviko Zamenhof en Bjalistoko."
        assert contains_gold(sentence, "Zamenhof") is True     # legacy: PASS
        assert exact_match(sentence, "Zamenhof") is False      # honest:  FAIL

    def test_wrong_answer_fails(self):
        assert not exact_match("Hendriks", "Zamenhof")


class TestTokenF1:
    def test_perfect_answer(self):
        assert token_f1("Ludoviko Zamenhof", "Ludoviko Zamenhof") == 1.0

    def test_partial_credit(self):
        f1 = token_f1("Zamenhof", "Ludoviko Zamenhof")
        assert 0.0 < f1 < 1.0

    def test_verbosity_is_penalized(self):
        """A 9-token answer for a 1-token gold span must score poorly. Recall
        alone would give it 1.0 — the precision term is what stops the
        dump-the-whole-passage strategy from winning."""
        sentence = "En 1859 naskiĝis Ludoviko Zamenhof en la urbo Bjalistoko"
        f1 = token_f1(sentence, "Zamenhof")
        assert f1 < 0.25

    def test_no_overlap_is_zero(self):
        assert token_f1("Hendriks", "Zamenhof") == 0.0


class TestScoreExtraction:
    def test_unscorable_without_a_gold_span(self):
        """A legacy set with no gold span is UNSCORABLE — the metrics come back
        None, never 0.0. A missing label must never be silently read as a
        failure; that would blame the extractor for the test set's gap."""
        s = score_extraction("Zamenhof", None, gold_retrieved=True)
        assert s["scorable"] is False
        assert s["exact_match"] is None
        assert s["em_given_retrieved"] is None

    def test_conditional_is_none_when_gold_not_retrieved(self):
        """If retrieval never surfaced the gold passage, the extractor was
        never given the chance to succeed. Scoring it as a failure would
        attribute a retrieval problem to extraction."""
        s = score_extraction("something wrong", "Zamenhof", gold_retrieved=False)
        assert s["scorable"] is True
        assert s["exact_match"] is False        # unconditional: still a miss
        assert s["em_given_retrieved"] is None  # conditional: not the extractor's fault

    def test_conditional_is_scored_when_gold_retrieved(self):
        s = score_extraction("Zamenhof", "Zamenhof", gold_retrieved=True)
        assert s["em_given_retrieved"] is True

    def test_legacy_gap_is_visible(self):
        """The legacy number and the honest number must both be reported, so
        the gap between them is legible."""
        sentence = "En 1859 naskiĝis Ludoviko Zamenhof."
        s = score_extraction(sentence, "Zamenhof", gold_retrieved=True)
        assert s["legacy_contains"] is True
        assert s["exact_match"] is False


class TestAggregate:
    def test_reports_unscorable_set_loudly(self):
        rows = [score_extraction("x", None, True) for _ in range(3)]
        agg = aggregate_extraction(rows)
        assert agg["scorable_questions"] == 0
        assert "UNSCORABLE" in agg["note"]

    def test_conditional_denominator_is_reported(self):
        """A conditional metric with an unstated denominator is a number you
        can talk yourself into believing."""
        rows = [
            score_extraction("Zamenhof", "Zamenhof", gold_retrieved=True),
            score_extraction("", "Kabe", gold_retrieved=False),
        ]
        agg = aggregate_extraction(rows)
        assert agg["scorable_questions"] == 2
        assert agg["gold_retrieved_n"] == 1          # denominator, stated
        assert agg["em_given_retrieved"] == 1.0      # 1/1 — the extractor did its job
        assert agg["exact_match"] == 0.5             # 1/2 — but overall we missed one

    def test_never_retrieved_says_the_failure_is_upstream(self):
        rows = [score_extraction("", "Zamenhof", gold_retrieved=False)]
        agg = aggregate_extraction(rows)
        assert agg["em_given_retrieved"] is None
        assert "upstream" in agg["note"]


class TestEsperantoNumberFolding:
    """#899: digit <-> Esperanto number-word equivalence in scoring."""

    def test_digit_matches_number_word(self):
        assert exact_match("16", "dek ses")
        assert exact_match("6", "ses")
        assert exact_match("dudek tri", "23")
        assert exact_match("cent dudek tri", "123")

    def test_ordinals_and_diacritics(self):
        assert exact_match("naŭ", "9")
        assert exact_match("8", "ok")

    def test_non_numbers_unaffected(self):
        # a number word inside a phrase folds, the rest is untouched
        assert normalize("Zamenhof") == "zamenhof"
        assert normalize("la tri musketeroj") == "la 3 musketeroj"

    def test_token_f1_credits_number_word_answer(self):
        # gold '16', system says 'dek ses' — should get full credit now
        assert token_f1("dek ses", "16") == 1.0
