"""Tests for the R13 language-quality checks added by #791.

Every test here names a real pair that the auditor PASSED before these checks
existed. The auditor reported 8/8 PASS on a batch containing an English
possessive, three foreign titles with the accusative -n glued on, a double
accusative, and an adjective as the gold answer span.

The quality standard's own escalation rule is the point:

    "The auditor is the source of truth — if a question feels wrong but passes
     audit, ADD A NEW CHECK to the auditor."
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))

from audit_language_quality import (  # noqa: E402
    check_answer_is_nominal,
    check_esperanto_orthography,
    check_foreign_title_not_inflected,
    check_no_accusative_after_de,
)


class TestEsperantoOrthography:
    def test_english_possessive_is_rejected(self):
        # PASSED the old auditor.
        ok, why = check_esperanto_orthography("Kiu venkis Rorke's Driftn?")
        assert not ok
        assert "apostrophe" in why

    def test_foreign_letters_rejected_in_running_text(self):
        # PASSED the old auditor.
        ok, why = check_esperanto_orthography("Kiu gajnis The World Is Not Overn?")
        assert not ok
        assert "non-Esperanto letters" in why

    def test_foreign_letters_allowed_INSIDE_a_quoted_work(self):
        """R1 actively prefers quoted works, and a quoted title may legitimately
        be foreign. The rule is about bare running text, not about quotes."""
        ok, _ = check_esperanto_orthography('Kiu inventis «Nonograms»?')
        assert ok
        ok, _ = check_esperanto_orthography('Kiu verkis «The Waste Land»?')
        assert ok

    def test_clean_esperanto_passes(self):
        ok, _ = check_esperanto_orthography("Kiu fondis Esperanton?")
        assert ok


class TestForeignTitleNotInflected:
    def test_accusative_glued_to_accented_foreign_name(self):
        # PASSED the old auditor.
        ok, why = check_foreign_title_not_inflected("Kiu reĝisoris Théâtre des Variétésn?")
        assert not ok
        assert "-n" in why

    def test_esperanto_supersigns_are_not_foreign_accents(self):
        """Ĉeĥion is a properly assimilated Esperanto accusative. The check must
        not confuse Esperanto's own supersigns with foreign diacritics."""
        ok, _ = check_foreign_title_not_inflected("Kiu vizitis Ĉeĥion?")
        assert ok

    def test_native_accusative_passes(self):
        ok, _ = check_foreign_title_not_inflected("Kiu fondis Esperanton?")
        assert ok


class TestNoAccusativeAfterDe:
    def test_double_accusative_under_de_is_rejected(self):
        # PASSED the old auditor.
        ok, why = check_no_accusative_after_de("Kiu fondis Ĉambron de Arton?")
        assert not ok
        assert "de" in why

    def test_genitive_de_phrase_passes(self):
        ok, _ = check_no_accusative_after_de("Kiu fondis Ĉambron de Arto?")
        assert ok

    def test_coordinated_objects_are_NOT_rejected(self):
        """A blanket 'one accusative per clause' rule would be WRONG —
        coordinated objects legitimately take two. The defect is specifically an
        accusative governed by a preposition."""
        ok, _ = check_no_accusative_after_de("Mi vidis la domon kaj la ĝardenon.")
        assert ok


class TestAnswerIsNominal:
    def test_adjective_answer_is_rejected(self):
        """`Britaj` ("British") became a gold_answer_span. It parses as
        propra_nomo ONLY because it is capitalized — the heuristic masks its
        adjectival morphology."""
        ok, why = check_answer_is_nominal({"gold_answer_span": "Britaj"})
        assert not ok
        assert "adjectival plural" in why

    def test_adverb_answer_is_rejected(self):
        ok, _ = check_answer_is_nominal({"gold_answer_span": "rapide"})
        assert not ok

    @pytest.mark.parametrize("name", ["Maria", "Anna", "Zamenhof", "Thomas Fritsch"])
    def test_real_names_survive(self, name):
        """THE BOUNDARY, encountered in the wild.

        `Brita` (adjective, root brit-) and `Maria` / `Anna` (names) are
        morphologically identical — both lowercase to `adjektivo`. Telling them
        apart requires world knowledge, not rules: it is the proper-noun
        disambiguation residue VISION.md names as irreducible.

        So the check only rejects what Esperanto's regularity DOES settle: the
        adjectival plural -aj/-ajn, which no proper name takes. The singular -a
        case is deliberately left alone. A name gazetteer here would not solve
        the residue — it would relocate it.
        """
        ok, _ = check_answer_is_nominal({"gold_answer_span": name})
        assert ok
