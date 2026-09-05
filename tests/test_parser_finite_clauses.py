"""Finite mood, not the presence of tense, determines clause predicates."""

import pytest
from klareco.parser import parse


@pytest.mark.parametrize(
    "text,verb",
    [
        ("Venu!", "Venu"),
        ("Mi venus.", "venus"),
        ("Li venus se ŝi invitus lin.", "venus"),
    ],
)
def test_tenseless_finite_predicates(text, verb):
    ast = parse(text)
    token = next(w for w in ast["vortoj"] if w["plena_vorto"] == verb)
    assert (token["kapo"], token["rolo"]) == (0, "root")
    assert ast["propozicioj"]


def test_nested_infinitives_attach_to_the_immediately_governing_verb():
    ast = parse("Li provis ĉesi fumi.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["fumi"]["kapo"] == words["ĉesi"]["id"]
    assert words["fumi"]["rolo"] == "xcomp"
