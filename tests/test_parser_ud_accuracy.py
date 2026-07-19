"""
Parser-quality regression guard against the UD Esperanto gold treebanks.

WHY THIS EXISTS
---------------
The parser is the universal contract — every downstream stage reads the AST it
produces, and `subj_radiko` (which DuckDBRetriever and every reranker key on) is
literally the parser's subject slot. For months parser correctness was asserted
only by hand-written unit examples ("parser is deterministic so it must be
right"). The Universal Dependencies Esperanto treebanks (Prago in-corpus, Cairo
HELD-OUT) are the only EXTERNAL, linguist-curated ground truth we have. This
test wires them into the suite as FLOORS so any change that degrades parse
quality — POS tagging, subject/object role labelling, or dependency attachment
(UAS/LAS) — fails loudly instead of silently.

WHAT IT MEASURES (three layers of AST/parse-tree quality)
  - POS (word class)                       eval_ud_prago.evaluate
  - subject/object role labelling (F1)     eval_ud_roles.evaluate   ← retrieval reads this
  - dependency attachment (UAS/LAS)        eval_conllu.evaluate     ← the parse TREE

The numbers are DETERMINISTIC (frozen fixtures + deterministic parser), so the
floors are set AT the recorded baseline (2026-07-19, bench_history) with `>=`:
a regression fails; an improvement passes and should prompt a floor bump.

Marker: `accuracy` (auto-applied by tests/conftest.py) — quality vs. baseline,
part of the merge gate `-m "perf or accuracy"`. Unlike the store-backed accuracy
tests, this one needs NO production indexes: only klareco.parser + the committed
fixtures under tests/fixtures/ud/. So it runs in the fast CI too.

Related: #726 (test set too small — these 131+20 sentences are a good CEILING
and regression ruler but too few to DETECT sub-treebank incremental wins, e.g.
#871's 182k-corpus fix is invisible here). See bench_history 2026-07-19.

Last Updated: 2026-07-19
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "ud"
PRAGO = FIXTURES / "eo_prago-ud-test.conllu"
CAIRO = FIXTURES / "eo_cairo-ud-test.conllu"

# The evaluators live under scripts/eval (excluded from collection). Import their
# single-source-of-truth evaluate() functions rather than re-implementing metrics.
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import eval_conllu  # noqa: E402
import eval_ud_prago  # noqa: E402
import eval_ud_roles  # noqa: E402

from klareco.parser import parse  # noqa: E402

# ---------------------------------------------------------------------------
# FROZEN BASELINE FLOORS — recorded 2026-07-19 (bench_history "PARSER BASELINE
# on UD Esperanto gold treebanks"). Values are deterministic. `>=` guard: raise
# a floor only when an INTENDED parser improvement clears it.
# ---------------------------------------------------------------------------
BASELINE = {
    "prago": {  # in-corpus
        "pos_strict": 0.813, "pos_adjusted": 0.947,
        "subj_f1": 0.687, "subj_recall": 0.683, "obj_f1": 0.760,
        "uas": 0.695, "las": 0.623,
    },
    "cairo": {  # HELD-OUT — the honest generalization ruler
        "pos_strict": 0.803, "pos_adjusted": 0.959,
        "subj_f1": 0.930, "subj_recall": 0.909, "obj_f1": 0.880,
        "uas": 0.738, "las": 0.664,
    },
}
# Small tolerance so a floating-point last-digit wobble in a deterministic metric
# never red-flags; a real regression is far larger than this.
EPS = 0.005

_PATHS = {"prago": PRAGO, "cairo": CAIRO}


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"UD fixture missing: {path} (should be committed)")


# --- cache the three evaluations per treebank (parse once, assert many) ------
_POS_CACHE: dict[str, dict] = {}
_ROLE_CACHE: dict[str, dict] = {}
_DEP_CACHE: dict[str, dict] = {}


def _pos(tb: str) -> dict:
    _require(_PATHS[tb])
    if tb not in _POS_CACHE:
        _POS_CACHE[tb] = eval_ud_prago.evaluate(_PATHS[tb])
    return _POS_CACHE[tb]


def _roles(tb: str) -> dict:
    _require(_PATHS[tb])
    if tb not in _ROLE_CACHE:
        sents = eval_ud_roles.load_conllu(str(_PATHS[tb]))
        _ROLE_CACHE[tb] = eval_ud_roles.evaluate(sents)
    return _ROLE_CACHE[tb]


def _dep(tb: str) -> dict:
    _require(_PATHS[tb])
    if tb not in _DEP_CACHE:
        _DEP_CACHE[tb] = eval_conllu.evaluate(str(_PATHS[tb]))
    return _DEP_CACHE[tb]


TREEBANKS = ["prago", "cairo"]


# --- POS / word-class quality ------------------------------------------------
@pytest.mark.parametrize("tb", TREEBANKS)
def test_pos_accuracy_floor(tb):
    r = _pos(tb)
    b = BASELINE[tb]
    assert r["pos_strict"] + EPS >= b["pos_strict"], (
        f"{tb}: POS strict {r['pos_strict']:.4f} < floor {b['pos_strict']}")
    assert r["pos_adjusted"] + EPS >= b["pos_adjusted"], (
        f"{tb}: POS scheme-adjusted {r['pos_adjusted']:.4f} < floor {b['pos_adjusted']}")


# --- role labelling (subj/obj) — what retrieval reads ------------------------
@pytest.mark.parametrize("tb", TREEBANKS)
def test_subject_role_f1_floor(tb):
    r = _roles(tb)
    b = BASELINE[tb]
    assert r["subj"]["f1"] + EPS >= b["subj_f1"], (
        f"{tb}: subject F1 {r['subj']['f1']:.4f} < floor {b['subj_f1']} "
        f"(subj_radiko is the field DuckDBRetriever + rerankers key on)")
    assert r["subj"]["recall"] + EPS >= b["subj_recall"], (
        f"{tb}: subject recall {r['subj']['recall']:.4f} < floor {b['subj_recall']}")


@pytest.mark.parametrize("tb", TREEBANKS)
def test_object_role_f1_floor(tb):
    r = _roles(tb)
    assert r["obj"]["f1"] + EPS >= BASELINE[tb]["obj_f1"], (
        f"{tb}: object F1 {r['obj']['f1']:.4f} < floor {BASELINE[tb]['obj_f1']}")


# --- dependency attachment (the parse TREE) ---------------------------------
@pytest.mark.parametrize("tb", TREEBANKS)
def test_dependency_uas_las_floor(tb):
    r = _dep(tb)
    b = BASELINE[tb]
    assert r["uas"] + EPS >= b["uas"], (
        f"{tb}: UAS {r['uas']:.4f} < floor {b['uas']}")
    assert r["las"] + EPS >= b["las"], (
        f"{tb}: LAS {r['las']:.4f} < floor {b['las']}")


# --- per-sentence AST quality: the parser must not CRASH on gold text --------
@pytest.mark.parametrize("tb", TREEBANKS)
def test_no_parse_crashes_and_well_formed_ast(tb):
    _require(_PATHS[tb])
    texts = _gold_texts(_PATHS[tb])
    assert texts, f"{tb}: no sentences read from fixture"
    crashes = []
    for t in texts:
        try:
            ast = parse(t)
        except Exception as e:  # a crash on gold text is always a bug
            crashes.append((t[:60], repr(e)))
            continue
        # every AST is a well-formed frazo dict carrying parse statistics
        assert isinstance(ast, dict), f"{tb}: AST is not a dict for {t[:60]!r}"
        assert ast.get("tipo") == "frazo", f"{tb}: AST tipo != frazo for {t[:60]!r}"
        assert "parse_statistics" in ast, f"{tb}: no parse_statistics for {t[:60]!r}"
    assert not crashes, f"{tb}: parser crashed on {len(crashes)} gold sentences: {crashes[:3]}"


def _gold_texts(path: Path) -> list[str]:
    """Extract the `# text = ...` lines (the raw sentences) from a CoNLL-U file."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            out.append(line[len("# text = "):].strip())
    return out
