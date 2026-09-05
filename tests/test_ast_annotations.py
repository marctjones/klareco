"""Annotation ownership, source binding, and storage integrity contracts."""

from copy import deepcopy
import json

import pytest

from klareco.ast_annotations import annotation_basis, with_annotation_layer
from klareco.parser import compact_ast, expand_ast, parse


def layer(ast, target=None):
    return {
        "version": 1,
        "id": "review:1",
        "schema": "urn:test:linguistic-notes:1",
        "status": "draft",
        "producer": {
            "name": "test annotator",
            "version": "1",
            "method": "human",
            "artifacts": {},
        },
        "basis": annotation_basis(ast),
        "annotations": [
            {
                "id": "a1",
                "target": target or {"kind": "tokens", "ids": [1]},
                "value": {"note": "Independent annotation", "alternatives": []},
            }
        ],
    }


def test_annotation_storage_keeps_prediction_and_human_note_separate():
    ast = parse("Mi vidas vin.")
    note = layer(ast)
    enriched = with_annotation_layer(ast, note)
    note["annotations"][0]["value"]["note"] = "Changed externally"
    assert "annotation_layers" not in ast
    assert enriched["vortoj"] == ast["vortoj"]
    restored = expand_ast(json.loads(json.dumps(compact_ast(enriched))))
    assert restored == enriched
    assert restored["subjekto"]["kerno"] is restored["vortoj"][0]
    assert (
        restored["annotation_layers"][0]["annotations"][0]["value"]["note"]
        == "Independent annotation"
    )


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "tokens", "ids": [99]},
        {"kind": "tokens", "ids": [True]},
        {"kind": "tokens", "ids": [1, 1]},
        {"kind": "tokens", "ids": [2, 1]},
        {"kind": "edge", "dependent": 1, "head": 1},
        {"kind": "edge", "dependent": 1, "head": 99},
        {"kind": "span", "spans": [[0, 99]]},
        {"kind": "span", "spans": [[0, 2], [1, 3]]},
        {"kind": "span", "spans": [[False, 2]]},
        {"kind": "unknown"},
    ],
)
def test_invalid_targets_cannot_be_stored(target):
    ast = parse("Mi venas.")
    with pytest.raises(ValueError):
        with_annotation_layer(ast, layer(ast, target))


def test_character_spans_survive_retokenization_but_token_ids_do_not():
    ast = parse("Mi venas.")
    spans = layer(ast, {"kind": "span", "spans": [[0, 2], [3, 8]]})
    spans["basis"] = annotation_basis(ast, tokens=False)
    other = deepcopy(ast)
    other["vortoj"][0]["normalized_span"] = None
    assert with_annotation_layer(other, spans)["annotation_layers"]
    with pytest.raises(ValueError, match="Stale annotation"):
        with_annotation_layer(other, layer(ast))
    changed_source = parse("Vi venas.")
    with pytest.raises(ValueError, match="Stale annotation"):
        with_annotation_layer(changed_source, spans)


def test_syntax_dependent_annotation_cannot_follow_a_changed_parse():
    ast = parse("Mi vidas vin.")
    note = layer(ast)
    note["basis"] = annotation_basis(ast, syntax=True)
    note["basis"]["dependencies_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Stale annotation"):
        with_annotation_layer(ast, note)


def test_reviewed_status_requires_independent_review_record():
    ast = parse("Mi venas.")
    note = layer(ast)
    note["status"] = "reviewed"
    for review in [None, {"annotator": "A", "reviewer": " a ", "notes": "Reviewed."}]:
        note["review"] = review
        with pytest.raises(ValueError, match="distinct review"):
            with_annotation_layer(ast, note)
    note["review"] = {"annotator": "A", "reviewer": "B", "notes": "Adjudicated."}
    assert (
        with_annotation_layer(ast, note)["annotation_layers"][0]["status"] == "reviewed"
    )


def test_duplicate_layer_ids_and_unknown_schema_versions_fail():
    ast = parse("Mi venas.")
    note = layer(ast)
    enriched = with_annotation_layer(ast, note)
    with pytest.raises(ValueError, match="duplicate annotation layer"):
        with_annotation_layer(enriched, note)
    note["version"] = 99
    with pytest.raises(ValueError, match="version"):
        with_annotation_layer(ast, note)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), (1, 2), {1: "key"}, {1}, object()]
)
def test_lossy_or_non_json_annotation_values_are_rejected(value):
    ast = parse("Mi venas.")
    ast["extra"] = value
    with pytest.raises(ValueError, match="JSON"):
        compact_ast(ast)


def test_cyclic_annotations_fail_with_an_explicit_error():
    ast = parse("Mi venas.")
    ast["extra"] = ast
    with pytest.raises(ValueError, match="Cyclic JSON"):
        compact_ast(ast)


def test_storage_rejects_annotations_with_stale_source_hashes():
    ast = parse("Mi venas.")
    packed = compact_ast(with_annotation_layer(ast, layer(ast)))
    packed["structure"]["annotation_layers"][0]["basis"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Stale annotation"):
        expand_ast(packed)
