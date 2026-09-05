"""Versioned stand-off annotations bound to their actual source and tokenization.

An annotation layer supplements a parse; it never overwrites its token registry.
Payload schemas belong to the producer. This module validates the shared envelope,
JSON values, references, provenance, and review records, not linguistic truth.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Literal, TypedDict

LAYER_VERSION = 1


class Producer(TypedDict):
    name: str
    version: str
    method: Literal["rule", "human"]
    artifacts: dict[str, str]


class SentenceTarget(TypedDict):
    kind: Literal["sentence"]


class TokenTarget(TypedDict):
    kind: Literal["tokens"]
    ids: list[int]


class SpanTarget(TypedDict):
    kind: Literal["span"]
    spans: list[list[int]]


class EdgeTarget(TypedDict):
    kind: Literal["edge"]
    dependent: int
    head: int


class _SourceBasis(TypedDict):
    source_sha256: str


class AnnotationBasis(_SourceBasis, total=False):
    tokenization_sha256: str
    dependencies_sha256: str


class Review(TypedDict):
    annotator: str
    reviewer: str
    notes: str


class Annotation(TypedDict):
    id: str
    target: SentenceTarget | TokenTarget | SpanTarget | EdgeTarget
    value: dict[str, Any]


class _RequiredLayer(TypedDict):
    version: int
    id: str
    schema: str
    status: Literal["predicted", "draft", "reviewed"]
    producer: Producer
    basis: AnnotationBasis
    annotations: list[Annotation]


class AnnotationLayer(_RequiredLayer, total=False):
    review: Review


def validate_json(value: Any) -> None:
    """Reject values JSON would silently coerce, or cannot preserve at all."""
    active: set[int] = set()
    finished: set[int] = set()

    def visit(item, depth):
        if depth > 256:
            raise ValueError("AST exceeds the JSON nesting limit of 256")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if not isinstance(item, (dict, list)):
            raise ValueError(
                f"AST requires finite JSON values, got {type(item).__name__}"
            )
        identity = id(item)
        if identity in active:
            raise ValueError("Cyclic JSON structure in AST")
        if identity in finished:
            return
        active.add(identity)
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("AST JSON object keys must be strings")
            children = item.values()
        else:
            children = item
        for child in children:
            visit(child, depth + 1)
        active.remove(identity)
        finished.add(identity)

    visit(value, 0)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def annotation_basis(
    ast: dict, *, tokens: bool = True, syntax: bool = False
) -> dict[str, str]:
    """Span annotations survive retokenization; token annotations cannot."""
    source = ast.get("source", {}).get("original")
    if not isinstance(source, str):
        raise ValueError("Annotation requires original source text")
    basis = {"source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}
    if tokens or syntax:
        basis["tokenization_sha256"] = _digest(
            {
                "normalized": ast["source"]["normalized"],
                "tokens": [
                    [
                        w["id"],
                        w["plena_vorto"],
                        w.get("normalized_span"),
                        w.get("original_span"),
                    ]
                    for w in ast["vortoj"]
                ],
            }
        )
    if syntax:
        basis["dependencies_sha256"] = _digest(
            [[w["id"], w["kapo"], w["rolo"]] for w in ast["vortoj"]]
        )
    return basis


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_layers(ast: dict) -> None:
    layers = ast.get("annotation_layers", [])
    if not isinstance(layers, list):
        raise ValueError("Annotation layers must be a list")
    if not layers:
        return
    registry = {w["id"] for w in ast["vortoj"]}
    expected = annotation_basis(ast, syntax=True)
    text = ast["source"]["original"]
    layer_ids = set()
    required = {"version", "id", "schema", "status", "producer", "basis", "annotations"}
    for layer in layers:
        if (
            not isinstance(layer, dict)
            or not required <= layer.keys()
            or set(layer) - required - {"review"}
        ):
            raise ValueError("Invalid annotation layer envelope")
        if type(layer["version"]) is not int or layer["version"] != LAYER_VERSION:
            raise ValueError("Unsupported annotation layer version")
        if not _nonempty(layer["id"]) or layer["id"] in layer_ids:
            raise ValueError("Invalid or duplicate annotation layer id")
        layer_ids.add(layer["id"])
        if not _nonempty(layer["schema"]) or ":" not in layer["schema"]:
            raise ValueError("Annotation payload schema must be a qualified identifier")
        if layer["status"] not in ("predicted", "draft", "reviewed"):
            raise ValueError("Invalid annotation status")
        producer = layer["producer"]
        if (
            not isinstance(producer, dict)
            or set(producer) != {"name", "version", "method", "artifacts"}
            or not all(_nonempty(producer[k]) for k in ("name", "version"))
            or producer["method"] not in ("rule", "human")
            or not isinstance(producer["artifacts"], dict)
            or not all(
                _nonempty(k) and _sha256(v) for k, v in producer["artifacts"].items()
            )
        ):
            raise ValueError("Invalid annotation producer provenance")
        basis = layer["basis"]
        if (
            not isinstance(basis, dict)
            or "source_sha256" not in basis
            or set(basis) - expected.keys()
            or any(value != expected[key] for key, value in basis.items())
        ):
            raise ValueError("Stale annotation source, tokenization, or dependencies")
        if "dependencies_sha256" in basis and "tokenization_sha256" not in basis:
            raise ValueError("Syntax annotation requires tokenization binding")
        if layer["status"] == "reviewed":
            review = layer.get("review")
            if (
                not isinstance(review, dict)
                or set(review) != {"annotator", "reviewer", "notes"}
                or not all(_nonempty(v) for v in review.values())
                or review["annotator"].strip().casefold()
                == review["reviewer"].strip().casefold()
            ):
                raise ValueError(
                    "Reviewed annotations require distinct review identities and notes"
                )
        elif "review" in layer:
            raise ValueError("Review records require reviewed status")
        if not isinstance(layer["annotations"], list):
            raise ValueError("Layer annotations must be a list")
        ids = set()
        for record in layer["annotations"]:
            if (
                not isinstance(record, dict)
                or set(record) != {"id", "target", "value"}
                or not _nonempty(record["id"])
                or record["id"] in ids
                or not isinstance(record["value"], dict)
            ):
                raise ValueError("Invalid or duplicate annotation record")
            ids.add(record["id"])
            target = record["target"]
            if not isinstance(target, dict):
                raise ValueError("Invalid annotation target")
            kind = target.get("kind")
            if kind == "sentence" and set(target) == {"kind"}:
                continue
            if kind == "span" and set(target) == {"kind", "spans"}:
                spans = target["spans"]
                if not isinstance(spans, list) or not spans:
                    raise ValueError("Annotation spans must be nonempty")
                previous = 0
                for span in spans:
                    if (
                        not isinstance(span, list)
                        or len(span) != 2
                        or any(type(i) is not int for i in span)
                        or not previous <= span[0] < span[1] <= len(text)
                    ):
                        raise ValueError(
                            "Invalid, overlapping, or unordered annotation spans"
                        )
                    previous = span[1]
                continue
            if "tokenization_sha256" not in basis:
                raise ValueError(
                    "Token and edge annotations require tokenization binding"
                )
            if kind == "tokens" and set(target) == {"kind", "ids"}:
                token_ids = target["ids"]
                if (
                    not isinstance(token_ids, list)
                    or not token_ids
                    or any(type(i) is not int or i not in registry for i in token_ids)
                    or token_ids != sorted(set(token_ids))
                ):
                    raise ValueError("Invalid annotation token references")
            elif kind == "edge" and set(target) == {"kind", "dependent", "head"}:
                dep, head = target["dependent"], target["head"]
                if (
                    type(dep) is not int
                    or dep not in registry
                    or type(head) is not int
                    or head not in registry | {0}
                    or dep == head
                ):
                    raise ValueError("Invalid annotation edge references")
            else:
                raise ValueError("Unsupported annotation target")


def with_annotation_layer(ast: dict, layer: AnnotationLayer) -> dict:
    """Return an owned, validated copy; never relabel the source parse in place."""
    from .syntax_graph import validate_ast

    result = deepcopy(ast)
    result.setdefault("annotation_layers", []).append(deepcopy(layer))
    validate_json(result)
    validate_ast(result)
    validate_layers(result)
    return result
