"""Conservative dependency refinements with explicit before/after evidence.

These rules consume final surface IDs. They do not infer semantic classes or
claim that the remaining attachment choices exhaust the language's ambiguity.
"""

from __future__ import annotations

from typing import TypedDict


class Edge(TypedDict):
    head_id: int
    relation: str


class AttachmentChange(TypedDict):
    token_id: int
    rule: str
    before: Edge
    after: Edge
    evidence_token_ids: list[int]


def refine_dependencies(tokens: list[dict]) -> list[AttachmentChange]:
    from .syntax_graph import validate_tokens

    validate_tokens(tokens)
    registry = {word["id"]: word for word in tokens}
    changes: list[AttachmentChange] = []

    def attach(word, head, relation, rule, evidence):
        before = Edge(head_id=word["kapo"], relation=word["rolo"])
        after = Edge(head_id=head, relation=relation)
        if before == after:
            return
        ancestor = head
        while ancestor:
            if ancestor == word["id"]:
                return
            ancestor = registry[ancestor]["kapo"]
        word["kapo"], word["rolo"] = head, relation
        changes.append(
            AttachmentChange(
                token_id=word["id"],
                rule=rule,
                before=before,
                after=after,
                evidence_token_ids=sorted(set(evidence)),
            )
        )

    for index, word in enumerate(tokens):
        # A zero head is the root relation even in a verbless fragment. The
        # serializer must not invent an attachment absent from the AST.
        if word["kapo"] == 0:
            attach(word, 0, "root", "root-relation-v1", [word["id"]])
        head = registry.get(word["kapo"])
        if word["rolo"] == "advmod" and head and head["rolo"] in ("cop", "aux"):
            attach(
                word,
                head["kapo"],
                "advmod",
                "predicate-adverb-v1",
                [word["id"], head["id"], head["kapo"]],
            )

        if word["rolo"] == "cc" and 0 < index < len(tokens) - 1:
            left, right = tokens[index - 1], tokens[index + 1]
            if (
                left.get("vortspeco") == right.get("vortspeco") == "adjektivo"
                and left.get("kazo") == right.get("kazo")
                and left.get("nombro") == right.get("nombro")
                and right["rolo"] in ("amod", "xcomp", "nmod")
            ):
                attach(
                    right,
                    left["id"],
                    "conj",
                    "adjacent-adjective-coordination-v1",
                    [left["id"], word["id"], right["id"]],
                )
                if right["kapo"] == left["id"] and right["rolo"] == "conj":
                    attach(
                        word,
                        right["id"],
                        "cc",
                        "coordinator-dependent-v1",
                        [left["id"], word["id"], right["id"]],
                    )

        root = word.get("radiko")
        if root not in ("ĉi", "ajn") or word.get("vortspeco") != "partiklo":
            continue
        neighbours = tokens[max(0, index - 1) : index]
        if root == "ĉi":
            neighbours += tokens[index + 1 : index + 2]
        candidates = [
            other
            for other in neighbours
            if other.get("vortspeco") == "korelativo"
            and (root == "ajn" or other.get("korelativo_prefikso") == "ti")
        ]
        if len(candidates) == 1:
            other = candidates[0]
            attach(
                word,
                other["id"],
                "advmod",
                "correlative-particle-v1",
                [word["id"], other["id"]],
            )
    return changes
