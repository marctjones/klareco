"""Conservative dependency refinements with explicit before/after evidence.

These rules consume final surface IDs. They do not infer semantic classes or
claim that the remaining attachment choices exhaust the language's ambiguity.
"""

from __future__ import annotations

from typing import TypedDict


_LIST_NOMINAL_CLASSES = frozenset(("substantivo", "propra_nomo"))
_LIST_COORDINATORS = frozenset(("kaj", "aŭ", "nek"))


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

    def same_inflection(left: dict, right: dict) -> bool:
        """Return whether the visible nominal agreement supports coordination."""
        return (
            left.get("kazo") == right.get("kazo")
            and left.get("nombro") == right.get("nombro")
        )

    def nearest_list_nominal(words: list[dict]) -> dict | None:
        return next(
            (
                candidate
                for candidate in reversed(words)
                if candidate.get("vortspeco") in _LIST_NOMINAL_CLASSES
            ),
            None,
        )

    # A comma-delimited nominal member followed by a coordinator is a list, not
    # a nominal modifier. Restrict the rule to one member between this comma and
    # its coordinator: longer enumerations require an explicit UD head-policy
    # decision (chain versus first-member anchoring), which surface grammar does
    # not settle.
    for comma_index, comma in enumerate(tokens):
        if comma.get("plena_vorto") != ",":
            continue
        prefix = tokens[comma_index + 1 : comma_index + 7]
        candidate = next(
            (
                word
                for word in prefix
                if word.get("vortspeco") in _LIST_NOMINAL_CLASSES
            ),
            None,
        )
        if candidate is None or candidate.get("rolo") != "nmod":
            continue
        candidate_index = tokens.index(candidate)
        if any(
            word.get("vortspeco") in {"verbo", "prepozicio", "konjunkcio"}
            for word in tokens[comma_index + 1 : candidate_index]
        ):
            continue
        tail = tokens[candidate_index + 1 : candidate_index + 10]
        coordinator_index = next(
            (
                candidate_index + 1 + offset
                for offset, word in enumerate(tail)
                if word.get("radiko") in _LIST_COORDINATORS
            ),
            None,
        )
        if coordinator_index is None:
            continue
        between = tokens[candidate_index + 1 : coordinator_index]
        if any(
            word.get("vortspeco") in {
                "verbo", "prepozicio", "konjunkcio", *_LIST_NOMINAL_CLASSES
            }
            for word in between
        ):
            continue
        final_member = next(
            (
                word
                for word in tokens[coordinator_index + 1 : coordinator_index + 5]
                if word.get("vortspeco") in _LIST_NOMINAL_CLASSES
            ),
            None,
        )
        predecessor_slice = tokens[max(0, comma_index - 7) : comma_index]
        predecessor = nearest_list_nominal(predecessor_slice)
        if (
            predecessor is None
            or final_member is None
            or not same_inflection(predecessor, candidate)
            or not same_inflection(candidate, final_member)
            or any(
                word.get("vortspeco") in {"verbo", "prepozicio", "konjunkcio"}
                for word in tokens[tokens.index(predecessor) + 1 : comma_index]
            )
        ):
            continue
        attach(
            candidate,
            predecessor["id"],
            "conj",
            "punctuated-nominal-enumeration-v1",
            [predecessor["id"], comma["id"], candidate["id"],
             tokens[coordinator_index]["id"], final_member["id"]],
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
