"""Validated dependency registry and derived clause/phrase views.

Token IDs are sentence-local surface positions. Dependencies are authoritative;
frames are projections and never another source of attachment decisions.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib

VERSION = 2
SUPPORTED_VERSIONS = (1, VERSION)
CLAUSE_RELATIONS = frozenset(
    ("root", "conj", "parataxis", "acl", "advcl", "ccomp", "xcomp", "csubj")
)
PHRASE_MODIFIERS = frozenset(
    ("amod", "nmod", "det", "nummod", "appos", "compound", "flat")
)


def _phrase_members(head_id: int, children: dict, predicates: set[int]) -> list[int]:
    members, pending = [], [head_id]
    while pending:
        token_id = pending.pop()
        members.append(token_id)
        pending.extend(
            c["id"]
            for c in children[token_id]
            if c["id"] not in predicates and c["rolo"] != "punct"
        )
    return sorted(members)


def validate_tokens(tokens: list[dict]) -> list[int]:
    if not isinstance(tokens, list) or any(
        not isinstance(w, dict) or "id" not in w for w in tokens
    ):
        raise ValueError("Syntax graph requires a token registry")
    if any(type(w["id"]) is not int or w["id"] < 1 for w in tokens):
        raise ValueError("Syntax graph requires unique positive token ids")
    registry = {w["id"]: w for w in tokens}
    if len(registry) != len(tokens):
        raise ValueError("Syntax graph requires unique positive token ids")
    for word in tokens:
        head = word.get("kapo")
        if type(head) is not int or (head != 0 and head not in registry):
            raise ValueError(f"Dangling dependency at token {word['id']}: {head}")
        if not isinstance(word.get("rolo"), str) or not word["rolo"]:
            raise ValueError(f"Missing dependency relation at token {word['id']}")
    # Each node is visited once even for deep or discontinuous dependency trees.
    complete = set()
    for word in tokens:
        seen = set()
        node = word["id"]
        while node and node not in complete:
            if node in seen:
                raise ValueError(f"Dependency cycle at token {node}")
            seen.add(node)
            node = registry[node]["kapo"]
        complete.update(seen)
    return [w["id"] for w in tokens if w["kapo"] == 0]


def project(ast: dict, original: str, normalized: str) -> dict:
    tokens = ast["vortoj"]
    roots = validate_tokens(tokens)
    registry = {w["id"]: w for w in tokens}
    children = {i: [] for i in registry}
    for word in tokens:
        if word["kapo"]:
            children[word["kapo"]].append(word)
    predicates = {
        w["id"]
        for w in tokens
        if w["rolo"] in CLAUSE_RELATIONS
        and (
            w.get("vortspeco") == "verbo"
            or any(c["rolo"] in ("cop", "aux") for c in children[w["id"]])
        )
    }

    owners = {0: None, **{i: i for i in predicates}}

    def owner(i):
        path = []
        while i not in owners:
            path.append(i)
            i = registry[i]["kapo"]
        for token_id in path:
            owners[token_id] = owners[i]
        return owners[i]

    membership = {pid: [] for pid in predicates}
    for word in tokens:
        pid = owner(word["id"])
        if pid is not None:
            membership[pid].append(word["id"])

    phrases = {}

    def phrase(word):
        if word["id"] in phrases:
            return phrases[word["id"]]
        modifiers = [c for c in children[word["id"]] if c["rolo"] in PHRASE_MODIFIERS]
        group = {
            "tipo": "vortgrupo",
            "kerno": word,
            "head_id": word["id"],
            "token_ids": _phrase_members(word["id"], children, predicates),
            "modifier_ids": [c["id"] for c in modifiers],
            "case_marker_ids": [
                c["id"] for c in children[word["id"]] if c["rolo"] == "case"
            ],
            "priskriboj": [c for c in modifiers if c.get("vortspeco") != "artikolo"],
        }
        phrases[word["id"]] = group
        articles = [c for c in modifiers if c.get("vortspeco") == "artikolo"]
        if articles:
            group["artikolo"] = articles[0]["plena_vorto"].lower()
        return group

    frames = []
    for pid in sorted(predicates):
        pred = registry[pid]
        dependents = children[pid]
        verbs = [c for c in dependents if c["rolo"] in ("cop", "aux")]
        verb = verbs[0] if verbs else pred
        arguments = {
            role: [phrase(c) for c in dependents if c["rolo"].split(":")[0] == role]
            for role in ("nsubj", "obj", "iobj", "csubj")
        }
        others = []
        for child in dependents:
            if child["rolo"].split(":")[0] in (
                "nsubj",
                "obj",
                "iobj",
                "csubj",
                "cop",
                "aux",
                "punct",
            ):
                continue
            if child["id"] in predicates:
                continue
            cases = [c for c in children[child["id"]] if c["rolo"] == "case"]
            if cases:
                others.extend(cases)
                others.append(child)
            else:
                others.append(child)
        if pred is not verb:
            others.insert(0, pred)
        members = membership[pid]
        frames.append(
            {
                "tipo": "propozicio",
                "predikato": pred,
                "verbo": verb,
                "subjekto": next(iter(arguments["nsubj"]), None),
                "objekto": next(iter(arguments["obj"]), None),
                "argumentoj": arguments,
                "aliaj": others,
                "rolo": {"root": "ĉefa", "conj": "kunordigita", "acl": "rilativa"}.get(
                    pred["rolo"], "subordigita"
                ),
                "fonto": "regulo",
                "derivation": "dependency-projection-v1",
                "token_ids": members,
                "parent_predicate_id": owner(pred["kapo"]),
                "complement_predicate_ids": [
                    c["id"]
                    for c in dependents
                    if c["id"] in predicates
                    and c["rolo"] in ("ccomp", "xcomp", "csubj")
                ],
                "negita": any(
                    registry[i].get("radiko") == "ne"
                    or (registry[i].get("radiko") or "").startswith("neni")
                    for i in members
                ),
            }
        )
    for frame in frames:
        pred = frame["predikato"]
        if pred["rolo"] == "acl" and pred["kapo"]:
            relative = dict(frame, tipo="rilata_subfrazo")
            relative["rilata_pronomo"] = next(
                (
                    registry[i]
                    for i in frame["token_ids"]
                    if registry[i].get("radiko") in ("kiu", "kio")
                ),
                None,
            )
            phrase(registry[pred["kapo"]])["priskriboj"].append(relative)
    ast["propozicioj"] = frames
    main = next((f for f in frames if f["predikato"]["kapo"] == 0), None)
    if main:
        for field in ("subjekto", "verbo", "objekto", "aliaj", "negita"):
            ast[field] = main[field]
    ast["syntax"] = {
        "version": VERSION,
        "authority": "vortoj.kapo/rolo",
        "roots": roots,
        "status": "forest" if len(roots) > 1 else "tree" if roots else "empty",
        "unassigned_token_ids": [w["id"] for w in tokens if owner(w["id"]) is None],
        "alternatives": [],
        "alternatives_status": "not_enumerated",
        "trace_coverage": "refinements_only",
        "attachment_candidates": [],
    }
    for word in tokens:
        if not word.get("alligo_opcioj"):
            continue
        options = []
        for option in word["alligo_opcioj"]:
            head = registry.get(option["kapo"])
            relation = (
                "root"
                if not option["kapo"]
                else "obl" if head and head.get("vortspeco") == "verbo" else "nmod"
            )
            option["rolo"] = relation
            edge = {"head_id": option["kapo"], "relation": relation}
            if edge not in options:
                options.append(edge)
        selected = {"head_id": word["kapo"], "relation": word["rolo"]}
        if selected not in options:
            options.append(selected)
        ast["syntax"]["attachment_candidates"].append(
            {
                "token_id": word["id"],
                "selected": selected,
                "options": options,
                "status": "unresolved",
                "completeness": "partial",
                "generator": "pp-candidates-v1",
                "selection_rule": "nominal-proximity-v1",
            }
        )
    for word in tokens:
        if word.get("vortspeco") in ("substantivo", "propra_nomo", "pronomo"):
            phrase(word)
    ast["phrases"] = [phrases[i] for i in sorted(phrases)]
    ast["source"] = {
        "original": original,
        "normalized": normalized,
        "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "offset_unit": "unicode_codepoint",
        "span_convention": "half_open",
    }
    # Spans index normalized text; original text is retained verbatim rather than
    # pretending normalization preserves character offsets.
    offsets = {}
    for tag, a, b, c, d in SequenceMatcher(
        None, original, normalized, autojunk=False
    ).get_opcodes():
        for j in range(c, d):
            offsets[j] = (a + j - c, a + j - c + 1) if tag == "equal" else (a, b)
    folded = ""
    folded_offsets = []
    for index, char in enumerate(normalized):
        folded += char.lower()
        folded_offsets.extend([index] * len(char.lower()))
    cursor = 0
    for word in tokens:
        form = word["plena_vorto"]
        folded_start = folded.find(form.lower(), cursor)
        start = folded_offsets[folded_start] if folded_start >= 0 else -1
        end = (
            folded_offsets[folded_start + len(form.lower()) - 1] + 1
            if start >= 0
            else -1
        )
        word["normalized_span"] = [start, end] if start >= 0 else None
        word["original_span"] = None
        if start >= 0:
            cursor = folded_start + len(form.lower())
            word["surface_form"] = normalized[start:end]
            word["original_span"] = [offsets[start][0], offsets[end - 1][1]]
    return ast


def main_clause_tokens(ast: dict) -> list[dict]:
    """Surface tokens of the main clause, excluding embedded interrogatives."""
    main = next(
        (
            frame
            for frame in ast.get("propozicioj", [])
            if frame.get("predikato", {}).get("kapo") == 0
        ),
        None,
    )
    ids = set(main["token_ids"]) if main else None
    return [word for word in ast.get("vortoj", []) if ids is None or word["id"] in ids]


def validate_ast(ast: dict) -> None:
    """Check versioned graph and projection consistency at storage boundaries."""
    try:
        _validate_ast(ast)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed AST structure: {exc}") from exc


def _validate_ast(ast: dict) -> None:
    if "syntax" not in ast:
        return
    if (
        type(ast["syntax"].get("version")) is not int
        or ast["syntax"]["version"] not in SUPPORTED_VERSIONS
    ):
        raise ValueError("Unsupported syntax graph version")
    roots = validate_tokens(ast["vortoj"])
    if roots != ast["syntax"].get("roots"):
        raise ValueError("Syntax root registry disagrees with dependencies")
    registry = {w["id"]: w for w in ast["vortoj"]}
    owners = {}
    for frame in ast["propozicioj"]:
        pred = frame["predikato"]
        if registry.get(pred.get("id")) != pred:
            raise ValueError("Clause predicate differs from its registry token")
        for token_id in frame["token_ids"]:
            if token_id not in registry or token_id in owners:
                raise ValueError("Invalid or duplicate clause membership")
            owners[token_id] = pred["id"]
        for slot, role in [("subjekto", "nsubj"), ("objekto", "obj")]:
            if frame.get(slot):
                token = frame[slot]["kerno"]
                if (
                    registry.get(token.get("id")) != token
                    or token["kapo"] != pred["id"]
                    or token["rolo"].split(":")[0] != role
                ):
                    raise ValueError("Clause argument contradicts dependencies")
    predicates = {f["predikato"]["id"] for f in ast["propozicioj"]}

    def owner(token_id):
        while token_id and token_id not in predicates:
            token_id = registry[token_id]["kapo"]
        return token_id or None

    for token_id in registry:
        if owners.get(token_id) != owner(token_id):
            raise ValueError("Clause membership contradicts dependencies")
    for frame in ast["propozicioj"]:
        if frame.get("parent_predicate_id") != owner(frame["predikato"]["kapo"]):
            raise ValueError("Clause parent contradicts dependencies")
    if ast["syntax"].get("unassigned_token_ids") != [
        i for i in registry if i not in owners
    ]:
        raise ValueError("Unassigned token registry contradicts clause membership")
    main = next((f for f in ast["propozicioj"] if f["predikato"]["kapo"] == 0), None)
    if main:
        for slot in ("subjekto", "verbo", "objekto", "aliaj", "negita"):
            if ast.get(slot) != main.get(slot):
                raise ValueError("Sentence frame contradicts its main clause")
    if ast["syntax"]["version"] == VERSION:
        _validate_v2(ast, registry, roots)


def _validate_v2(ast: dict, registry: dict, roots: list[int]) -> None:
    """Validate new contracts without asserting them for historical v1 blobs."""
    source = ast.get("source")
    if (
        not isinstance(source, dict)
        or not all(isinstance(source.get(k), str) for k in ("original", "normalized"))
        or source.get("sha256")
        != hashlib.sha256(source["original"].encode("utf-8")).hexdigest()
        or source.get("offset_unit") != "unicode_codepoint"
        or source.get("span_convention") != "half_open"
    ):
        raise ValueError("Invalid source provenance or offset convention")
    expected_status = "forest" if len(roots) > 1 else "tree" if roots else "empty"
    if (
        ast["syntax"].get("authority") != "vortoj.kapo/rolo"
        or ast["syntax"].get("status") != expected_status
    ):
        raise ValueError("Syntax status or authority contradicts dependencies")
    children = {i: [] for i in registry}
    last_end = 0
    for word in registry.values():
        _validate_morphemes(word)
        if (word["kapo"] == 0) != (word["rolo"] == "root"):
            raise ValueError("Root head and relation disagree")
        if word["kapo"]:
            children[word["kapo"]].append(word)
        span = word.get("normalized_span")
        original_span = word.get("original_span")
        for candidate, name in [(span, "normalized"), (original_span, "original")]:
            if candidate is not None and (
                not isinstance(candidate, list)
                or len(candidate) != 2
                or any(type(i) is not int for i in candidate)
                or not 0 <= candidate[0] < candidate[1] <= len(source[name])
            ):
                raise ValueError("Invalid source token span")
        if span is not None:
            if (
                span[0] < last_end
                or source["normalized"][slice(*span)] != word.get("surface_form")
                or word["surface_form"].lower() != word["plena_vorto"].lower()
            ):
                raise ValueError("Token span contradicts surface text or token order")
            last_end = span[1]

    expected_predicates = {
        i
        for i, w in registry.items()
        if w["rolo"] in CLAUSE_RELATIONS
        and (
            w.get("vortspeco") == "verbo"
            or any(c["rolo"] in ("cop", "aux") for c in children[i])
        )
    }
    frames = ast["propozicioj"]
    if [f["predikato"]["id"] for f in frames] != sorted(expected_predicates):
        raise ValueError("Missing, duplicate, or unexpected predicate frames")
    phrase_registry = {}
    for group in ast["phrases"]:
        token_id = group["head_id"]
        if (
            type(token_id) is not int
            or token_id not in registry
            or token_id in phrase_registry
            or group["kerno"] != registry[token_id]
            or group["token_ids"]
            != _phrase_members(token_id, children, expected_predicates)
            or group["modifier_ids"]
            != [w["id"] for w in children[token_id] if w["rolo"] in PHRASE_MODIFIERS]
            or group["case_marker_ids"]
            != [w["id"] for w in children[token_id] if w["rolo"] == "case"]
        ):
            raise ValueError("Phrase registry contradicts dependencies")
        phrase_registry[token_id] = group
    for frame in frames:
        pid = frame["predikato"]["id"]
        arguments = frame.get("argumentoj")
        if not isinstance(arguments, dict) or set(arguments) != {
            "nsubj",
            "obj",
            "iobj",
            "csubj",
        }:
            raise ValueError("Invalid clause argument registry")
        for role, groups in arguments.items():
            expected = [w for w in children[pid] if w["rolo"].split(":")[0] == role]
            if (
                not isinstance(groups, list)
                or [g.get("kerno") for g in groups if isinstance(g, dict)] != expected
                or any(g != phrase_registry.get(g["kerno"]["id"]) for g in groups)
            ):
                raise ValueError("Clause argument registry contradicts dependencies")
        for slot, role in [("subjekto", "nsubj"), ("objekto", "obj")]:
            if frame[slot] != next(iter(arguments[role]), None):
                raise ValueError("Clause slot contradicts argument registry")
        verbs = [w for w in children[pid] if w["rolo"] in ("cop", "aux")]
        if frame["verbo"] != (verbs[0] if verbs else registry[pid]):
            raise ValueError("Clause verb contradicts dependency predicate")
        if frame.get("complement_predicate_ids") != [
            w["id"]
            for w in children[pid]
            if w["id"] in expected_predicates
            and w["rolo"] in ("ccomp", "xcomp", "csubj")
        ]:
            raise ValueError("Clause complement registry contradicts dependencies")

    def edge_ok(token_id, edge):
        if (
            not isinstance(edge, dict)
            or set(edge) != {"head_id", "relation"}
            or type(edge["head_id"]) is not int
            or edge["head_id"] not in registry.keys() | {0}
            or edge["head_id"] == token_id
            or not isinstance(edge["relation"], str)
            or not edge["relation"]
            or (edge["head_id"] == 0) != (edge["relation"] == "root")
        ):
            raise ValueError("Invalid attachment edge reference")

    candidates = ast["syntax"].get("attachment_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Missing attachment candidate registry")
    if (
        ast["syntax"].get("trace_coverage") != "refinements_only"
        or ast["syntax"].get("alternatives_status") != "not_enumerated"
        or ast["syntax"].get("alternatives") != []
    ):
        raise ValueError("Unsupported trace or joint-alternative coverage")
    ids = set()
    for candidate in candidates:
        token_id = candidate.get("token_id")
        if type(token_id) is not int or token_id not in registry or token_id in ids:
            raise ValueError("Invalid or duplicate attachment candidate token")
        ids.add(token_id)
        word = registry[token_id]
        if (
            candidate.get("selected")
            != {"head_id": word["kapo"], "relation": word["rolo"]}
            or not isinstance(candidate.get("options"), list)
            or candidate["selected"] not in candidate["options"]
        ):
            raise ValueError("Attachment selection contradicts dependencies")
        expected_options = [
            {"head_id": option["kapo"], "relation": option["rolo"]}
            for option in word.get("alligo_opcioj", [])
        ]
        expected_options = [
            option
            for index, option in enumerate(expected_options)
            if option not in expected_options[:index]
        ]
        if candidate["selected"] not in expected_options:
            expected_options.append(candidate["selected"])
        if (
            candidate["options"] != expected_options
            or candidate.get("status") != "unresolved"
            or candidate.get("completeness") != "partial"
            or candidate.get("generator") != "pp-candidates-v1"
            or candidate.get("selection_rule") != "nominal-proximity-v1"
        ):
            raise ValueError(
                "Attachment candidates contradict their recorded generator"
            )
        for edge in candidate["options"]:
            edge_ok(token_id, edge)
            seen = {token_id}
            node = edge["head_id"]
            while node:
                if node in seen:
                    raise ValueError(
                        "Attachment alternative creates a dependency cycle"
                    )
                seen.add(node)
                node = registry[node]["kapo"]
    if ids != {i for i, w in registry.items() if w.get("alligo_opcioj")}:
        raise ValueError("Attachment candidate registry omitted token alternatives")
    current = {
        i: {"head_id": w["kapo"], "relation": w["rolo"]} for i, w in registry.items()
    }
    for change in reversed(ast.get("attachment_trace", [])):
        token_id = change.get("token_id")
        if (
            type(token_id) is not int
            or token_id not in registry
            or current[token_id] != change.get("after")
            or not isinstance(change.get("rule"), str)
            or not change["rule"]
            or not isinstance(change.get("evidence_token_ids"), list)
            or any(
                type(i) is not int or i not in registry
                for i in change["evidence_token_ids"]
            )
        ):
            raise ValueError("Attachment trace contradicts dependencies or evidence")
        edge_ok(token_id, change["after"])
        # The previous stage may have emitted an inconsistent root relation;
        # root-relation-v1 explicitly documents its repair.
        before = change.get("before")
        if change["rule"] != "root-relation-v1":
            edge_ok(token_id, before)
        elif (
            not isinstance(before, dict)
            or set(before) != {"head_id", "relation"}
            or before["head_id"] != 0
            or not isinstance(before["relation"], str)
        ):
            raise ValueError("Invalid root repair trace")
        current[token_id] = before


def _validate_morphemes(word: dict) -> None:
    def surface(parts):
        if not isinstance(parts, list) or not parts:
            raise ValueError("Missing structured morphemes")
        for part in parts:
            if (
                not isinstance(part, dict)
                or set(part) != {"form", "kind", "pos"}
                or not isinstance(part["form"], str)
                or part["kind"]
                not in ("radiko", "prefikso", "sufikso", "finaĵo", "kunmeto")
                or not (part["pos"] is None or isinstance(part["pos"], str))
            ):
                raise ValueError("Invalid structured morpheme")
        return "".join(p["form"] for p in parts)

    if "morfemoj" in word and surface(word["morfemoj"]) != word.get("morfologia_formo"):
        raise ValueError("Morphemes do not reconstruct their analyzed form")
    choices = word.get("alternativoj", {})
    if "version" not in choices:
        return  # Earlier token analyses did not carry the complete candidates.
    if type(choices["version"]) is not int or choices["version"] != 1:
        raise ValueError("Unsupported morphology candidate version")
    options = choices.get("opcioj")
    applied = choices.get("aplikita")
    if (
        not isinstance(options, list)
        or type(applied) is not int
        or not 0 <= applied < len(options)
    ):
        raise ValueError("Invalid applied morphology candidate")
    selected = choices.get("elektita")
    if selected is not None and (
        type(selected) is not int or not 0 <= selected < len(options)
    ):
        raise ValueError("Invalid selected morphology candidate")
    if (
        choices.get("nivelo") != "morfemo"
        or choices.get("selection_policy") != "lexical-types-and-morpheme-cost-v1"
        or choices.get("completeness") != "bounded_lexicon_search"
        or choices.get("selection_status")
        != ("unresolved" if selected is None else "heuristic")
        or (selected is not None and selected != applied)
    ):
        raise ValueError("Morphology candidate selection metadata is inconsistent")
    for option in options:
        if surface(option["morfemoj"]) != option["surface"] or option[
            "surface"
        ] != word.get("morfologia_formo"):
            raise ValueError("Morphology candidate changed its analyzed form")
    if options[applied]["morfemoj"] != word.get("morfemoj"):
        raise ValueError("Applied morphology differs from its candidate")
