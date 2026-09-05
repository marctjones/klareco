"""Validated dependency registry and derived clause/phrase views.

Token IDs are sentence-local surface positions. Dependencies are authoritative;
frames are projections and never another source of attachment decisions.
"""

from __future__ import annotations

from difflib import SequenceMatcher

VERSION = 1
CLAUSE_RELATIONS = frozenset(
    ("root", "conj", "parataxis", "acl", "advcl", "ccomp", "xcomp", "csubj")
)


def validate_tokens(tokens: list[dict]) -> list[int]:
    registry = {w["id"]: w for w in tokens}
    if len(registry) != len(tokens) or any(
        type(i) is not int or i < 1 for i in registry
    ):
        raise ValueError("Syntax graph requires unique positive token ids")
    for word in tokens:
        head = word.get("kapo")
        if type(head) is not int or (head != 0 and head not in registry):
            raise ValueError(f"Dangling dependency at token {word['id']}: {head}")
        if not isinstance(word.get("rolo"), str) or not word["rolo"]:
            raise ValueError(f"Missing dependency relation at token {word['id']}")
        seen = set()
        node = word["id"]
        while node:
            if node in seen:
                raise ValueError(f"Dependency cycle at token {node}")
            seen.add(node)
            node = registry[node]["kapo"]
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

    def owner(i):
        while i and i not in predicates:
            i = registry[i]["kapo"]
        return i or None

    phrases = {}

    def phrase(word):
        if word["id"] in phrases:
            return phrases[word["id"]]
        modifiers = [
            c
            for c in children[word["id"]]
            if c["rolo"]
            in ("amod", "nmod", "det", "nummod", "appos", "compound", "flat")
        ]
        group = {
            "tipo": "vortgrupo",
            "kerno": word,
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
        members = [w["id"] for w in tokens if owner(w["id"]) == pid]
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
                "negita": any(
                    w.get("radiko") == "ne" or w.get("radiko", "").startswith("neni")
                    for w in tokens
                    if w["id"] in members
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
    }
    ast["phrases"] = [
        phrase(w)
        for w in tokens
        if w.get("vortspeco") in ("substantivo", "propra_nomo", "pronomo")
    ]
    ast["source"] = {"original": original, "normalized": normalized}
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
    if "syntax" not in ast:
        return
    if (
        type(ast["syntax"].get("version")) is not int
        or ast["syntax"]["version"] != VERSION
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
    if ast["syntax"].get("unassigned_token_ids") != [i for i in registry if i not in owners]:
        raise ValueError("Unassigned token registry contradicts clause membership")
    main = next((f for f in ast["propozicioj"] if f["predikato"]["kapo"] == 0), None)
    if main:
        for slot in ("subjekto", "verbo", "objekto", "aliaj", "negita"):
            if ast.get(slot) != main.get(slot):
                raise ValueError("Sentence frame contradicts its main clause")
