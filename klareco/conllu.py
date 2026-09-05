"""Serialize the selected dependency graph as CoNLL-U.

Native word classes remain in MISC; UPOS is an explicit scheme projection.
The serializer never repairs heads or chooses a different root. A forest can be
exported for diagnostics, or rejected by the strict single-tree export gate.
"""

from __future__ import annotations

from klareco.parser import parse

# --- vortspeco -> UPOS ------------------------------------------------------
_UPOS = {
    # Punctuation is SYNTAX, not typography: UD gold has 454 PUNCT tokens and
    # every one of them carries a HEAD. We used to delete them at tokenization.
    "interpunkcio": "PUNCT",
    "substantivo": "NOUN",
    "propra_nomo": "PROPN",
    "verbo": "VERB",
    "adjektivo": "ADJ",
    "adverbo": "ADV",
    "pronomo": "PRON",
    "prepozicio": "ADP",
    "konjunkcio": "CCONJ",
    "artikolo": "DET",
    "numero": "NUM",
    "partiklo": "PART",
    "interjekcio": "INTJ",
    "nekonata": "X",
    "fremda_vorto": "X",
}

# UD splits the correlative table across three tags by its SUFFIX. Esperanto does
# not — it is one paradigm — but we emit UD's view so the score is comparable.
_KORELATIVO_UPOS = {
    "u": "DET",  # kiu, tiu, ĉiu  — "which individual"
    "a": "DET",  # kia, tia       — "of which kind"
    "o": "PRON",  # kio, tio       — "which thing"
    "e": "ADV",  # kie, tie       — "where"
    "am": "ADV",  # kiam, tiam     — "when"
    "al": "ADV",  # kial, tial     — "why"
    "el": "ADV",  # kiel, tiel     — "how"
    "om": "ADV",  # kiom, tiom     — "how much"
    "es": "DET",  # kies, ties     — "whose"
}

# `estas` is a copula in UD, not a full verb.
_COPULA_ROOTS = {"est"}

# Subordinating vs coordinating — UD splits these; Esperanto's `konjunkcio` does not.
_SCONJ = {"ke", "ĉar", "se", "kvankam", "dum", "ĝis", "apenaŭ", "kvazaŭ", "ol"}


def upos(w: dict) -> str:
    vs = w.get("vortspeco")
    if vs == "korelativo":
        suffix = w.get("korelativo_sufikso") or ""
        # The suffix table gives the default UD projection, but -u/-a/-es
        # forms are DET only when they modify a nominal (`ĉiu lingvo`).  Once
        # dependency analysis has assigned a nominal role (`kiu` as nsubj,
        # `kies` as nmod), the same forms are pronouns.  Use the selected role
        # rather than surface morphology alone so POS reflects syntax.
        if suffix in ("u", "a", "es") and w.get("rolo") not in ("det", "amod"):
            return "PRON"
        return _KORELATIVO_UPOS.get(suffix, "PRON")
    if vs == "konjunkcio" and (w.get("radiko") or "").lower() in _SCONJ:
        return "SCONJ"
    if vs == "verbo" and (w.get("radiko") or "").lower() in _COPULA_ROOTS:
        return "AUX"
    return _UPOS.get(vs, "X")


# Only NOMINALS inflect for case and number. The parser fills `kazo`/`nombro` on
# every node with a default, so emitting them unconditionally puts
# `Case=Nom|Number=Sing` on a VERB — which is not wrong so much as meaningless,
# and UD would count it against us.
_NOMINAL = {
    "substantivo",
    "propra_nomo",
    "adjektivo",
    "pronomo",
    "korelativo",
    "numero",
    "artikolo",
}


def feats(w: dict) -> str:
    """Esperanto marks case, number and tense ON THE SURFACE. This is free —
    English parsers spend real effort recovering what `-n` and `-j` just say."""
    f = []
    if w.get("vortspeco") in _NOMINAL:
        if w.get("kazo") == "akuzativo":
            f.append("Case=Acc")
        elif w.get("kazo") == "nominativo":
            f.append("Case=Nom")
        if w.get("nombro") == "pluralo":
            f.append("Number=Plur")
        elif w.get("nombro") == "singularo":
            f.append("Number=Sing")
    if w.get("vortspeco") == "verbo":
        t = {
            "prezenco": "Tense=Pres",
            "preterito": "Tense=Past",
            "futuro": "Tense=Fut",
        }.get(w.get("tempo") or "")
        if t:
            f.append(t)
        mood = {"infinitivo": None, "kondicionalo": "Cnd", "imperativo": "Imp"}.get(
            w.get("modo")
        )
        if w.get("modo") == "infinitivo":
            f.append("VerbForm=Inf")
        elif t or mood:
            f.extend(["VerbForm=Fin", f'Mood={mood or "Ind"}'])
    return "|".join(sorted(f)) or "_"


def _kern(node):
    if not isinstance(node, dict):
        return None
    return node.get("kerno", node)


def ast_to_conllu(ast: dict, sent_id: str = "1", *, strict: bool = False) -> str:
    """Export an existing expanded AST without reparsing or changing its analysis.

    Strict mode requires a single complete dependency tree. Diagnostic mode keeps
    forests explicit; multiple-root output is not a valid UD gold tree.
    """
    from .syntax_graph import validate_ast, validate_tokens

    validate_ast(ast)
    ordered = ast.get("vortoj", [])
    roots = validate_tokens(ordered)
    if (
        not isinstance(sent_id, str)
        or not sent_id
        or any(c in sent_id for c in "\r\n\t")
    ):
        raise ValueError("CoNLL-U sentence id must be a single nonempty field")
    if [w["id"] for w in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("CoNLL-U export requires contiguous surface token IDs")
    if any((w["kapo"] == 0) != (w["rolo"] == "root") for w in ordered):
        raise ValueError("CoNLL-U root relation disagrees with the AST head")
    if strict and (
        len(roots) != 1 or any(w.get("normalized_span") is None for w in ordered)
    ):
        raise ValueError(
            "Strict CoNLL-U export requires one complete tree with aligned spans"
        )

    def field(value):
        value = str(value)
        if any(c in value for c in "\r\n\t"):
            raise ValueError("CoNLL-U field contains a line break or tab")
        return value or "_"

    source = ast.get("source", {})
    text = source.get("normalized", " ".join(w["plena_vorto"] for w in ordered))
    # A CoNLL-U comment is one line; offsets still refer to the stored source.
    comment_text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    status = "tree" if len(roots) == 1 else "forest" if roots else "empty"
    lines = [
        f"# sent_id = {sent_id}",
        f"# text = {comment_text}",
        f"# parse_status = {status}",
    ]
    for index, word in enumerate(ordered):
        misc = [f"Vortspeco={field(word.get('vortspeco', '_'))}"]
        span = word.get("normalized_span")
        if span is not None:
            misc.extend([f"StartChar={span[0]}", f"EndChar={span[1]}"])
            following = (
                ordered[index + 1].get("normalized_span")
                if index + 1 < len(ordered)
                else None
            )
            if following is not None and span[1] == following[0]:
                misc.append("SpaceAfter=No")
        lines.append(
            "\t".join(
                [
                    str(word["id"]),
                    field(word.get("surface_form", word["plena_vorto"])),
                    field(word.get("radiko") or "_"),
                    upos(word),
                    "_",
                    feats(word),
                    str(word["kapo"]),
                    field(word["rolo"]),
                    "_",
                    "|".join(sorted(misc)),
                ]
            )
        )
    return "\n".join(lines) + "\n\n"


def to_conllu(text: str, sent_id: str = "1", *, strict: bool = False) -> str:
    """Parse once, then serialize exactly the selected dependency graph."""
    return ast_to_conllu(parse(text), sent_id, strict=strict)
