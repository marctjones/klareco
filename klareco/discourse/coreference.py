"""Deterministic candidate generation for cross-sentence pronoun reference.

VISION.md lists "cross-sentence coreference" as a suspected residue and
says it has never been tested. This module is that test, at the smallest
scope that can be measured: does the deterministic information already in
the corpus (sentence order within a document, grammatical number, the
thin entity-type ontology, morphological affixes) resolve a 3rd-person
pronoun's antecedent, and when it doesn't, how far off is it?

Scope, deliberately narrow:
- Esperanto's PERSONAL pronouns li/ŝi/ĝi/ili only. Reflexive `si` is
  EXCLUDED on purpose: it is bound to the subject of its OWN clause by
  grammar, never a cross-sentence reference, so including it would not be
  testing coreference at all. Demonstratives (tiu/tio) are a distinct
  closed class and out of scope for this first pass.
- A candidate antecedent is any NOMINAL (substantivo/propra_nomo) token in
  one of the `window` sentences immediately preceding the pronoun's own
  sentence, in the same document (matched by DB `article_id`, not by the
  parser -- the parser has no concept of "document").
- Compatibility is grammatical NUMBER (always reliable in Esperanto: an
  explicit, unambiguous morphological marker) plus an ontology-informed,
  NON-EXCLUSIONARY animacy check. The entity-type ontology
  (`ontology_edges` WHERE rel='HAVAS_ENTECAN_TIPON') covers 2,337 roots and
  is hand-seeded and thin (CLAUDE.md) -- an UNCLASSIFIED root is never
  treated as evidence against a candidate, only a classified one is.

Ranking preference (nearer sentence, then subject role over object role)
follows Centering Theory (Grosz, Joshi & Weinstein 1995; Brennan, Friedman
& Pollard 1987): the current utterance's subject is the single strongest
predictor of what a following pronoun refers to. This module does not
implement full Centering Theory -- no discourse-segment tracking, no
transition typing -- it borrows exactly the one, well-established ranking
signal that is cheap and deterministic here.

This is NOT wired into the default orchestrator pipeline. Per the
project's contract, a new capability stays default-OFF until it passes
the contract suite and carries a measured number.
"""
from __future__ import annotations

from typing import Any, Optional

# Esperanto's closed personal-pronoun set and the grammatical number each
# one requires of its antecedent. `si` (reflexive) is deliberately absent.
_PRONOUN_NUMBER: dict[str, str] = {
    "li": "singularo",
    "ŝi": "singularo",
    "ĝi": "singularo",
    "ili": "pluralo",
}

_NOMINAL_VORTSPECOJ = ("substantivo", "propra_nomo")

# Roles that make a nominal the SUBJECT of its clause, for the Centering-
# Theory-derived ranking preference. Matches the relation labels this
# parser actually assigns (see klareco/parser.py's UD-style `rolo` values).
# NOTE: 'root' is deliberately EXCLUDED. In a copular sentence ("Kongō Gumi
# estis ... firmao"), 'root' is the PREDICATE NOMINAL ('firmao'), not the
# subject -- the real subject is still 'nsubj' ('Kongō'). Treating 'root'
# as subject-preferred would rank a predicate above the actual subject.
_SUBJECT_ROLES = frozenset({"nsubj", "csubj"})

MODULE_NAME = "klareco.discourse.coreference"
MODULE_VERSION = "1"


def pronoun_radiko(word: dict) -> Optional[str]:
    """Return 'li'/'ŝi'/'ĝi'/'ili' if `word` is one of those, else None."""
    if not isinstance(word, dict):
        return None
    radiko = (word.get("radiko") or "").lower()
    return radiko if radiko in _PRONOUN_NUMBER else None


def _iter_slot_words(ast: dict):
    """Yield every word-shaped node in subjekto/objekto/aliaj -- the slots
    a pronoun can occupy. Mirrors klareco.dialog.state's `_kerno` unwrap."""
    for role in ("subjekto", "objekto"):
        node = ast.get(role)
        if isinstance(node, dict):
            yield (role, node.get("kerno") if node.get("tipo") == "vortgrupo" else node)
    for item in ast.get("aliaj") or []:
        if isinstance(item, dict):
            yield ("aliaj", item.get("kerno") if item.get("tipo") == "vortgrupo" else item)


def find_pronoun_mentions(sid: int, ast: dict) -> list[dict]:
    """Every li/ŝi/ĝi/ili occurrence in one sentence, as a mention record."""
    mentions = []
    for role, w in _iter_slot_words(ast):
        radiko = pronoun_radiko(w)
        if radiko is None:
            continue
        mentions.append({
            "sid": sid,
            "token_id": w.get("id"),
            "pronoun": radiko,
            "surface": w.get("plena_vorto"),
            "role": role,
        })
    return mentions


def _nominal_candidates(sid: int, ast: dict) -> list[dict]:
    """Every substantivo/propra_nomo token in one sentence's AST, PLUS one
    synthetic PLURAL candidate per coordinated nominal chain ("Maria kaj
    Petro" -> one group candidate with nombro='pluralo'), so 'ili' can
    match a coordinated pair of singular names -- the single most common
    source of a plural antecedent in ordinary text. Coordination is read
    directly off the dependency tree (rolo='conj' pointing at another
    nominal's head), not re-detected with a separate heuristic."""
    tokens = {w["id"]: w for w in ast.get("vortoj", []) if isinstance(w, dict)}
    out = []
    conj_children: dict[int, list[dict]] = {}
    for w in tokens.values():
        if w.get("vortspeco") not in _NOMINAL_VORTSPECOJ:
            continue
        out.append({
            "sid": sid,
            "token_id": w.get("id"),
            "radiko": w.get("radiko"),
            "surface": w.get("plena_vorto"),
            "nombro": w.get("nombro"),
            "rolo": w.get("rolo"),
            "vortspeco": w.get("vortspeco"),
            "sufiksoj": w.get("sufiksoj") or [],
        })
        if w.get("rolo") == "conj":
            head = tokens.get(w.get("kapo"))
            if isinstance(head, dict) and head.get("vortspeco") in _NOMINAL_VORTSPECOJ:
                conj_children.setdefault(head["id"], []).append(w)
    for head_id, children in conj_children.items():
        members = [tokens[head_id]] + children
        out.append({
            "sid": sid,
            "token_id": head_id,
            "radiko": "+".join((m.get("radiko") or "") for m in members),
            "surface": " kaj ".join((m.get("plena_vorto") or "") for m in members),
            "nombro": "pluralo",
            "rolo": tokens[head_id].get("rolo"),
            "vortspeco": "grupo_kunordigita",
            "sufiksoj": [],
            "member_token_ids": [m["id"] for m in members],
        })
    return out


def _animacy_class(candidate: dict, entity_types: dict[str, str]) -> Optional[str]:
    """'persono' / 'non_person' / None (unclassified -- the ontology is
    thin; silence is never treated as evidence)."""
    cls = entity_types.get((candidate.get("radiko") or "").lower())
    if cls == "persono":
        return "persono"
    if cls in ("loko", "besto", "planto", "malsano", "lingvo"):
        return "non_person"
    return None


def is_compatible(pronoun: str, candidate: dict, entity_types: dict[str, str]) -> bool:
    """NUMBER agreement is required (a hard, always-marked Esperanto
    signal). Animacy agreement for li/ŝi/ĝi is ontology-informed.

    The default direction differs by candidate kind, on purpose:
    - PROPER NOUNS default to ELIGIBLE for li/ŝi when unclassified: most
      capitalized names in this corpus denote a person, place, or
      organization, and there is no better default for a name specifically.
    - COMMON NOUNS default to INELIGIBLE for li/ŝi when unclassified: the
      ordinary vocabulary is overwhelmingly non-human referents (objects,
      places, abstract concepts), so treating "the ontology has no entry"
      as "assume it's a person" would manufacture false positives (an
      earlier version of this rule let 'ĝi'-only nouns like 'matematiko'
      compete for 'ŝi' purely because the thin ontology never classified
      them). A common noun needs a POSITIVE 'persono' classification to be
      eligible for li/ŝi -- this is a real, named residue (the ontology's
      entity-type coverage of ordinary vocabulary is thin), not a silent
      guess either way.
    'ĝi' keeps the permissive default in both directions: it is the
    general "it", expected to cover most unclassified common nouns, and
    only excluded by a POSITIVE 'persono' classification.
    """
    if candidate.get("nombro") != _PRONOUN_NUMBER[pronoun]:
        return False
    if pronoun == "ili":
        return True
    animacy = _animacy_class(candidate, entity_types)
    if pronoun == "ĝi":
        return animacy != "persono"
    # li / ŝi below.
    if animacy == "non_person":
        return False
    if animacy is None and candidate.get("vortspeco") == "substantivo":
        return False
    # Esperanto's -in- suffix is the one deterministic sex marker, and it
    # is directional: PRESENT means feminine (excludes 'li'); ABSENT means
    # nothing (the unmarked form is not male-exclusive, so it must not
    # exclude 'ŝi'). This is a real, named residue, not a bug: li/ŝi
    # disambiguation for a referent with no -in- suffix and no ontology
    # entry is not decidable from grammar alone.
    if pronoun == "li" and "in" in candidate.get("sufiksoj", []):
        return False
    return True


def _rank_key(candidate: dict, distance: int) -> tuple:
    is_subject = candidate.get("rolo") in _SUBJECT_ROLES
    return (distance, 0 if is_subject else 1, candidate.get("token_id", 0))


def resolve_mention(
    mention: dict,
    preceding_sentences: list[tuple[int, dict]],
    entity_types: dict[str, str],
    *,
    window: int = 5,
    chain_candidate: Optional[dict] = None,
) -> dict:
    """Search up to `window` sentences immediately before the mention's own
    sentence (nearest first) for compatible nominal candidates.

    `preceding_sentences` must already be ordered nearest-first (index 0 =
    the sentence immediately before the mention's sentence). `chain_candidate`,
    if given, is "the entity the last same-type pronoun in this document
    resolved to" (see `resolve_document`) -- it competes on equal footing
    with fresh nominal mentions, ranked as if at distance 0 (Centering
    Theory's CONTINUE preference: an already-established topic beats a
    same-distance competing noun, but a genuinely closer or better-ranked
    fresh mention is still listed alongside it, not hidden).

    Returns {status: 'resolved'|'ambiguous'|'unresolved', candidates: [...]}.
    `candidates` is sorted by rank and is empty iff status is 'unresolved'.
    """
    scored: list[tuple[tuple, dict]] = []
    if chain_candidate is not None and is_compatible(
        mention["pronoun"], chain_candidate, entity_types
    ):
        entry = dict(chain_candidate)
        entry["distance_sentences"] = 0
        entry["source"] = "pronoun_chain"
        scored.append((_rank_key(chain_candidate, 0), entry))
    for distance, (csid, cast) in enumerate(preceding_sentences[:window], start=1):
        for cand in _nominal_candidates(csid, cast):
            if is_compatible(mention["pronoun"], cand, entity_types):
                entry = dict(cand)
                entry["distance_sentences"] = distance
                entry["source"] = "nominal_mention"
                scored.append((_rank_key(cand, distance), entry))
    scored.sort(key=lambda pair: pair[0])
    candidates = [entry for _, entry in scored]
    if not candidates:
        status = "unresolved"
    elif len(candidates) == 1:
        status = "resolved"
    else:
        status = "ambiguous"
    return {"status": status, "candidates": candidates}


def resolve_document(
    sentences: list[tuple[int, dict]],
    entity_types: dict[str, str],
    *,
    window: int = 5,
) -> list[tuple[dict, dict]]:
    """Resolve every li/ŝi/ĝi/ili mention in a document-ordered list of
    (sid, ast) pairs, tracking one PRONOUN CHAIN per pronoun type.

    Why a chain: a long stretch of biographical or narrative text often
    refers to its subject with a run of pronouns ("Ŝi studis ... Ŝi estis
    ... Ŝi elmigris ...") without re-mentioning the name in between. A
    pure "look at nearby NOUNS" search never finds a nominal candidate for
    the second, third, ... mention in that run, even though the answer is
    obvious to a reader tracking WHO is being talked about. Once a
    same-type pronoun resolves (uniquely, or as the top-ranked guess among
    several), later same-type pronouns in the SAME document get that
    resolution offered again as a `chain_candidate`, competing on equal
    footing with any fresh nominal mention (see `resolve_mention`) --
    never overriding a genuinely closer or unambiguous fresh candidate,
    only filling the gap when there isn't one.

    Returns a list of (mention, resolution) pairs in document order.
    """
    chain: dict[str, dict] = {}
    results: list[tuple[dict, dict]] = []
    for i, (sid, ast) in enumerate(sentences):
        mentions = find_pronoun_mentions(sid, ast)
        if not mentions:
            continue
        preceding = [(psid, past) for psid, past in reversed(sentences[:i])]
        for m in mentions:
            res = resolve_mention(
                m, preceding, entity_types, window=window,
                chain_candidate=chain.get(m["pronoun"]),
            )
            results.append((m, res))
            if res["candidates"]:
                chain[m["pronoun"]] = res["candidates"][0]
    return results


def build_entity_type_lookup(con) -> dict[str, str]:
    """One query, reused across a whole document/corpus pass -- the
    ontology-query pattern CLAUDE.md asks for, not a hardcoded list."""
    rows = con.execute(
        "SELECT radiko, class_id FROM ontology_edges "
        "WHERE rel = 'HAVAS_ENTECAN_TIPON'"
    ).fetchall()
    lookup: dict[str, str] = {}
    for radiko, class_id in rows:
        # A root can carry more than one class_id (the ontology is noisy
        # as well as thin); 'persono' wins ties since it's the only class
        # this module treats as an exclusion signal for 'ĝi'.
        if radiko not in lookup or class_id == "persono":
            lookup[(radiko or "").lower()] = class_id
    return lookup


def make_annotation_layer(
    mention_ast: dict,
    mention: dict,
    resolution: dict,
    *,
    artifact_hashes: Optional[dict[str, str]] = None,
) -> dict:
    """Build one mention's resolution as an annotation_layers entry per
    klareco.ast_annotations's contract, ready for
    `with_annotation_layer(mention_ast, layer)`.

    `mention_ast` is the pronoun's OWN sentence's ast (used only to compute
    the required source/tokenization basis hashes). The candidates
    themselves live in `value`, referencing their own (different, earlier)
    sentence by `sid` -- the envelope's TokenTarget can only address token
    ids within the annotated ast's own tokenization, so a genuinely
    cross-sentence reference has to live in the free-form payload, not the
    structural `target`.
    """
    from klareco.ast_annotations import annotation_basis

    return {
        "version": 1,
        "id": f"corefcand:{mention['sid']}:{mention['token_id']}",
        "schema": "urn:klareco:coref-candidates:1",
        "status": "predicted",
        "producer": {
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "method": "rule",
            "artifacts": artifact_hashes or {},
        },
        "basis": annotation_basis(mention_ast, tokens=True),
        "annotations": [{
            "id": f"mention:{mention['sid']}:{mention['token_id']}",
            "target": {"kind": "tokens", "ids": [mention["token_id"]]},
            "value": {
                "pronoun": mention["pronoun"],
                "surface": mention["surface"],
                "role": mention["role"],
                "resolution_status": resolution["status"],
                "candidates": resolution["candidates"],
            },
        }],
    }
