"""#871: a fronted PREPOSITIONAL phrase must not null the main clause.

Dual-use words (post/antaŭ/dum/por/malgraŭ) are both prepositions and
subordinating conjunctions; as prepositions they head a fronted PP that must
keep the main clause's subjekto+verbo. ~182k store sentences (4% of corpus)
had lost their subject to this."""
import pytest

from klareco.parser import parse


def _roles(text):
    a = parse(text)

    def kern(n):
        if isinstance(n, dict):
            return n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
        return {}
    s = kern(a.get('subjekto')) or {}
    v = a.get('verbo') or {}
    return s.get('radiko'), v.get('radiko')


@pytest.mark.parametrize("text, subj, verb", [
    ("Post la milito multaj homoj revenis hejmen.", "hom", "ven"),   # re+ven
    ("Post la milito, multaj homoj revenis hejmen.", "hom", "ven"),
    ("Dum la vojaĝo ŝi lernis Esperanton.", "ŝi", "lern"),
    ("Antaŭ la domo staras arbo.", "arb", "star"),
    ("Por la paco ni laboras.", "ni", "labor"),
])
def test_fronted_pp_keeps_main_clause(text, subj, verb):
    s, v = _roles(text)
    assert s == subj and v == verb, f"{text!r} -> subj={s!r} verb={v!r}"


@pytest.mark.parametrize("text", [
    "Zamenhof kreis Esperanton.",              # control: no fronting
    "Hieraŭ la hundo manĝis.",                 # control: fronted adverb (already OK)
])
def test_controls_unregressed(text):
    s, v = _roles(text)
    assert s is not None and v is not None, f"regressed: {text!r}"


def test_genuine_fronted_conjunction_stays_subordinate():
    # "Post kiam ... finiĝis, ..." — real subordinate clause; the dual-use guard
    # must NOT treat this as a PP (an explicit ki-correlative follows).
    from klareco.parser import _is_fronted_pp_not_clause, parse as _p
    words = _p("Post kiam la milito finiĝis, homoj revenis.")['vortoj']
    assert _is_fronted_pp_not_clause(words, 0) is False
    # and the PP case is correctly classified as a PP
    words_pp = _p("Post la milito homoj revenis.")['vortoj']
    assert _is_fronted_pp_not_clause(words_pp, 0) is True
