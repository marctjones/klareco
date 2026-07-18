"""
Shared fixtures for the contract suite (#883, #886).

The whole point of this suite is to test THE ORCHESTRATOR — real stages,
threading a real thought — WITHOUT the 49 GB production indexes. So we build a
tiny real store + Whoosh index in a tmp dir from a dozen hand-written Esperanto
sentences, using the SAME parser / shred / schema the production build uses.
That makes these tests CI-runnable and fast, and it means a stage that drifts
from the contract fails here, loudly, in under a second.
"""
from __future__ import annotations

import json

import pytest

# A tiny, self-contained corpus. Each sentence is a plausible answer to one of
# the CANONICAL_QUESTIONS below, so the full pipeline has something to retrieve,
# rerank, and extract. Kept deterministic and small on purpose.
MINI_CORPUS = [
    (1,  "Zamenhof kreis Esperanton en 1887.",            "Zamenhof",  "wiki"),
    (2,  "Ludoviko Zamenhof estis pola okulisto.",         "Zamenhof",  "wiki"),
    (3,  "Esperanto estas internacia planlingvo.",         "Esperanto", "wiki"),
    (4,  "La Fundamento aperis en 1905.",                  "Fundamento","wiki"),
    (5,  "Kalocsay verkis multajn poemojn en Esperanto.",  "Kalocsay",  "wiki"),
    (6,  "Parizo estas la ĉefurbo de Francio.",            "Parizo",    "wiki"),
    (7,  "La suno estas stelo.",                           "Suno",      "wiki"),
    (8,  "Hundoj kaj katoj estas bestoj.",                 "Bestoj",    "wiki"),
    (9,  "Zamenhof naskiĝis en Bjalistoko.",               "Zamenhof",  "wiki"),
    (10, "La unua libro aperis en Varsovio.",              "Unua Libro","wiki"),
    (11, "Akvo estas necesa por vivo.",                    "Akvo",      "wiki"),
    (12, "Montoj estas altaj kaj rokaj.",                  "Montoj",    "wiki"),
]

CANONICAL_QUESTIONS = [
    "Kiu kreis Esperanton?",
    "Kio estas Esperanto?",
    "Kie naskiĝis Zamenhof?",
    "Kiom estas du plus tri?",          # exercises the math short-circuit
]


@pytest.fixture(scope="session")
def mini_store(tmp_path_factory):
    """Build a real DuckDB store + Whoosh index from MINI_CORPUS.

    Returns (duckdb_path, whoosh_dir). Session-scoped: built once per run.
    Uses the production parser + shred + schema so the retriever's real SQL
    runs against a real (tiny) table.
    """
    import duckdb
    from whoosh import index as whoosh_index
    from whoosh.fields import ID, TEXT, Schema

    from klareco.parser import parse
    from scripts.index.build_duckdb_store import ensure_schema, shred

    # Column order matching ensure_schema() in build_duckdb_store.py.
    COLS = (
        "sid", "text",
        "subj_radiko", "subj_vortspeco", "subj_propranoma_kat", "subj_kazo",
        "verb_radiko", "verb_tempo", "obj_radiko", "obj_kazo",
        "aliaj_json", "success_rate", "ast_json",
        "source_name", "source_type", "article_title", "article_id",
        "section", "quality",
    )

    d = tmp_path_factory.mktemp("mini_store")
    db_path = d / "store.db"
    whoosh_dir = d / "whoosh"
    whoosh_dir.mkdir()

    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    for sid, text, title, source in MINI_CORPUS:
        ast = parse(text)
        row = shred(ast)
        row.update({
            "sid": sid, "text": text, "ast_json": json.dumps(ast, ensure_ascii=False),
            "source_name": source, "source_type": source,
            "article_title": title, "article_id": str(sid),
            "section": "", "quality": "ok",
        })
        cols = list(COLS)
        con.execute(
            f"INSERT INTO sentences ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [row.get(c) for c in cols],
        )
    con.close()

    ix = whoosh_index.create_in(
        str(whoosh_dir),
        Schema(id=ID(stored=True, unique=True), text=TEXT(stored=True)),
    )
    w = ix.writer()
    for sid, text, _title, _src in MINI_CORPUS:
        w.add_document(id=str(sid), text=text)
    w.commit()

    return db_path, whoosh_dir


@pytest.fixture(scope="session")
def mini_pipeline(mini_store):
    """A real Orchestrator over the mini store (factory-parity, no preflight).

    Deliberately bypasses build_default_pipeline's ARTIFACT preflight — the
    mini store is intentionally tiny — but uses the exact same stage classes
    in the same order, so it is a faithful orchestration test target.
    """
    from klareco.orchestrator.mini import build_mini_pipeline
    db_path, whoosh_dir = mini_store
    return build_mini_pipeline(whoosh_dir=whoosh_dir, duckdb_path=db_path)
