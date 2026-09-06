from pathlib import Path

from scripts.eval.build_parser_corpus_pilot import read_rows, select


def _row(source, title, sentence, kind="originala"):
    return {
        "source": source,
        "source_title": title,
        "sentence": sentence,
        "kind": kind,
        "licence": "public domain",
    }


def test_selection_is_deterministic_and_document_disjoint_by_split():
    rows = [
        _row("book", f"book-{i}", f"La libro numero {i} enhavas klaran frazon.")
        for i in range(20)
    ]
    rows += [
        _row("news", f"news-{i}", f"La novaĵo numero {i} estas interesa hodiaŭ.")
        for i in range(20)
    ]
    first = select(rows, 20)
    second = select(rows, 20)
    assert first == second
    assert len({item["text_sha256"] for item in first}) == 20
    development = {
        (item["source"], item["source_title"])
        for item in first if item["split"] == "development"
    }
    heldout = {
        (item["source"], item["source_title"])
        for item in first if item["split"] == "heldout"
    }
    assert development.isdisjoint(heldout)


def test_read_rows_preserves_provenance_and_filters_extremes(tmp_path: Path):
    path = tmp_path / "sentences.jsonl"
    path.write_text(
        '{"sentence":"Tiu estas valida frazo.","source":"book",'
        '"source_title":"Libro","licence":"CC BY"}\n'
        '{"sentence":"Tro mallonga","source":"book",'
        '"source_title":"Libro","licence":"CC BY"}\n',
        encoding="utf-8",
    )
    rows = read_rows([path])
    assert len(rows) == 1
    assert rows[0]["_path"] == str(path)
    assert rows[0]["_line"] == 1
