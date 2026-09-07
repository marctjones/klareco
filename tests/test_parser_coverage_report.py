from scripts.eval.parser_coverage_report import evaluate


def rows():
    return [
        {"id": "a", "text": "La hundo kuras.", "source": "book", "kind": "originala"},
        {"id": "b", "text": "La kato dormas.", "source": "news", "kind": "originala"},
    ]


def test_coverage_report_records_success_roundtrip_and_determinism():
    calls = []

    def parser(text):
        calls.append(text)
        return {"text": text, "vortoj": []}

    report = evaluate(rows(), parser)
    assert report["sentences"] == 2
    assert report["parse_failures"] == 0
    assert report["complete_storage_round_trips"] == 2
    assert report["deterministic_reparses"] == 2
    assert len(calls) == 4


def test_coverage_report_keeps_failures_as_inventory():
    def parser(text):
        if text.startswith("La kato"):
            raise ValueError("fixture failure")
        return {"text": text, "vortoj": []}

    report = evaluate(rows(), parser)
    assert report["parse_failures"] == 1
    assert report["parse_success_rate"] == 0.5
    assert report["failures"][0]["id"] == "b"
