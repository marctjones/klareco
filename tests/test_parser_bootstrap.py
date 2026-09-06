from pathlib import Path

from scripts.eval.parser_bootstrap_report import bootstrap, sentence_scores


def test_sentence_scores_reproduce_report_metrics():
    path = Path("tests/fixtures/ud/eo_cairo-ud-test.conllu")
    scores, overall = sentence_scores(path)
    assert sum(row["tokens"] for row in scores) == overall["gold_tokens"]
    assert sum(row["las"] for row in scores) / overall["gold_tokens"] == overall["las_all"]
    assert sum(row["uas"] for row in scores) / overall["gold_tokens"] == overall["uas_all"]
    assert sum(row["upos"] for row in scores) / overall["gold_tokens"] == overall["upos"]


def test_bootstrap_is_seeded_and_returns_ordered_intervals():
    scores = [{"tokens": 2, "uas": 2, "las": 1, "upos": 2} for _ in range(5)]
    first = bootstrap(scores, seed=7, draws=200)
    second = bootstrap(scores, seed=7, draws=200)
    assert first == second
    for metric in ("uas", "las", "upos"):
        assert first[metric]["p025"] <= first[metric]["p975"]
