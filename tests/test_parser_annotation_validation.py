import hashlib
import pytest
from scripts.eval.validate_parser_annotations import validate


def row():
    text = "Mi venas."
    return dict(
        id="1",
        annotation_status="reviewed",
        annotator="a",
        reviewer="b",
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        gold_conllu="1\tMi\tmi\tPRON\t_\t_\t2\tnsubj\t_\t_\n2\tvenas\tveni\tVERB\t_\t_\t0\troot\t_\t_\n3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_",
        phenomena=[],
        review_notes="Reviewed tokenization and attachments.",
    )


def test_reviewed_gold_exports():
    assert validate(row()).endswith("\n\n")


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewer", "a"),
        ("annotation_status", "unreviewed"),
        ("text", "Vi venas."),
        ("gold_conllu", None),
    ],
)
def test_unreviewed_or_changed_gold_rejected(field, value):
    record = row()
    record[field] = value
    with pytest.raises(ValueError):
        validate(record)


def test_gold_cycles_rejected():
    record = row()
    record["gold_conllu"] = record["gold_conllu"].replace("2\tnsubj", "1\tnsubj")
    with pytest.raises(ValueError, match="Cycle"):
        validate(record)
