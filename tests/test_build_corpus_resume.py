import json
from pathlib import Path

from scripts.build_corpus_with_sources import build_corpus, _save_checkpoint


def test_build_corpus_resumes(tmp_path: Path):
    # Prepare a small cleaned file
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    sample = cleaned_dir / "cleaned_sample.txt"
    sample.write_text("Unu linio\nDua linio\nTria linio\nKvara linio\n", encoding="utf-8")

    output = tmp_path / "corpus.jsonl"
    checkpoint = tmp_path / "checkpoint.json"

    texts = [("cleaned_sample.txt", "Sample")]

    # First run: process first two lines then checkpoint
    _save_checkpoint(checkpoint, "cleaned_sample.txt", 2, 2)
    total = build_corpus(
        cleaned_dir,
        output,
        texts,
        skip_empty=True,
        skip_metadata=True,
        min_length=0,
        checkpoint_path=checkpoint,
    )
    # After resume, it should write remaining two lines; total written tracked in checkpoint
    assert total == 4

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # only two new lines appended this run
    records = [json.loads(line) for line in lines]
    assert records[0]["text"].startswith("Tria")
    assert records[1]["text"].startswith("Kvara")
