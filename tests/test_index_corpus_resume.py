import json
import numpy as np
from pathlib import Path

from scripts.index_corpus import CorpusIndexer
from klareco.parser import parse


class DummyIndexer(CorpusIndexer):
    """Indexer stub that skips model/graph work for testing resume behavior."""

    def load_model(self):
        # Skip model loading
        self.model = True

    def encode_sentence(self, sentence: str):
        ast = parse(sentence)
        # Simple deterministic embedding
        emb = np.array([len(sentence), 0, 0], dtype=np.float32)
        return emb, ast

    def build_faiss_index(self):
        # Skip FAISS in tests
        return


def test_index_corpus_resume(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Mi amas vin.\nVi kuras.\nLi legas.\n", encoding="utf-8")

    output_dir = tmp_path / "index"
    output_dir.mkdir()

    # Pre-create embeddings/metadata and checkpoint for first two sentences
    np.save(output_dir / "embeddings.npy", np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, text in enumerate(["Mi amas vin.", "Vi kuras."]):
            f.write(json.dumps({"sentence": text, "idx": idx, "embedding_idx": idx}) + "\n")
    checkpoint = {
        "total_sentences": 3,
        "processed": 2,
        "successful": 2,
        "failed": 0,
    }
    (output_dir / "indexing_checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    indexer = DummyIndexer(
        model_path=str(tmp_path / "dummy.pt"),
        output_dir=str(output_dir),
        batch_size=1,
        embedding_dim=3,
    )
    indexer.index_corpus(str(corpus), resume=True)

    # New embedding appended
    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape[0] == 3

    # Metadata appended
    with (output_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3

    # Checkpoint updated to processed all sentences
    final_ckpt = json.loads((output_dir / "indexing_checkpoint.json").read_text(encoding="utf-8"))
    assert final_ckpt["processed"] == 3
    assert final_ckpt["successful"] == 3
