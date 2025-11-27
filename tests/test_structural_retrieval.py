import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from klareco.parser import parse
from klareco.structural_index import build_structural_metadata
from klareco.rag.retriever import KlarecoRetriever
from klareco.ast_to_graph import ASTToGraphConverter


class DummyRetriever(KlarecoRetriever):
    """Retriever stub that skips model loading and uses fixed embeddings."""

    def __init__(self, *args, embeddings=None, **kwargs):
        self._synthetic_embeddings = embeddings
        super().__init__(*args, **kwargs)

    def _load_tree_lstm(self):
        self.encoder = None
        self.converter = ASTToGraphConverter()

    def _encode_ast(self, ast):
        # Map any query to the first embedding for deterministic ordering
        return self._synthetic_embeddings[0]


@pytest.mark.skipif(not faiss, reason="faiss not installed")
def test_structural_filter_prefers_slot_overlap(tmp_path: Path):
    sentences = [
        "La hundo vidas la katon.",
        "La kato dormas.",
        "La knabo legas libron.",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    metadata = []
    for s in sentences:
        ast = parse(s)
        m = build_structural_metadata(ast)
        m["sentence"] = s
        metadata.append(m)

    # Write index artifacts
    np.save(tmp_path / "embeddings.npy", embeddings)
    with open(tmp_path / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    norm_embeddings = embeddings.copy()
    faiss.normalize_L2(norm_embeddings)
    index = faiss.IndexFlatIP(norm_embeddings.shape[1])
    index.add(norm_embeddings)
    faiss.write_index(index, str(tmp_path / "faiss_index.bin"))

    retriever = DummyRetriever(
        index_dir=str(tmp_path),
        model_path=str(tmp_path / "dummy.pt"),
        mode="tree_lstm",
        device="cpu",
        embeddings=embeddings,
    )

    results = retriever.retrieve("La hundo vidas la katon.", k=2)
    assert results[0]["text"].startswith("La hundo vidas")
    # Second result should share overlap (kato dormas) before unrelated sentence
    assert "kato" in results[1]["text"]
