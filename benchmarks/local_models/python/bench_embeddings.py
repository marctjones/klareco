#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def mrr_at_k(rankings: list[list[str]], gold: list[set[str]], k: int) -> float:
    total = 0.0
    for ranked, relevant in zip(rankings, gold):
        reciprocal = 0.0
        for index, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in relevant:
                reciprocal = 1.0 / index
                break
        total += reciprocal
    return total / len(rankings)


def recall_at_k(rankings: list[list[str]], gold: list[set[str]], k: int) -> float:
    hits = 0
    for ranked, relevant in zip(rankings, gold):
        if any(doc_id in relevant for doc_id in ranked[:k]):
            hits += 1
    return hits / len(rankings)


def prefixed(texts: list[str], prefix: str) -> list[str]:
    return [prefix + text for text in texts]


def run_suite(model_entry: dict[str, Any], suite_name: str, suite: dict[str, Any]) -> dict[str, Any]:
    load_start = time.perf_counter()
    model = SentenceTransformer(model_entry["model"], trust_remote_code=True)
    load_seconds = time.perf_counter() - load_start

    document_prefix = model_entry.get("document_prefix", "")
    query_prefix = model_entry.get("query_prefix", "")

    doc_ids = [item["id"] for item in suite["documents"]]
    docs = prefixed([item["text"] for item in suite["documents"]], document_prefix)
    queries = prefixed([item["text"] for item in suite["queries"]], query_prefix)
    gold = [set(item["relevant"]) for item in suite["queries"]]

    encode_start = time.perf_counter()
    cpu_start = time.process_time()
    doc_embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=False)
    query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=False)
    cpu_seconds = time.process_time() - cpu_start
    encode_seconds = time.perf_counter() - encode_start

    doc_embeddings = normalize(np.asarray(doc_embeddings))
    query_embeddings = normalize(np.asarray(query_embeddings))
    scores = query_embeddings @ doc_embeddings.T
    rankings = [
        [doc_ids[index] for index in np.argsort(-query_scores)]
        for query_scores in scores
    ]

    top1 = [ranking[0] for ranking in rankings]
    return {
        "runtime": "python_sentence_transformers",
        "category": "embedding",
        "suite": suite_name,
        "model_id": model_entry["id"],
        "model": model_entry["model"],
        "producer": model_entry.get("producer"),
        "region": model_entry.get("region"),
        "family": model_entry.get("family"),
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_mb": rss_mb(),
        "embedding_dim": int(doc_embeddings.shape[1]),
        "documents": len(docs),
        "queries": len(queries),
        "docs_per_second": len(docs) / encode_seconds if encode_seconds else None,
        "queries_per_second": len(queries) / encode_seconds if encode_seconds else None,
        "recall_at_1": recall_at_k(rankings, gold, 1),
        "recall_at_3": recall_at_k(rankings, gold, 3),
        "mrr_at_3": mrr_at_k(rankings, gold, 3),
        "top1": top1,
        "rankings": rankings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding models on separate English and Esperanto retrieval suites.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="Optional model ids to run.")
    args = parser.parse_args()

    models = read_json(args.models)["embeddings"]
    if args.only:
        allowed = set(args.only)
        models = [model for model in models if model["id"] in allowed]
    tasks = read_json(args.tasks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for model_entry in models:
            for suite_name, suite in tasks.items():
                row = run_suite(model_entry, suite_name, suite)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
