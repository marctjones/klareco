"""
Parallel Klareco extractive-QA evaluation on Modal.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu DB, orchestrator pipeline (klareco.orchestrator),
                 Modal client 1.4+
DEPENDENCIES: Whoosh FTS index, Kuzu DB (uploaded to Modal Volume `klareco-indexes`)
STAGE: Evaluation

Description:
    Runs the same per-question logic as scripts/evaluate_extractive_qa.py,
    but fans out across Modal containers so an N-question test set finishes
    in roughly max(cold_start, max_per_question_latency) instead of
    N x avg_latency. Each container loads the orchestrator once on @enter,
    then receives questions via Function.map().

Pipeline Position:
    Test JSONL  ->  modal_eval (parallel)  ->  eval_results JSON
    (Output schema is identical to evaluate_extractive_qa.py.)

Usage:
    # 1) One-time: upload indexes to the Modal Volume
    ./scripts/modal_upload_indexes.sh

    # 2) Run the eval (general-knowledge set, default)
    modal run scripts/modal_eval.py \\
        --test-set data/test_sets/general_knowledge_30_keyed.jsonl \\
        --output data/eval_results/general_knowledge_30_modal.json

    # Smoke test (first 3 questions)
    modal run scripts/modal_eval.py --limit 3

Inputs:
    - JSONL test set: {id, question, expected_keywords, question_type, ...}
    - Modal Volume `klareco-indexes` containing:
        whoosh_fts/                  (Whoosh FTS, ~17 GB)
        v2.1_kuzu_index_full         (Kuzu DB file, ~9.9 GB)

Outputs:
    - JSON with summary, by_type breakdown, per-question results
      (same schema as data/eval_results/general_knowledge_30_baseline.json)

Quality Checks:
    - Aggregate metrics computed identically to evaluate_extractive_qa.py
      (answer_accuracy, retrieval_recall, MRR, rank_distribution)
    - Per-question latency captured; Modal logs show cold-start cost

Last Updated: 2026-05-02
See Also: scripts/evaluate_extractive_qa.py, scripts/modal_upload_indexes.sh
"""

import modal

VOLUME_NAME = "klareco-indexes"
WHOOSH_REMOTE = "/indexes/whoosh_fts"
KUZU_REMOTE = "/indexes/v2.1_kuzu_index_full"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "kuzu",
        "Whoosh==2.7.4",
        "faiss-cpu",
        "tqdm",
        "psutil>=5.9.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir("klareco", "/root/klareco")
    # Code-data files the parser/extractor read at startup. NOT the corpus or
    # indexes (those live on the volume). Keep this list minimal.
    .add_local_file("data/vocabularies/merged_roots.json",
                    "/root/data/vocabularies/merged_roots.json")
    .add_local_file("data/vocabularies/fundamento_roots.json",
                    "/root/data/vocabularies/fundamento_roots.json")
    .add_local_file("data/vocabularies/protected_roots.json",
                    "/root/data/vocabularies/protected_roots.json")
    .add_local_file("data/raw/eo/dictionaries/revo/revo_semantic_relations.json",
                    "/root/data/raw/eo/dictionaries/revo/revo_semantic_relations.json")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

app = modal.App("klareco-eval")


@app.cls(
    image=image,
    volumes={"/indexes": volume},
    timeout=900,
    # Default fan-out: one container per input up to 100. For batch eval we
    # want minimum wall-clock; cold-cache penalty per container is amortized
    # because containers run in parallel.
    max_containers=100,
    # Kuzu uses internal multithreading + a RAM-backed buffer pool. Default
    # Modal containers (~0.125 vCPU, ~128 MB RAM) starve a 9.9 GB graph DB.
    # 8 vCPU + 16 GB lets Cypher execute in parallel and lets the hot pages
    # stay resident across queries.
    cpu=8.0,
    memory=16384,
)
class Evaluator:
    top_k: int = modal.parameter(default=10)

    @modal.enter()
    def setup(self) -> None:
        import logging
        import os
        import sys
        import time

        sys.path.insert(0, "/root")
        # Tell semantic_bridge's hardcoded singleton where Kuzu lives —
        # orchestrator's kuzu_db_path arg doesn't reach that module.
        os.environ["KLARECO_KUZU_DB_PATH"] = KUZU_REMOTE
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
        logging.getLogger("klareco").setLevel(logging.WARNING)

        from klareco.orchestrator import build_default_pipeline

        t0 = time.time()
        self.pipeline = build_default_pipeline(
            whoosh_index_dir=WHOOSH_REMOTE,
            kuzu_db_path=KUZU_REMOTE,
            top_k=self.top_k,
        )
        # Tag this container so we can see how questions distributed across
        # containers (cold vs warm-cache effect).
        import uuid
        self._container_id = uuid.uuid4().hex[:8]
        self._call_seq = 0
        print(f"Pipeline ready in {time.time() - t0:.1f}s "
              f"(top_k={self.top_k}, container={self._container_id})")

    @modal.method()
    def evaluate(self, entry: dict) -> dict:
        from klareco.eval import evaluate_question
        result = evaluate_question(self.pipeline, entry)
        self._call_seq += 1
        result["_container_id"] = self._container_id
        result["_container_call_seq"] = self._call_seq  # 1 = cold, >1 = warm
        return result


@app.local_entrypoint()
def main(
    test_set: str = "data/test_sets/general_knowledge_30_keyed.jsonl",
    output: str = "data/eval_results/general_knowledge_30_modal.json",
    top_k: int = 10,
    limit: int = 0,
) -> None:
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from klareco.eval import summarize, print_summary

    test_path = Path(test_set)
    if not test_path.exists():
        raise SystemExit(f"ERROR: test set not found at {test_path}")

    entries = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if limit:
        entries = entries[:limit]

    print(f"Dispatching {len(entries)} questions to Modal "
          f"(top_k={top_k}, app=klareco-eval)...")

    evaluator = Evaluator(top_k=top_k)
    t0 = time.time()
    results = list(evaluator.evaluate.map(entries, order_outputs=True))
    wall = time.time() - t0
    print(f"\nWall-clock fan-out: {wall:.1f}s for {len(entries)} questions")

    summary = summarize(results)
    by_type_groups: dict = {}
    for r in results:
        qt = r.get("question_type") or "OTHER"
        by_type_groups.setdefault(qt, []).append(r)
    by_type = {qt: summarize(rs) for qt, rs in by_type_groups.items()}

    print_summary(summary, by_type)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "by_type": by_type, "results": results,
           "wall_clock_sec": round(wall, 2)}
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWritten to {out_path}")
