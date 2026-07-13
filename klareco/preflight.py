"""
Artifact preflight — fail loudly instead of degrading silently.

VERSION: v2.x
COMPATIBLE WITH: DuckDB store (v2.2 schema), Whoosh v2 index
DEPENDENCIES: duckdb; no models
STAGE: Utility

Description:
    Validates every artifact the pipeline depends on, and RAISES if one is
    missing or empty.

    This module exists because of a specific, expensive failure. Several data
    artifacts were lost in a June 2026 laptop migration. **None of them
    crashed.** Every loader used the pattern

        if path.exists():
            load(path)
        # ...and silently carried on without it

    so the pipeline kept returning plausible-looking answers while quietly
    getting worse. The parser over-decomposed `Esperanton` into `esper` + a
    participle suffix for a month, the entire semantic ontology was absent at
    runtime, and nobody noticed — because nothing ever said so.

    An artifact that silently degrades output is worse than one that crashes:
    the failure is invisible and the numbers still look reasonable.

    Hence the decision principle (DESIGN.md): **a silently-degrading dependency
    is a bug.** See issue #779.

Two artifact classes:
    REQUIRED  — the pipeline cannot function without it. Always raises.
    DEGRADING — the pipeline runs, but output is measurably worse and the
                degradation is invisible. Raises by default; can be downgraded
                to a loud, itemized banner with an explicit opt-in.

The opt-in is deliberately explicit and deliberately noisy. You may run
degraded — you may not do so by accident.

Usage:
    from klareco.preflight import preflight, PreflightError

    preflight(duckdb_path, whoosh_index_dir)              # raises on any problem
    preflight(duckdb_path, whoosh_index_dir,
              allow_degraded=True)                         # warns, itemized

    # or from the shell:
    python -m klareco.preflight
    python -m klareco.preflight --allow-degraded

Environment:
    KLARECO_ALLOW_DEGRADED=1   equivalent to allow_degraded=True

Last Updated: 2026-07-13
Related Issues: #779
See Also: DESIGN.md ("Current state"), AGENTS.md ("Fail loudly")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent


class PreflightError(RuntimeError):
    """A required artifact is missing, or a degrading artifact is missing and
    degraded operation was not explicitly permitted."""


@dataclass
class Finding:
    """One artifact problem."""
    name: str
    required: bool
    detail: str          # what is wrong
    consequence: str     # what it costs us — stated in terms of OUTPUT, not code
    remedy: str          # how to fix it

    def render(self) -> str:
        tag = "REQUIRED" if self.required else "DEGRADED"
        return (f"  [{tag}] {self.name}\n"
                f"      problem:     {self.detail}\n"
                f"      consequence: {self.consequence}\n"
                f"      remedy:      {self.remedy}")


# --------------------------------------------------------------------------
# Individual checks. Each returns a Finding, or None when the artifact is OK.
# --------------------------------------------------------------------------

def _check_file(name: str, path: Path, *, required: bool,
                consequence: str, remedy: str) -> Optional[Finding]:
    if path.exists() and path.stat().st_size > 0:
        return None
    detail = (f"{path} does not exist" if not path.exists()
              else f"{path} is empty (0 bytes)")
    return Finding(name, required, detail, consequence, remedy)


def _check_parser_vocabularies() -> list[Finding]:
    """The parser's protected roots and proper-noun dictionary.

    Both were lost in the June 2026 migration. Without them the parser
    over-decomposes: `Esperanton` -> radiko `esper` + participle suffix `ant`,
    which is not merely inelegant — it changes the root that retrieval matches
    on, so the query and the index stop agreeing.
    """
    findings: list[Finding] = []

    f = _check_file(
        "protected_roots.json",
        _PROJECT_ROOT / "data" / "vocabularies" / "protected_roots.json",
        required=False,
        consequence=("Parser over-decomposes roots whose tail mimics an affix: "
                     "`esperant` -> `esper` + `ant`, `banan` -> `ban` + `an`. "
                     "~10 parser tests fail. Retrieval roots shift."),
        remedy=("Restore the file. NOTE: the DuckDB store was ALSO built "
                "without it (it contains `esper`, not `esperant`), so question "
                "and index currently agree. Restoring it WITHOUT a corpus "
                "reparse will break retrieval — see DESIGN.md 'Current state'."),
    )
    if f:
        findings.append(f)

    # proper_nouns.py has a documented fallback chain: v3 -> v2 -> legacy.
    # Only report if the whole chain is absent.
    pn_paths = [
        _PROJECT_ROOT / "data" / "proper_nouns_dynamic_v3.json",
        _PROJECT_ROOT / "data" / "proper_nouns_dynamic_v2.json",
        _PROJECT_ROOT / "data" / "proper_nouns_dynamic.json",
    ]
    if not any(p.exists() for p in pn_paths):
        findings.append(Finding(
            "proper_nouns_dynamic_v*.json",
            required=False,
            detail="none of v3 / v2 / legacy exists (whole fallback chain absent)",
            consequence=("Proper-noun dictionary loads as None. The parser falls "
                         "back to capitalization heuristics alone, so it cannot "
                         "distinguish a name from a sentence-initial common noun. "
                         "Every downstream stage keyed on propra_nomo is degraded."),
            remedy="Restore the dictionary, or rebuild it (see #779).",
        ))

    return findings


def _check_duckdb(duckdb_path: Path) -> list[Finding]:
    """Store presence, plus column-population assertions.

    A column that exists but is 0% populated is exactly the silent failure this
    module is for: every consumer of `verb_klaso` no-ops, and the reranker that
    scores on it produces the same ranking as one that doesn't. That is how nine
    rerankers came to be tied without anyone noticing.
    """
    findings: list[Finding] = []

    f = _check_file(
        "duckdb_store.db", duckdb_path, required=True,
        consequence="No retrieval is possible.",
        remedy="Build it: python scripts/index/build_duckdb_store.py",
    )
    if f:
        return [f]   # nothing else is checkable

    try:
        import duckdb
    except ImportError:
        return [Finding("duckdb", True, "duckdb package not importable",
                        "No retrieval is possible.",
                        "pip install -r requirements.txt")]

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        total = con.execute("SELECT count(*) FROM sentences").fetchone()[0]
        if not total:
            return [Finding("sentences", True, "table is empty",
                            "No retrieval is possible.",
                            "Rebuild the store.")]

        # Columns that must carry data for the stage that reads them to do
        # anything at all. (column, min_fraction, who_reads_it)
        POPULATION_CONTRACTS: list[tuple[str, float, str]] = [
            ("ast_json",   0.99, "every stage that inspects sentence structure"),
            ("aliaj_json", 0.90, "KIE/KIAM answer-slot matching"),
            ("subj_radiko", 0.80, "subject-role retrieval and KIU reranking"),
            ("verb_klaso", 0.01, "ast_aware_reranker's verb-class generalization"),
        ]
        for col, min_frac, reader in POPULATION_CONTRACTS:
            n = con.execute(
                f"SELECT count(*) FROM sentences "
                f"WHERE {col} IS NOT NULL AND CAST({col} AS VARCHAR) <> ''"
            ).fetchone()[0]
            frac = n / total
            if frac < min_frac:
                findings.append(Finding(
                    f"sentences.{col}",
                    required=False,
                    detail=f"{frac:.1%} populated ({n:,} of {total:,}); "
                           f"contract requires >= {min_frac:.0%}",
                    consequence=(f"{reader} silently no-ops. A scoring function "
                                 f"that reads this column contributes nothing, "
                                 f"and looks identical to one that doesn't."),
                    remedy=f"Populate {col} (see #777 for verb_klaso).",
                ))

        # The ontology tables: present-but-empty is the trap.
        for tbl in ("ontology_nodes", "ontology_edges"):
            exists = con.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_name = ?", [tbl]).fetchone()[0]
            n = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0] if exists else 0
            if n == 0:
                findings.append(Finding(
                    tbl, required=False,
                    detail="table missing" if not exists else "table exists but has 0 rows",
                    consequence=("The semantic ontology is absent at runtime. Entity "
                                 "types, verb classes, thematic roles, and schema-slot "
                                 "weights all resolve to nothing; consumers fall back "
                                 "to hardcoded lists or no-op."),
                    remedy="Load the ontology snapshot — see #777.",
                ))

        # entity_facts: absent means BiographyFormatStage *does* crash — one of
        # the few honest failures in the system.
        has_ef = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'entity_facts'").fetchone()[0]
        if not has_ef:
            findings.append(Finding(
                "entity_facts", required=False,
                detail="table does not exist",
                consequence="BiographyFormatStage raises a Catalog Error on 'kio estas X' "
                            "and 'diru pri X' questions.",
                remedy="python scripts/index/extract_entity_facts.py  (see #745)",
            ))
    finally:
        con.close()

    return findings


def _check_whoosh(whoosh_index_dir: Path) -> list[Finding]:
    if not whoosh_index_dir.exists():
        return [Finding("whoosh index", True,
                        f"{whoosh_index_dir} does not exist",
                        "BM25 retrieval is impossible.",
                        "python scripts/index/rebuild_whoosh_from_duckdb.py")]
    try:
        from whoosh import index as whoosh_index
        if not whoosh_index.exists_in(str(whoosh_index_dir)):
            return [Finding("whoosh index", True,
                            f"{whoosh_index_dir} exists but holds no index",
                            "BM25 retrieval is impossible.",
                            "python scripts/index/rebuild_whoosh_from_duckdb.py")]
        if whoosh_index.open_dir(str(whoosh_index_dir)).doc_count() == 0:
            return [Finding("whoosh index", True, "index has 0 documents",
                            "BM25 retrieval returns nothing.",
                            "python scripts/index/rebuild_whoosh_from_duckdb.py")]
    except ImportError:
        return [Finding("whoosh", True, "whoosh package not importable",
                        "BM25 retrieval is impossible.",
                        "pip install -r requirements.txt")]
    return []


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_checks(duckdb_path: Path | str = "data/indexes/duckdb_store.db",
               whoosh_index_dir: Path | str = "data/indexes/whoosh_v2",
               ) -> list[Finding]:
    """Run every check and return the findings. Never raises on a bad artifact."""
    return [
        *_check_parser_vocabularies(),
        *_check_duckdb(Path(duckdb_path)),
        *_check_whoosh(Path(whoosh_index_dir)),
    ]


def preflight(duckdb_path: Path | str = "data/indexes/duckdb_store.db",
              whoosh_index_dir: Path | str = "data/indexes/whoosh_v2",
              allow_degraded: Optional[bool] = None) -> list[Finding]:
    """Validate the pipeline's artifacts.

    Raises PreflightError if a REQUIRED artifact is missing, or if a DEGRADING
    artifact is missing and degraded operation was not explicitly permitted.

    allow_degraded : True permits degraded operation, printing a loud itemized
        banner. None (the default) reads KLARECO_ALLOW_DEGRADED from the
        environment. A REQUIRED artifact always raises regardless.
    """
    if allow_degraded is None:
        allow_degraded = os.environ.get("KLARECO_ALLOW_DEGRADED", "") not in ("", "0")

    findings = run_checks(duckdb_path, whoosh_index_dir)
    if not findings:
        return []

    required = [f for f in findings if f.required]
    degrading = [f for f in findings if not f.required]

    if required or not allow_degraded:
        header = ("Preflight failed — the pipeline's artifacts are not intact.\n\n"
                  "Klareco does not run degraded by accident. Fix the items below,\n"
                  "or, if you are knowingly working on a degraded system, opt in\n"
                  "explicitly with allow_degraded=True (or KLARECO_ALLOW_DEGRADED=1)\n"
                  "— note that a REQUIRED artifact cannot be waived.\n")
        body = "\n".join(f.render() for f in findings)
        raise PreflightError(f"{header}\n{body}\n")

    # Degraded, but knowingly. Say so loudly, and say what it costs.
    banner = ("\n" + "=" * 72 +
              "\nRUNNING DEGRADED — you opted in. Output is measurably worse.\n" +
              "=" * 72 + "\n" +
              "\n".join(f.render() for f in degrading) +
              "\n" + "=" * 72)
    log.warning(banner)
    print(banner, flush=True)
    return degrading


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--duckdb-path", default="data/indexes/duckdb_store.db")
    ap.add_argument("--whoosh-index-dir", default="data/indexes/whoosh_v2")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="permit degraded operation (still prints what is degraded)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        findings = preflight(args.duckdb_path, args.whoosh_index_dir,
                             allow_degraded=args.allow_degraded)
    except PreflightError as e:
        print(str(e))
        return 1

    if not findings:
        print("Preflight OK — every artifact is present and populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
