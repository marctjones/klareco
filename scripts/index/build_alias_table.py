"""
Build Alias Table from Wikipedia Redirects — canonical-name bridge for retrieval

VERSION: v2.1
COMPATIBLE WITH: v2.1 DuckDB store (sentences.article_title), whoosh_v2
DEPENDENCIES: data/raw/eo/wikipedia/eo_wikipedia.xml.bz2 (MediaWiki dump). No models.
STAGE: Index

Description:
    Streams the Esperanto Wikipedia dump and extracts every redirect
    (<page><title>ALIAS</title><redirect title="CANONICAL"/></page>) into a
    static alias->canonical map. Redirects are human-curated aliases — the
    deterministic bridge the alias_variant question band needs (#865): the
    question names an entity under one surface form ("Spirited Away"), the gold
    sentence uses another ("Sen to Chihiro no Kamikakushi"). No ML.

    Oracle probe (bench_history 2026-07-18) put the ceiling at 93% top-20 on the
    82-question alias_variant band IF a perfect bridge exists. This builds the
    real (redirect-derived) bridge so the realistic yield can be measured under
    the merge gate.

Pipeline Position:
    eo_wikipedia.xml.bz2 -> [THIS SCRIPT] -> alias_table.json -> duckdb_retriever
    (KLARECO_ALIAS_BRIDGE) -> multi_reranker_bench on alias_variant band

Usage:
    python scripts/index/build_alias_table.py            # build
    python scripts/index/build_alias_table.py --resume   # skip if output exists

Inputs:
    - data/raw/eo/wikipedia/eo_wikipedia.xml.bz2 (MediaWiki XML, bz2)

Outputs:
    - data/indexes/alias_table.json  {alias_lower: canonical_title}

Quality Checks:
    - Reports pair count, drops self-redirects and empty targets.
    - Coverage against alias_variant band is measured separately by the probe.

Last Updated: 2026-07-18
Author: Marc Jones (with Claude Fable 5)
Related Issues: #865
See Also: scripts/index/build_duckdb_store.py, klareco/rag/duckdb_retriever.py
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
from pathlib import Path

DUMP = Path('data/raw/eo/wikipedia/eo_wikipedia.xml.bz2')
OUT = Path('data/indexes/alias_table.json')

# A redirect page is: <title>ALIAS</title> ... <redirect title="CANONICAL" />
_TITLE = re.compile(r'<title>(.*?)</title>')
_REDIR = re.compile(r'<redirect title="(.*?)"\s*/>')


def _unescape(s: str) -> str:
    return (s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            .replace('&quot;', '"').replace('&#039;', "'"))


def build(resume: bool = False) -> None:
    if resume and OUT.exists():
        print(f"[skip] {OUT} exists (--resume)")
        return
    if not DUMP.exists():
        sys.exit(f"FATAL: dump missing: {DUMP}")

    alias: dict[str, str] = {}
    cur_title = None
    pages = redirects = 0
    with bz2.open(DUMP, 'rt', encoding='utf-8') as fh:
        for line in fh:
            mt = _TITLE.search(line)
            if mt:
                cur_title = _unescape(mt.group(1).strip())
                if cur_title.startswith(('Vikipedio:', 'Ŝablono:', 'Kategorio:',
                                         'Dosiero:', 'MediaWiki:', 'Modulo:',
                                         'Helpo:', 'Portalo:')):
                    cur_title = None
                pages += 1
                continue
            mr = _REDIR.search(line)
            if mr and cur_title:
                canon = _unescape(mr.group(1).strip())
                a = cur_title.lower()
                # drop self-redirects, empties, and case-only variants
                if canon and a and a != canon.lower():
                    alias[a] = canon
                    redirects += 1
                cur_title = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(alias, f, ensure_ascii=False)
    tmp.rename(OUT)
    print(f"[done] scanned ~{pages} titles, {redirects} redirects -> "
          f"{len(alias)} unique aliases -> {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    build(resume=ap.parse_args().resume)
