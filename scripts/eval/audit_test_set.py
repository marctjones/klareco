#!/usr/bin/env python3
"""
Test-set validity auditor

VERSION: v2.1
COMPATIBLE WITH: v2.1 corpus (Whoosh index data/indexes/whoosh_fts)
DEPENDENCIES: data/indexes/whoosh_fts ; klareco.parser
STAGE: Evaluation

For an arbitrary QA test set, reports per question:
  - corpus_answer: does the Esperanto corpus contain a sentence that
    plausibly answers it? (BM25 over keywords + question content terms;
    a question with NO answer-bearing sentence is UNWINNABLE by any
    retriever and is pure metric-rot)
  - source_ok: if the set carries a source_sentence_id, does that id
    exist in the corpus and does the expected answer text appear in it?
  - coherence flags (mechanical, high-precision only):
      * kiu_nonperson : "Kiu" (who) question whose keyed answer is not
        a person/proper-noun  -> likely should be "Kio"
      * unparsed_content : a content word the parser cannot analyze and
        which is not a recognized proper noun -> possible vocabulary
        error in the question itself

Whoosh-only (no Kuzu) so it is safe to run alongside an eval.
Aggregate output: unwinnable rate + flagged ids — the numbers needed to
judge how much of a set is defective vs. measuring the model.

Usage:
    python scripts/eval/audit_test_set.py data/test_sets/qa_test_set_50.jsonl
    python scripts/eval/audit_test_set.py PATH --show 5

Last Updated: 2026-05-16
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from klareco.parser import parse

STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis estos '
           'la de en al el ĉu por kaj aŭ ke ne'.split())
PERSON_HINT = re.compile(r'[A-ZĈĜĤĴŜŬ]')  # capitalized -> likely proper noun


def norm(r: dict) -> dict:
    return {
        'id': r.get('id', '?'),
        'question': r.get('question', ''),
        'answer': (r.get('answer') or r.get('expected_answer')
                   or r.get('expected_answer_pattern') or ''),
        'keywords': r.get('expected_keywords') or [],
        'qtype': (r.get('question_type') or '').upper(),
        'source_id': r.get('source_sentence_id'),
        'source_text': r.get('source_sentence_text', ''),
    }


def content_terms(q: str) -> list[str]:
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in STOP and len(t) > 2]


def coherence_flags(n: dict) -> list[str]:
    flags = []
    # Kiu (who) but keyed answer is not a person/proper-noun
    if n['question'].split() and n['question'].split()[0].lower() == 'kiu':
        ans = str(n['answer'])
        kws = ' '.join(map(str, n['keywords']))
        if ans and not PERSON_HINT.search(ans) and not PERSON_HINT.search(kws):
            flags.append('kiu_nonperson')
    # content word the parser cannot analyze (possible bad vocabulary)
    try:
        ast = parse(n['question'])

        def walk(node, out):
            if not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                out.append(node)
                return
            if node.get('kerno'):
                walk(node['kerno'], out)
            for c in node.get('priskriboj', []) or []:
                walk(c, out)
        words = []
        for k in ('subjekto', 'verbo', 'objekto'):
            if ast.get(k):
                walk(ast[k], words)
        for a in ast.get('aliaj', []) or []:
            walk(a, words)
        for w in words:
            st = w.get('analizstato')
            vs = w.get('vortspeco')
            if st and st != 'sukceso' and vs != 'propra_nomo':
                flags.append(f"unparsed:{w.get('plena_vorto')}")
    except Exception:
        pass
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--whoosh', default='data/indexes/whoosh_fts')
    ap.add_argument('--show', type=int, default=0,
                    help='print top-N corpus hits per question')
    args = ap.parse_args()

    rows = [norm(json.loads(l)) for l in open(args.path) if l.strip()]
    ix = open_dir(args.whoosh)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    unwinnable, src_bad, flagged = [], [], []
    with ix.searcher() as s:
        valid_ids = None  # lazily only if needed
        for n in rows:
            terms = content_terms(n['question']) + [str(k) for k in n['keywords']]
            q = qp.parse(' OR '.join(
                f'"{t}"' if ' ' in t else t for t in terms) or '""')
            hits = s.search(q, limit=max(args.show, 5))

            # corpus-answer heuristic: a sentence is answer-bearing only
            # if it contains BOTH an expected keyword AND a salient
            # question topic-term (co-occurrence = on-topic + has the
            # answer). Keyword-anywhere is far too lenient — common
            # words like "sep"/"herbo" appear in thousands of unrelated
            # sentences and produce false "winnable" verdicts.
            kws = [str(k).lower() for k in n['keywords']]
            qterms = [t for t in content_terms(n['question'])]
            has_ans = False
            for h in hits:
                tl = h['text'].lower()
                if (any(k in tl for k in kws)
                        and any(t in tl for t in qterms)):
                    has_ans = True
                    break
            if not has_ans:
                unwinnable.append(n['id'])

            # source_sentence_id check
            if n['source_id'] is not None:
                doc = s.document(id=str(n['source_id']))
                if not doc:
                    src_bad.append((n['id'], 'source id not in corpus'))
                elif n['keywords']:
                    tl = (doc.get('text') or '').lower()
                    if not any(str(k).lower() in tl for k in n['keywords']):
                        src_bad.append((n['id'], 'answer not in source sentence'))

            fl = coherence_flags(n)
            if fl:
                flagged.append((n['id'], fl))

            if args.show:
                print(f"\n{n['id']} [{n['qtype']}] {n['question']}")
                print(f"  answer={n['answer']!r} kws={n['keywords']} "
                      f"corpus_answer={'Y' if has_ans else 'N'}")
                for h in hits[:args.show]:
                    print(f"   id={h['id']} {' '.join(h['text'].split())[:150]}")

    total = len(rows)
    print(f"\n{'='*64}\n{args.path}  ({total} questions)")
    print(f"  UNWINNABLE (no answer-bearing sentence in corpus): "
          f"{len(unwinnable)}/{total} ({len(unwinnable)/total*100:.0f}%)")
    if unwinnable:
        print(f"    ids: {unwinnable}")
    if src_bad:
        print(f"  SOURCE-SENTENCE problems: {len(src_bad)}")
        for i, why in src_bad[:20]:
            print(f"    {i}: {why}")
    if flagged:
        print(f"  COHERENCE-flagged: {len(flagged)}")
        for i, fl in flagged[:30]:
            print(f"    {i}: {fl}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
