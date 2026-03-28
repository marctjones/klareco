#!/usr/bin/env python3
"""
Corpus Coverage Audit using Kuzu Database

Directly queries the Kuzu database to find sentences containing expected answers.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Install with: pip install kuzu")
    sys.exit(1)

def load_test_set(path):
    """Load JSONL test set."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def search_kuzu(conn, keywords):
    """Search Kuzu database for sentences containing keywords."""

    # Build search query - case insensitive search
    # We'll search for each keyword separately then combine results
    all_matches = []

    for keyword in keywords:
        query = f"""
        MATCH (f:Frazoteksto)
        WHERE lower(f.teksto) CONTAINS lower('{keyword}')
        RETURN f.teksto AS text, f.id AS id
        LIMIT 20
        """

        try:
            result = conn.execute(query)
            while result.has_next():
                row = result.get_next()
                all_matches.append({
                    'text': row[0],
                    'id': row[1],
                    'keyword': keyword
                })
        except Exception as e:
            print(f"    Warning: Search for '{keyword}' failed: {e}")
            continue

    return all_matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--db', default='data/indexes/v2.1_kuzu_index_full')
    args = parser.parse_args()

    # Connect to Kuzu
    print(f"Opening Kuzu database: {args.db}")
    try:
        db = kuzu.Database(args.db)
        conn = kuzu.Connection(db)
    except Exception as e:
        print(f"ERROR: Failed to open Kuzu database: {e}")
        return

    # Load test set
    print(f"Loading test set from {args.test_set}")
    questions = load_test_set(args.test_set)
    print(f"Loaded {len(questions)} questions\n")

    # Audit each question
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        print(f"  Expected: {q['answer']}")
        print(f"  Keywords: {', '.join(q['expected_keywords'])}")

        # Search for each expected keyword
        matches = search_kuzu(conn, q['expected_keywords'])

        if matches:
            print(f"  ✓ FOUND {len(matches)} sentences in corpus")
            # Show first match
            first = matches[0]
            print(f"    First match (keyword: {first['keyword']}): {first['text'][:100]}...")
        else:
            print(f"  ✗ NOT FOUND in corpus")

        results.append({
            'question': q,
            'found': len(matches) > 0,
            'matches': matches
        })
        print()

    # Summary
    total = len(results)
    found = sum(1 for r in results if r['found'])
    print("="*80)
    print(f"SUMMARY: {found}/{total} ({found/total*100:.1f}%) questions have answers in corpus")
    print("="*80)

    # Write detailed report
    output_path = '/tmp/corpus_coverage_audit.md'
    with open(output_path, 'w') as f:
        f.write("# Corpus Coverage Audit (Kuzu Database)\n\n")
        f.write(f"- Test set: {args.test_set}\n")
        f.write(f"- Database: {args.db}\n")
        f.write(f"- Results: {found}/{total} ({found/total*100:.1f}%) found\n\n")
        f.write("---\n\n")

        for i, r in enumerate(results, 1):
            q = r['question']
            status = "✓" if r['found'] else "✗"

            f.write(f"## {status} Q{i}: {q['question']}\n\n")
            f.write(f"**Expected**: {q['answer']}\n\n")
            f.write(f"**Keywords**: {', '.join(q['expected_keywords'])}\n\n")

            if r['found']:
                f.write(f"**Status**: FOUND - {len(r['matches'])} sentences containing answer\n\n")

                # Group by keyword
                by_keyword = {}
                for m in r['matches']:
                    kw = m['keyword']
                    if kw not in by_keyword:
                        by_keyword[kw] = []
                    by_keyword[kw].append(m)

                for kw, matches in by_keyword.items():
                    f.write(f"### Keyword: `{kw}` ({len(matches)} sentences)\n\n")
                    for m in matches[:3]:  # Show up to 3 per keyword
                        f.write(f"> {m['text']}\n\n")
                    if len(matches) > 3:
                        f.write(f"*...and {len(matches)-3} more*\n\n")
            else:
                f.write(f"**Status**: NOT FOUND in corpus\n\n")
                f.write(f"**Diagnosis**: Corpus coverage gap - searched for keywords {q['expected_keywords']} but found no matching sentences\n\n")

            f.write("---\n\n")

    print(f"\nDetailed report: {output_path}")

if __name__ == '__main__':
    main()
