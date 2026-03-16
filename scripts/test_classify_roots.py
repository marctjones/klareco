#!/usr/bin/env python3
"""
Test classification logic on small sample before full run.

Tests:
1. Load all tier0 words correctly
2. Load Unua Libro roots correctly
3. Classification logic works
4. Database updates work
5. Both nivelo and fonto are set correctly

Usage:
    python scripts/test_classify_roots.py --kuzu data/indexes/v2.1_kuzu_index_full --limit 1000
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    exit(1)


def load_tier0_grammatical_words(vocab_dir: Path) -> dict:
    """Load all tier0 grammatical words."""
    tier0_file = vocab_dir / 'tier0_grammatical_words.json'

    if not tier0_file.exists():
        print(f"ERROR: {tier0_file} not found")
        exit(1)

    with open(tier0_file, 'r') as f:
        tier0_data = json.load(f)

    word_to_tier0 = {}

    for category_key in ['tier0_afikso', 'tier0_korelativo', 'tier0_pronomo',
                          'tier0_prepozicio', 'tier0_konjunkcio', 'tier0_partiklo',
                          'tier0_nombro', 'tier0_finaco', 'tier0_artikolo', 'tier0_komparativo']:
        if category_key in tier0_data:
            category_data = tier0_data[category_key]
            words = category_data.get('words', [])

            if isinstance(words, dict):
                for word in words.keys():
                    word_to_tier0[word] = category_key
            elif isinstance(words, list):
                for word in words:
                    word_to_tier0[word] = category_key

            for key in ['suffixes', 'prefixes']:
                if key in category_data:
                    for word in category_data[key].keys():
                        word_to_tier0[word] = category_key

    print(f"✓ Loaded {len(word_to_tier0)} tier0 grammatical words")
    return word_to_tier0


def load_unua_libro_roots(vocab_dir: Path) -> set:
    """Load the 787 Unua Libro lexical roots."""
    ub_file = vocab_dir / 'unua_libro_original_roots.json'

    if not ub_file.exists():
        print(f"ERROR: {ub_file} not found")
        exit(1)

    with open(ub_file, 'r') as f:
        ub_data = json.load(f)

    roots = set(ub_data.keys())
    print(f"✓ Loaded {len(roots)} Unua Libro lexical roots")
    return roots


def load_classification_data(vocab_dir: Path) -> tuple:
    """Load all classification sources."""
    print("\n=== Loading Classification Data ===")

    tier0_words = load_tier0_grammatical_words(vocab_dir)
    unua_libro_roots = load_unua_libro_roots(vocab_dir)

    class_file = vocab_dir / 'root_classification.json'
    if not class_file.exists():
        print(f"ERROR: {class_file} not found")
        exit(1)

    with open(class_file, 'r') as f:
        classification = json.load(f)

    fundamento_all = set(classification['tier1_fundamento']['roots'])
    tier1b_roots = fundamento_all - unua_libro_roots - set(tier0_words.keys())
    revo_roots = set(classification['tier2_revo']['roots']) - fundamento_all - set(tier0_words.keys())
    corpus_roots = set(classification['tier3_corpus']['roots']) - fundamento_all - revo_roots - set(tier0_words.keys())
    proper_names = set(classification['tier4_proper_names']['roots'])

    failure_file = vocab_dir / 'parse_failures.json'
    parse_failures = set()
    if failure_file.exists():
        with open(failure_file, 'r') as f:
            failures = json.load(f)
        parse_failures = set(failures.keys())

    corpus_file = vocab_dir / 'corpus_validated_roots_clean.json'
    corpus_usage = {}
    if corpus_file.exists():
        with open(corpus_file, 'r') as f:
            corpus_data = json.load(f)
        corpus_usage = {root: data.get('usage', 0) for root, data in corpus_data.items()}

    print(f"✓ Tier 0 (grammatical): {len(tier0_words):,}")
    print(f"✓ Tier 1a (Unua Libro): {len(unua_libro_roots):,}")
    print(f"✓ Tier 1b (Fundamento): {len(tier1b_roots):,}")
    print(f"✓ Tier 2 (ReVo): {len(revo_roots):,}")
    print(f"✓ Tier 3 (Corpus): {len(corpus_roots):,}")
    print(f"✓ Tier 4 (Proper names): {len(proper_names):,}")
    print(f"✓ Tier 5 (Garbage): {len(parse_failures):,}")

    return (tier0_words, unua_libro_roots, tier1b_roots, revo_roots,
            corpus_roots, proper_names, parse_failures, corpus_usage)


def determine_classification(root: str, tier0_words, unua_libro_roots, tier1b_roots,
                             revo_roots, corpus_roots, proper_names, parse_failures) -> tuple:
    """Determine tier and source."""
    if root in tier0_words:
        return (tier0_words[root], 'unua_libro')
    if root in unua_libro_roots:
        return ('tier1a_unua_libro', 'unua_libro')
    if root in tier1b_roots:
        return ('tier1b_fundamento', 'fundamento')
    if root in revo_roots:
        return ('tier2_revo', 'revo')
    if root in corpus_roots:
        return ('tier3_korpuso', 'korpuso')
    if root in proper_names:
        return ('tier4_propranomo', 'propranomo')
    if root in parse_failures:
        return ('tier5_rubaĵo', None)
    return ('tier6_nekonata', None)


def test_classification_logic(classification_data):
    """Test classification logic with known examples."""
    print("\n=== Testing Classification Logic ===")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    test_cases = [
        # (root, expected_nivelo, expected_fonto, description)
        ('mi', 'tier0_pronomo', 'unua_libro', 'pronoun'),
        ('kaj', 'tier0_konjunkcio', 'unua_libro', 'conjunction'),
        ('mal', 'tier0_afikso', 'unua_libro', 'prefix'),
        ('iĝ', 'tier0_afikso', 'unua_libro', 'suffix'),
        ('kio', 'tier0_korelativo', 'unua_libro', 'correlative'),
        ('la', 'tier0_artikolo', 'unua_libro', 'article'),
        ('hund', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
        ('dom', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
    ]

    all_passed = True
    for root, expected_nivelo, expected_fonto, desc in test_cases:
        nivelo, fonto = determine_classification(
            root, tier0_words, unua_libro_roots, tier1b_roots,
            revo_roots, corpus_roots, proper_names, parse_failures
        )

        if nivelo == expected_nivelo and fonto == expected_fonto:
            print(f"  ✓ {root:10s} → {nivelo:25s} fonto={fonto:15s} ({desc})")
        else:
            print(f"  ✗ {root:10s} → Expected: {expected_nivelo}, {expected_fonto}")
            print(f"              → Got:      {nivelo}, {fonto}")
            all_passed = False

    if all_passed:
        print("\n✓ All classification tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1)


def test_database_update(conn, classification_data, limit):
    """Test updating database with small sample."""
    print(f"\n=== Testing Database Updates (limit={limit}) ===")

    tier0_words, unua_libro_roots, tier1b_roots, revo_roots, \
    corpus_roots, proper_names, parse_failures, corpus_usage = classification_data

    # Get sample of roots
    result = conn.execute(f"""
        MATCH (r:Radiko)
        RETURN r.radiko
        LIMIT {limit}
    """)

    roots = []
    while result.has_next():
        (radiko,) = result.get_next()
        roots.append(radiko)

    print(f"  Fetched {len(roots)} sample roots")

    # Classify and update
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    for radiko in roots:
        nivelo, fonto = determine_classification(
            radiko, tier0_words, unua_libro_roots, tier1b_roots,
            revo_roots, corpus_roots, proper_names, parse_failures
        )

        ofteco = corpus_usage.get(radiko, 0)

        radiko_escaped = radiko.replace("'", "\\'")
        nivelo_escaped = nivelo.replace("'", "\\'")

        set_clause = f"r.nivelo = '{nivelo_escaped}', r.ofteco = {ofteco}"
        if fonto:
            fonto_escaped = fonto.replace("'", "\\'")
            set_clause += f", r.fonto = '{fonto_escaped}'"

        conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{radiko_escaped}'}})
            SET {set_clause}
        """)

        tier_counts[nivelo] += 1
        if fonto:
            source_counts[fonto] += 1

    print(f"  Updated {len(roots)} roots")

    # Show distribution
    print("\n  Tier distribution in sample:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / len(roots)
        print(f"    {tier:25s}: {count:4d} ({pct:5.1f}%)")

    print("\n  Source distribution in sample:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        pct = 100 * count / len(roots)
        print(f"    {source:15s}: {count:4d} ({pct:5.1f}%)")


def verify_sample(conn, limit):
    """Verify classification was applied correctly."""
    print("\n=== Verifying Sample ===")

    # Check some specific examples
    test_words = ['mi', 'kaj', 'hund', 'dom', 'mal', 'la', 'kio']

    for word in test_words:
        result = conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{word}'}})
            RETURN r.nivelo, r.fonto, r.ofteco
            LIMIT 1
        """)

        if result.has_next():
            nivelo, fonto, ofteco = result.get_next()
            nivelo_str = nivelo if nivelo else 'NULL'
            fonto_str = fonto if fonto else 'NULL'
            ofteco_str = str(ofteco) if ofteco is not None else '0'
            print(f"  {word:6s} → nivelo={nivelo_str:25s} fonto={fonto_str:15s} ofteco={ofteco_str}")
        else:
            print(f"  {word:6s} → NOT FOUND")

    # Show tier distribution in database (limited to our sample)
    print(f"\n  Tier distribution (all nodes with nivelo set):")
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo IS NOT NULL
        WITH r.nivelo as tier, count(r) as cnt
        RETURN tier, cnt
        ORDER BY tier
        LIMIT 20
    """)

    total = 0
    while result.has_next():
        tier, count = result.get_next()
        total += count
        print(f"    {tier:25s}: {count:,}")

    print(f"\n  Total nodes classified: {total:,}")


def main():
    parser = argparse.ArgumentParser(description='Test classification on small sample')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'))
    parser.add_argument('--limit', type=int, default=1000, help='Number of nodes to test')

    args = parser.parse_args()

    # Load classification data
    classification_data = load_classification_data(args.vocab_dir)

    # Test classification logic
    test_classification_logic(classification_data)

    # Connect to database
    print(f"\n=== Connecting to Database ===")
    print(f"  Database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu))
    conn = kuzu.Connection(db)
    print("  ✓ Connected")

    # Add schema properties if needed
    print("\n=== Ensuring Schema Properties ===")
    for table, prop, dtype in [
        ('Radiko', 'nivelo', 'STRING'),
        ('Radiko', 'fonto', 'STRING'),
        ('Radiko', 'ofteco', 'INT64'),
        ('Vorto', 'radiko_nivelo', 'STRING'),
        ('Vorto', 'radiko_fonto', 'STRING'),
        ('Vorto', 'radiko_ofteco', 'INT64'),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD {prop} {dtype}")
            print(f"  ✓ Added {table}.{prop}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  ✓ {table}.{prop} already exists")
            else:
                print(f"  ✗ Error: {e}")

    # Test database updates
    test_database_update(conn, classification_data, args.limit)

    # Verify
    verify_sample(conn, args.limit)

    print("\n✓ Test complete!")
    print(f"\nIf everything looks good, run full classification:")
    print(f"  python scripts/classify_roots_v2.py --kuzu {args.kuzu}")


if __name__ == '__main__':
    main()
