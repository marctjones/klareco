#!/usr/bin/env python3
"""
Validate root classification results.

Checks:
1. Known examples (mi, kaj, hund, dom, mal, la, kio)
2. Tier counts vs expected
3. Unua Libro coverage (should be 937 total)
4. Vorto propagation worked
5. No NULL values where unexpected
6. Tier boundaries make sense

Usage:
    python scripts/validate_classification.py --kuzu data/indexes/v2.1_kuzu_index_full
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    exit(1)


def validate_known_examples(conn):
    """Test known examples match expected classifications."""
    print("\n=== Test 1: Known Examples ===")

    test_cases = [
        # (radiko, expected_nivelo, expected_fonto, description)
        ('mi', 'tier0_pronomo', 'unua_libro', 'pronoun'),
        ('kaj', 'tier0_konjunkcio', 'unua_libro', 'conjunction'),
        ('mal', 'tier0_afikso', 'unua_libro', 'prefix'),
        ('iĝ', 'tier0_afikso', 'unua_libro', 'suffix'),
        ('kio', 'tier0_korelativo', 'unua_libro', 'correlative'),
        ('la', 'tier0_artikolo', 'unua_libro', 'article'),
        ('hund', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
        ('dom', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
        ('amik', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
        ('bon', 'tier1a_unua_libro', 'unua_libro', 'Unua Libro root'),
    ]

    all_passed = True
    for radiko, expected_nivelo, expected_fonto, desc in test_cases:
        result = conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{radiko}'}})
            RETURN r.nivelo, r.fonto, r.ofteco
        """)

        if result.has_next():
            nivelo, fonto, ofteco = result.get_next()

            if nivelo == expected_nivelo and fonto == expected_fonto:
                print(f"  ✓ {radiko:10s} → {nivelo:25s} fonto={fonto:15s} ofteco={ofteco:,} ({desc})")
            else:
                print(f"  ✗ {radiko:10s} → Expected: {expected_nivelo}, {expected_fonto}")
                print(f"              → Got:      {nivelo}, {fonto}")
                all_passed = False
        else:
            print(f"  ✗ {radiko:10s} → NOT FOUND")
            all_passed = False

    if all_passed:
        print("\n✓ All known examples passed!")
    else:
        print("\n✗ Some examples failed!")
        return False

    return True


def validate_tier_counts(conn):
    """Check tier counts are reasonable."""
    print("\n=== Test 2: Tier Count Validation ===")

    expected_ranges = {
        'tier0_': (180, 200),  # ~190 total across 10 subcategories
        'tier1a_unua_libro': (700, 800),  # ~750 (some may be missing)
        'tier1b_fundamento': (1200, 1600),  # ~1,403
        'tier2_revo': (5000, 25000),  # ~7,730 (ReVo subset)
        'tier3_korpuso': (50000, 80000),  # ~66,555
        'tier4_propranomo': (60000, 80000),  # ~68,981
        'tier5_rubaĵo': (900000, 1100000),  # ~1,016,666
    }

    all_passed = True

    # Get tier counts
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo IS NOT NULL
        WITH r.nivelo as tier, count(r) as cnt
        RETURN tier, cnt
        ORDER BY tier
    """)

    tier_counts = {}
    while result.has_next():
        tier, count = result.get_next()
        tier_counts[tier] = count

    # Check tier0 total (sum of all subcategories)
    tier0_total = sum(count for tier, count in tier_counts.items() if tier.startswith('tier0_'))
    min_val, max_val = expected_ranges['tier0_']
    if min_val <= tier0_total <= max_val:
        print(f"  ✓ Tier 0 (grammatical): {tier0_total} (expected {min_val}-{max_val})")
    else:
        print(f"  ✗ Tier 0 (grammatical): {tier0_total} (expected {min_val}-{max_val})")
        all_passed = False

    # Check specific tiers
    for tier_prefix, (min_val, max_val) in expected_ranges.items():
        if tier_prefix == 'tier0_':
            continue  # Already checked

        count = tier_counts.get(tier_prefix, 0)
        if min_val <= count <= max_val:
            print(f"  ✓ {tier_prefix:20s}: {count:,} (expected {min_val:,}-{max_val:,})")
        else:
            print(f"  ✗ {tier_prefix:20s}: {count:,} (expected {min_val:,}-{max_val:,})")
            all_passed = False

    if all_passed:
        print("\n✓ All tier counts within expected ranges!")
    else:
        print("\n✗ Some tier counts outside expected ranges!")

    return all_passed


def validate_unua_libro_coverage(conn):
    """Check Unua Libro total (tier0 + tier1a)."""
    print("\n=== Test 3: Unua Libro Coverage ===")

    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.fonto = 'unua_libro'
        RETURN count(r)
    """)

    unua_libro_total = result.get_next()[0]

    # Expected: 937 (190 tier0 + 787 tier1a, but some may be missing)
    # Allow range 900-980
    if 900 <= unua_libro_total <= 980:
        print(f"  ✓ Unua Libro total: {unua_libro_total} (expected ~937)")
        print(f"    This includes tier0 grammatical words + tier1a lexical roots")
        return True
    else:
        print(f"  ✗ Unua Libro total: {unua_libro_total} (expected ~937)")
        return False


def validate_vorto_propagation(conn):
    """Check Vorto nodes received classifications."""
    print("\n=== Test 4: Vorto Propagation ===")

    # Check total Vorto nodes
    result = conn.execute("MATCH (v:Vorto) RETURN count(v)")
    total_vorto = result.get_next()[0]

    # Check how many have classifications
    result = conn.execute("""
        MATCH (v:Vorto)
        WHERE v.radiko_nivelo IS NOT NULL
        RETURN count(v)
    """)
    classified_vorto = result.get_next()[0]

    pct = 100 * classified_vorto / total_vorto if total_vorto > 0 else 0

    # Most Vorto nodes should have classifications (>90%)
    if pct > 90:
        print(f"  ✓ Vorto propagation: {classified_vorto:,} / {total_vorto:,} ({pct:.1f}%)")
        return True
    else:
        print(f"  ⚠ Vorto propagation: {classified_vorto:,} / {total_vorto:,} ({pct:.1f}%)")
        print(f"    Expected >90% to have classifications")
        return False


def validate_no_unexpected_nulls(conn):
    """Check for NULL values where unexpected."""
    print("\n=== Test 5: NULL Value Check ===")

    all_passed = True

    # All Radiko nodes should have nivelo
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo IS NULL
        RETURN count(r)
    """)
    null_nivelo = result.get_next()[0]

    if null_nivelo == 0:
        print(f"  ✓ All Radiko nodes have nivelo")
    else:
        print(f"  ✗ {null_nivelo:,} Radiko nodes missing nivelo")
        all_passed = False

    # Tier0-4 should have fonto
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo IN ['tier0_afikso', 'tier0_korelativo', 'tier0_pronomo',
                           'tier0_prepozicio', 'tier0_konjunkcio', 'tier0_partiklo',
                           'tier0_nombro', 'tier0_finaco', 'tier0_artikolo', 'tier0_komparativo',
                           'tier1a_unua_libro', 'tier1b_fundamento',
                           'tier2_revo', 'tier3_korpuso', 'tier4_propranomo']
          AND r.fonto IS NULL
        RETURN count(r)
    """)
    null_fonto = result.get_next()[0]

    if null_fonto == 0:
        print(f"  ✓ All tier0-4 nodes have fonto")
    else:
        print(f"  ⚠ {null_fonto:,} tier0-4 nodes missing fonto")
        all_passed = False

    # Tier5-6 should NOT have fonto (garbage/unknown)
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo IN ['tier5_rubaĵo', 'tier6_nekonata']
          AND r.fonto IS NOT NULL
        RETURN count(r)
    """)
    unexpected_fonto = result.get_next()[0]

    if unexpected_fonto == 0:
        print(f"  ✓ Tier5-6 (garbage/unknown) have NULL fonto")
    else:
        print(f"  ⚠ {unexpected_fonto:,} tier5-6 nodes have unexpected fonto")
        all_passed = False

    if all_passed:
        print("\n✓ No unexpected NULL values!")
    else:
        print("\n⚠ Some unexpected NULL values found")

    return all_passed


def validate_tier_boundaries(conn):
    """Spot check tier boundaries make sense."""
    print("\n=== Test 6: Tier Boundary Spot Checks ===")

    all_passed = True

    # Sample some tier1a_unua_libro roots
    print("\n  Tier 1a (Unua Libro) samples:")
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo = 'tier1a_unua_libro'
        RETURN r.radiko
        ORDER BY r.ofteco DESC
        LIMIT 5
    """)

    samples = []
    while result.has_next():
        samples.append(result.get_next()[0])
    print(f"    {', '.join(samples)}")

    # Sample some tier4_propranomo (proper names)
    print("\n  Tier 4 (Proper names) samples:")
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo = 'tier4_propranomo'
        RETURN r.radiko
        ORDER BY r.ofteco DESC
        LIMIT 5
    """)

    samples = []
    while result.has_next():
        samples.append(result.get_next()[0])
    print(f"    {', '.join(samples)}")

    # Check that tier5 (garbage) has low ofteco
    result = conn.execute("""
        MATCH (r:Radiko)
        WHERE r.nivelo = 'tier5_rubaĵo'
        RETURN avg(r.ofteco) as avg_ofteco, max(r.ofteco) as max_ofteco
    """)

    avg_ofteco, max_ofteco = result.get_next()
    print(f"\n  Tier 5 (Parse failures) ofteco stats:")
    print(f"    Average: {avg_ofteco:.1f}")
    print(f"    Maximum: {max_ofteco:,}")

    # Tier5 should generally have low usage
    if avg_ofteco < 10:
        print(f"    ✓ Low average usage (expected for parse failures)")
    else:
        print(f"    ⚠ Higher than expected average usage")
        all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Validate classification results')
    parser.add_argument('--kuzu', type=Path, required=True, help='Kuzu database path')

    args = parser.parse_args()

    print("=" * 60)
    print("CLASSIFICATION VALIDATION")
    print("=" * 60)

    # Connect to database (read-only)
    print(f"\nOpening database: {args.kuzu}")
    db = kuzu.Database(str(args.kuzu), read_only=True)
    conn = kuzu.Connection(db)

    # Run all validation tests
    results = []

    results.append(("Known Examples", validate_known_examples(conn)))
    results.append(("Tier Counts", validate_tier_counts(conn)))
    results.append(("Unua Libro Coverage", validate_unua_libro_coverage(conn)))
    results.append(("Vorto Propagation", validate_vorto_propagation(conn)))
    results.append(("NULL Values", validate_no_unexpected_nulls(conn)))
    results.append(("Tier Boundaries", validate_tier_boundaries(conn)))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} - {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n  {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ ALL VALIDATION TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} validation test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
