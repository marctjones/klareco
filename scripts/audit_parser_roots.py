#!/usr/bin/env python3
"""
Audit KNOWN_ROOTS in parser for non-Esperanto words.

Checks for:
1. English-only words (not also valid Esperanto)
2. Proper names (characters, places)
3. Words that look suspicious
"""

import re
from pathlib import Path

# Known problematic patterns
SUSPICIOUS_PATTERNS = {
    # English words that might not be Esperanto
    'english_suspects': [
        'help', 'bird', 'turn', 'respond', 'demand', 'region',
        'best', 'just', 'pet', 'man', 'cel', 'long', 'mal jung'
    ],

    # Proper names (should NOT be in KNOWN_ROOTS)
    'proper_names': [
        'gandalf', 'bilbo', 'frodo', 'aragorn', 'tolkien', 'poe',
        'alice', 'hobbit', 'mordor', 'shire', 'rivendell'
    ],

    # Fantasy/fictional terms
    'fantasy_terms': [
        'elf', 'elvish', 'orc', 'dwarf', 'wizard', 'ring',
        'silmaril', 'palantir', 'mithril'
    ],

    # Compound words (should be built from roots, not listed as roots)
    'potential_compounds': [
        'maljung',  # mal + jung
        'malbon',   # mal + bon
        'malbel',   # mal + bel
    ]
}

# Valid Esperanto roots that LOOK English but ARE legitimate
VALID_INTERNATIONAL_ROOTS = {
    'help': 'helpi (to help) - borrowed from Germanic languages',
    'bird': 'birdo (bird) - borrowed from English',
    'turn': 'turni (to turn) - international word',
    'respond': 'respondi (to respond) - from Latin/Romance',
    'demand': 'demandi (to demand) - from Latin/Romance',
    'region': 'regiono (region) - international word',
    'best': 'besto (beast) - international word',
    'just': 'justa (just) - from Latin',
    'pet': 'peti (to request) - from Latin "petere"',
    'man': 'mano (hand) - from Latin "manus"',
    'cel': 'celo (aim/goal) - from Latin',
    'dialog': 'dialogo (dialogue) - from Greek',
    'alfabet': 'alfabeto (alphabet) - from Greek',
    'punkt': 'punkto (point) - from Latin "punctum"',
    'fakt': 'fakto (fact) - from Latin "factum"',
    'long': 'longa (long) - from Romance languages',
    'sven': 'sveni (to faint/swoon) - standard Esperanto',
    'disting': 'distingi (to distinguish) - from Latin "distinguere"',
}

# Questionable roots that need verification (REMOVED from parser after audit)
# These were removed because they're not standard Esperanto:
# - 'propriet'     # Not standard, should be "propraĵo" or "posedaĵo"
# - 'proprietant'  # Not standard
# - 'kareks'       # Too specialized, not in standard dictionaries
# - 'vicest'       # Questionable, "vic-" is typically a prefix
# - 'mal jung'     # Had a space - clearly an error
NEEDS_VERIFICATION = []


def extract_roots_from_parser():
    """Extract all roots from parser.py KNOWN_ROOTS."""
    parser_file = Path(__file__).parent.parent / 'klareco' / 'parser.py'

    with open(parser_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find KNOWN_ROOTS section
    match = re.search(r'KNOWN_ROOTS = \{(.*?)\n\}', content, re.DOTALL)
    if not match:
        print("ERROR: Could not find KNOWN_ROOTS in parser.py")
        return []

    roots_section = match.group(1)

    # Extract roots with comments
    roots = []
    for line in roots_section.split('\n'):
        # Match pattern: "root", # comment
        match = re.match(r'\s*"([^"]+)",?\s*#\s*(.+)', line)
        if match:
            root = match.group(1)
            comment = match.group(2).strip()
            roots.append((root, comment))

    return roots


def audit_roots(roots):
    """Audit roots for potential issues."""

    issues = {
        'proper_names_found': [],
        'fantasy_terms_found': [],
        'suspicious_english': [],
        'needs_verification': [],
        'compounds_as_roots': [],
    }

    print(f"\n{'='*70}")
    print("PARSER ROOTS AUDIT")
    print(f"{'='*70}\n")
    print(f"Total roots to check: {len(roots)}\n")

    for root, comment in roots:
        root_lower = root.lower()

        # Check for proper names
        if root_lower in SUSPICIOUS_PATTERNS['proper_names']:
            issues['proper_names_found'].append((root, comment))

        # Check for fantasy terms
        if root_lower in SUSPICIOUS_PATTERNS['fantasy_terms']:
            issues['fantasy_terms_found'].append((root, comment))

        # Check for suspicious English words
        if root_lower in SUSPICIOUS_PATTERNS['english_suspects']:
            if root_lower not in VALID_INTERNATIONAL_ROOTS:
                issues['suspicious_english'].append((root, comment))

        # Check for words needing verification
        if root_lower in [v.lower() for v in NEEDS_VERIFICATION]:
            issues['needs_verification'].append((root, comment))

        # Check for compound words with "mal" prefix
        if root.startswith('mal') and len(root) > 3:
            # Check if it's in the compounds list
            if root_lower in SUSPICIOUS_PATTERNS['potential_compounds']:
                issues['compounds_as_roots'].append((root, comment))

    # Report findings
    print(f"{'='*70}")
    print("AUDIT RESULTS")
    print(f"{'='*70}\n")

    # Proper names
    if issues['proper_names_found']:
        print("❌ CRITICAL: Proper names found in KNOWN_ROOTS:")
        for root, comment in issues['proper_names_found']:
            print(f"   - '{root}' # {comment}")
        print()
    else:
        print("✅ No proper names found\n")

    # Fantasy terms
    if issues['fantasy_terms_found']:
        print("❌ CRITICAL: Fantasy/fictional terms found:")
        for root, comment in issues['fantasy_terms_found']:
            print(f"   - '{root}' # {comment}")
        print()
    else:
        print("✅ No fantasy terms found\n")

    # Suspicious English
    if issues['suspicious_english']:
        print("⚠️  WARNING: English-looking words (not in valid list):")
        for root, comment in issues['suspicious_english']:
            print(f"   - '{root}' # {comment}")
        print()
    else:
        print("✅ No suspicious English words\n")

    # Needs verification
    if issues['needs_verification']:
        print("⚠️  INFO: Roots needing manual verification:")
        for root, comment in issues['needs_verification']:
            print(f"   - '{root}' # {comment}")
        print()

    # Compounds
    if issues['compounds_as_roots']:
        print("ℹ️  NOTE: Compound words listed as roots (intentional?):")
        for root, comment in issues['compounds_as_roots']:
            print(f"   - '{root}' # {comment}")
        print()

    # Valid international roots that look English
    print(f"\n{'='*70}")
    print("VALID INTERNATIONAL ROOTS (look English but are Esperanto)")
    print(f"{'='*70}\n")

    for root, comment in roots:
        if root.lower() in VALID_INTERNATIONAL_ROOTS:
            explanation = VALID_INTERNATIONAL_ROOTS[root.lower()]
            print(f"✓ '{root}' # {comment}")
            print(f"  → {explanation}\n")

    return issues


def main():
    roots = extract_roots_from_parser()

    if not roots:
        print("ERROR: No roots extracted")
        return

    issues = audit_roots(roots)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    total_issues = sum([
        len(issues['proper_names_found']),
        len(issues['fantasy_terms_found']),
        len(issues['suspicious_english']),
    ])

    if total_issues == 0:
        print("✅ No critical issues found!")
        print("✅ Parser vocabulary appears clean")
    else:
        print(f"⚠️  Found {total_issues} potential issues")
        print("   Review the warnings above and verify roots")

    if issues['needs_verification']:
        print(f"\nℹ️  {len(issues['needs_verification'])} roots need manual verification")
        print("   These should be checked against ReVo or other Esperanto dictionary")


if __name__ == '__main__':
    main()
