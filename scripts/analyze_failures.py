"""
Analyze Parse Failures and Suggest Root Additions

Analyzes failed parse attempts to identify missing roots.
"""

from pathlib import Path
from collections import Counter
import re

# Known missing roots from analysis
MISSING_ROOTS_TO_ADD = {
    # From failure analysis
    "kareks",      # sedge (plant)
    "propriet",    # property/proprietor
    "region",      # region
    "trankv",      # calm, tranquil
    "alfabet",     # alphabet
    "liĝ",         # law
    "punkt",       # point
    "manier",      # manner
    "preciz",      # precise
    "sven",        # faint, swoon
    "distingi",    # distinguish (full root? or disting?)
    "disting",     # distinguish
    "renkont",     # encounter, meet
    "distanc",     # distance
    "proprietant", # proprietor (alt form)
    "demand",      # ask, demand
    "bord",        # edge, border
    "skrib",       # write
    "miz",         # misery
    "memor",       # memory
    "fakt",        # fact
    "mir",         # wonder, marvel
    "ofer",        # offer, sacrifice
    "kord",        # cord, heart
    "nask",        # birth, be born
    "labor",       # work
    "vicest",      # vice-, deputy
    "redakt",      # edit, redact
    "prezid",      # preside
    "akademi",     # academy
    "vekiĝ",       # awaken (compound?)

    # Common roots that should be there
    "aŭd",         # hear
    "sent",        # feel
    "don",         # give
    "vok",         # call
    "pens",        # think
    "parol",       # speak
    "respond",     # respond
    "demand",      # ask
    "konfirm",     # confirm
    "absolut",     # absolute
    "dialog",      # dialogue
    "sistematik",  # systematic
}

def suggest_roots():
    """Suggest roots to add based on failure analysis."""
    print("="*70)
    print("Missing Roots to Add")
    print("="*70)
    print()

    print(f"Total roots to add: {len(MISSING_ROOTS_TO_ADD)}")
    print()

    print("Roots (alphabetically sorted):")
    for root in sorted(MISSING_ROOTS_TO_ADD):
        print(f'    "{root}",')

    print()
    print("="*70)
    print("To add to parser.py KNOWN_ROOTS, or better yet, merge with")
    print("data/merged_vocabulary.py")
    print("="*70)

if __name__ == '__main__':
    suggest_roots()
