"""Discourse-level (cross-sentence) deterministic analysis.

Everything in `klareco/parser.py` operates on ONE sentence at a time — by
design (VISION.md: attempt each capability deterministically, at the
smallest scope that can be measured, before reaching for anything else).
This package is the next scope up: rules that need to see MULTIPLE
sentences from the same document to do their job, starting with pronoun
coreference candidate generation (`coreference.py`).

Nothing here is wired into the default orchestrator pipeline. Per the
project's contract, a new capability is default-OFF until it passes the
contract suite and carries a measured number (CLAUDE.md, "THE ORCHESTRATION
CONTRACT").
"""
