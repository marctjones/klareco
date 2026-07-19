# UD Esperanto gold treebanks — parser-quality test fixtures

These two CoNLL-U files are committed **as test fixtures** so the parser-quality
regression suite (`tests/test_parser_ud_accuracy.py`, marker `accuracy`) is
self-contained and runs in CI without the (gitignored) `data/external/` tree.

| file | source treebank | sentences | tokens |
|------|-----------------|-----------|--------|
| `eo_prago-ud-test.conllu` | UD_Esperanto-Prago (test split) | 131 | ~3,166 |
| `eo_cairo-ud-test.conllu`  | UD_Esperanto-Cairo (test split, **held-out**) | 20 | ~177 |

## Provenance & license

Universal Dependencies Esperanto treebanks, released **CC-BY-SA 4.0**.
See Oya, *"UD Treebanks for Esperanto as a Natural Language"*, UDW/SyntaxFest
2025. Redistributed here under the same CC-BY-SA 4.0 license with attribution.
These are linguist-curated gold-standard annotations — the only external ground
truth the deterministic parser is measured against.

**Do not edit these files.** They are the frozen ruler. If the upstream UD
release changes, update `data/external/` first, re-run the evaluators, and copy
fresh fixtures here in a single deliberate commit (and re-baseline the floors in
the test).

Cairo is treated as **held-out** — it must never inform parser rule changes; it
is the honest generalization check. Prago is in-corpus (the parser has been
tuned with awareness of it).
