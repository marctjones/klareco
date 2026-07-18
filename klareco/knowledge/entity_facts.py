"""
entity_facts schema adapter (#881).

The live `entity_facts` table uses a TRIPLE schema
(sid, entito, rilato, valoro, klauzo, klaso, fonto), but three consumers —
generation (discourse.py), the planner, and reasoning — were written against an
older SLOT schema (entity_radiko, slot, value, value_radiko, source_sid,
confidence, pattern_name). The mismatch raised BinderException on every call and
was silently swallowed, so the whole fact-consuming symbolic layer was dead in
production (probed 2026-07-18).

Rather than scatter the mapping across every SQL string (that is exactly how the
drift went unnoticed), it lives HERE, once. Consumers select `FROM {SLOTS} ef`.
This is the read-only compatibility path — non-destructive, no store write.

Caveat (honest): the triple schema stores only the radiko-form value (`valoro`),
so `value` and `value_radiko` both map to it — generated text shows radikoj
(e.g. "grup", not "grupojn") until surface forms are stored. `confidence` is not
in the triple schema, so it defaults to 1.0. This is a compatibility bridge; the
target is facts flowing as FactFragments in the thought (MVP-1).
"""
from __future__ import annotations

# A slot-schema projection of the live triple table. Insert as the FROM target:
#     conn.execute(f"SELECT ... FROM {SLOTS} ef WHERE ef.entity_radiko = ?")
# It contains no `{}` so it is safe to interpolate into an f-string alongside
# `?` placeholders.
SLOTS = """(
    SELECT sid                    AS source_sid,
           entito                 AS entity_radiko,
           rilato                 AS slot,
           valoro                 AS value,
           valoro                 AS value_radiko,
           CAST(1.0 AS DOUBLE)    AS confidence,
           fonto                  AS pattern_name,
           klaso,
           klauzo
    FROM entity_facts
)"""

# The underlying triple columns, for anything that needs to know the real schema
# (e.g. dependency declarations / the eventual FactFragment loader).
TRIPLE_COLUMNS = ('sid', 'entito', 'rilato', 'valoro', 'klauzo', 'klaso', 'fonto')
