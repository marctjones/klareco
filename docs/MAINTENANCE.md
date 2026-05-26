# Klareco Maintenance

Routine tasks to keep the project healthy on disk. All commands are run
from the project root.

## At a glance

| Task | Frequency | Command |
|---|---|---|
| Audit disk usage | Anytime | `du -sh data/* logs/* models/* results/` |
| Preview deletions | Before any cleanup | `./scripts/util/cleanup_stale.sh` |
| Reclaim safe space | Monthly | `./scripts/util/cleanup_stale.sh --apply` |
| Check for DuckDB bloat | Quarterly or after bulk schema changes | `python scripts/util/compact_duckdb.py --dry-run` |
| Compact DuckDB | When bloat is significant (>20% dead pages) | `python scripts/util/compact_duckdb.py --apply` |

## Weekly: lightweight check

```bash
df -h /
du -sh data/indexes/duckdb_store.db data/enhanced_corpus/ data/staging/ logs/ results/
```

If `data/staging/*.jsonl` is large and the corresponding `build_*` script
has already applied to the DB, that staging file is regenerable. Same for
old per-run logs in `logs/`.

## Monthly: safe cleanup

```bash
# Preview what would be deleted (dry-run is the default):
./scripts/util/cleanup_stale.sh

# Inspect the list, then if it looks fine:
./scripts/util/cleanup_stale.sh --apply

# Tune the windows if needed:
./scripts/util/cleanup_stale.sh --apply --log-days 14 --results-days 30
```

What this removes:
- Per-run logs older than 30 days (configurable)
- Per-run bench result JSON/JSONL older than 60 days
- Known applied staging files (currently `data/staging/entity_postings.jsonl`)
- Orphaned `*.tmp` checkpoint files older than 60 minutes
- Old `/tmp/claude-1000/-home-marc-Projects-klareco/*` task files older than 7 days

What it never touches: `data/raw/`, `data/cleaned/`, `data/extracted/`,
`data/corpus/`, `data/enhanced_corpus/`, `data/indexes/`, `data/dictionaries/`,
`data/vocabularies/`, `data/proper_nouns_dynamic*.json`, `models/`.

## Quarterly: DuckDB compaction

DuckDB doesn't auto-vacuum. In-place `ALTER TABLE ADD COLUMN` + `UPDATE`
+ `CREATE INDEX` leaves dead pages that grow the file beyond the actual
data size. Compaction does an `EXPORT DATABASE` → fresh `IMPORT DATABASE`
round-trip, reclaiming the dead pages.

```bash
# Step 1: confirm Phases A + B (read-only, doesn't touch the live DB)
python scripts/util/compact_duckdb.py --dry-run

# Step 2: actually compact (Phases A → B → C → D)
python scripts/util/compact_duckdb.py --apply
```

The script's own preflight refuses to start unless free disk is at least
**1.5× the current DB size** — that's the working space needed for the
Parquet export plus the imported fresh DB alongside the original until
verification passes. On a typical state (DB ~30 GB), you need ~45 GB free.

Run after:
- Any bulk schema change run via in-place `ALTER`
- A backfill that touches >10% of rows
- The DB file size becomes noticeably larger than `row_count * avg_row_size`

## After bulk schema changes (irregular)

If you absolutely must do an in-place ALTER + UPDATE + INDEX (rather than
the preferred new-table-swap pattern documented in CLAUDE.md), follow it
with a compaction:

```bash
python scripts/util/compact_duckdb.py --apply
```

## Disk-space failure mode (what to do)

When disk hits 100%:

1. Stop any running write operations (you'll see ENOSPC errors).
2. Run the safe cleanup — usually buys back 1-5 GB:
   ```bash
   ./scripts/util/cleanup_stale.sh --apply
   ```
3. If still tight, audit and consider:
   - `gzip data/enhanced_corpus/corpus_with_metadata.jsonl` → ~15 GB back,
     but rebuilds need a `zcat` step thereafter.
   - Delete `data/enhanced_corpus/` entirely if the DB is current and
     you're confident in the parse pipeline.
   - `python scripts/util/compact_duckdb.py --apply` → up to ~30 GB back,
     but needs ~45 GB working space — chicken-and-egg if disk is already
     full.
4. Once recovered, investigate root cause. A common one: scripts running
   `ALTER + UPDATE + INDEX` in place. See CLAUDE.md "Disk-space
   conventions" for the rewrite pattern.

## Preflight conventions for new scripts

Any new long-running script should source `preflight_disk.sh` at the
top and refuse to start without enough headroom. Conservative estimates:

| Script type | Min free GB |
|---|---:|
| Parse a corpus | 30 |
| Build DuckDB store from corpus | 50 |
| Rebuild Whoosh from DuckDB | 10 |
| Bulk schema change on `sentences` (5M rows) | 35 |
| Full pipeline rebuild | 70 |
| DuckDB compaction (self-preflights at 1.5× DB size) | (script handles it) |

Usage at the top of a bash script:

```bash
source scripts/util/preflight_disk.sh
require_disk_gb 50 "build_duckdb_store needs working space"
```

Or as a guard:

```bash
./scripts/util/preflight_disk.sh 50 "reason" || exit 1
```

## Related

- `CLAUDE.md` → "Disk-space conventions" section — never-delete list, the
  new-table-swap pattern, the maintenance toolkit reference.
- EPIC issue: disk-space hygiene (filed as #746 in May 2026).
