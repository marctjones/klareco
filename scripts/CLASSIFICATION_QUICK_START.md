# Root Classification Quick Start

## Run Classification

**In a separate terminal window:**

```bash
cd /home/marc/Projects/klareco
./scripts/classify_roots.sh
```

The script will:
- ✅ Activate venv automatically
- ✅ Show summary of what will be classified
- ✅ Give you 5 seconds to cancel (Ctrl+C)
- ✅ Log output to `logs/classification_YYYYMMDD_HHMMSS.log`
- ✅ Take ~50 minutes to complete

## Resume from Checkpoint

If interrupted (Ctrl+C or crash):

```bash
./scripts/classify_roots.sh --resume
```

Checkpoint saved in: `data/vocabularies/classification_checkpoint_v2.json`

## Monitor Progress

**While running, in another terminal:**

```bash
# Follow live progress
tail -f logs/classification_*.log

# Check most recent log
ls -lt logs/classification_*.log | head -1

# Check memory usage
ps aux | grep classify_roots | grep -v grep
```

## What Gets Updated

**1.2M Radiko nodes** get these properties:
- `nivelo`: Tier (tier0_pronomo, tier1a_unua_libro, etc.)
- `fonto`: Source (unua_libro, fundamento, revo, korpuso)
- `ofteco`: Usage frequency (integer)

**77.9M Vorto nodes** get propagated properties:
- `radiko_nivelo`: Copied from connected Radiko
- `radiko_fonto`: Copied from connected Radiko
- `radiko_ofteco`: Copied from connected Radiko

## Expected Output

```
=== Loading Classification Data ===
✓ Loaded 190 tier0 grammatical words
✓ Loaded 787 Unua Libro lexical roots
...

=== Classifying Radiko nodes ===
  Total Radiko nodes: 1,248,082
  Progress: 10,000 / 1,248,082 (0.8%) - 250 nodes/sec
  Progress: 20,000 / 1,248,082 (1.6%) - 245 nodes/sec
  ...

=== Propagating to Vorto nodes ===
  Total Vorto nodes: 77,929,426
  Batch 1/78 (1.3%) - 28.5s - ETA: 36.2m
  ...

=== Verification ===
Radiko tier distribution:
  tier0_afikso: 45
  tier0_artikolo: 1
  tier0_konjunkcio: 7
  ...

✓ Classification complete!
```

## After Completion

Test some queries:

```bash
# All Unua Libro words (lexical + grammatical)
python -c "
import kuzu
db = kuzu.Database('data/indexes/v2.1_kuzu_index_full')
conn = kuzu.Connection(db)
result = conn.execute('''
    MATCH (r:Radiko)
    WHERE r.fonto = \"unua_libro\"
    RETURN count(r)
''')
print('Unua Libro words:', result.get_next()[0])
"

# Most frequent Unua Libro words
python -c "
import kuzu
db = kuzu.Database('data/indexes/v2.1_kuzu_index_full')
conn = kuzu.Connection(db)
result = conn.execute('''
    MATCH (r:Radiko)
    WHERE r.fonto = \"unua_libro\"
    RETURN r.radiko, r.nivelo, r.ofteco
    ORDER BY r.ofteco DESC
    LIMIT 20
''')
print('Top 20 most frequent Unua Libro words:')
while result.has_next():
    radiko, nivelo, ofteco = result.get_next()
    print(f\"  {radiko:15s} {nivelo:25s} {ofteco:,}\")
"
```

## Troubleshooting

**Script fails with "No venv found":**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Database locked error:**
- Another process is using the database
- Close other Kuzu connections first

**Out of memory:**
- Script is designed to be memory-safe
- If OOM happens, check system resources: `free -h`
- Resume with `--resume` after freeing memory

**Checkpoint file corrupt:**
- Delete: `rm data/vocabularies/classification_checkpoint_v2.json`
- Start fresh: `./scripts/classify_roots.sh`

**No progress updates appearing:**
- Fixed in latest version with unbuffered output
- Scripts now use `python -u` and `stdbuf -oL -eL`
- Progress updates appear in real-time in both terminal and log file

## Files Created/Modified

**Created:**
- `logs/classification_YYYYMMDD_HHMMSS.log` - Execution log
- `data/vocabularies/classification_checkpoint_v2.json` - Resume checkpoint

**Modified in database:**
- `Radiko` nodes: nivelo, fonto, ofteco properties
- `Vorto` nodes: radiko_nivelo, radiko_fonto, radiko_ofteco properties

**NOT modified:**
- Database schema (properties already added by test script)
- Relationships (no changes to graph structure)
- Other node properties (only adds/updates classification properties)
