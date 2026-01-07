---
id: 90
title: Fix legacy Retriever - add lazy metadata loading for 4.4M corpus
state: closed
created: '2026-01-06T19:50:58.799881Z'
labels:
- bug
- retrieval
- memory
priority: medium
---
## Problem

The legacy `Retriever` class in `klareco/rag/retriever.py` loads **ALL metadata into RAM** at startup. With 4.4M documents, this causes OOM crashes.

## Code Location

`retriever.py:128-136`:
```python
# Load metadata
metadata_path = index_dir / "metadata.jsonl"
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found: {metadata_path}")
metadata = []
with open(metadata_path) as f:
    for line in f:
        metadata.append(json.loads(line))  # 4.4M dicts in RAM!
logger.info(f"  Loaded metadata: {len(metadata)} entries")
```

## Impact

- **~10-15GB RAM** for 4.4M metadata dicts
- OOM crash on machines with <16GB RAM
- Even with enough RAM, slow startup (~30-60 seconds)

## Proposed Fix

Add lazy metadata loading with byte offset index (same pattern as slot retrievers):

```python
def _load_metadata_offsets(self, metadata_path: Path):
    """Build offset index for O(1) metadata lookup."""
    self.metadata_offsets = []
    with open(metadata_path, 'rb') as f:
        offset = 0
        for line in f:
            self.metadata_offsets.append(offset)
            offset += len(line)

def _get_metadata(self, idx: int) -> Dict:
    """Load single metadata entry by index."""
    with open(self.metadata_path, 'rb') as f:
        f.seek(self.metadata_offsets[idx])
        line = f.readline()
        return json.loads(line)
```

## Acceptance Criteria

- [ ] Retriever can load 4.4M corpus without OOM
- [ ] Memory usage <2GB at startup
- [ ] Search still works correctly
- [ ] Metadata loaded lazily on demand

## Alternative: Deprecate

This is a **legacy retriever** that doesn't use AST slots. Consider:
1. Deprecating in favor of slot-based retrievers
2. Adding deprecation warning
3. Removing from `__init__.py` exports

## Priority

**P1** - Blocking for legacy index usage, but slot-based retrievers are preferred
