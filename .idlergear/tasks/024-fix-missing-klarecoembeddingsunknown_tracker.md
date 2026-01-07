---
id: 24
title: Fix missing klareco.embeddings.unknown_tracker module
state: closed
created: '2026-01-05T00:13:57.463941Z'
labels:
- bug
- missing-module
priority: high
---
**Problem**: `klareco.embeddings.unknown_tracker` module is imported but doesn't exist.

**Error**:
```
ModuleNotFoundError: No module named 'klareco.embeddings.unknown_tracker'
```

**Location**: Imported in `klareco/embeddings/__init__.py`:
```python
from .unknown_tracker import UnknownRootTracker, get_tracker, log_unknown_root
```

**Impact**: Blocks `scripts/diagnose_query_matching.py` and any other code importing from `klareco.embeddings`

**Fix options**:
1. Create the missing module file
2. Remove the import if feature was deprecated
3. Comment out import if temporarily disabled

**Investigation needed**: Check git history to see if this was deleted or never created.

**Priority**: High (P1) - blocks diagnostic scripts
**Labels**: bug, missing-module
