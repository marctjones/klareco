---
id: 149
title: Refactor parser to load vocabulary from data files instead of hardcoded Python
state: open
created: '2026-01-08T15:43:24.991248Z'
labels:
- enhancement
- refactoring
- parser
priority: low
---
## Problem
The parser has large vocabulary sets hardcoded in `klareco/parser.py`:
- `KNOWN_ROOTS`: 953,569 entries
- `PROTECTED_PREFIX_ROOTS`: ~100 entries  
- `PROTECTED_SUFFIX_ROOTS`: ~220 entries
- `_FUNDAMENTO_ROOTS`: ~900 entries

This makes updates difficult and mixes data with code.

## Proposed Structure
```
data/vocabularies/
├── fundamento_roots.json       # Core Esperanto roots (~900)
├── known_roots.json            # Extended vocabulary (~950K)
├── protected_prefix_roots.json # Don't strip prefixes
├── protected_suffix_roots.json # Don't strip suffixes (elefant, etc.)
└── stopword_roots.json         # High-freq roots to skip in retrieval
```

## Benefits
1. **Easier updates** - edit JSON, not Python code
2. **Separate versioning** - vocabulary changes tracked separately from code
3. **User extensibility** - users can add custom roots without modifying code
4. **Different vocabularies** - swap vocabularies for different use cases
5. **Smaller parser.py** - currently bloated with 950K+ entries

## Implementation
1. Create JSON files with current vocabulary data
2. Add `VocabularyLoader` class to parser.py
3. Load at module import time with caching
4. Provide fallback to embedded minimal set if files missing
5. Add CLI command to validate/update vocabulary files

## Migration
- Extract current hardcoded sets to JSON files
- Keep backward compatibility during transition
- Update CLAUDE.md with new vocabulary file locations

## Files
- `klareco/parser.py` - refactor to load from files
- `data/vocabularies/*.json` - new vocabulary files
- `scripts/export_vocabulary.py` - one-time export script

## Priority
Low - this is a refactoring task, not blocking functionality
