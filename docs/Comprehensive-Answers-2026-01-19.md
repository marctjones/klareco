# Comprehensive Answers to Your Questions

## Question 1: How are the two synonym approaches used together?

**Answer**: They compose in layers - graph-based first (currently active), then embedding-based (planned).

### Current Implementation (Graph-Based Only)

**Code**: `klareco/rag/kuzu_inverted_index.py:921-956`

```python
def _build_concepts_from_graph(self, roots, use_graph=True):
    for root, weight in roots.items():
        concept = SemanticConcept(original_root=root, weight=weight)

        # Layer 1: Graph-based synonyms (ACTIVE NOW)
        if use_graph and self._conn:
            synonyms = self.get_synonyms_transitive(root, max_hops=2)
            for syn in synonyms:
                if syn in index_roots:
                    concept.equivalent_roots.add(syn)

        # Layer 2: Embedding-based (PLANNED - Task #7)
        # Would add here when implemented
```

### How They Will Work Together (Task #7)

**Example Query**: "Kiu amas hundojn?" (Who loves dogs?)

**Step 1**: Parser extracts roots → ["am", "hund"]

**Step 2: Layer 1 - Graph expansion** (working now)
```
"am" → {"ŝat", "preferi"} (via ReVo SYNONYM relations)
"hund" → {"besti", "kanid"} (via ReVo SYNONYM relations)
Expanded: ["am", "ŝat", "preferi", "hund", "besti", "kanid"]
```

**Step 3: Layer 2 - Embedding expansion** (planned)
```python
if fallback_mode in [FallbackMode.EMBEDDING, FallbackMode.FULL]:
    # For roots not found by graph OR to augment graph results
    similar = stage1.find_similar("am", threshold=0.6, top_k=10)
    # Returns: [("ador", 0.71), ("kar", 0.68), ...]

    similar = stage1.find_similar("hund", threshold=0.6, top_k=10)
    # Returns: [("kato", 0.72), ("animalo", 0.65), ...]

Final expanded: ["am", "ŝat", "preferi", "ador", "kar",
                 "hund", "besti", "kanid", "kato", "animalo"]
```

**Result**: Retriever finds sentences containing ANY of these roots.

### When Each Layer Activates

**Graph-based** (default, always on):
- High precision (dictionary-sourced)
- Fast (O(1) graph traversal)
- Transitive chains (synonym-of-synonym)
- Hypernym hierarchies

**Embedding-based** (opt-in via `FallbackMode`):
- Higher recall (learned associations)
- Catches creative usage, metaphors
- Slower (embedding lookup)
- Quality depends on Stage 1 training

### Configuration

```python
# Pure deterministic (graph only) - DEFAULT
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.NONE)

# Add embeddings for OOV roots
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.EMBEDDING)

# Use embeddings for reranking
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.RERANK)

# All fallbacks enabled
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.FULL)
```

**See**: `docs/Synonym-Expansion-Architecture.md` for complete details.

---

## Question 2: Are there markdown files in /docs that should migrate to wiki?

**Answer**: YES! Several files belong in wiki.

### Files to Migrate

| File | Location | → Wiki Page | Reason |
|------|----------|-------------|--------|
| ✅ **SEMANTIC_KNOWLEDGE_GRAPH.md** | `docs/` | Semantic-Knowledge-Graph.md | Explains Kuzu structure, educational |
| ✅ **Synonym-Expansion-Architecture.md** | `docs/` | Synonym-Expansion-Architecture.md | Architectural concept explanation |
| ✅ **wiki_templates/Model-Overview.md** | `docs/wiki_templates/` | Model-Overview.md | Overview of M0/Stage1/M1/M2/M3 |
| ✅ **wiki_templates/Stage-1-Root-Embeddings.md** | `docs/wiki_templates/` | Stage-1-Root-Embeddings.md | Model documentation |
| ✅ **wiki_templates/M1-Selectional-Preferences.md** | `docs/wiki_templates/` | M1-Selectional-Preferences.md | Model documentation |
| ✅ **wiki_templates/Understanding-Model-Metrics.md** | `docs/wiki_templates/` | Understanding-Model-Metrics.md | Educational reference |

### Files to KEEP in docs/

| File | Reason |
|------|--------|
| ❌ **M1-Integration-Guide.md** | Tied to specific code paths, integration examples |
| ❌ **RETRAINING_WITH_TIER0.md** | Step-by-step operational guide, tied to scripts |
| ❌ **RAG-Status-2026-01-19.md** | Snapshot in time, belongs in Git history |

### Files to Move to Discussions

| File | → Discussion Title |
|------|-------------------|
| 📝 **wiki_templates/Training-Results-2026-01-18.md** | "Lab Notebook: M1 Training Results 2026-01-18" |
| 📝 **wiki_templates/M1-Investigation-2026-01-18.md** | "M1 Debugging Session 2026-01-18" |

**See**: `docs/Wiki-Migration-Plan.md` for complete migration plan.

---

## Question 3: Is GitHub wiki enabled?

**Answer**: YES! ✅

```bash
$ gh repo view --json hasWikiEnabled --jq '.hasWikiEnabled'
true

$ idlergear wiki status
✓ Wiki is in sync
Last commit: 3154bec Sync from IdlerGear at 2026-01-17 09:56:00
```

**Wiki URL**: https://github.com/marctjones/klareco/wiki

**Current pages**:
- Home
- Current-Architecture
- Development-History
- Semantic-Query-Patterns

---

## Question 4: Can you enable it if not?

**Answer**: Already enabled! No action needed.

If it weren't enabled, you could enable via:
```bash
gh api repos/marctjones/klareco --method PATCH \
  --field has_wiki=true
```

But it's already working.

---

## Question 5: Is it enabled as a backend in idlergear?

**Answer**: YES! ✅ Wiki backend is working.

### Evidence

**Commands available**:
```bash
$ idlergear wiki --help
Commands:
  push     Push IdlerGear references to GitHub Wiki
  pull     Pull GitHub Wiki pages into IdlerGear references
  sync     Bidirectional sync between IdlerGear and GitHub Wiki
  status   Show GitHub Wiki sync status
  config   Get or set wiki configuration
```

**Current status**:
```bash
$ idlergear wiki status
✓ Wiki is in sync
```

**References synced**:
```bash
$ idlergear reference list
- design (pinned from DESIGN.md)
- readme (pinned from README.md)
- vision (pinned from VISION.md)
- semantic-query-patterns (from wiki)
```

### How to Use

**Add reference → automatically syncs to wiki**:
```bash
idlergear reference add "Model Overview" \
  --body "$(cat docs/wiki_templates/Model-Overview.md)"

idlergear wiki push  # Push to wiki
```

**Pull wiki changes**:
```bash
idlergear wiki pull  # Get updates from wiki
```

**Bidirectional sync**:
```bash
idlergear wiki sync  # Two-way sync
```

---

## Question 6: Can you integrate parenthetical English translations into demos?

**Answer**: YES! Done! ✅

### Changes Made

Updated `scripts/demo_rag_with_m1.py` with:

1. **Translation function** (lines 199-277):
   - Simple word-by-word translation
   - ~50 common Esperanto words mapped
   - Unknown words shown in brackets

2. **Query translation** (lines 284-288):
   ```
   Query: Kiu fondis Esperanton?
     → who founded Esperanto?
   ```

3. **Answer translation** (lines 182-186):
   ```
   ✓ PLAUSIBLE
   Ludoviko Lazaro Zamenhof fondis Esperanton.
     → [ludoviko] [lazaro] zamenhof founded Esperanto.
   Source: gutenberg_12
   ```

4. **CLI flag** to disable translations:
   ```bash
   python scripts/demo_rag_with_m1.py --no-translate
   ```

### Usage Examples

**With translations (default)**:
```bash
python scripts/demo_rag_with_m1.py -i
# Shows both Esperanto and English

python scripts/demo_rag_with_m1.py "Kiu fondis Esperanton?"
# Query: Kiu fondis Esperanton?
#   → who founded Esperanto?
```

**Without translations**:
```bash
python scripts/demo_rag_with_m1.py -i --no-translate
# Shows only Esperanto

python scripts/demo_rag_with_m1.py "Kiu fondis Esperanton?" --no-translate
# Query: Kiu fondis Esperanton?
# (no English translation shown)
```

### Translation Quality

**Current vocabulary (~50 words)**:
- Question words: kiu, kio, kia, kie, kiam, kial, kiom, kiel
- Common verbs: estas, fondis, naskiĝis, kreis, amas, ŝatas, manĝas
- Common nouns: esperanto, zamenhof, hundo, kato, libro, lingvo
- Function words: la, kaj, aŭ, sed, en, de, al

**Unknown words**: Shown in brackets like `[ludoviko]`

**Limitations**:
- Word-by-word (not grammatical)
- Limited vocabulary (easily extensible)
- No context awareness

**Good enough for demos**: Users can understand the gist!

**To extend**: Add more words to the `word_map` dictionary in `translate_to_english()`.

---

## Summary of All Actions Taken

### Files Created:
1. ✅ `docs/Synonym-Expansion-Architecture.md` - Explains how graph + embedding work together
2. ✅ `docs/Wiki-Migration-Plan.md` - Complete migration guide
3. ✅ `docs/Comprehensive-Answers-2026-01-19.md` - This file

### Files Modified:
1. ✅ `scripts/demo_rag_with_m1.py` - Added English translations with --no-translate flag

### Investigations:
1. ✅ Checked graph-based synonym expansion code
2. ✅ Verified embedding-based expansion is planned (Task #7)
3. ✅ Confirmed GitHub wiki is enabled
4. ✅ Confirmed idlergear wiki backend is working
5. ✅ Identified files for wiki migration

---

## Next Steps

### Immediate:
1. **Migrate wiki templates**: Use commands from `docs/Wiki-Migration-Plan.md`
2. **Try RAG demo with translations**: `python scripts/demo_rag_with_m1.py -i`
3. **Extend translation vocabulary**: Add more words to `translate_to_english()` as needed

### Short-term:
1. **Task #7**: Implement embedding-based synonym expansion
2. **Task #8**: Complete wiki migration (close task after migration)
3. **Update CLAUDE.md**: Reference new docs

### Medium-term:
1. Compare retrieval quality: graph-only vs graph+embedding
2. Benchmark latency impact of embedding fallback
3. Create evaluation dataset for synonym expansion

---

## Files for Reference

- **Synonym architecture**: `docs/Synonym-Expansion-Architecture.md`
- **Wiki migration**: `docs/Wiki-Migration-Plan.md`
- **RAG demo with translations**: `scripts/demo_rag_with_m1.py`
- **Graph expansion code**: `klareco/rag/kuzu_inverted_index.py:313-360`
- **Concept building**: `klareco/rag/kuzu_inverted_index.py:921-956`
