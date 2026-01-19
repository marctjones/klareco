# Wiki Migration Plan

## GitHub Wiki Status

**Wiki is ENABLED and synced**: ✅
- Last sync: 2026-01-17 09:56:00 (commit 3154bec)
- IdlerGear wiki backend: Working
- Status: `idlergear wiki status` shows "in sync"

**IdlerGear references**:
- 4 references currently: design, readme, vision, semantic-query-patterns
- Command: `idlergear reference list` to see all
- Sync command: `idlergear reference sync` or `idlergear wiki push`

---

## Files to Migrate to Wiki

### Tier 1: Educational/Conceptual (→ Wiki)

These explain concepts, algorithms, and theory - perfect for wiki.

| File | Current Location | Proposed Wiki Page | Reason |
|------|------------------|-------------------|--------|
| `SEMANTIC_KNOWLEDGE_GRAPH.md` | `docs/` | `Semantic-Knowledge-Graph.md` | Explains Kuzu graph structure, ReVo/ConceptNet integration |
| `docs/wiki_templates/Model-Overview.md` | `docs/wiki_templates/` | `Model-Overview.md` | Overview of all models (M0/Stage1/M1/M2/M3) |
| `docs/wiki_templates/Stage-1-Root-Embeddings.md` | `docs/wiki_templates/` | `Stage-1-Root-Embeddings.md` | Stage 1 architecture, training, metrics |
| `docs/wiki_templates/M1-Selectional-Preferences.md` | `docs/wiki_templates/` | `M1-Selectional-Preferences.md` | M1 architecture, training, usage |
| `docs/wiki_templates/Understanding-Model-Metrics.md` | `docs/wiki_templates/` | `Understanding-Model-Metrics.md` | How to interpret metrics (correlation, accuracy, etc.) |
| `Synonym-Expansion-Architecture.md` | `docs/` | `Synonym-Expansion-Architecture.md` | How graph-based + embedding-based work together |

### Tier 2: Operational/Process (Stay in docs/)

These are tied to specific code/scripts - better in versioned docs.

| File | Location | Keep Because |
|------|----------|--------------|
| `M1-Integration-Guide.md` | `docs/` | Tied to specific code paths, integration examples |
| `RETRAINING_WITH_TIER0.md` | `docs/` | Step-by-step guide tied to specific scripts |
| `RAG-Status-2026-01-19.md` | `docs/` | Snapshot in time, belongs in Git history |

### Tier 3: Session Notes (→ GitHub Discussions)

These are lab notes/progress reports - better in Discussions.

| File | Current Location | Proposed Destination | Reason |
|------|------------------|----------------------|--------|
| `docs/wiki_templates/Training-Results-2026-01-18.md` | `docs/wiki_templates/` | Discussion: "Lab Notebook 2026-01-18" | Experimental results, time-stamped |
| `docs/wiki_templates/M1-Investigation-2026-01-18.md` | `docs/wiki_templates/` | Discussion: "M1 Debugging Session 2026-01-18" | Investigation notes, historical record |

---

## Migration Commands

### Using IdlerGear Wiki Commands

```bash
# Add reference doc (will sync to wiki)
idlergear reference add "Semantic Knowledge Graph" \
  --body "$(cat docs/SEMANTIC_KNOWLEDGE_GRAPH.md)"

# Push all references to wiki
idlergear wiki push

# Check sync status
idlergear wiki status

# Pull wiki changes back
idlergear wiki pull

# Bidirectional sync
idlergear wiki sync
```

### Manual Migration (Alternative)

If you prefer direct control:

```bash
# Clone wiki repo
git clone https://github.com/marctjones/klareco.wiki.git
cd klareco.wiki

# Copy files
cp ../klareco/docs/SEMANTIC_KNOWLEDGE_GRAPH.md Semantic-Knowledge-Graph.md
cp ../klareco/docs/wiki_templates/M1-Selectional-Preferences.md M1-Selectional-Preferences.md
# ... etc

# Commit and push
git add .
git commit -m "Migrate documentation from main repo"
git push origin master
```

**Note**: Always `git pull` before editing wiki manually to avoid merge conflicts if others edited via web UI.

---

## Recommended Migration Process

### Phase 1: Core Model Documentation (Now)
```bash
# Add model docs to wiki via idlergear
idlergear reference add "Model Overview" \
  --body "$(cat docs/wiki_templates/Model-Overview.md)"

idlergear reference add "Stage 1 Root Embeddings" \
  --body "$(cat docs/wiki_templates/Stage-1-Root-Embeddings.md)"

idlergear reference add "M1 Selectional Preferences" \
  --body "$(cat docs/wiki_templates/M1-Selectional-Preferences.md)"

idlergear reference add "Understanding Model Metrics" \
  --body "$(cat docs/wiki_templates/Understanding-Model-Metrics.md)"

idlergear reference add "Synonym Expansion Architecture" \
  --body "$(cat docs/Synonym-Expansion-Architecture.md)"

# Push to wiki
idlergear wiki push
```

### Phase 2: Semantic Knowledge Graph
```bash
idlergear reference add "Semantic Knowledge Graph" \
  --body "$(cat docs/SEMANTIC_KNOWLEDGE_GRAPH.md)"

idlergear wiki push
```

### Phase 3: Lab Notes to Discussions
Create GitHub Discussions for session notes:
```bash
# Via gh CLI
gh api repos/marctjones/klareco/discussions \
  --method POST \
  --field title="Lab Notebook: M1 Training Results 2026-01-18" \
  --field body="$(cat docs/wiki_templates/Training-Results-2026-01-18.md)" \
  --field category_id="<lab-notebook-category-id>"
```

### Phase 4: Cleanup
```bash
# Remove migrated files from main repo
git rm docs/wiki_templates/*.md
git rm docs/SEMANTIC_KNOWLEDGE_GRAPH.md
git rm docs/Synonym-Expansion-Architecture.md

# Update README with wiki links
# Commit changes
git commit -m "Migrate documentation to wiki"
```

---

## Wiki Organization

### Proposed Wiki Structure

**Home Page**:
- Project overview (from README)
- Links to main sections

**Architecture Section**:
- Current-Architecture (already exists)
- Model-Overview (new)
- Synonym-Expansion-Architecture (new)
- Semantic-Knowledge-Graph (new)

**Model Documentation**:
- Stage-1-Root-Embeddings (new)
- M1-Selectional-Preferences (new)
- M2-Taxonomic-Model (placeholder)
- M3-Orchestration (placeholder)

**Training & Evaluation**:
- Understanding-Model-Metrics (new)
- Development-History (already exists)

**Reference**:
- Semantic-Query-Patterns (already exists)
- 16-Esperanto-Rules (from main repo)

---

## Why This Structure?

**Wiki = Educational, Timeless**
- Explains concepts that don't change often
- Architectural decisions
- Model documentation
- How things work

**docs/ = Operational, Versioned**
- Step-by-step guides tied to code
- Integration examples
- Current status snapshots

**Discussions = Temporal, Exploratory**
- Lab notebooks
- Experiment results
- Investigation notes
- Questions and answers

---

## IdlerGear Wiki Backend Configuration

**Current Status**: Working, but config command has a bug

**Config file**: `.idlergear/config.toml`
```toml
idlergear_version = "0.4.12"

# Wiki config would go here (but currently minimal)
# [wiki]
# enabled = true
# auto_sync = false
# sync_interval = 3600
```

**Note**: Config command has a TypeError bug, but wiki sync works fine via `idlergear wiki push/pull/sync`.

---

## After Migration

**Update references in code/docs**:
```bash
# Replace local doc links with wiki links
# Old: See docs/SEMANTIC_KNOWLEDGE_GRAPH.md
# New: See wiki: https://github.com/marctjones/klareco/wiki/Semantic-Knowledge-Graph

# Update CLAUDE.md, README.md, etc.
```

**Add wiki links to README**:
```markdown
## Documentation

| Document | Purpose |
|----------|---------|
| [Wiki: Model Overview](https://github.com/marctjones/klareco/wiki/Model-Overview) | Architecture overview |
| [Wiki: Stage 1 Embeddings](https://github.com/marctjones/klareco/wiki/Stage-1-Root-Embeddings) | Root embedding model |
| [Wiki: M1 Selectional](https://github.com/marctjones/klareco/wiki/M1-Selectional-Preferences) | Selectional preference model |
| `docs/M1-Integration-Guide.md` | Integration code examples |
| `docs/RETRAINING_WITH_TIER0.md` | Retraining process |
```

---

## Next Steps

1. ✅ **Verify wiki is working**: `idlergear wiki status` (Done - it's working!)
2. 📝 **Phase 1 migration**: Add core model docs to wiki
3. 🧹 **Clean up wiki_templates/**: Remove README, update paths
4. 📚 **Phase 2 migration**: Add architectural docs
5. 💬 **Phase 3**: Move session notes to Discussions
6. 🔗 **Phase 4**: Update links, remove migrated files
7. ✅ **Close Task #8**: Wiki pages documentation complete

**Estimated time**: 30-60 minutes for full migration
