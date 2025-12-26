# Documentation Migration Plan

**Status**: Ready to Execute
**Date**: 2025-12-23

## Migration Strategy by File

### TO WIKI (Timeless Educational Content)

| File | Wiki Page Name | Action | Priority |
|------|---------------|--------|----------|
| `docs/wiki-drafts/Esperanto-Grammar-Reference.md` | Esperanto Grammar Reference | ✅ Ready | P0 |
| `docs/wiki-drafts/LLM-Architecture-Fundamentals.md` | LLM Architecture Fundamentals | ✅ Ready | P0 |
| `COMPOSITIONAL_EMBEDDINGS.md` | Compositional Embeddings | Migrate | P0 |
| `DESIGN.md` (sections) | AST Structure and Theory | Extract & migrate | P1 |
| `VISION.md` (sections) | Deterministic vs Learned Boundaries | Extract & migrate | P1 |
| `docs/TWO_STAGE_RETRIEVAL.md` | Retrieval Strategies | Migrate | P2 |

### TO DISCUSSIONS (Research Findings & Ideas)

| File | Discussion Title | Category | Action | Priority |
|------|------------------|----------|--------|----------|
| `CORPUS_V2_RESULTS.md` | Research: Corpus V2 Build Results | Research Findings | Migrate | P0 |
| `CORPUS_AND_AST_AUDIT.md` | Research: Corpus Quality Analysis | Research Findings | Migrate | P0 |
| `RAG_STATUS.md` | Research: Two-Stage Retrieval Performance | Research Findings | Migrate | P1 |
| Future: Semantic training results | Research: Semantic Similarity Training | Research Findings | Create | P1 |

### TO ISSUES (Actionable Items)

Already complete - Epics #4-9 created ✅

### KEEP IN REPO (Code Documentation)

| File | Purpose | Action |
|------|---------|--------|
| `README.md` | Main entry point | Keep, maintain |
| `CLAUDE.md` | AI assistant guide | Keep, maintain |
| `VISION.md` | Architectural vision | Keep (reference for wiki) |
| `DESIGN.md` | Technical design | Keep (reference for wiki) |
| `ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md` | Implementation phases | Keep, maintain |
| `DOCUMENTATION_INDEX.md` | Documentation navigation | Keep, update after migration |
| `docs/CORPUS_INVENTORY.md` | Corpus catalog | Keep, maintain |
| `docs/CORPUS_MANAGEMENT.md` | Corpus management guide | Keep |
| `docs/RAG_SYSTEM.md` | RAG architecture | Keep |
| `docs/RETRIEVAL_GUIDE.md` | Retrieval guide | Keep |

### DELETE AFTER MIGRATION

| File | Reason | Migrated To |
|------|--------|-------------|
| `docs/wiki-drafts/*.md` | After wiki migration | GitHub Wiki |
| `COMPOSITIONAL_EMBEDDINGS.md` | After wiki migration | Wiki page |
| `CORPUS_V2_RESULTS.md` | After discussion creation | Discussion |
| `CORPUS_AND_AST_AUDIT.md` | After discussion creation | Discussion |
| `RAG_STATUS.md` | After discussion creation | Discussion |
| `docs/TWO_STAGE_RETRIEVAL.md` | After wiki migration | Wiki page |

## Execution Steps

### Step 1: Clone Wiki Repo
```bash
cd /tmp
git clone https://github.com/marctjones/klareco.wiki.git
```

### Step 2: Create Wiki Pages
Copy and commit:
1. Home.md (navigation)
2. Esperanto-Grammar-Reference.md
3. LLM-Architecture-Fundamentals.md
4. Compositional-Embeddings.md (from COMPOSITIONAL_EMBEDDINGS.md)
5. AST-Structure-and-Theory.md (extract from DESIGN.md)
6. Deterministic-vs-Learned-Boundaries.md (extract from VISION.md)
7. Retrieval-Strategies.md (from docs/TWO_STAGE_RETRIEVAL.md)

### Step 3: Create Discussions
Use `gh` CLI:
1. Corpus V2 Build Results
2. Corpus Quality Analysis
3. Two-Stage Retrieval Performance

### Step 4: Update Documentation
1. Update DOCUMENTATION_INDEX.md with wiki/discussion links
2. Delete migrated files
3. Commit changes

### Step 5: Verify
1. Check wiki pages render correctly
2. Check discussions are in right category
3. Verify links in DOCUMENTATION_INDEX.md work
