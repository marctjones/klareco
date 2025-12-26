# docs/ Directory Migration Plan

Following the four-tier knowledge management system from CLAUDE.md.

## Current docs/ Files - Where Should They Go?

### ✅ KEEP IN REPO (Tier 4: Code Documentation)

**Files that should stay as markdown in repo:**

1. **CORPUS_BUILDING.md** ✅ KEEP
   - Why: Step-by-step technical guide for running scripts
   - Audience: Developers running corpus rebuild
   - Tied to: Specific scripts in repo
   - Version-controlled: Yes, updates with script changes

### 📚 MOVE TO WIKI (Tier 1: Educational, Timeless)

**Files that belong in Wiki (educational/reference):**

1. **TWO_STAGE_RETRIEVAL.md** → Wiki
   - Why: Explains the two-stage retrieval algorithm
   - Content: How structural + neural retrieval works
   - Audience: Anyone learning about the system
   - Wiki page: "Two-Stage Retrieval Architecture"

2. **RAG_SYSTEM.md** → Wiki
   - Why: Explains RAG architecture and components
   - Content: How retrieval works, what each part does
   - Audience: Understanding the system
   - Wiki page: "RAG System Overview"

3. **RETRIEVAL_GUIDE.md** → Wiki (merge with above)
   - Why: How to use retrieval system
   - Content: Usage patterns, examples
   - Merge into: "Two-Stage Retrieval" wiki page

### 💬 MOVE TO DISCUSSIONS (Tier 2: Lab Notes, Results)

**Files that are session summaries or results:**

1. **SESSION_SUMMARY.md** → Discussion
   - Why: Session notes from development
   - Content: What was built, decisions made
   - Audience: Historical record
   - Discussion: "Lab Notebook: [Date] - Session Summary"

2. **docs/archive/*.md** → Discussions (most of them)
   - PHASE3_SESSION_SUMMARY.md → Discussion
   - OVERNIGHT_PROGRESS_REPORT.md → Discussion
   - RESULTS_COMPARISON.md → Discussion
   - OPTIMAL_TRAINING_PARAMS.md → Discussion (experiment results)
   - FINAL_IMPROVEMENTS_REPORT.md → Discussion
   - TWO_STAGE_IMPLEMENTATION_SUMMARY.md → Discussion

   Pattern: Create "Lab Notebook: [Topic]" discussions for each

### 📋 MOVE TO ISSUES (Tier 3: Actionable)

**None currently** - Session summaries describe completed work, not future tasks

### 📊 CORPUS INVENTORY - Special Case

**CORPUS_INVENTORY.md** - KEEP but simplify
- Why: Documents what data we have
- Current: Too detailed, looks like lab notes
- Action: Keep simplified version in repo, move details to Discussion
- Keep: List of sources, file locations, parse rates
- Move to Discussion: Detailed statistics, quality analysis

### 📦 CORPUS_MANAGEMENT.md

**CORPUS_MANAGEMENT.md** → Wiki
- Why: How to manage corpus (conceptual)
- Content: Building, indexing, quality control
- Wiki page: "Corpus Management Guide"

## Migration Actions

### IMMEDIATE (Do Now)

1. Create Wiki pages:
   - "Two-Stage Retrieval Architecture" (from TWO_STAGE_RETRIEVAL.md + RETRIEVAL_GUIDE.md)
   - "RAG System Overview" (from RAG_SYSTEM.md)
   - "Corpus Management Guide" (from CORPUS_MANAGEMENT.md)

2. Create Discussions:
   - "Lab Notebook: Session Summary [Date]" (from SESSION_SUMMARY.md)
   - "Lab Notebook: Archive" (consolidate docs/archive/*.md)

3. Simplify CORPUS_INVENTORY.md:
   - Keep: Source list, locations, basic stats
   - Move detailed analysis to Discussion

4. Keep in repo:
   - CORPUS_BUILDING.md (technical guide for scripts)
   - CORPUS_INVENTORY.md (simplified)

### FINAL docs/ Structure

```
docs/
├── CORPUS_BUILDING.md          # How to run corpus rebuild scripts
├── CORPUS_INVENTORY.md         # Simple list of data sources
└── archive/                    # Delete after migration to Discussions
```

## Reasoning

### Why Wiki for RAG/Retrieval docs?
- **Timeless**: Explains how the system works (won't change often)
- **Educational**: Teaches concepts
- **Reference**: People look up "how does retrieval work?"
- **Not tied to code**: Conceptual understanding, not implementation guide

### Why Discussion for Session Summaries?
- **Lab notes**: Record of what was done
- **Results**: Experiment outcomes, training params
- **Not actionable**: Describes past work, not future tasks
- **Historical**: Valuable record but not reference material

### Why Keep CORPUS_BUILDING.md in repo?
- **Script documentation**: Directly tied to scripts in repo
- **Version-controlled**: Changes when scripts change
- **Implementation guide**: How to run the code, not concepts
- **Not educational**: Operational, not conceptual

## Migration Script

Would you like me to:
1. Create the Wiki pages with migrated content?
2. Create Discussions with consolidated archive content?
3. Simplify CORPUS_INVENTORY.md?
4. Delete migrated files from docs/?

This follows the pattern:
- **Wiki**: "How does it work?" (concepts)
- **Discussions**: "What did we learn?" (results)
- **Issues**: "What needs doing?" (tasks)
- **Markdown**: "How do I run it?" (operational docs)
