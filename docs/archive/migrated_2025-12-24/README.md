# Migrated Documentation (2025-12-24)

These files were migrated to Wiki and Discussions following the four-tier knowledge management system.

## Files Migrated to Wiki

### RAG_SYSTEM.md → Wiki: "RAG System Overview"
- **Why**: Conceptual explanation of RAG architecture
- **Wiki page**: https://github.com/marctjones/klareco/wiki/RAG-System-Overview

### TWO_STAGE_RETRIEVAL.md → Wiki: "Two-Stage Retrieval Architecture"
- **Why**: Educational content about retrieval algorithm
- **Wiki page**: https://github.com/marctjones/klareco/wiki/Two-Stage-Retrieval-Architecture

### RETRIEVAL_GUIDE.md → Wiki: "Two-Stage Retrieval Architecture" (merged)
- **Why**: Usage guide merged into architecture page
- **Wiki page**: https://github.com/marctjones/klareco/wiki/Two-Stage-Retrieval-Architecture

### CORPUS_MANAGEMENT.md → Wiki: "Corpus Management"
- **Why**: Conceptual guide to corpus management
- **Wiki page**: https://github.com/marctjones/klareco/wiki/Corpus-Management

## Files Migrated to Discussions

### SESSION_SUMMARY.md → Discussion: "Lab Notebook: Development Sessions Archive"
- **Why**: Session notes and historical record
- **Discussion**: https://github.com/marctjones/klareco/discussions/15

## Files Simplified (Remain in docs/)

### CORPUS_INVENTORY.md → Simplified to operational reference
- **Why**: Operational reference (file locations, quick stats)
- **Location**: `docs/CORPUS_INVENTORY.md`
- **Change**: Removed detailed analysis, kept essentials

## Files That Stay in docs/

### CORPUS_BUILDING.md → Stays
- **Why**: Technical guide for running scripts (tied to code)
- **Location**: `docs/CORPUS_BUILDING.md`

## Migration Reasoning

Following CLAUDE.md four-tier system:
- **Wiki (Tier 1)**: "How does it work?" - concepts, algorithms, education
- **Discussions (Tier 2)**: "What did we learn?" - lab notes, results
- **Issues (Tier 3)**: "What needs doing?" - actionable tasks
- **Markdown (Tier 4)**: "How do I run it?" - operational guides

## Restoration

If needed, these files can be restored from this archive or from git history:
```bash
git log --all --full-history -- "docs/RAG_SYSTEM.md"
```
