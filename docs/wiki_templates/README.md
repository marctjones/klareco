# Wiki Page Templates

This directory contains **template wiki pages** that should be copied to the project wiki once training completes and actual results are available.

## Purpose

These templates provide comprehensive documentation for Klareco's trained models with:
- Architecture descriptions
- Training data composition
- Performance metrics (with placeholders for actual results)
- Usage examples
- Troubleshooting guides
- Integration documentation

## Files

| File | Description | Status |
|------|-------------|--------|
| `Model-Overview.md` | Entry point - overview of all models | ✅ Template complete |
| `Stage-1-Root-Embeddings.md` | Complete Stage 1 documentation | ✅ Template complete |
| `M1-Selectional-Preferences.md` | Complete M1 documentation | ✅ Template complete |
| `Understanding-Model-Metrics.md` | Educational guide to metrics | ✅ Template complete |

## How to Use

### 1. Wait for Training to Complete

Current training status (as of 2026-01-18):
- Stage 1: 🔄 In progress (~2-4 hours)
- M1: ⏳ Pending (trains after Stage 1)

### 2. Populate Placeholders

Search for `[PLACEHOLDER]` in each file and replace with actual values:

**Stage-1-Root-Embeddings.md**:
- `[PLACEHOLDER]` → Final correlation (e.g., 0.8518)
- `[PLACEHOLDER]` → Positive similarity (e.g., 0.529)
- `[PLACEHOLDER]` → Negative similarity (e.g., 0.031)
- `[PLACEHOLDER]` → Separation gap (e.g., 0.498)
- `[PLACEHOLDER]` → Mean pairwise similarity (e.g., 0.342)
- `[PLACEHOLDER]` → Training epochs (e.g., 47)
- `[PLACEHOLDER]` → Training time (e.g., "3.2 hours")

**M1-Selectional-Preferences.md**:
- `[PLACEHOLDER]` → Validation accuracy (e.g., 0.8654)
- `[PLACEHOLDER]` → Test accuracy (e.g., 0.8571)
- `[PLACEHOLDER]` → Score mean (e.g., 0.523)
- `[PLACEHOLDER]` → Score std (e.g., 0.187)
- `[PLACEHOLDER]` → Component losses (e.g., 0.245, 0.238, 0.251)
- `[PLACEHOLDER]` → Training epochs (e.g., 32)
- `[PLACEHOLDER]` → Training time (e.g., "47 minutes")

**Model-Overview.md**:
- Replace status indicators (🔄, ⏳) with ✅ when complete
- Update "Latest Training Run" table with actual values
- Add date stamps (YYYY-MM-DD format)

### 3. Clone Wiki Repository

```bash
# From project root
cd ..
git clone https://github.com/marctjones/klareco.wiki.git
cd klareco.wiki
```

### 4. Copy Files to Wiki

```bash
# From klareco.wiki directory
cp ../klareco/docs/wiki_templates/Model-Overview.md ./Model-Overview.md
cp ../klareco/docs/wiki_templates/Stage-1-Root-Embeddings.md ./Stage-1-Root-Embeddings.md
cp ../klareco/docs/wiki_templates/M1-Selectional-Preferences.md ./M1-Selectional-Preferences.md
cp ../klareco/docs/wiki_templates/Understanding-Model-Metrics.md ./Understanding-Model-Metrics.md
```

### 5. Update Wiki Home

Edit `Home.md` to add links to new pages:

```markdown
## Model Documentation

- [Model Overview](Model-Overview) - All models at a glance
- [Stage 1: Root Embeddings](Stage-1-Root-Embeddings) - Semantic vectors
- [M1: Selectional Preferences](M1-Selectional-Preferences) - Plausibility scoring
- [Understanding Model Metrics](Understanding-Model-Metrics) - How to interpret results
```

### 6. Commit and Push

```bash
git add *.md
git commit -m "Add comprehensive model documentation with training results"
git push origin master
```

## Current Status (2026-01-18)

### Completed Training

- ✅ **Stage 1**: Correlation 0.8491 (target: > 0.80) - **Production ready**
- ⚠️ **M1**: Accuracy 70.2% (target: > 82%) - **Needs retraining**

### M1 Issue and Solution

M1 accuracy is below target due to insufficient model capacity (128d hidden dimension). The model is biased toward predicting "implausible" for most inputs.

**To retrain M1 with improved hyperparameters:**

```bash
cd /path/to/klareco
./scripts/retrain_m1_improved.sh --fresh
```

This retrains with:
- Hidden dimension: 256d (double capacity)
- Dropout: 0.2 (better regularization)
- Patience: 20 (more training time)

**After retraining**, re-run step 2 above to populate M1 placeholders with new results.

See `Training-Results-2026-01-18.md` for detailed analysis.

## Maintenance

### When to Update

- **After each training run**: Update metrics with latest results
- **When models change**: Update architecture descriptions
- **When adding new models**: Create new pages following template structure
- **When troubleshooting**: Add solutions to troubleshooting sections

### Version History

Add to "Changelog" section at bottom of each page:

```markdown
## Changelog

- **2026-01-18**: Initial training with tier0 + ReVo (correlation: 0.8518)
- **2026-02-XX**: Retrained with expanded corpus (correlation: 0.8623)
- **2026-03-XX**: Added M2 integration (no impact on metrics)
```

## Template Structure

Each model page follows this structure:

1. **Overview** - What the model does
2. **Architecture** - Technical details
3. **Training Data** - Data composition
4. **Training Configuration** - Hyperparameters
5. **Performance Metrics** - Actual results
6. **Usage** - Code examples
7. **Integration** - How it fits in pipeline
8. **Troubleshooting** - Common issues
9. **Retraining** - When and how
10. **Testing** - Quality validation
11. **Files & Paths** - Locations
12. **References** - Related docs
13. **Changelog** - Version history

## Notes

- **Keep placeholders visible** until actual data is available
- **Don't guess values** - only fill in with measured results
- **Include context** - explain why metrics matter
- **Add examples** - real code that works
- **Cross-reference** - link related pages
- **Stay current** - update as models evolve

## Questions?

See:
- [Understanding-Model-Metrics.md](Understanding-Model-Metrics.md) - Explains what metrics mean
- [CLAUDE.md](../../CLAUDE.md) - Development guide
- Task #8: "Create wiki pages documenting each trained model"
