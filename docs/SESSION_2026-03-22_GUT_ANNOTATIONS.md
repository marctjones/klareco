# Session Summary: Gut Annotations Design (2026-03-22)

## What This Session Explored

This Claude Code session explored how to add **semantic intuition annotations** ("gut feelings") to Klareco's AST system, using pure Esperanto terminology.

## Key Insights

### 1. Philosophical Foundation: Meaning Grounded in Goals

**Question:** Where does meaning come from beyond circular verbal definitions?

**Answer:** Meaning is grounded in "what matters" to a system with goals:
- Traditional theories: referential, embodied, distributional
- **Our addition:** Meaning grounded in desires, values, goals (axiological grounding)

**Example:** "Important" means "increases goal satisfaction" - this requires having goals!

### 2. Dual-Process Cognition for AI

**Key insight:** ASTs are "thoughts" (explicit, verbalizable), embeddings are "gut feelings" (implicit, subsymbolic).

- **Thoughts (ASTs):** Structured, rule-based, slow, explainable (System 2)
- **Gut (Embeddings):** Pattern-based, fast, confidence-weighted (System 1)
- **Integration:** Gut guides which thoughts to pursue, thoughts verify gut

### 3. Goals Must Match Capabilities

**Critical insight:** Can't implement sophisticated goals (autonomy, curiosity, social relationships) for systems without the capabilities to ground them!

**For current Klareco:**
- ✅ CAN meaningfully care about: parse accuracy, grammatical correctness, vocabulary coverage
- ❌ CANNOT meaningfully care about: long-term relationships, exploration, survival

**Design principle:** Match goals to actual capabilities, avoid premature sophistication.

### 4. Pure Esperanto Annotations

**Key decision:** All annotation types and values should be in Esperanto, linked to Root table.

**Why:**
- Circular vocabulary (annotations explained in Esperanto)
- Self-describing system
- Philosophical purity
- Already supported by v2.1 schema!

## The Design: Intuiciaj Annotacioj

### Four Annotation Types (All Esperanto)

1. **certeco** (confidence) - How well do I know this word?
   - Based on: vocabulary tier, parse status, frequency
   - No embeddings needed!

2. **graveco** (salience) - How important is this for the query?
   - Based on: semantic similarity (60%) + structural role (40%)
   - Requires embeddings

3. **kohereco** (coherence) - How well does this fit the context?
   - Based on: average similarity to other words
   - Requires embeddings

4. **surprizo** (surprise) - How unexpected is this?
   - Based on: inverse frequency
   - No embeddings needed!

### Implementation Strategy

**Uses existing v2.1 annotation system - NO schema changes!**

- Completely modular (can enable/disable each feature)
- Backward compatible (disabled by default)
- Can start with "minimal mode" (only confidence, no embeddings)
- Then add "full mode" (all features) when embeddings stable

## Deliverables

### Primary Document

**`docs/INTUICIAJ_ANNOTACIOJ_DESIGN.md`** - Complete design specification including:
- Philosophical foundation
- Pure Esperanto annotation types/values
- Technical architecture
- Computation methods
- Integration points
- Implementation phases
- Coordination strategy with parallel development
- Testing strategy
- Example code and queries

### Key Design Decisions

1. **No schema changes** - Uses v2.1 AnnotationType/AnnotationValue tables
2. **Modular** - Can enable/disable each feature independently
3. **Esperanto-first** - All types/values linked to Root table
4. **Backward compatible** - Default is disabled
5. **Phased implementation** - Start minimal (no embeddings), add full later

## Coordination Notes

### Parallel Development Detected

Git status shows another session working on:
- `klareco/embeddings/` - Hybrid embeddings
- Training scripts - Fundamento training
- Documentation - Model naming

### Conflict Avoidance Strategy

**Safe to implement (no conflicts):**
- Phase 1: Minimal mode (confidence + surprise)
  - Uses v2.1 tier classification
  - Uses corpus stats
  - No embeddings dependency

**Coordinate for:**
- Phase 2: Full mode (salience + coherence)
  - Imports from `klareco.embeddings.compositional`
  - Wait for embeddings session to stabilize
  - Use interface pattern to avoid coupling

### Recommended Approach

**Option 1: Implement Phase 1 now**
- Low risk (no embeddings dependency)
- High value (vocabulary coverage awareness)
- 8-12 hours implementation time

**Option 2: Defer entirely**
- Let embeddings work complete first
- Implement all phases together later
- Lower coordination overhead

**Option 3: Document and review**
- Review design with user/team
- Get feedback on Esperanto naming
- Decide timing after review

## Questions for Other Session

1. Is embeddings work stable enough to import from?
2. Should we implement minimal mode now or defer all?
3. Any concerns about adding annotations to v2.1 database?
4. Preferred coordination method (branches, tasks, direct communication)?

## Files Created

- `docs/INTUICIAJ_ANNOTACIOJ_DESIGN.md` - Complete design spec (8000+ words)
- `docs/SESSION_2026-03-22_GUT_ANNOTATIONS.md` - This summary

## Files NOT Modified

- No code changes made
- No schema changes made
- No git branches created
- No conflicts introduced

**Status:** DESIGN COMPLETE, awaiting review and implementation decision

---

**Recommendation:** Review the design document, then decide:
- Implement Phase 1 (minimal) now?
- Implement full design later?
- Modify design before implementing?

The design is modular and backward-compatible, so low risk either way.
