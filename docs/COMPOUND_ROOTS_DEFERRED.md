# Compound Roots - Deferred Feature

## The Issue

**Esperanto has productive compounding**, but our current parser treats compounds as atomic roots:
- "sunfloro" (sunflower) → stored as single root `sunflor`
- "retpoŝto" (email) → stored as single root `retpoŝt`
- "komputilo" (computer) → stored as single root `komput` + suffix `il`

**We're missing the internal structure**: "sun+flor", "ret+poŝt"

## Why It Matters

1. **Compositional embeddings** - Could build "retpoŝto" from "ret" + "poŝt"
2. **Neologism validation** - "blogospaco" = "blogo" + "spaco" → both valid → probably legitimate
3. **Semantic drift detection** - Compare compositional vs. actual usage
4. **Smaller vocabulary** - Don't need separate embeddings for every compound
5. **Training efficiency** - Learn ~2K roots instead of ~20K compounds

## Semantic Drift Examples

Most compounds ARE compositional (meaning = sum of parts):
- sunfloro = sun + flower
- papermonero = paper + money
- retpoŝto = net + mail

But some MAY drift:
- komputilo = compute + tool (now fixed term for "computer")
- sangpremo = blood + pressure (medical term, specific meaning)
- sunfloro = specific plant species (not just any sun-loving flower)

## Schema Changes Needed

### Option A: Properties
```
Vorto:
  - estas_kunmetaĵo: BOOLEAN
  - radikoj: STRING[] (e.g., ["sun", "flor"])
  - kunmetstruktur: STRING (e.g., "sun+flor")
```

### Option B: Relationships (PREFERRED)
```
(v:Vorto)-[:KUNMETITA_EL]->(r:Radiko)
```

Example:
```
(sunfloro:Radiko {nivelo: 'tier3_korpuso', estas_kunmetaĵo: true})
  -[:KONSISTAS_EL]->(sun:Radiko {nivelo: 'tier1a_unua_libro'})
  -[:KONSISTAS_EL]->(flor:Radiko {nivelo: 'tier1a_unua_libro'})
```

## Implementation Approaches

**Option A: Modify parser** (HARDER)
- Parser splits compounds during parsing
- Requires dictionary lookup (is "sun" a known root?)
- Ambiguity issues: "komforto" = "kom+fort+o" or "komfort+o"?

**Option B: Post-processing** (EASIER)
- After loading to Kuzu, run compound detection pass
- Try all possible splits, check if components are known roots
- Add relationships: `(v:Vorto)-[:KUNMETITA_EL]->(r:Radiko)`
- Can use dynamic programming for optimal splits

**Option C: Hybrid** (RECOMMENDED when we do this)
- Parser marks POTENTIAL compounds (heuristics)
- Separate analysis pass does decomposition
- Best of both: parser stays simple, analysis is thorough

## Algorithm Sketch (Post-Processing)

```python
def decompose_compound(root: str, known_roots: set) -> list[str]:
    """Find optimal decomposition of compound root."""
    # Dynamic programming: find longest valid splits
    # Example: "sunfloro" → try all splits
    #   - "s" + "unfloro" (s not valid)
    #   - "su" + "nfloro" (neither valid)
    #   - "sun" + "floro" (sun valid, floro valid) ✓
    #   - "sunf" + "loro" (neither valid)
    #   ...
    # Return: ["sun", "flor"] (best split)
```

## Why Deferred

**Complexity**:
- Ambiguous splits (multiple valid decompositions)
- Edge cases (borrowed compounds, abbreviations)
- Need robust validation (false positives)

**Current Priority**:
- ✅ Get basic graph working first
- ✅ Establish tier classification
- ✅ Load all data
- ⏭ Compound decomposition later

## When to Implement

**Prerequisites**:
1. ✅ Complete tier classification (know which roots are valid)
2. ✅ Load full database
3. ⏭ Have frequency data (validate splits by checking if components are used)
4. ⏭ Train initial root embeddings (establish baseline)

**Then**:
- Run post-processing compound detection
- Add `KONSISTAS_EL` relationships
- Re-train compositional embeddings
- Measure improvement

## References

- Parser: `klareco/parser.py` (currently treats compounds as atomic)
- Schema: `klareco/schema/kuzu_ast_schema_v2_1.py` (would need new relationship)
- Current Radiko nodes: All compounds stored as single roots

## Decision

**DEFER compound decomposition until after:**
1. Basic tier classification complete
2. Database fully loaded
3. Initial embeddings trained
4. Have baseline to measure improvement against

**Tracked in: Issue #615**

**Document here so we don't forget!**
