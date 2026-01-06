---
id: 51
title: Create semantic relation database for Esperanto
state: open
created: '2026-01-05T15:47:14.222033Z'
labels:
- enhancement
- data
- retrieval
priority: high
---
## Objective
Build a database of semantic relations (synonyms, agent nouns, related concepts) to support flexible pattern matching.

## Problem
Pattern matcher (Task #50) needs to know:
- fondis ≈ kreis ≈ establis (synonyms)
- fondis → fondinto (verb to agent noun)
- aŭtoro ≈ kreinto ≈ fondinto (related agent nouns)

Current system has no way to know these relations.

## Solution
Create a structured database of Esperanto semantic relations.

## Relation Types

### 1. Verb Synonyms
```json
{
  "fond": {
    "synonyms": ["krei", "establi", "starigi"],
    "related": ["naski", "komenci"],
    "context": "creation/founding"
  },
  "verk": {
    "synonyms": ["skribi", "komponi", "krei"],
    "related": ["eldoni", "publikigi"],
    "context": "writing/creating text"
  }
}
```

### 2. Verb → Agent Noun Transformations
```json
{
  "fond": {
    "agent_noun": "fondinto",
    "related_nouns": ["fondanto", "kreinto", "establinto"],
    "semantic_equivalents": ["aŭtoro", "kreanto"]
  },
  "verk": {
    "agent_noun": "verkinto",
    "related_nouns": ["verkanto"],
    "semantic_equivalents": ["aŭtoro", "skribinto"]
  }
}
```

### 3. Noun Relations
```json
{
  "aŭtoro": {
    "hypernym": "persono",
    "related": ["verkinto", "kreinto", "fondinto"],
    "domain": "creation/authorship"
  },
  "lingvo": {
    "hypernym": "sistemo",
    "related": ["idiomo", "parolmaniero"],
    "examples": ["Esperanto", "angla", "franca"]
  }
}
```

### 4. Context-Specific Relations
```json
{
  "esperanto_creation": {
    "entities": ["Esperanto", "Zamenhof", "Fundamento"],
    "actions": ["fond", "krei", "publikigi", "eldoni"],
    "roles": {
      "creator": ["Zamenhof", "D-ro Zamenhof", "Ludoviko Zamenhof"],
      "creation": ["Esperanto", "la lingvo", "lingvo internacia"]
    }
  }
}
```

## Data Sources

### Automatic Extraction
1. **From corpus**: Extract co-occurrence patterns
   - Words appearing in similar contexts → synonyms
   - Words appearing together frequently → related

2. **From morphology**: Deterministic transformations
   - -i → -into (verb to agent noun)
   - -i → -ado (verb to process noun)
   - mal- prefix (opposites)

3. **From ReVo dictionary**: Parse definitions for synonyms

### Manual Curation
High-value relations for common domains:
- Esperanto creation (fond, krei, Zamenhof, etc.)
- Time expressions (kiam, antaŭ, post, etc.)
- Definitions (esti, signifi, difini, etc.)

## Implementation

```python
class SemanticRelationDB:
    """Database of Esperanto semantic relations."""
    
    def __init__(self, db_path: Path):
        self.relations = self._load_relations(db_path)
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """Get synonyms for a word."""
        return self.relations['synonyms'].get(word, [])
    
    def get_agent_noun(self, verb_root: str) -> Optional[str]:
        """Convert verb to agent noun (fondis → fondinto)."""
        # Check database first
        if verb_root in self.relations['verb_to_agent']:
            return self.relations['verb_to_agent'][verb_root]
        
        # Fallback: morphological rule
        return f"{verb_root}into"
    
    def get_related_concepts(self, word: str, relation: str = 'any') -> List[str]:
        """Get semantically related words."""
        # relation can be: 'synonym', 'hypernym', 'hyponym', 'related'
        pass
    
    def are_semantically_related(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically related."""
        # Check direct synonyms
        if word2 in self.get_synonyms(word1):
            return True
        
        # Check shared hypernym
        if self._share_hypernym(word1, word2):
            return True
        
        # Check co-occurrence in same contexts
        if self._share_contexts(word1, word2):
            return True
        
        return False
```

## File Structure
```
data/semantic_relations/
├── verb_synonyms.json       # Verb synonym groups
├── verb_to_agent.json        # Verb → agent noun mappings
├── noun_relations.json       # Noun hierarchies and relations
├── context_groups.json       # Domain-specific clusters
└── esperanto_specific.json   # High-value manual curations
```

## Deliverable
- `data/semantic_relations/*.json` files
- `klareco/semantics/relation_db.py` - Python interface
- Seed data for top 100 most common verbs/nouns
- Unit tests for relation queries

## Success Criteria
```python
db = SemanticRelationDB()

# Synonyms
assert 'krei' in db.get_synonyms('fond')
assert 'establi' in db.get_synonyms('fond')

# Agent nouns
assert db.get_agent_noun('fond') == 'fondinto'
assert 'aŭtoro' in db.get_related_concepts('fondinto')

# Semantic relatedness
assert db.are_semantically_related('aŭtoro', 'verkinto')
assert db.are_semantically_related('fond', 'krei')
```

## Prioritization
**Phase 1** (do first): Manual curation for benchmark questions
- Top 20 verbs from benchmark (fond, verk, aperi, naski, etc.)
- Top 20 nouns from benchmark (Esperanto, Fundamento, etc.)

**Phase 2**: Automatic extraction from corpus
- Co-occurrence analysis
- Morphological patterns

**Phase 3**: ReVo dictionary parsing

## Dependencies
None - standalone resource

## Effort
- Phase 1 (manual): ~6 hours
- Phase 2 (auto): ~8 hours  
- Phase 3 (ReVo): ~4 hours

**Recommendation**: Start with Phase 1 for immediate impact
