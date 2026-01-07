---
id: 102
title: Extend semantic role expansion beyond creator role
state: open
created: '2026-01-06T23:08:14.913361Z'
labels:
- enhancement
- retrieval
- semantic-roles
priority: medium
---
## Problem

Currently only the CREATOR semantic role is implemented:
```python
CREATOR_VERBS = {'fond', 'kre', 'invent', 'establ', 'inici'}
CREATOR_NOUNS = {'aŭtor', 'kreint', 'fondint', 'inventint', 'iniciatint'}
```

Other important semantic roles are not handled:

## Proposed Semantic Roles to Add

### 1. LOCATION role
For "Kie..." (WHERE) questions:
```python
LOCATION_VERBS = {'situ', 'trov', 'lok', 'loĝ', 'est'}  # situi, troviĝi, lokiĝi, loĝi
LOCATION_NOUNS = {'loko', 'urbo', 'lando', 'region', 'adres'}
LOCATION_PREPS = {'en', 'ĉe', 'apud', 'sur'}
```

Example: "Kie situas UEA?" should match "UEA havas sidejon en Roterdamo"

### 2. TEMPORAL role
For "Kiam..." (WHEN) questions:
```python
TEMPORAL_VERBS = {'okaz', 'komenc', 'fin', 'daŭr', 'naski'}
TEMPORAL_NOUNS = {'dato', 'jaro', 'tago', 'epoko', 'periodo'}
TEMPORAL_MARKERS = {'en 18..', 'en 19..', 'en 20..', 'la ...-an de'}
```

Example: "Kiam naskiĝis Zamenhof?" should match "Zamenhof (1859-1917)"

### 3. DEFINITION role
For "Kio estas..." questions:
```python
DEFINITION_VERBS = {'est', 'signif', 'reprezent'}
DEFINITION_PATTERNS = ['X estas Y', 'X signifas Y', 'X = Y']
```

### 4. QUANTITY role
For "Kiom..." questions:
```python
QUANTITY_MARKERS = {'nombro', 'kvanto', 'multe', 'miliono', 'mil'}
NUMBER_PATTERNS = [r'\d+', r'dek', r'cent', r'mil']
```

## Implementation Approach

```python
class SemanticRoleExpander:
    """Expand semantic roles for improved retrieval matching."""
    
    ROLE_DEFINITIONS = {
        'CREATOR': {
            'verbs': {'fond', 'kre', 'invent', 'establ', 'inici'},
            'nouns': {'aŭtor', 'kreint', 'fondint', 'inventint'},
        },
        'LOCATION': {
            'verbs': {'situ', 'trov', 'lok', 'loĝ'},
            'nouns': {'loko', 'urbo', 'lando', 'adres', 'sidej'},
            'preps': {'en', 'ĉe', 'apud'},
        },
        'TEMPORAL': {
            'verbs': {'okaz', 'komenc', 'fin', 'daŭr'},
            'nouns': {'dato', 'jaro', 'tago', 'epoko'},
        },
        # ... etc
    }
    
    def detect_role_from_question(self, question_ast) -> str:
        """Detect which semantic role the question is seeking."""
        q_word = self._get_question_word(question_ast)
        
        if q_word == 'kiu':
            return 'CREATOR'  # or AGENT
        elif q_word == 'kie':
            return 'LOCATION'
        elif q_word == 'kiam':
            return 'TEMPORAL'
        elif q_word == 'kio':
            return 'DEFINITION'
        elif q_word == 'kiom':
            return 'QUANTITY'
        
        return None
    
    def expand_keywords_for_role(self, role: str, keywords: Set[str]) -> Set[str]:
        """Add role-specific terms to keyword set."""
        if role not in self.ROLE_DEFINITIONS:
            return keywords
        
        role_def = self.ROLE_DEFINITIONS[role]
        expanded = keywords.copy()
        expanded.update(role_def.get('verbs', set()))
        expanded.update(role_def.get('nouns', set()))
        
        return expanded
```

## Files to Modify

- `klareco/rag/ast_aware_retriever.py` - Add new role definitions
- Create new `klareco/rag/semantic_roles.py` for centralized role definitions
- `klareco/rag/question_classifier.py` - Integrate with question classification

## Acceptance Criteria

- [ ] LOCATION role implemented and tested
- [ ] TEMPORAL role implemented and tested
- [ ] DEFINITION role implemented and tested
- [ ] QUANTITY role implemented and tested
- [ ] Role detection integrated with question classifier
- [ ] Benchmark queries using these roles show improvement
