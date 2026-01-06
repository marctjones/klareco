---
id: 50
title: Build AST pattern matcher for flexible structural matching
state: open
created: '2026-01-05T15:46:39.588933Z'
labels:
- enhancement
- retrieval
priority: high
---
## Objective
Create a pattern matching system that can match AST structures with transformations (passive voice, appositives, relative clauses, fragments).

## Problem
Current slot matching is too rigid:
- Question: "Kiu fondis Esperanton?" (SUBJ=kiu, VERB=fond, OBJ=esperant)
- Document: "ZAMENHOF, Aŭtoro de la lingvo Esperanto" (SUBJ=ZAMENHOF, VERB=null, OBJ=null)
- **No match** because no verb!

## Solution
Pattern matching that recognizes equivalent structures:

```python
class ASTPatternMatcher:
    """Match AST patterns with structural transformations."""
    
    def create_patterns_for_question(self, question_ast: Dict) -> List[Pattern]:
        """
        Generate multiple equivalent patterns for a question.
        
        For "Kiu fondis Esperanton?":
        - Pattern 1: {PERSON} fondis Esperanton (active)
        - Pattern 2: Esperanto estis fondita de {PERSON} (passive)
        - Pattern 3: {PERSON}, fondinto de Esperanto (appositive)
        - Pattern 4: {PERSON}, aŭtoro/kreinto de Esperanto (synonym)
        """
        patterns = []
        
        # Extract core elements
        q_word = self._get_question_word(question_ast)
        verb = self._get_main_verb(question_ast)
        obj = self._get_object(question_ast)
        
        if q_word == 'kiu' and verb and obj:
            # Active voice pattern
            patterns.append(Pattern(
                agent=Placeholder('PERSON'),
                action=verb['radiko'],
                theme=obj['radiko']
            ))
            
            # Passive voice pattern
            patterns.append(Pattern(
                theme=obj['radiko'],
                action=f"{verb['radiko']}_passive",
                agent_prepositional=Placeholder('PERSON')
            ))
            
            # Appositive pattern (no verb)
            patterns.append(Pattern(
                agent=Placeholder('PERSON'),
                appositive=self._get_agent_noun_for_verb(verb),  # fondis → fondinto
                prepositional_object=obj['radiko']
            ))
            
            # Synonym patterns
            for synonym in self._get_verb_synonyms(verb['radiko']):
                patterns.append(Pattern(
                    agent=Placeholder('PERSON'),
                    action=synonym,
                    theme=obj['radiko']
                ))
        
        return patterns
    
    def match(self, pattern: Pattern, document_ast: Dict) -> Optional[Match]:
        """
        Try to match a pattern against a document AST.
        
        Returns Match with:
        - matched: True/False
        - bindings: {placeholder: value} (e.g., {'PERSON': 'ZAMENHOF'})
        - confidence: 0.0-1.0
        """
        # Try different matching strategies
        
        # Strategy 1: Direct structural match
        if self._structural_match(pattern, document_ast):
            return Match(matched=True, bindings=self._extract_bindings(...))
        
        # Strategy 2: Appositive/fragment match
        if pattern.has_appositive and self._appositive_match(pattern, document_ast):
            return Match(matched=True, bindings=self._extract_bindings(...))
        
        # Strategy 3: Relative clause match
        if self._relative_clause_match(pattern, document_ast):
            return Match(matched=True, bindings=self._extract_bindings(...))
        
        return None
```

## Pattern Examples

### Pattern 1: Active Voice
```
Question: "Kiu fondis Esperanton?"
Pattern: {PERSON} [fond] Esperanton
Match: "Zamenhof fondis Esperanton en 1887" ✓
Binding: PERSON=Zamenhof
```

### Pattern 2: Appositive (No Verb)
```
Question: "Kiu fondis Esperanton?"
Pattern: {PERSON}, [fondinto|aŭtoro|kreinto] de Esperanto
Match: "ZAMENHOF, Aŭtoro de la lingvo Esperanto" ✓
Binding: PERSON=ZAMENHOF
```

### Pattern 3: Passive Voice
```
Question: "Kiu fondis Esperanton?"
Pattern: Esperanto estis fondita de {PERSON}
Match: "Esperanto estis kreita de Zamenhof" ✓
Binding: PERSON=Zamenhof
```

### Pattern 4: Relative Clause
```
Question: "Kiu fondis Esperanton?"
Pattern: {PERSON}, kiu fondis Esperanton
Match: "Zamenhof, kiu fondis Esperanton..." ✓
Binding: PERSON=Zamenhof
```

## Transformation Types to Support

1. **Passive ↔ Active**
   - "X fondis Y" ↔ "Y estis fondita de X"

2. **Verb → Participle/Noun**
   - fondis → fondinto (founder)
   - kreis → kreinto (creator)
   - verkis → verkinto (author)

3. **Synonym Expansion**
   - fondis ≈ kreis ≈ establis ≈ "esti aŭtoro de"

4. **Fragment/Appositive**
   - Full sentence → title/description fragment

## Deliverable
- `klareco/rag/ast_pattern_matcher.py`
- Pattern DSL for expressing structural patterns
- Unit tests for each transformation type
- Integration with question classifier (Task #49)

## Success Criteria
```python
# Should match all these variants
matcher = ASTPatternMatcher()
patterns = matcher.create_patterns_for_question(parse("Kiu fondis Esperanton?"))

assert matcher.match(patterns, parse("Zamenhof fondis Esperanton")) ✓
assert matcher.match(patterns, parse("ZAMENHOF, Aŭtoro de Esperanto")) ✓
assert matcher.match(patterns, parse("Esperanto estis kreita de Zamenhof")) ✓
assert matcher.match(patterns, parse("Zamenhof, kiu fondis Esperanton")) ✓
```

## Dependencies
- Task #49 (question classifier) for generating appropriate patterns

## Effort
~8 hours (complex pattern matching logic)
