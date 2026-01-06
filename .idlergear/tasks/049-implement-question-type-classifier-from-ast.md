---
id: 49
title: Implement question type classifier from AST
state: open
created: '2026-01-05T15:46:08.653720Z'
labels:
- enhancement
- retrieval
priority: high
---
## Objective
Build a deterministic classifier that analyzes question ASTs to identify the question type and what information is being sought.

## Question Types to Detect

### Entity-Seeking Questions
- **WHO** (kiu): Seeking person/agent
  - "Kiu fondis Esperanton?" → seeking PERSON who did ACTION
  - "Kiu verkis la poemon?" → seeking PERSON (author)

- **WHAT** (kio): Seeking thing/concept  
  - "Kio estas akuzativo?" → seeking DEFINITION
  - "Kio signifas mal-?" → seeking MEANING

- **WHERE** (kie): Seeking location
  - "Kie naskiĝis Zamenhof?" → seeking PLACE
  - "Kie situas UEA?" → seeking LOCATION

- **WHEN** (kiam): Seeking time/date
  - "Kiam aperis la Fundamento?" → seeking DATE/YEAR
  - "En kiu jaro naskiĝis X?" → seeking YEAR

### Property-Seeking Questions
- **HOW MANY** (kiom): Seeking quantity
  - "Kiom da reguloj havas Esperanto?" → seeking NUMBER

- **HOW** (kiel): Seeking method/manner
  - "Kiel oni formas la pasintecon?" → seeking PROCESS/METHOD

### Boolean Questions
- **YES/NO** (ĉu): Seeking confirmation
  - "Ĉu ekzistas nedifina artikolo?" → seeking BOOLEAN

## Implementation Approach

```python
class QuestionClassifier:
    """Deterministic question type classifier using AST structure."""
    
    def classify(self, question_ast: Dict) -> QuestionType:
        """
        Classify question from AST structure.
        
        Returns:
            QuestionType with:
            - category: WHO/WHAT/WHERE/WHEN/HOW/BOOLEAN
            - seeking: PERSON/THING/PLACE/TIME/NUMBER/METHOD/BOOLEAN
            - intent: more specific (e.g., "definition", "biography")
        """
        # Check fraztipo
        if question_ast.get('fraztipo') != 'demando':
            raise ValueError("Not a question")
        
        # Extract question word from AST
        q_word = self._find_question_word(question_ast)
        
        # Classify based on question word
        if q_word == 'kiu':
            return self._classify_kiu(question_ast)
        elif q_word == 'kio':
            return self._classify_kio(question_ast)
        elif q_word == 'kie':
            return QuestionType(category='WHERE', seeking='PLACE')
        elif q_word == 'kiam':
            return QuestionType(category='WHEN', seeking='TIME')
        elif q_word == 'kiom':
            return QuestionType(category='HOW_MANY', seeking='NUMBER')
        elif q_word == 'kiel':
            return self._classify_kiel(question_ast)
        elif q_word == 'ĉu':
            return QuestionType(category='BOOLEAN', seeking='YES_NO')
        
        return QuestionType(category='UNKNOWN', seeking='UNKNOWN')
    
    def _classify_kiu(self, ast: Dict) -> QuestionType:
        """'Kiu' can seek person or thing depending on context."""
        # Default: person
        return QuestionType(category='WHO', seeking='PERSON')
    
    def _classify_kio(self, ast: Dict) -> QuestionType:
        """'Kio' classification based on verb."""
        verb = self._get_main_verb(ast)
        
        # "Kio estas X?" → definition
        if verb and verb.get('radiko') == 'est':
            return QuestionType(
                category='WHAT', 
                seeking='DEFINITION',
                intent='definition'
            )
        
        # "Kio signifas X?" → meaning
        if verb and verb.get('radiko') == 'signif':
            return QuestionType(
                category='WHAT',
                seeking='MEANING',
                intent='meaning'
            )
        
        return QuestionType(category='WHAT', seeking='THING')
```

## Deliverable
- `klareco/rag/question_classifier.py`
- Unit tests covering all question types from benchmark
- Should achieve 100% accuracy on the 50 benchmark questions

## Success Criteria
```python
# Test cases
assert classify("Kiu fondis Esperanton?").seeking == 'PERSON'
assert classify("Kiam aperis la Fundamento?").seeking == 'TIME'
assert classify("Kio estas akuzativo?").intent == 'definition'
assert classify("Kiom da reguloj havas Esperanto?").seeking == 'NUMBER'
```

## Dependencies
None - uses existing AST structure

## Effort
~4 hours (implementation + testing)
