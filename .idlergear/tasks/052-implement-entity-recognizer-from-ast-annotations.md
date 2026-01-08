---
id: 52
title: Implement entity recognizer from AST annotations
state: closed
created: '2026-01-05T15:47:58.406980Z'
labels:
- enhancement
- retrieval
priority: high
---
## Objective
Build a deterministic entity recognizer that extracts and classifies entities (persons, places, organizations, times) from parsed AST structure.

## Problem
Pattern matcher needs to identify entities to:
- Find person names for WHO questions
- Find places for WHERE questions  
- Find dates/times for WHEN questions
- Boost relevance when query entities match document entities

## Solution
Use AST parse annotations and heuristics to classify entities.

## Entity Types to Recognize

### 1. Persons
```python
# From AST clues:
- parse_status == 'failed' AND category == 'proper_name'
- parse_status == 'proper_name_unknown'
- Capitalized words (not sentence-initial)
- Preceded by titles: D-ro, Sinjoro, Profesoro
- Common person indicators: -ano, -isto (in some contexts)

# Examples:
"ZAMENHOF" → PERSON (capitalized, unknown)
"D-ro Zamenhof" → PERSON (title + name)
"Ludoviko Lazaro Zamenhof" → PERSON (multi-word capitalized)
```

### 2. Places
```python
# From AST clues:
- Proper names with locative prepositions: en, al, de, ĝis
- Known place suffixes: -ujo (countries), -io (places)
- Geography-related words nearby: urbo, lando, maro, montaro

# Examples:
"en Bjalistoko" → PLACE (en + proper name)
"Roterdamo" → PLACE (city context)
"Pollando" → PLACE (-o ending, geography context)
```

### 3. Organizations
```python
# Acronyms (all caps, 2-5 letters)
"UEA" → ORGANIZATION
"UN" → ORGANIZATION

# Organizational keywords
"Asocio", "Societo", "Organizo", "Instituto"
```

### 4. Temporal Expressions
```python
# Years
"1887", "1905", "2024" → YEAR

# Relative dates
"hodiaŭ", "hieraŭ", "morgaŭ" → DATE

# Month/day patterns
"la 28-an de majo" → DATE
"en julio" → MONTH

# Temporal prepositions
"antaŭ 100 jaroj" → TIME_EXPRESSION
"dum la milito" → TIME_EXPRESSION
```

### 5. Works/Publications
```python
# Quoted or titled
"Unua Libro" → WORK
"Fundamento de Esperanto" → WORK
"La Espero" → WORK (with La + capitalized)
```

## Implementation

```python
class EntityRecognizer:
    """Extract and classify entities from Esperanto AST."""
    
    def __init__(self, semantic_db: SemanticRelationDB = None):
        self.semantic_db = semantic_db
        # Load gazetteers
        self.known_persons = self._load_gazetteer('persons')
        self.known_places = self._load_gazetteer('places')
    
    def extract_entities(self, ast: Dict) -> List[Entity]:
        """
        Extract all entities from an AST.
        
        Returns:
            List of Entity objects with:
            - text: the entity mention
            - type: PERSON/PLACE/ORG/TIME/WORK
            - confidence: 0.0-1.0
            - context: surrounding words
        """
        entities = []
        
        # Walk AST and extract candidates
        for word_ast in self._walk_words(ast):
            # Check if proper noun
            if self._is_proper_noun(word_ast):
                entity_type = self._classify_proper_noun(word_ast, ast)
                entities.append(Entity(
                    text=word_ast['plena_vorto'],
                    type=entity_type,
                    confidence=self._compute_confidence(word_ast, entity_type)
                ))
            
            # Check for temporal expressions
            if self._is_temporal(word_ast):
                entities.append(Entity(
                    text=word_ast['plena_vorto'],
                    type='TIME',
                    confidence=0.95
                ))
        
        # Merge multi-word entities
        entities = self._merge_multiword_entities(entities, ast)
        
        return entities
    
    def _classify_proper_noun(self, word_ast: Dict, context_ast: Dict) -> str:
        """Classify a proper noun as PERSON/PLACE/ORG/WORK."""
        word = word_ast['plena_vorto']
        
        # Check gazetteers first
        if word in self.known_persons:
            return 'PERSON'
        if word in self.known_places:
            return 'PLACE'
        
        # Check context clues
        preceding = self._get_preceding_word(word_ast, context_ast)
        following = self._get_following_word(word_ast, context_ast)
        preposition = self._get_governing_preposition(word_ast, context_ast)
        
        # Person indicators
        if preceding and preceding['radiko'] in ['sinjor', 'doktor', 'd-r', 'profesor']:
            return 'PERSON'
        
        # Place indicators
        if preposition and preposition in ['en', 'al', 'de', 'el']:
            return 'PLACE'
        
        # Organization (acronyms)
        if word.isupper() and 2 <= len(word) <= 5:
            return 'ORG'
        
        # Default: heuristic based on context
        return self._heuristic_classify(word, context_ast)
    
    def has_entity_type(self, ast: Dict, entity_type: str) -> bool:
        """Check if AST contains entity of given type."""
        entities = self.extract_entities(ast)
        return any(e.type == entity_type for e in entities)
    
    def get_entities_by_type(self, ast: Dict, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entities = self.extract_entities(ast)
        return [e for e in entities if e.type == entity_type]
```

## Gazetteers (Seed Data)

Create small gazetteers for high-value entities:

```json
// data/gazetteers/persons.json
[
  "Zamenhof", "Ludoviko Zamenhof", "L.L. Zamenhof",
  "Grabowski", "Waringhien", "Kalocsay"
]

// data/gazetteers/places.json
[
  "Bjalistoko", "Bialystok", "Roterdamo", "Rotterdam",
  "Pollando", "Nederlando", "Varsovio", "Parizo"
]

// data/gazetteers/organizations.json
[
  "UEA", "SAT", "TEJO", "ILEI"
]
```

## Deliverable
- `klareco/rag/entity_recognizer.py`
- `data/gazetteers/*.json` (seed data)
- Unit tests with examples from benchmark questions
- Integration with AST pattern matcher (Task #50)

## Success Criteria
```python
recognizer = EntityRecognizer()

# Persons
entities = recognizer.extract_entities(parse("Zamenhof fondis Esperanton"))
assert any(e.type == 'PERSON' and 'Zamenhof' in e.text for e in entities)

# Places
entities = recognizer.extract_entities(parse("Zamenhof naskiĝis en Bjalistoko"))
assert any(e.type == 'PLACE' and 'Bjalistoko' in e.text for e in entities)

# Times
entities = recognizer.extract_entities(parse("La Fundamento aperis en 1905"))
assert any(e.type == 'TIME' and '1905' in e.text for e in entities)

# Organizations
entities = recognizer.extract_entities(parse("UEA estas en Roterdamo"))
assert any(e.type == 'ORG' and 'UEA' in e.text for e in entities)
```

## Dependencies
- Task #51 (semantic relation DB) - optional enhancement

## Effort
~6 hours (implementation + gazetteers + testing)
