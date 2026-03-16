# Schema-Aware Summarization Design

## The Problem

**Previous approach**: Query-oriented fact ranking (good for "Who founded Esperanto?" but not for summarization)

**User insight**: "How does this help summarize 20 sentences into 4 sentences?"

**Answer**: Need linguistic theory about what information is **generally important** in summaries, not just query-specific importance.

## Linguistic Foundations

### 1. Rhetorical Structure Theory (RST)

**Source**: [RST Deep Dive](https://www.numberanalytics.com/blog/rhetorical-structure-theory-deep-dive), [Wikipedia](https://en.wikipedia.org/wiki/Rhetorical_structure_theory)

**Core insight**: Text has hierarchical structure where units are either:
- **Nucleus**: Essential information, comprehensible on its own
- **Satellite**: Supporting/contextual information, incomprehensible without nucleus

**RST Relations** (examples):
```
Elaboration:  [Zamenhof fondis Esperanton]NUCLEUS [en 1887]SATELLITE
Background:   [Li estis kuracisto]SATELLITE [kiu fondis Esperanton]NUCLEUS
Evidence:     [Esperanto estas facila]NUCLEUS [ĉar ĝi havas regulan gramatikon]SATELLITE
Cause:        [Zamenhof kreis Esperanton]NUCLEUS [por mondpaco]SATELLITE
```

**For summarization**: Include all nucleus units first, then add satellites based on length target.

### 2. Content Schemas (Schema Theory)

**Source**: [Schema Theory](https://www.educationcorner.com/schema-theory/)

**Core insight**: Different text types have expected information structures. Readers expect certain types of information.

**Biographical Schema**:
```
1. Identification → Who they are (name, profession, nationality)
2. Major achievement → What they're known for
3. Key dates → Birth, death, major events
4. Context → Historical/geographical setting
5. Significance → Why they matter
```

**Definitional Schema**:
```
1. Category → What kind of thing ("estas [category]")
2. Essential properties → Defining characteristics (always true)
3. Typical properties → Common features (usually true)
4. Function/behavior → What it does
5. Examples → Instances or subtypes
```

**Event Schema**:
```
1. Main event → What happened
2. Participants → Who was involved
3. Time/place → When/where
4. Cause → Why it happened
5. Consequences → Results/impact
```

### 3. Inverted Pyramid (Journalism)

**Source**: [Inverted Pyramid Structure](https://fiveable.me/newsroom/unit-4/structure-news-articles-inverted-pyramid-formats/study-guide/HAdB1vi8CyJIR4pq)

**Core insight**: Most important information first. Essential facts answer the **5 W's + H**:
- **Who**: Entities involved
- **What**: Main event/action
- **When**: Time
- **Where**: Location
- **Why**: Cause/motivation
- **How**: Manner/method

**For summarization**: Ensure summary answers the 5 W's + H for the topic.

### 4. Information Structure

**Core insight**: Information has pragmatic status:
- **Given information**: Already mentioned/known → lower importance
- **New information**: Novel content → higher importance
- **Topic**: What sentence is about → continuity
- **Comment**: What's said about topic → new contribution

**For summarization**: Prioritize new information, penalize redundancy.

---

## Architecture: Schema-Aware Fact Ranking

### Pipeline Overview

```
Input: Query/Topic + 20 Retrieved Sentences
  ↓
Step 1: Classify Summary Type (Deterministic)
  → biographical | definitional | event | query_answer
  ↓
Step 2: Parse Sentences → Extract Facts (Deterministic)
  → 50-100 atomic facts from 20 sentences
  ↓
Step 3: Schema-Aware Fact Ranking (95% Deterministic)
  → Rank using: schema slot + RST role + information status + centrality
  ↓
Step 4: Select Top Facts (Deterministic)
  → Select N facts for target length (1 sentence = 5 facts, 4 sentences = 15 facts)
  ↓
Step 5: Cluster Facts (Deterministic)
  → Group by shared entities/topics
  ↓
Step 6: Synthesize Sentences (Deterministic)
  → Build ASTs from fact clusters → Deparse to text
  ↓
Output: 1-4 Sentence Summary
```

---

## Step 1: Classify Summary Type (100% Deterministic)

```python
class SummaryTypeClassifier:
    """
    Determine what kind of summary is needed.
    Different types need different information.
    """

    def classify(self, query_or_topic):
        """
        Classify from query patterns.
        Returns: 'biographical' | 'definitional' | 'event' | 'query_answer' | 'generic'
        """

        if not query_or_topic:
            return 'generic'  # Generic summarization of input

        query = query_or_topic.lower()

        # Biographical queries
        if any(phrase in query for phrase in ['rakontu pri', 'tell me about', 'kiu estas', 'who is']):
            # Check if topic is a person
            if self._is_person_entity(query):
                return 'biographical'
            else:
                return 'topical'  # Topical summary (not person)

        # Definitional queries
        elif any(phrase in query for phrase in ['kio estas', 'what is', 'difinu', 'define']):
            return 'definitional'

        # Event queries
        elif any(phrase in query for phrase in ['kio okazis', 'what happened', 'kiam', 'when']):
            return 'event'

        # Specific factoid queries (not summarization)
        elif query.startswith(('kiu', 'who', 'kie', 'where', 'kial', 'why', 'kiel', 'how')):
            return 'query_answer'

        else:
            return 'generic'

    def _is_person_entity(self, text):
        """Check if text mentions a person (heuristics)"""
        person_indicators = [
            'zamenhof', 'person', 'homo', 'viro', 'virino',
            # Check if proper noun
        ]
        return any(indicator in text.lower() for indicator in person_indicators)
```

---

## Step 3: Schema-Aware Fact Ranking (95% Deterministic)

### Schema Definitions

```python
class SchemaAwareFactRanker:
    """
    Rank facts based on what's important for the summary type.
    Uses linguistic theory: schemas, RST, information structure.
    """

    # Schema importance weights (from linguistic theory)
    BIOGRAPHICAL_SCHEMA = {
        'identification': 1.0,      # Who they are (name, profession, nationality)
        'major_achievement': 0.95,  # What they're known for
        'birth_death': 0.85,        # Key dates (birth, death)
        'profession': 0.80,         # Occupation
        'nationality': 0.75,        # Origin/ethnicity
        'education': 0.70,          # Where they studied
        'context': 0.60,            # Historical/geographical setting
        'family': 0.50,             # Family relationships
        'minor_detail': 0.30        # Trivia
    }

    DEFINITIONAL_SCHEMA = {
        'category': 1.0,            # "X estas [category]"
        'essential_property': 0.90, # Always-true characteristics
        'typical_property': 0.70,   # Usually-true characteristics
        'function': 0.75,           # What it does
        'parts': 0.65,              # Components ("havas X")
        'example': 0.50,            # Instances/subtypes
        'comparison': 0.55          # Similar/different to Y
    }

    EVENT_SCHEMA = {
        'main_event': 1.0,          # What happened (main predication)
        'participants': 0.90,       # Who was involved
        'time': 0.85,               # When it happened
        'place': 0.80,              # Where it happened
        'cause': 0.75,              # Why it happened
        'consequence': 0.70,        # Result/impact
        'manner': 0.60,             # How it happened
        'detail': 0.40              # Minor information
    }

    GENERIC_SCHEMA = {
        'main_predication': 1.0,    # Central statements
        'entity_property': 0.70,    # Properties of main entities
        'relationship': 0.65,       # Relationships between entities
        'context': 0.50,            # Background/setting
        'detail': 0.30              # Minor info
    }
```

### Ranking Algorithm

```python
    def rank_facts(self, facts, summary_type, query=None):
        """
        Rank facts using schema-based importance.
        Returns facts sorted by importance (0-1).
        """

        # Get appropriate schema
        schema = self._get_schema(summary_type)

        # Track entities mentioned (for given/new detection)
        mentioned_entities = set()

        for fact in facts:
            importance = 0.0

            # Factor 1: Schema slot importance (40%)
            slot = self._classify_into_schema(fact, summary_type)
            schema_weight = schema.get(slot, 0.5)
            importance += 0.40 * schema_weight

            # Factor 2: RST role - nucleus vs satellite (25%)
            rst_role = self._determine_rst_role(fact, summary_type)
            if rst_role == 'nucleus':
                importance += 0.25 * 1.0
            elif rst_role == 'satellite':
                importance += 0.25 * 0.5
            else:
                importance += 0.25 * 0.7  # Unknown

            # Factor 3: Information status - given vs new (15%)
            if self._is_new_information(fact, mentioned_entities):
                importance += 0.15 * 1.0  # New information
            else:
                importance += 0.15 * 0.3  # Redundant/given

            # Factor 4: Centrality - how connected is this fact? (10%)
            centrality = self._compute_centrality(fact, facts)
            importance += 0.10 * centrality

            # Factor 5: Sentence rank from retriever (10%)
            rank_score = 1.0 / (1.0 + fact.get('sentence_rank', 10))
            importance += 0.10 * rank_score

            # Optional: Query relevance (if doing query answering, not pure summarization)
            if query and summary_type == 'query_answer':
                query_overlap = self._compute_query_overlap(fact, query)
                importance += 0.20 * query_overlap

            fact['importance'] = min(importance, 1.0)

            # Track entities for given/new detection
            self._update_mentioned_entities(fact, mentioned_entities)

        # Sort by importance
        return sorted(facts, key=lambda f: f['importance'], reverse=True)
```

### Schema Classification (Deterministic)

```python
    def _classify_into_schema(self, fact, summary_type):
        """
        Classify fact into schema slot.
        100% deterministic - uses AST patterns.
        """

        if summary_type == 'biographical':
            return self._classify_biographical(fact)
        elif summary_type == 'definitional':
            return self._classify_definitional(fact)
        elif summary_type == 'event':
            return self._classify_event(fact)
        else:
            return self._classify_generic(fact)

    def _classify_biographical(self, fact):
        """Classify into biographical schema slot"""

        # Identification: "X estas [profession/nationality]"
        if fact['type'] == 'predication' and fact['predicate'] == 'est':
            obj = fact.get('object', '')

            # Profession
            if obj in ['kuracisto', 'verkisto', 'sciencisto', 'artisto', 'politikisto']:
                return 'profession'

            # Nationality
            elif obj in ['pola', 'franca', 'germana', 'usona', 'brita']:
                return 'nationality'

            else:
                return 'identification'

        # Major achievement: "fondis", "kreis", "skribis", "inventis"
        elif fact['type'] == 'predication' and fact['predicate'] in [
            'fond', 'kre', 'skriv', 'invent', 'malkovr'
        ]:
            return 'major_achievement'

        # Birth/death dates
        elif fact['type'] == 'temporal':
            if fact.get('event') in ['naskigx', 'mort']:
                return 'birth_death'
            else:
                return 'context'

        # Location (birthplace, residence)
        elif fact['type'] == 'locative':
            if fact.get('event') in ['naskigx', 'viv']:
                return 'context'
            else:
                return 'minor_detail'

        # Education
        elif fact['type'] == 'predication' and fact['predicate'] in ['stud', 'lern']:
            return 'education'

        # Family
        elif fact['type'] == 'possession' and fact.get('possessed') in ['patro', 'patrino', 'frato', 'fratino', 'edzo', 'edzino']:
            return 'family'

        else:
            return 'minor_detail'

    def _classify_definitional(self, fact):
        """Classify into definitional schema slot"""

        # Category: "X estas [category]"
        if fact['type'] == 'predication' and fact['predicate'] == 'est':
            return 'category'

        # Essential vs typical property
        elif fact['type'] == 'property':
            if self._is_essential_property(fact):
                return 'essential_property'
            else:
                return 'typical_property'

        # Function: "X [action]" - what it does
        elif fact['type'] == 'predication' and fact['predicate'] in [
            'mangx', 'trinkt', 'viv', 'kres', 'malgrandigt'
        ]:
            return 'function'

        # Parts: "havas X"
        elif fact['type'] == 'possession':
            return 'parts'

        # Example: "ekzemple", "kiel"
        elif fact.get('is_example'):
            return 'example'

        else:
            return 'typical_property'

    def _classify_event(self, fact):
        """Classify into event schema slot"""

        # Main event: first predication from highest-ranked sentence
        if fact['type'] == 'predication' and fact.get('sentence_rank', 999) <= 2:
            return 'main_event'

        # Participants: facts mentioning entities
        elif fact.get('subject_entity') or fact.get('object_entity'):
            return 'participants'

        # Time
        elif fact['type'] == 'temporal':
            return 'time'

        # Place
        elif fact['type'] == 'locative':
            return 'place'

        # Cause
        elif fact['type'] == 'cause':
            return 'cause'

        # Consequence
        elif fact['type'] == 'purpose':
            return 'consequence'

        else:
            return 'detail'

    def _is_essential_property(self, fact):
        """
        Is this property essential (always true) or typical (usually true)?
        Heuristic: essential properties are often part of the definition.
        """

        # Essential properties tend to be:
        # - About category membership: "karnivora", "mamulo"
        # - About fundamental structure: "havas kvar piedojn"

        essential_roots = [
            'karnivoro', 'herbivoro', 'mamulo', 'besto', 'planto',
            'viva', 'vivanta', 'mortebla'
        ]

        return fact.get('property') in essential_roots
```

### RST Role Detection (Deterministic)

```python
    def _determine_rst_role(self, fact, summary_type):
        """
        Is this fact a nucleus (essential) or satellite (supporting)?
        Uses RST heuristics.
        """

        # Rule 1: Predications are usually nuclei
        if fact['type'] == 'predication':
            # Especially "estas" predications (definitions)
            if fact['predicate'] == 'est':
                return 'nucleus'
            else:
                return 'nucleus'

        # Rule 2: Properties can be nucleus or satellite
        elif fact['type'] == 'property':
            # Essential properties are nuclei
            if self._is_essential_property(fact):
                return 'nucleus'
            else:
                return 'satellite'

        # Rule 3: Temporal/locative are usually satellites (elaboration/background)
        elif fact['type'] in ['temporal', 'locative']:
            # Exception: in event summaries, time/place can be nuclei
            if summary_type == 'event':
                return 'nucleus'
            else:
                return 'satellite'

        # Rule 4: Cause/purpose can be nuclei if they're high-importance
        elif fact['type'] in ['cause', 'purpose']:
            # If this explains the main event, it's nucleus
            if fact.get('sentence_rank', 999) <= 3:
                return 'nucleus'
            else:
                return 'satellite'

        # Rule 5: Possessions are usually satellites (elaboration)
        elif fact['type'] == 'possession':
            return 'satellite'

        return 'unknown'
```

### Information Status (Deterministic)

```python
    def _is_new_information(self, fact, mentioned_entities):
        """
        Is this fact introducing new information or repeating known info?
        Uses given/new information tracking.
        """

        # Extract entities from fact
        fact_entities = self._extract_entities(fact)

        # All entities are new = definitely new information
        if not mentioned_entities:
            return True

        # None of the entities have been mentioned = new information
        if not (fact_entities & mentioned_entities):
            return True

        # All entities already mentioned = likely redundant
        if fact_entities <= mentioned_entities:
            return False

        # Mix of new and given = partially new (count as new)
        return True

    def _update_mentioned_entities(self, fact, mentioned_entities):
        """Track which entities have been mentioned"""
        entities = self._extract_entities(fact)
        mentioned_entities.update(entities)

    def _extract_entities(self, fact):
        """Extract all entities from fact"""
        entities = set()
        for key in ['subject', 'object', 'entity', 'possessor', 'possessed']:
            if fact.get(key):
                entities.add(fact[key])
        return entities
```

---

## Example: Biographical Summarization

### Input

**Query**: "Rakontu al mi pri Zamenhof"
**Retrieved**: 20 sentences about Zamenhof

### Step 1: Classify

```python
summary_type = classify_summary_type("Rakontu al mi pri Zamenhof")
# → 'biographical'
```

### Step 2: Extract Facts (from 20 sentences → ~50 facts)

```python
facts = [
    {'type': 'predication', 'subject': 'Zamenhof', 'predicate': 'fond', 'object': 'Esperanto', 'rank': 1},
    {'type': 'property', 'entity': 'Zamenhof', 'property': 'kuracisto', 'rank': 2},
    {'type': 'property', 'entity': 'kuracisto', 'property': 'pola', 'rank': 2},
    {'type': 'temporal', 'event': 'fond', 'time': '1887', 'rank': 3},
    {'type': 'temporal', 'event': 'naskigx', 'time': '1859', 'rank': 5},
    {'type': 'locative', 'entity': 'Zamenhof', 'location': 'Bjalistoko', 'rank': 4},
    {'type': 'predication', 'subject': 'Zamenhof', 'predicate': 'hav', 'object': 'celo', 'rank': 8},
    # ... 43 more facts
]
```

### Step 3: Schema-Aware Ranking

```python
# Fact 1: (Zamenhof, fondis, Esperanto)
slot = 'major_achievement'     # Schema weight: 0.95
rst_role = 'nucleus'            # RST bonus: 1.0
is_new = True                   # New info bonus: 1.0
centrality = 0.85               # High centrality
sentence_rank = 1               # Top rank: 1.0

importance = 0.40*0.95 + 0.25*1.0 + 0.15*1.0 + 0.10*0.85 + 0.10*1.0
           = 0.38 + 0.25 + 0.15 + 0.085 + 0.10
           = 0.965

# Fact 2: (Zamenhof, estas, kuracisto)
slot = 'profession'             # Schema weight: 0.80
rst_role = 'nucleus'            # RST bonus: 1.0
is_new = False                  # Already mentioned Zamenhof: 0.3
centrality = 0.70
sentence_rank = 2               # 0.5

importance = 0.40*0.80 + 0.25*1.0 + 0.15*0.3 + 0.10*0.70 + 0.10*0.5
           = 0.32 + 0.25 + 0.045 + 0.07 + 0.05
           = 0.735

# Fact 3: (fondis, tempo, 1887)
slot = 'context'                # Schema weight: 0.60
rst_role = 'satellite'          # RST: 0.5
is_new = True                   # New temporal info: 1.0
centrality = 0.40
sentence_rank = 3               # 0.33

importance = 0.40*0.60 + 0.25*0.5 + 0.15*1.0 + 0.10*0.40 + 0.10*0.33
           = 0.24 + 0.125 + 0.15 + 0.04 + 0.033
           = 0.588

# ... rank all 50 facts
```

### Step 4: Select Top Facts

For **4-sentence summary**, select top **15 facts** (~4 per sentence):

```python
top_facts = [
    (Zamenhof, fondis, Esperanto) - 0.965
    (Zamenhof, estas, kuracisto) - 0.735
    (kuracisto, property, pola) - 0.680
    (naskigx, tempo, 1859) - 0.650
    (fondis, tempo, 1887) - 0.588
    (naskigx, loko, Bjalistoko) - 0.570
    (Zamenhof, havas, celo_pri_mondpaco) - 0.540
    (Esperanto, estas, planlingvo) - 0.520
    (Zamenhof, uzis, pseuxdonimo) - 0.480
    (pseuxdonimo, estas, "Doktoro_Esperanto") - 0.470
    # ... 5 more facts
]
```

### Step 5: Cluster Facts

```python
clusters = [
    [  # Cluster 1: Identification + major achievement
        (Zamenhof, estas, kuracisto),
        (kuracisto, property, pola),
        (fondis, Esperanto),
    ],
    [  # Cluster 2: Birth + founding date
        (naskigx, tempo, 1859),
        (naskigx, loko, Bjalistoko),
        (fondis, tempo, 1887),
    ],
    [  # Cluster 3: Purpose
        (Esperanto, estas, planlingvo),
        (havas, celo_pri_mondpaco),
    ],
    [  # Cluster 4: Pseudonym
        (uzis, pseuxdonimo),
        (pseuxdonimo, estas, "Doktoro Esperanto"),
    ]
]
```

### Step 6: Synthesize

```python
# Cluster 1 → Sentence 1
ast1 = {
    'subjekto': {'radiko': 'Zamenhof', 'priskriboj': [{'radiko': 'pola'}, {'radiko': 'kuracisto'}]},
    'verbo': {'radiko': 'fond'},
    'objekto': {'radiko': 'Esperanto'}
}
sentence1 = "Zamenhof, pola kuracisto, fondis Esperanton."

# Cluster 2 → Sentence 2
ast2 = {
    'subjekto': {'radiko': 'li'},
    'verbo': {'radiko': 'naskigx'},
    'aliaj': [
        {'type': 'tempo', 'teksto': 'en 1859'},
        {'type': 'loko', 'teksto': 'en Bjalistoko'},
    ]
}
sentence2 = "Li naskiĝis en 1859 en Bjalistoko, kaj fondis Esperanton en 1887."

# Cluster 3 → Sentence 3
sentence3 = "Esperanto estas planlingvo kun celo pri internacia komunikado kaj mondpaco."

# Cluster 4 → Sentence 4
sentence4 = "Zamenhof uzis la pseŭdonimon 'Doktoro Esperanto' por publikigi la lingvon."
```

### Output

```
Zamenhof, pola kuracisto, fondis Esperanton.
Li naskiĝis en 1859 en Bjalistoko, kaj fondis Esperanton en 1887.
Esperanto estas planlingvo kun celo pri internacia komunikado kaj mondpaco.
Zamenhof uzis la pseŭdonimon 'Doktoro Esperanto' por publikigi la lingvon.
```

**Success!** 4-sentence biographical summary from 20 input sentences, with:
- ✅ Identification (who he was)
- ✅ Major achievement (founded Esperanto)
- ✅ Key dates (1859, 1887)
- ✅ Purpose (why he created it)

---

## Optional: Semantic Enhancement Model (2M params)

**When needed**: If deterministic schema-based ranking has systematic errors.

**What it does**: Adjusts importance scores using semantic features (from existing 320K root embeddings).

**Architecture**:
```python
class SemanticSchemaRanker:
    def __init__(self, base_ranker, embedding_model):
        self.base_ranker = base_ranker
        self.embedding_model = embedding_model  # Existing 320K

        # Small adjustment model
        self.adjustment_model = nn.Sequential(
            nn.Linear(10, 64),   # det_score + 9 semantic features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def score_fact(self, fact, all_facts, summary_type):
        # Get deterministic schema-based score
        base_score = self.base_ranker.rank_facts([fact], summary_type)[0]['importance']

        # Extract semantic features
        semantic_features = self._extract_semantic_features(fact)

        # Get adjustment
        adjustment = self.adjustment_model(torch.tensor([base_score] + semantic_features)) * 0.2

        return base_score + adjustment.item()
```

**Training data**: 5,000-8,000 (facts, summary_type, gold_importance_score) examples.

---

## Summary: Deterministic Processing

| Component | Deterministic? | Notes |
|-----------|---------------|-------|
| Summary type classification | 100% ✅ | Query patterns |
| Fact extraction | 100% ✅ | AST structure |
| Schema slot classification | 100% ✅ | AST patterns |
| RST role detection | 100% ✅ | Fact type + context heuristics |
| Information status | 100% ✅ | Entity tracking |
| Centrality | 100% ✅ | Co-occurrence counting |
| Fact ranking | 95% ✅ | Schema + RST + info status |
| Clustering | 100% ✅ | Shared entities |
| Synthesis | 100% ✅ | AST construction |
| **Total** | **~95%** | Optional 2M semantic model for refinement |

---

## Implementation Plan

### Phase 1: Core Schema Implementation (Week 1-2)

1. Implement schema definitions (biographical, definitional, event)
2. Implement `classify_summary_type()` from query patterns
3. Implement `classify_into_schema()` for fact→slot mapping
4. Implement `determine_rst_role()` for nucleus/satellite detection
5. Implement information status tracking (given/new)
6. Update fact ranking to use schema importance + RST + info status

### Phase 2: Testing (Week 3)

1. Test on biographical summaries (20 sentences → 4 sentences)
2. Test on definitional summaries (15 sentences → 3 sentences)
3. Test on event summaries (20 sentences → 4 sentences)
4. Evaluate: Does summary include essential information? Is it coherent?

### Phase 3: Evaluation (Week 4)

- Where does deterministic approach fail?
- Are failures systematic?
- Would semantic model help?

### Phase 4: Optional Semantic Model (Week 5)

- Train 2M adjustment model (only if needed)
- Measure improvement

---

## Success Criteria

**Must have**:
- ✅ Can summarize 20 sentences about a person into 4-sentence biography
- ✅ Includes essential information per schema (who, profession, achievement, dates)
- ✅ Drops non-essential details (trivia, redundant info)
- ✅ Output is coherent and readable (not just fact list)

**Nice to have**:
- ✅ Works for different summary types (biographical, definitional, event)
- ✅ Adjustable length (1-4 sentences)
- ✅ Explainable (can show why facts were selected based on schema)

---

## References

1. **Rhetorical Structure Theory**: [RST Deep Dive](https://www.numberanalytics.com/blog/rhetorical-structure-theory-deep-dive)
2. **RST Wikipedia**: [RST Overview](https://en.wikipedia.org/wiki/Rhetorical_structure_theory)
3. **RST and Summarization**: [ScienceDirect Review](https://www.sciencedirect.com/science/article/abs/pii/S0957417420302451)
4. **Inverted Pyramid**: [Journalism Structure](https://fiveable.me/newsroom/unit-4/structure-news-articles-inverted-pyramid-formats/study-guide/HAdB1vi8CyJIR4pq)
5. **Schema Theory**: [Education Corner](https://www.educationcorner.com/schema-theory/)
6. **Discourse and Summarization**: [ResearchGate Paper](https://www.researchgate.net/publication/220048957_Discourse_Automatic_Annotation_of_Texts_an_Application_to_Summarization)
