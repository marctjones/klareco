# Abstractive Summarization: Writing NEW Sentences

## The Real Goal

**Input**: Top 20 sentences from reranker (already relevant)
**Output**: 3-4 NEW sentences that synthesize the information
**NOT**: Just picking the top 3 sentences (that's extractive - reranker already did that)
**YES**: Writing new sentences that combine and reorganize information

## Example: Abstractive vs Extractive

### Input (Top 5 sentences from reranker)
```
1. "Kato estas malgranda hejma besto."
2. "Katoj havas kvar piedojn."
3. "Ili havas longan voston."
4. "Katoj estas karnovoruloj."
5. "Ili ĉasas musojn kaj birdojn."
```

### Extractive (WRONG approach for us)
```
Output: Just pick top 3:
1. "Kato estas malgranda hejma besto."
2. "Katoj havas kvar piedojn."
3. "Ili havas longan voston."

→ Choppy, disconnected
```

### Abstractive (WHAT WE WANT)
```
Output: Write NEW sentences synthesizing information:
"Kato estas malgranda hejma besto kun kvar piedoj kaj longa vosto.
Ili estas karnovoruloj, kiuj ĉasas musojn kaj birdojn."

→ Smooth, synthesized, NEW sentences!
```

## Why AST Makes This Possible (Unique to Esperanto)

**In English**: Need large language model (100M+ params) to generate fluent text

**In Esperanto + AST**: Can construct NEW sentences by **manipulating ASTs** deterministically!

### The Key Insight: AST Construction

```python
# Instead of generating text...
# We CONSTRUCT new ASTs from facts, then deparse!

# Extract facts from input sentences:
fact1 = {"entity": "kato", "property": "estas", "value": "besto"}
fact2 = {"entity": "kato", "property": "havas", "value": "piedoj", "count": "kvar"}
fact3 = {"entity": "kato", "property": "havas", "value": "vosto", "modifier": "longa"}

# Combine facts into NEW AST:
new_ast = {
    "subjekto": {"radiko": "kato"},
    "verbo": {"radiko": "est"},
    "objekto": {
        "radiko": "besto",
        "priskriboj": ["malgranda", "hejma"],
        "posedoj": [  # WITH properties (new combination!)
            {"radiko": "piedo", "nombro": "kvar"},
            {"radiko": "vosto", "priskriboj": ["longa"]}
        ]
    }
}

# Deparse to NEW sentence:
"Kato estas malgranda hejma besto kun kvar piedoj kaj longa vosto."
```

**This is NOT text generation - it's AST construction!**

## The Pipeline: Facts → Graph → New ASTs → Sentences

```
┌──────────────────────────────────────────────────────────┐
│ Input: Top 20 Retrieved Sentences (from reranker)       │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 1: Parse to ASTs (Deterministic)                   │
│   Extract structural information                         │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 2: Extract Facts (Deterministic from AST)          │
│   Break down each sentence into atomic facts/propositions│
│                                                          │
│   "Kato estas malgranda hejma besto."                   │
│   → fact1: (kato, estas, besto)                         │
│   → fact2: (besto, havas-priskribon, malgranda)         │
│   → fact3: (besto, havas-priskribon, hejma)             │
│                                                          │
│   "Katoj havas kvar piedojn."                           │
│   → fact4: (kato, havas, piedo)                         │
│   → fact5: (piedo, nombro, kvar)                        │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 3: Build Fact Graph (Deterministic + Kuzu)         │
│   Nodes: Facts (propositions)                           │
│   Edges: Relationships between facts                    │
│                                                          │
│   Example edges:                                        │
│   - fact1 and fact2 share entity "besto" → connect      │
│   - fact4 and fact5 describe same thing → connect       │
│   - fact1 and fact4 both about "kato" → connect         │
│                                                          │
│   Query Kuzu: Which facts co-occur in corpus?           │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 4: Score Fact Importance (Deterministic)           │
│   - Query relevance (overlaps with query roots)         │
│   - Centrality in fact graph (PageRank)                 │
│   - Entity salience (Kuzu graph)                        │
│   - Information type (definition > description > trivia)│
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 5: Cluster Facts by Topic (Deterministic)          │
│   Group related facts that should go in same sentence   │
│                                                          │
│   Cluster 1: Definition & physical features             │
│   - (kato, estas, besto)                                │
│   - (besto, priskribo, malgranda)                       │
│   - (kato, havas, piedo)                                │
│   - (kato, havas, vosto)                                │
│                                                          │
│   Cluster 2: Behavior & diet                            │
│   - (kato, estas, karnovoro)                            │
│   - (kato, ĉasas, muso)                                 │
│   - (kato, ĉasas, birdo)                                │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 6: Select Top Facts (Deterministic Threshold)      │
│   Pick most important facts from each cluster           │
│   - Cluster 1: Top 5 facts (enough for 1 sentence)      │
│   - Cluster 2: Top 3 facts (enough for 1 sentence)      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 7: Construct NEW ASTs (Deterministic Synthesis)    │
│   Combine facts into coherent AST structures            │
│                                                          │
│   From Cluster 1 facts, build:                          │
│   {                                                      │
│     subjekto: "kato",                                   │
│     verbo: "estas",                                     │
│     objekto: {                                          │
│       kerno: "besto",                                   │
│       priskriboj: ["malgranda", "hejma"],              │
│       kun: [                                            │
│         {"radiko": "piedo", "nombro": "kvar"},         │
│         {"radiko": "vosto", "priskriboj": ["longa"]}   │
│       ]                                                 │
│     }                                                   │
│   }                                                     │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 8: Deparse to NEW Sentences (Deterministic)        │
│   AST → Esperanto text using grammar rules              │
│                                                          │
│   Output: "Kato estas malgranda hejma besto kun kvar   │
│            piedoj kaj longa vosto."                      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Step 9: Order Sentences (Deterministic)                 │
│   - Definition first                                    │
│   - Physical features second                            │
│   - Behavior third                                      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ Output: 3-4 NEW Synthesized Sentences                   │
└──────────────────────────────────────────────────────────┘
```

## Detailed Steps

### Step 2: Extract Facts from ASTs

```python
def extract_facts_from_ast(ast):
    """
    Break down AST into atomic facts (propositions).
    Returns list of facts.
    """
    facts = []

    # Fact 1: Main predication (subject-verb-object)
    if 'verbo' in ast and 'objekto' in ast:
        facts.append({
            'type': 'predication',
            'subject': ast['subjekto']['radiko'],
            'predicate': ast['verbo']['radiko'],
            'object': ast['objekto']['radiko'],
            'source_sentence': ast['sentence_id']
        })

    # Fact 2: Subject properties
    if 'subjekto' in ast and 'priskriboj' in ast['subjekto']:
        for priskribo in ast['subjekto']['priskriboj']:
            facts.append({
                'type': 'property',
                'entity': ast['subjekto']['radiko'],
                'property': priskribo['radiko'],
                'source_sentence': ast['sentence_id']
            })

    # Fact 3: Object properties
    if 'objekto' in ast and 'priskriboj' in ast['objekto']:
        for priskribo in ast['objekto']['priskriboj']:
            facts.append({
                'type': 'property',
                'entity': ast['objekto']['radiko'],
                'property': priskribo['radiko'],
                'source_sentence': ast['sentence_id']
            })

    # Fact 4: Possessions (havas constructions)
    if ast['verbo']['radiko'] == 'hav':
        facts.append({
            'type': 'possession',
            'possessor': ast['subjekto']['radiko'],
            'possessed': ast['objekto']['radiko'],
            'quantity': ast['objekto'].get('nombro'),
            'source_sentence': ast['sentence_id']
        })

    # Fact 5: Temporal/locative modifiers
    for modifier in ast.get('aliaj', []):
        if modifier['type'] in ['tempo', 'loko']:
            facts.append({
                'type': modifier['type'],
                'entity': ast['subjekto']['radiko'],
                'value': modifier['valoro'],
                'source_sentence': ast['sentence_id']
            })

    return facts
```

**Example**:
```python
sentence = "Kato estas malgranda hejma besto."
ast = parse(sentence)
facts = extract_facts_from_ast(ast)

# Results:
# fact1: {type: 'predication', subject: 'kato', predicate: 'est', object: 'besto'}
# fact2: {type: 'property', entity: 'besto', property: 'malgranda'}
# fact3: {type: 'property', entity: 'besto', property: 'hejma'}
```

### Step 3: Build Fact Graph

```python
class FactGraph:
    """Graph where nodes are facts, edges are relationships"""

    def __init__(self):
        self.nodes = []  # Facts
        self.edges = []  # Relationships

    def build_from_facts(self, facts, kuzu_db):
        self.nodes = facts

        # Build edges
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], start=i+1):

                # Edge type 1: Shared entity (deterministic)
                if self.share_entity(fact1, fact2):
                    self.add_edge(fact1, fact2, type='shared_entity', weight=1.0)

                # Edge type 2: Co-occurrence in corpus (Kuzu query)
                co_occur = kuzu_db.query("""
                    MATCH (s:Frazoteksto)-[:CONTAINS_FACT]->(f1:Fact)
                    MATCH (s)-[:CONTAINS_FACT]->(f2:Fact)
                    WHERE f1.subject = $subj1 AND f2.subject = $subj2
                    RETURN count(s) as co_count
                """, {'subj1': fact1['subject'], 'subj2': fact2['subject']})

                if co_occur > 5:
                    self.add_edge(fact1, fact2, type='co_occurrence', weight=0.8)

                # Edge type 3: Semantic similarity (existing embeddings)
                similarity = cosine_similarity(
                    get_root_embedding(fact1['subject']),
                    get_root_embedding(fact2['subject'])
                )
                if similarity > 0.6:
                    self.add_edge(fact1, fact2, type='semantic', weight=similarity)

    def share_entity(self, fact1, fact2):
        """Check if facts mention same entity (deterministic)"""
        entities1 = self.get_entities(fact1)
        entities2 = self.get_entities(fact2)
        return len(entities1 & entities2) > 0

    def get_entities(self, fact):
        """Extract all entities mentioned in fact"""
        entities = set()
        if 'subject' in fact:
            entities.add(fact['subject'])
        if 'object' in fact:
            entities.add(fact['object'])
        if 'entity' in fact:
            entities.add(fact['entity'])
        return entities
```

### Step 5: Cluster Facts by Topic

```python
def cluster_facts_by_topic(facts, fact_graph):
    """
    Group facts that should appear in same sentence.
    Uses deterministic clustering based on graph structure.
    """

    clusters = []
    used_facts = set()

    # Strategy: Find connected components in fact graph
    # Facts with strong connections → same cluster

    for fact in facts:
        if fact['id'] in used_facts:
            continue

        # Start new cluster
        cluster = [fact]
        used_facts.add(fact['id'])

        # Add strongly-connected facts
        for neighbor in fact_graph.get_neighbors(fact, min_weight=0.7):
            if neighbor['id'] not in used_facts:
                cluster.append(neighbor)
                used_facts.add(neighbor['id'])

        clusters.append(cluster)

    return clusters
```

### Step 7: Construct NEW ASTs (The Key Innovation!)

```python
def construct_ast_from_facts(facts):
    """
    Build NEW AST by synthesizing facts.
    This is the core of abstractive summarization!
    """

    # Identify main predication (subject-verb-object)
    main_fact = find_main_predication(facts)  # e.g., (kato, estas, besto)

    # Build base AST
    ast = {
        'subjekto': {'radiko': main_fact['subject']},
        'verbo': {'radiko': main_fact['predicate']},
        'objekto': {'radiko': main_fact['object']}
    }

    # Add properties to subject
    subject_properties = [f for f in facts if f['type'] == 'property' and f['entity'] == main_fact['subject']]
    if subject_properties:
        ast['subjekto']['priskriboj'] = [{'radiko': p['property']} for p in subject_properties]

    # Add properties to object
    object_properties = [f for f in facts if f['type'] == 'property' and f['entity'] == main_fact['object']]
    if object_properties:
        ast['objekto']['priskriboj'] = [{'radiko': p['property']} for p in object_properties]

    # Add possessions as "kun" phrase
    possessions = [f for f in facts if f['type'] == 'possession' and f['possessor'] == main_fact['subject']]
    if possessions:
        ast['objekto']['kun'] = []
        for poss in possessions:
            kun_phrase = {'radiko': poss['possessed']}
            if poss.get('quantity'):
                kun_phrase['nombro'] = poss['quantity']
            ast['objekto']['kun'].append(kun_phrase)

    return ast

def find_main_predication(facts):
    """Find the most important fact to use as main clause"""
    # Prefer "estas" (definition) facts
    for fact in facts:
        if fact['type'] == 'predication' and fact['predicate'] == 'est':
            return fact

    # Otherwise, use highest-importance fact
    return max(facts, key=lambda f: f['importance'])
```

**Example**:
```python
# Input facts:
facts = [
    {'type': 'predication', 'subject': 'kato', 'predicate': 'est', 'object': 'besto', 'importance': 0.95},
    {'type': 'property', 'entity': 'besto', 'property': 'malgranda', 'importance': 0.80},
    {'type': 'property', 'entity': 'besto', 'property': 'hejma', 'importance': 0.75},
    {'type': 'possession', 'possessor': 'kato', 'possessed': 'piedo', 'quantity': 'kvar', 'importance': 0.70},
    {'type': 'possession', 'possessor': 'kato', 'possessed': 'vosto', 'importance': 0.65},
    {'type': 'property', 'entity': 'vosto', 'property': 'longa', 'importance': 0.60}
]

# Construct NEW AST:
new_ast = construct_ast_from_facts(facts)

# Result:
{
    'subjekto': {'radiko': 'kato'},
    'verbo': {'radiko': 'est'},
    'objekto': {
        'radiko': 'besto',
        'priskriboj': [{'radiko': 'malgranda'}, {'radiko': 'hejma'}],
        'kun': [
            {'radiko': 'piedo', 'nombro': 'kvar'},
            {'radiko': 'vosto', 'priskriboj': [{'radiko': 'longa'}]}
        ]
    }
}

# Deparse:
"Kato estas malgranda hejma besto kun kvar piedoj kaj longa vosto."
```

**This is a NEW sentence synthesizing 6 facts from multiple input sentences!**

## Why This Works (Esperanto-Specific)

1. **Compositional syntax**: Can add properties, possessions, modifiers systematically
2. **Explicit case marking**: "kun" phrase unambiguous
3. **Regular word formation**: "malgranda hejma besto" always valid
4. **Conjunction rules**: "kaj" combines phrases predictably

**English would be much harder**:
- Word order constraints
- Agreement rules (singular/plural, articles)
- Idiomatic expressions
- Ambiguous constructions

## Complete Example

### Input (20 sentences from reranker)
```
1. "Kato estas besto."
2. "La kato estas malgranda."
3. "Katoj estas hejmaj bestoj."
4. "Katoj havas kvar piedojn."
5. "Ili havas voston."
6. "La vosto estas longa."
7. "Katoj estas karnovoruloj."
8. "Ili manĝas viandon."
9. "Katoj ĉasas musojn."
10. "Ili ankaŭ ĉasas birdojn."
... (10 more sentences)
```

### Step 2: Extract Facts
```
fact1: (kato, est, besto) [importance: 0.95]
fact2: (besto, property, malgranda) [importance: 0.80]
fact3: (besto, property, hejma) [importance: 0.75]
fact4: (kato, hav, piedo) [importance: 0.70]
fact5: (piedo, quantity, kvar) [importance: 0.70]
fact6: (kato, hav, vosto) [importance: 0.65]
fact7: (vosto, property, longa) [importance: 0.60]
fact8: (kato, est, karnovoro) [importance: 0.85]
fact9: (kato, ĉas, muso) [importance: 0.75]
fact10: (kato, ĉas, birdo) [importance: 0.70]
... (more facts)
```

### Step 5: Cluster Facts
```
Cluster 1 (Definition & Physical):
  fact1, fact2, fact3, fact4, fact5, fact6, fact7

Cluster 2 (Diet & Behavior):
  fact8, fact9, fact10
```

### Step 7: Construct NEW ASTs
```
AST 1 (from Cluster 1):
{
  subjekto: "kato",
  verbo: "est",
  objekto: {
    kerno: "besto",
    priskriboj: ["malgranda", "hejma"],
    kun: [
      {radiko: "piedo", nombro: "kvar"},
      {radiko: "vosto", priskriboj: ["longa"]}
    ]
  }
}

AST 2 (from Cluster 2):
{
  subjekto: "kato",
  verbo: {
    tipo: "kunmetita",
    verboj: [
      {radiko: "est", objekto: "karnovoro"},
      {radiko: "ĉas", objekto: ["muso", "birdo"]}
    ],
    ligilo: "kaj"
  }
}
```

### Step 8: Deparse
```
Sentence 1: "Kato estas malgranda hejma besto kun kvar piedoj kaj longa vosto."
Sentence 2: "Ili estas karnovoruloj kaj ĉasas musojn kaj birdojn."
```

**Output**: 2 NEW sentences synthesizing 10+ input sentences! ✓

## What We Need

### Data Structure: Fact Graph ✓

Yes! You were right - we need a graph:
- **Nodes**: Facts (atomic propositions)
- **Edges**: Shared entities, co-occurrence (Kuzu), semantic similarity

### Analysis: Importance Scoring ✓

Using:
- PageRank on fact graph (centrality)
- Entity salience (Kuzu)
- Information type (definition > description)

### Synthesis: AST Construction ✓

**This is the key innovation**: Don't generate text, construct ASTs from facts!

## Learned vs Deterministic

| Component | Deterministic? | Notes |
|-----------|---------------|-------|
| Extract facts | 100% | From AST structure |
| Build fact graph | 95% | Edges mostly deterministic, semantic uses existing embeddings |
| Score importance | 100% | PageRank + Kuzu + heuristics |
| Cluster facts | 100% | Graph connected components |
| Select facts | 100% | Threshold-based |
| Construct ASTs | 100% | Deterministic rules |
| Deparse | 100% | Grammar rules |

**Total new learned parameters**: 0 (reuses existing 320K root embeddings)

## This IS Unique to Esperanto

**Why this works**:
- AST structure makes facts explicit
- Regular grammar enables AST construction
- Can synthesize without language model!

**English would need**: Large generative model (100M+ params) for fluent synthesis

Ready to implement this approach? This is true abstractive summarization with minimal learning!
