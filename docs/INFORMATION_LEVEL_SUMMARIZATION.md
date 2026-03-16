# Information-Level Summarization Design

## The Real Problem

You identified the key insight: **Not all parts of a sentence are equally important, and importance depends on the query and topic context.**

### Example: Multi-Topic Relevance

```
Sentence: "Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887 por internacia komunikado."

Query 1: "Kiu fondis Esperanton?"
  → Most important: "Zamenhof fondis Esperanton"
  → Important: "pola kuracisto", "en 1887"
  → Less relevant: "el Bjalistoko", "por internacia komunikado"

Query 2: "Kial Zamenhof kreis Esperanton?"
  → Most important: "por internacia komunikado"
  → Important: "fondis Esperanton"
  → Less relevant: "pola kuracisto", "el Bjalistoko", "en 1887"

Query 3: "Kie vivis Zamenhof?"
  → Most important: "el Bjalistoko"
  → Important: "Zamenhof"
  → Less relevant: everything else
```

**The challenge**: Same sentence, different queries → different parts are important!

## AST Structure Enables Information-Level Granularity

**Key insight**: ASTs naturally decompose sentences into clauses, phrases, and facts. We can score importance at this granular level!

### AST Decomposition

```python
sentence = "Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887 por internacia komunikado."

ast = {
  "subjekto": {
    "kerno": {"radiko": "zamenhofo"},
    "apozicios": [  # Appositives (descriptive phrases)
      {
        "kerno": {"radiko": "kuracisto"},
        "priskriboj": [{"radiko": "pola"}],
        "loko": {"radiko": "bjalistoko"}
      }
    ]
  },
  "verbo": {"radiko": "fond", "tempo": "pasinto"},
  "objekto": {"radiko": "esperanto"},
  "aliaj": [
    {"tipo": "tempo", "radiko": "en", "valoro": "1887"},  # Temporal modifier
    {"tipo": "celo", "radiko": "por", "subfrazo": {  # Purpose clause
      "tipo": "substantiva",
      "kerno": "komunikado",
      "priskriboj": [{"radiko": "internacia"}]
    }}
  ]
}

# Decompose into information units (facts)
information_units = [
  {
    "id": "iu1",
    "type": "subject",
    "path": "subjekto.kerno",
    "text": "Zamenhof",
    "ast": {"radiko": "zamenhofo"},
    "importance": None  # To be scored
  },
  {
    "id": "iu2",
    "type": "appositive",
    "path": "subjekto.apozicios[0]",
    "text": "pola kuracisto",
    "ast": {"kerno": "kuracisto", "priskriboj": ["pola"]},
    "importance": None
  },
  {
    "id": "iu3",
    "type": "location_detail",
    "path": "subjekto.apozicios[0].loko",
    "text": "el Bjalistoko",
    "ast": {"radiko": "bjalistoko"},
    "importance": None
  },
  {
    "id": "iu4",
    "type": "main_action",
    "path": "verbo + objekto",
    "text": "fondis Esperanton",
    "ast": {"verbo": "fond", "objekto": "esperanto"},
    "importance": None
  },
  {
    "id": "iu5",
    "type": "temporal",
    "path": "aliaj[0]",
    "text": "en 1887",
    "ast": {"tipo": "tempo", "valoro": "1887"},
    "importance": None
  },
  {
    "id": "iu6",
    "type": "purpose",
    "path": "aliaj[1]",
    "text": "por internacia komunikado",
    "ast": {"tipo": "celo", "subfrazo": {...}},
    "importance": None
  }
]
```

**Key advantage**: AST structure gives us natural boundaries for information units (clauses, phrases, modifiers).

## Models Needed

### Model 1: Information Unit Importance Scorer (5M params)

**Purpose**: Score each information unit's importance relative to the query.

```python
class InformationImportanceModel:
    """
    Score importance of each AST node (clause, phrase, modifier) for a query.

    Input:
      - Information unit AST
      - Query AST
      - Context (surrounding sentence, retrieved set)

    Output:
      - Importance score (0-1)
      - Topic relevance scores (if multi-topic)
    """

    def score_importance(self, info_unit, query_ast, context):
        # Extract features (deterministic)
        features = {
            # Semantic features
            'query_overlap': compute_root_overlap(info_unit['ast'], query_ast),
            'answer_type_match': matches_question_type(info_unit, query_ast),

            # Structural features
            'unit_type': info_unit['type'],  # main_action > appositive > modifier
            'depth_in_ast': len(info_unit['path'].split('.')),
            'is_main_clause': info_unit['path'].startswith('verbo') or info_unit['path'].startswith('subjekto.kerno'),

            # Contextual features
            'appears_in_top_k': is_in_top_results(info_unit, context, k=3),
            'frequency_in_results': count_occurrences(info_unit, context)
        }

        # Learned scoring (5M params)
        score = self.model.predict(features)

        return score
```

**Training data**: Human-labeled (information unit, query) pairs with importance scores.

**Example annotations**:
```
Query: "Kiu fondis Esperanton?"

Information unit: "Zamenhof" → Importance: 1.0 (directly answers WHO)
Information unit: "fondis Esperanton" → Importance: 0.95 (confirms action)
Information unit: "pola kuracisto" → Importance: 0.7 (descriptive detail)
Information unit: "en 1887" → Importance: 0.6 (temporal context)
Information unit: "por internacia komunikado" → Importance: 0.3 (motivation, not who)
```

### Model 2: Topic Clustering Model (10M params)

**Purpose**: Group information units by topics/themes.

```python
class TopicClusteringModel:
    """
    Cluster information units into coherent topics.

    Input:
      - Set of information units from retrieved sentences
      - Query (provides context for relevant topics)

    Output:
      - Topic clusters with labels
      - Information unit assignments to topics
    """

    def cluster_topics(self, information_units, query_ast):
        # Extract embeddings for each information unit
        embeddings = [self.encode(iu['ast']) for iu in information_units]

        # Cluster embeddings (learned clustering)
        clusters = self.cluster_model.fit_predict(embeddings)

        # Generate topic labels (extractive: pick most representative unit)
        topics = []
        for cluster_id in set(clusters):
            units_in_cluster = [iu for iu, c in zip(information_units, clusters) if c == cluster_id]

            # Pick centroid as topic label
            topic_label = self.pick_representative(units_in_cluster)

            topics.append({
                'id': cluster_id,
                'label': topic_label,
                'units': units_in_cluster
            })

        return topics
```

**Training approach**: Unsupervised clustering on information unit embeddings, or supervised with human-labeled topic groupings.

**Example**:
```
Query: "Kiu fondis Esperanton?"

Retrieved information units (20):
  iu1: "Zamenhof"
  iu2: "fondis Esperanton"
  iu3: "en 1887"
  iu4: "pola kuracisto"
  iu5: "el Bjalistoko"
  iu6: "por internacia komunikado"
  iu7: "vivis en Bjalistoko"
  iu8: "kreis la lingvon"
  ...

Clustering output:
Topic 1: "Founder Identity" (label from iu1)
  - iu1: "Zamenhof"
  - iu4: "pola kuracisto"
  - iu5: "el Bjalistoko"

Topic 2: "Creation Event" (label from iu2)
  - iu2: "fondis Esperanton"
  - iu3: "en 1887"
  - iu8: "kreis la lingvon"

Topic 3: "Motivation" (label from iu6)
  - iu6: "por internacia komunikado"
  - iu12: "por mondpaco"
```

### Model 3: Topic-Aware Importance Model (5M params)

**Purpose**: Score importance of each information unit relative to EACH topic (not just query).

```python
class TopicAwareImportanceModel:
    """
    Score how important an information unit is for each topic.

    Input:
      - Information unit
      - Topic (cluster of related units)
      - Query

    Output:
      - Importance score for this topic (0-1)
    """

    def score_topic_importance(self, info_unit, topic, query_ast):
        features = {
            # Topic relevance
            'in_topic_cluster': info_unit in topic['units'],
            'similarity_to_topic': compute_similarity(info_unit, topic['label']),

            # Query relevance
            'query_overlap': compute_root_overlap(info_unit['ast'], query_ast),

            # Structural features
            'unit_type': info_unit['type'],
            'is_main_clause': is_main_clause(info_unit)
        }

        score = self.model.predict(features)
        return score
```

**Why this matters**: Information unit "pola kuracisto" is:
- High importance for topic "Founder Identity" (0.9)
- Low importance for topic "Creation Event" (0.2)
- Irrelevant for topic "Motivation" (0.0)

### Model 4: Summary Length Controller (3M params)

**Purpose**: Decide how much detail to include based on requested summary length.

```python
class SummaryLengthController:
    """
    Given importance-scored topics and information units, select what to include
    for 1-sentence, 1-paragraph, or multi-paragraph summary.

    Input:
      - Topics with scored information units
      - Target length (sentences or words)

    Output:
      - Selected information units to include
      - Grouping into sentences/paragraphs
    """

    def select_content(self, topics, target_length):
        if target_length == 1:  # 1-sentence summary
            # Only include highest-importance info from top topic
            top_topic = topics[0]
            units = [iu for iu in top_topic['units'] if iu['importance'] > 0.8]
            return [{'sentence': 1, 'units': units[:3]}]  # Max 3 units in 1 sentence

        elif target_length <= 3:  # 1-paragraph (2-3 sentences)
            # Include top 2 topics, high-importance units only
            selected = []
            for topic in topics[:2]:
                units = [iu for iu in topic['units'] if iu['importance'] > 0.6]
                selected.append({'sentence': len(selected) + 1, 'units': units[:5]})
            return selected

        else:  # Multi-paragraph (4+ sentences)
            # Include all topics, organize by topic
            paragraphs = []
            for topic in topics:
                units = [iu for iu in topic['units'] if iu['importance'] > 0.4]
                paragraphs.append({'paragraph': len(paragraphs) + 1, 'units': units})
            return paragraphs
```

**Training data**: Human judgments of "good 1-sentence vs 3-sentence vs multi-paragraph summaries".

## Deterministic vs Learned Breakdown

| Component | Deterministic | Learned | Notes |
|-----------|--------------|---------|-------|
| **AST parsing** | 100% | 0 | Existing |
| **Information unit extraction** | 100% | 0 | AST decomposition (deterministic boundaries) |
| **Query overlap computation** | 100% | 0 | Root matching |
| **Answer type matching** | 100% | 0 | kiu→person, kio→object, etc. |
| **Information importance (base features)** | 80% | 5M | Heuristics cover basics, learned for nuance |
| **Topic clustering** | 20% | 10M | Could use deterministic (root overlap), but learned better |
| **Topic-aware importance** | 50% | 5M | Hybrid: deterministic features + learned scoring |
| **Summary length control** | 40% | 3M | Could use thresholds (deterministic), but learned adapts better |
| **AST fusion** | 100% | 0 | Existing (from previous design) |
| **Deparsing** | 100% | 0 | Existing |
| **Total** | **60-70%** | **23M** | Much more learned than sentence-level approach |

## Complete Pipeline Example

### Input
```
Query: "Kiu fondis Esperanton?"
Target length: 1 sentence

Retrieved sentences (10):
1. "Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887 por internacia komunikado."
2. "Ludoviko Lazaro Zamenhof estis la kreinto de la lingvo."
3. "Li vivis en Rusio dum sia infanaĝo."
4. "La lingvo estis publikigita en 1887."
5. "Esperanto havas regulan gramatikon."
...
```

### Step 1: Parse to ASTs (deterministic)
```python
asts = [parse(sent) for sent in retrieved_sentences]
```

### Step 2: Extract information units (deterministic)
```python
information_units = []
for ast in asts:
    units = extract_information_units(ast)
    information_units.extend(units)

# Result: 42 information units extracted from 10 sentences
```

### Step 3: Score information importance (learned 5M)
```python
query_ast = parse(query)

for iu in information_units:
    iu['importance'] = importance_model.score_importance(iu, query_ast, context=information_units)

# Top scored units:
# iu1: "Zamenhof" (1.0)
# iu4: "fondis Esperanton" (0.95)
# iu8: "Ludoviko Lazaro Zamenhof" (0.92)
# iu3: "en 1887" (0.75)
# iu2: "pola kuracisto" (0.70)
# ...
```

### Step 4: Cluster into topics (learned 10M)
```python
topics = topic_clustering_model.cluster_topics(information_units, query_ast)

# Result:
# Topic 1: "Founder Identity"
#   - "Zamenhof" (1.0)
#   - "Ludoviko Lazaro Zamenhof" (0.92)
#   - "pola kuracisto" (0.70)
# Topic 2: "Creation Event"
#   - "fondis Esperanton" (0.95)
#   - "en 1887" (0.75)
#   - "estis publikigita en 1887" (0.70)
```

### Step 5: Score topic-aware importance (learned 5M)
```python
for topic in topics:
    for iu in information_units:
        iu[f'importance_topic_{topic["id"]}'] = topic_aware_model.score_topic_importance(iu, topic, query_ast)

# Example: "pola kuracisto"
#   - importance_topic_1 (Founder Identity): 0.90
#   - importance_topic_2 (Creation Event): 0.20
```

### Step 6: Select content for target length (learned 3M)
```python
selected = length_controller.select_content(topics, target_length=1)

# For 1-sentence summary:
# Selected units from Topic 1 (top priority):
#   - "Ludoviko Lazaro Zamenhof" (0.92)  # Full name preferred over just "Zamenhof"
#   - "pola kuracisto" (0.70)  # Descriptive detail
#   - "fondis Esperanton" (0.95)  # Main action
#   - "en 1887" (0.75)  # Temporal context
```

### Step 7: Construct AST from selected units (deterministic)
```python
# Build AST by combining selected information units
summary_ast = {
    "subjekto": {
        "kerno": information_units[8]['ast'],  # "Ludoviko Lazaro Zamenhof"
        "apozicio": information_units[2]['ast']  # "pola kuracisto"
    },
    "verbo": information_units[4]['ast']['verbo'],  # "fondis"
    "objekto": information_units[4]['ast']['objekto'],  # "Esperanton"
    "aliaj": [information_units[3]['ast']]  # "en 1887"
}
```

### Step 8: Deparse to text (deterministic)
```python
summary_text = deparse(summary_ast)
# "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."
```

### Output
```
1-sentence summary:
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."

Information included:
  ✓ Who: Ludoviko Lazaro Zamenhof
  ✓ Description: pola kuracisto
  ✓ What: fondis Esperanton
  ✓ When: en 1887

Information excluded (lower importance for 1-sentence):
  ✗ Where: el Bjalistoko (0.65)
  ✗ Why: por internacia komunikado (0.30)
  ✗ Lived where: vivis en Rusio (0.40)
```

## Multi-Length Example

### 1-Sentence Summary
```
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."

Information units: 4
Topics covered: 1 (Founder Identity)
```

### 1-Paragraph Summary (3 sentences)
```
"Ludoviko Lazaro Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887. La lingvo estis kreita por internacia komunikado. Esperanto havas regulan kaj facile lerneblan gramatikon."

Information units: 8
Topics covered: 2 (Founder Identity, Language Features)
```

### Multi-Paragraph Summary (6 sentences)
```
Paragraph 1 (Founder Identity):
"Ludoviko Lazaro Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887. Li vivis en Rusio dum sia infanaĝo kaj studis medicinon."

Paragraph 2 (Creation Event):
"La lingvo estis kreita por internacia komunikado kaj mondpaco. Ĝi estis publikigita en 1887 sub la pseŭdonimo 'Doktoro Esperanto'."

Paragraph 3 (Language Features):
"Esperanto havas regulan kaj facile lerneblan gramatikon. Ĝi estas planlingvo sen naciaj aŭ etnaj ligoj."

Information units: 15
Topics covered: 3 (Founder Identity, Creation Event, Language Features)
```

## How It Addresses Your Concerns

### 1. "Don't include every word" ✅
- Information unit extraction creates granular pieces
- Importance scoring filters out low-importance units
- Example: "por internacia komunikado" (0.30) excluded from 1-sentence summary

### 2. "Multi-granularity (1-sentence vs paragraph)" ✅
- Length controller model selects units based on target length
- 1-sentence: Only top 3-4 units (importance > 0.7)
- Paragraph: Top 8-10 units (importance > 0.5)
- Multi-paragraph: 15+ units (importance > 0.3), organized by topic

### 3. "Organize by topics" ✅
- Topic clustering model groups related information
- Multi-paragraph summaries have one paragraph per topic
- Example: Paragraph 1 = Founder, Paragraph 2 = Creation, Paragraph 3 = Features

### 4. "Topic-relative importance" ✅
- Topic-aware importance model scores each unit per topic
- "pola kuracisto" is 0.9 for "Founder Identity" but 0.2 for "Creation Event"
- Allows smart selection: include in founder-focused summary, exclude otherwise

### 5. "Part of sentence relevant to one topic, other part to another" ✅
- Information unit extraction splits sentences at clause boundaries
- Example: "Zamenhof, pola kuracisto, fondis Esperanton en 1887 por komunikado"
  - Unit 1: "Zamenhof" → Topic: Founder Identity
  - Unit 2: "pola kuracisto" → Topic: Founder Identity
  - Unit 3: "fondis Esperanton en 1887" → Topic: Creation Event
  - Unit 4: "por komunikado" → Topic: Motivation

## Implementation Phases

### Phase 1: Information Unit Extraction (Week 1)
- Implement deterministic AST decomposition
- Extract clauses, phrases, modifiers as units
- Test: Can we get natural boundaries?

### Phase 2: Importance Scoring (Week 2)
- Start with deterministic heuristics (query overlap, answer type, structure)
- Collect human annotations (which units are important?)
- Train 5M param importance model
- Compare: Deterministic vs Learned

### Phase 3: Topic Clustering (Week 3)
- Try deterministic clustering (root overlap, hierarchical)
- Collect human topic groupings
- Train 10M param clustering model
- Compare approaches

### Phase 4: Topic-Aware Importance (Week 4)
- Implement hybrid model (deterministic features + learned scoring)
- Train 5M param model
- Test on multi-topic queries

### Phase 5: Length Control (Week 5)
- Implement deterministic thresholds (importance > X)
- Collect human judgments of good 1-sentence/paragraph/multi-paragraph summaries
- Train 3M param length controller
- End-to-end evaluation

## Training Data Requirements

### Dataset 1: Information Unit Importance
- **Size**: 5,000-10,000 (information unit, query) pairs
- **Annotation**: Importance score 0-1
- **Collection**: Show annotators query + information unit, ask "How important is this for answering the query?"

### Dataset 2: Topic Clustering
- **Size**: 1,000 queries with 10-20 information units each
- **Annotation**: Group units into topics, label each topic
- **Collection**: Show annotators query + units, ask "Which units belong together? What is each group about?"

### Dataset 3: Topic-Aware Importance
- **Size**: 3,000-5,000 (information unit, topic, query) triples
- **Annotation**: Importance score 0-1 for this unit in this topic
- **Collection**: Show query + topic + unit, ask "How important is this unit for this topic?"

### Dataset 4: Summary Length
- **Size**: 500 queries with multiple summary lengths
- **Annotation**: Human-written 1-sentence, 3-sentence, multi-paragraph summaries
- **Collection**: Show query + retrieved sentences, ask humans to write summaries at different lengths

## Comparison: Sentence-Level vs Information-Level

| Aspect | Sentence-Level | Information-Level |
|--------|----------------|-------------------|
| **Granularity** | Whole sentences | Clauses, phrases, modifiers |
| **Selection** | Include/exclude entire sentence | Pick parts of sentences |
| **Conciseness** | Less concise (redundant details) | More concise (only relevant facts) |
| **Multi-topic** | Hard (sentence spans multiple topics) | Easy (units assigned to topics) |
| **Learned params** | 0-10M | 23M |
| **Deterministic** | 90-95% | 60-70% |
| **Complexity** | Lower | Higher |
| **Quality** | Good for simple queries | Better for complex, multi-topic queries |

## Recommendation

**This is the right design for production-quality summarization**, but it's more complex:

1. **Start simple**: Sentence-level summarization (Phase 1-2)
   - Get baseline working
   - Evaluate: Is it sufficient?

2. **Add complexity if needed**: Information-level summarization (Phase 3+)
   - If sentence-level too verbose or can't handle multi-topic
   - Implement information unit extraction
   - Train models incrementally

**Hypothesis**: Sentence-level sufficient for 70% of queries, information-level needed for remaining 30% (complex, multi-topic queries).

**Trade-off**:
- Sentence-level: Simpler, more deterministic, faster to implement
- Information-level: Better quality, handles complexity, more learned

Your choice based on quality requirements and timeline!
