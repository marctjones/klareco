# Linguistic & Graph-Based Summarization Design

## Research Synthesis

Based on linguistic theory, information theory, and graph-based approaches, here's a comprehensive design for Klareco summarization.

## Part 1: Linguistic Foundation

### Information Structure Theory

From [Information structure - Wikipedia](https://en.wikipedia.org/wiki/Information_structure) and [Vallduví (2014)](https://www.upf.edu/documents/2983731/3019795/2014-infostruct-cup-handbook.pdf):

**Key concepts**:
1. **Topic (Theme)**: What the sentence is about (old/given information)
2. **Comment (Rheme)**: What is said about the topic (new information)
3. **Focus**: The most informative part that answers the implicit question
4. **Background**: Presupposed/given information

**Example**:
```
Sentence: "Zamenhof fondis ESPERANTON en 1887."
         [Topic]  [verb] [FOCUS]      [Background]

Query: "Kion fondis Zamenhof?" (What did Zamenhof found?)
  → Topic: "Zamenhof" (given from query)
  → Focus: "ESPERANTON" (new, answers question)
  → Background: "fondis", "en 1887" (supporting info)

Importance ranking:
  1. ESPERANTON (focus, answers query)
  2. Zamenhof (topic, anchors sentence)
  3. fondis (main verb, necessary for meaning)
  4. en 1887 (background, adds detail)
```

### Esperanto Information Structure

From [Esperanto word order - Lingolia](https://esperanto.lingolia.com/en/grammar/sentence-structure/word-order) and [Wikibooks](https://en.wikibooks.org/wiki/Esperanto:_A_Complete_and_Comprehensive_Grammar/Word_order):

**Key insight**: Esperanto uses **topic-comment** order flexibly due to case marking!

```
Standard SVO: "La hundo mordas la katon."
              [Topic]  [verb]  [Comment]

Topicalization (emphasize object):
"La katon mordas la hundo."
[TOPIC] [verb]  [Comment]

The accusative -n allows reordering without ambiguity!
```

**How to identify topic/focus in Esperanto**:
1. **Position**: Initial position = topic (what we're talking about)
2. **Case marking**: Accusative (-n) marks object even in non-standard position
3. **Stress/emphasis**: In speech, focus gets stress (we can't detect in text)
4. **Anaphora**: Pronouns (li, ŝi, ĝi) refer to topic

**Implementation**: AST structure captures this!
```python
# Topic = what appears first in AST traversal
# Focus = answers query (query-AST overlap)
# Background = modifiers, temporal/locative phrases
```

### Rhetorical Structure Theory (RST)

From [RST Wikipedia](https://en.wikipedia.org/wiki/Rhetorical_structure_theory) and [Taboada & Mann (2006)](https://journals.sagepub.com/doi/10.1177/1461445606064836):

**Key concepts**:
1. **Elementary Discourse Units (EDUs)**: Clauses/propositions
2. **Nucleus**: Essential information (most important)
3. **Satellite**: Supporting information (can be removed)
4. **Relations**: How units connect (elaboration, cause, contrast, etc.)

**Example RST tree**:
```
Sentence: "Zamenhof fondis Esperanton en 1887, ĉar li deziris mondpacon."
         [Nucleus: Zamenhof fondis Esperanton en 1887]
         [Satellite: ĉar li deziris mondpacon]
         Relation: CAUSE

Importance:
  Nucleus > Satellite
  "Zamenhof fondis Esperanton" = essential
  "ĉar li deziris mondpacon" = optional (adds reason)
```

**RST Relations in Esperanto** (deterministically identifiable):
- **Cause**: ĉar, pro (because)
- **Purpose**: por (for, in order to)
- **Condition**: se (if)
- **Concession**: kvankam (although)
- **Elaboration**: kiu-clauses (relative clauses)
- **Sequence**: poste, tiam (then, afterwards)

**Implementation**: Parse discourse connectives to build RST structure!

## Part 2: Graph-Based Approaches

### TextRank/LexRank Applied to ASTs

From [TextRank (Medium)](https://medium.com/@yassineerraji/understanding-textrank-a-deep-dive-into-graph-based-text-summarization-and-keyword-extraction-905d1fb5d266) and [LexRank (CMU)](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html):

**Traditional approach**:
1. Build graph: Sentences = nodes, Similarity = edges
2. Run PageRank to find central sentences
3. Select top-ranked sentences

**Our AST-based adaptation**:
1. Build graph: **Information units = nodes** (not sentences)
2. Edges: **AST relations** + **semantic similarity**
3. Run PageRank to find central information units
4. Select top-ranked units across topics

**Information Unit Graph Construction**:

```python
class InformationGraph:
    """
    Graph where nodes are information units (clauses, phrases, facts)
    and edges represent relationships.
    """

    def __init__(self):
        self.nodes = []  # Information units
        self.edges = []  # Relationships

    def build_from_asts(self, asts):
        """Build graph from retrieved sentence ASTs"""

        # Step 1: Extract information units (nodes)
        for ast in asts:
            units = extract_information_units(ast)
            self.nodes.extend(units)

        # Step 2: Add edges (relationships)
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:], start=i+1):

                # Edge type 1: AST structural relations (deterministic)
                if self.has_syntactic_relation(node1, node2):
                    self.add_edge(node1, node2,
                                 type='syntactic',
                                 weight=1.0)

                # Edge type 2: Coreference (deterministic)
                if self.are_coreferent(node1, node2):
                    self.add_edge(node1, node2,
                                 type='coreference',
                                 weight=0.9)

                # Edge type 3: Shared entity (Kuzu graph!)
                if self.share_entity(node1, node2):
                    self.add_edge(node1, node2,
                                 type='entity',
                                 weight=0.8)

                # Edge type 4: Semantic similarity (embeddings)
                similarity = self.compute_semantic_similarity(node1, node2)
                if similarity > 0.5:
                    self.add_edge(node1, node2,
                                 type='semantic',
                                 weight=similarity)

        return self

    def has_syntactic_relation(self, node1, node2):
        """Deterministic: Check if nodes have syntactic relation in AST"""
        # Examples:
        # - node1 is subject, node2 is verb → related
        # - node1 is main clause, node2 is subordinate → related
        # - node1 is noun, node2 is modifier of that noun → related

        if node1['sentence_id'] != node2['sentence_id']:
            return False  # Different sentences

        # Check if one is parent/child in AST
        if node2['path'].startswith(node1['path']):
            return True
        if node1['path'].startswith(node2['path']):
            return True

        return False

    def are_coreferent(self, node1, node2):
        """Deterministic: Check if nodes refer to same entity"""
        # Use Esperanto coreference rules:
        # - "Zamenhof" and "li" → coreferent
        # - "fundinto" (founder) and "Zamenhof" → coreferent

        entity1 = resolve_entity(node1)
        entity2 = resolve_entity(node2)

        return entity1 == entity2

    def share_entity(self, node1, node2):
        """Query Kuzu graph: Do these units mention same entity?"""
        # Check Kuzu knowledge graph
        entities1 = extract_entities(node1['ast'])
        entities2 = extract_entities(node2['ast'])

        shared = entities1 & entities2
        return len(shared) > 0

    def compute_semantic_similarity(self, node1, node2):
        """Use embeddings for semantic similarity"""
        emb1 = self.encode(node1['ast'])
        emb2 = self.encode(node2['ast'])

        return cosine_similarity(emb1, emb2)
```

**Graph edge types**:
1. ✅ **Syntactic**: AST structure (deterministic)
2. ✅ **Coreference**: Same entity reference (deterministic)
3. ✅ **Entity**: Shared entity in Kuzu graph (deterministic lookup)
4. ⚠️ **Semantic**: Embedding similarity (learned)

**Mostly deterministic!**

### PageRank for Information Unit Importance

From [PageRank centrality (Medium)](https://medium.com/@nassarhuda/pagerank-and-centrality-they-measure-importance-2458aa8b6eef):

**Standard PageRank**:
```
PR(node) = (1-d) + d * Σ(PR(incoming_node) / out_degree(incoming_node))

where:
  d = damping factor (0.85)
  incoming_node = nodes that link to this node
```

**Query-Biased PageRank** (for query-focused summarization):
```
PR(node) = (1-d) * query_relevance(node) + d * Σ(...)

where:
  query_relevance(node) = how well node matches query (deterministic!)
```

**Implementation**:

```python
def compute_importance_pagerank(info_graph, query_ast):
    """
    Use PageRank on information graph to score importance.

    Advantages over heuristics:
    - Considers global structure (not just local query match)
    - Nodes connected to many important nodes are important
    - Captures centrality in information network
    """

    # Initialize: Nodes matching query get high initial score
    initial_scores = {}
    for node in info_graph.nodes:
        initial_scores[node.id] = compute_query_relevance(node, query_ast)

    # Run query-biased PageRank
    pagerank_scores = pagerank(
        graph=info_graph,
        initial_scores=initial_scores,
        damping=0.85,
        iterations=100
    )

    return pagerank_scores

def compute_query_relevance(node, query_ast):
    """Deterministic query relevance (initialization for PageRank)"""

    score = 0.0

    # 1. Root overlap (deterministic)
    query_roots = extract_roots(query_ast)
    node_roots = extract_roots(node['ast'])
    overlap = len(query_roots & node_roots) / len(query_roots)
    score += overlap * 0.5

    # 2. Answer type match (deterministic)
    if matches_question_type(node, query_ast):
        score += 0.3  # kiu → person, kio → object, etc.

    # 3. Information structure (deterministic)
    if is_focus(node, query_ast):
        score += 0.2  # Focus = most informative

    return min(score, 1.0)
```

**Why PageRank is powerful**:
- Node with many incoming edges = central to information network
- Node connected to highly-ranked nodes = important by association
- Query-biased initialization ensures relevance to query

## Part 3: Knowledge Graph Integration

### Entity Salience from Kuzu Graph

From [GDELT entity salience](https://blog.gdeltproject.org/neural-entity-extraction-disambiguation-sentiment-salience-knowledge-graph-linking-and-contextualization-via-web-ngrams-3-0/):

**Key insight**: Entities that appear frequently in knowledge graph are more salient!

**Kuzu Graph Structure** (existing):
```
Nodes:
  - Radiko (roots): hund, kur, esperanto, zamenhofo
  - Vorto (words): hundo, kuras, Esperanton, Zamenhof
  - Frazoteksto (sentences): full sentences with ASTs

Edges:
  - (Vorto)-[:HAS_ROOT]->(Radiko)
  - (Frazoteksto)-[:CONTAINS_WORD]->(Vorto)
  - (Frazoteksto)-[:MENTIONS_ENTITY]->(Entity)  # NEW!
```

**Entity Salience Computation**:

```python
def compute_entity_salience_from_kuzu(entity, kuzu_db):
    """
    Query Kuzu graph to determine how salient/important an entity is.

    Measures:
    1. Frequency: How often does entity appear in corpus?
    2. Centrality: How connected is entity in knowledge graph?
    3. Distinctiveness: How unique/specific is entity? (IDF)
    """

    # Measure 1: Frequency (deterministic)
    frequency = kuzu_db.query("""
        MATCH (s:Frazoteksto)-[:MENTIONS_ENTITY]->(e:Entity {name: $entity})
        RETURN count(s) as freq
    """, {'entity': entity})

    # Measure 2: Centrality (deterministic)
    # Count how many other entities this entity co-occurs with
    centrality = kuzu_db.query("""
        MATCH (e1:Entity {name: $entity})<-[:MENTIONS_ENTITY]-(s:Frazoteksto)-[:MENTIONS_ENTITY]->(e2:Entity)
        WHERE e1 <> e2
        RETURN count(DISTINCT e2) as connections
    """, {'entity': entity})

    # Measure 3: Distinctiveness (IDF-like)
    total_sentences = kuzu_db.query("MATCH (s:Frazoteksto) RETURN count(s)")
    idf = log(total_sentences / (frequency + 1))

    # Combine measures
    salience = (
        0.4 * normalize(frequency) +      # Common entities are important
        0.3 * normalize(centrality) +     # Connected entities are important
        0.3 * normalize(idf)              # Distinctive entities are important
    )

    return salience
```

**Using Entity Salience for Information Units**:

```python
def boost_information_unit_by_entity_salience(info_unit, kuzu_db):
    """
    Boost importance of information unit if it mentions salient entities.
    """

    entities = extract_entities(info_unit['ast'])

    if not entities:
        return 0.0  # No entities, no boost

    # Get salience for each entity
    saliences = [compute_entity_salience_from_kuzu(e, kuzu_db) for e in entities]

    # Use max salience (most important entity)
    max_salience = max(saliences)

    return max_salience
```

**Example**:
```
Query: "Kiu fondis Esperanton?"

Information units extracted:
1. "Zamenhof" → mentions entity "Zamenhof"
   → Query Kuzu: "Zamenhof" appears in 500 sentences, connected to 50 entities
   → High salience (0.85)

2. "pola kuracisto" → mentions concept "kuracisto" (doctor)
   → Query Kuzu: "kuracisto" appears in 200 sentences, connected to 30 entities
   → Medium salience (0.60)

3. "el Bjalistoko" → mentions entity "Bjalistoko" (city)
   → Query Kuzu: "Bjalistoko" appears in 20 sentences, connected to 5 entities
   → Low salience (0.25)

Importance ranking boosted by entity salience:
1. "Zamenhof" (0.95 + 0.85 boost = 1.0)
2. "fondis Esperanton" (0.90 + 0.0 = 0.90) [no entity mention]
3. "pola kuracisto" (0.70 + 0.60 = 0.85)
4. "el Bjalistoko" (0.65 + 0.25 = 0.72)
```

### Information Theory: Entropy-Based Salience

From [Investigating Entropy for Extractive Document Summarization](https://arxiv.org/abs/2109.10886):

**Key insight**: Information units with low entropy (high certainty) relative to query are more important!

**Shannon Entropy**:
```
H(X) = -Σ P(x) * log(P(x))

Low entropy = predictable, certain
High entropy = uncertain, unpredictable
```

**Application to information units**:

```python
def compute_entropy_based_importance(info_unit, query_ast, all_units):
    """
    Use information theory to score importance.

    Intuition: Information units that reduce uncertainty about the answer
    are more important.
    """

    # Step 1: Compute probability distribution over entities/concepts
    # in information unit

    roots = extract_roots(info_unit['ast'])
    root_probs = {}

    for root in roots:
        # P(root | query) = How often does this root appear in relevant contexts?
        # Query Kuzu for co-occurrence statistics
        count = count_cooccurrences(root, query_roots, kuzu_db)
        root_probs[root] = count / total_relevant_sentences

    # Step 2: Compute entropy
    entropy = -sum(p * log(p) for p in root_probs.values() if p > 0)

    # Step 3: Importance = inverse of entropy
    # Low entropy = high certainty = high importance
    importance = 1.0 / (1.0 + entropy)

    return importance
```

**Example**:
```
Query: "Kiu fondis Esperanton?"

Unit 1: "Zamenhof"
  → P(Zamenhof | "kiu fondis Esperanton") = 0.95 (highly certain)
  → Entropy = -0.95*log(0.95) - 0.05*log(0.05) = 0.20 (low)
  → Importance = 1 / (1 + 0.20) = 0.83 (high)

Unit 2: "en Bjalistoko"
  → P(Bjalistoko | "kiu fondis Esperanton") = 0.10 (uncertain)
  → Entropy = 2.30 (high)
  → Importance = 1 / (1 + 2.30) = 0.30 (low)
```

## Part 4: Complete Hybrid Architecture

### The "Information Crystallization Graph"

Your insight: Build a graph that helps identify most important information!

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Query + Retrieved Sentences                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Parse to ASTs (Deterministic)                      │
│   - 16 Esperanto grammar rules                              │
│   - Output: Structured ASTs with roles                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Extract Information Units (Deterministic)          │
│   - Use AST structure to find clause boundaries             │
│   - Create information units (clauses, phrases, modifiers)  │
│   - Tag with linguistic features (topic/focus, nucleus/sat) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Build Information Graph (Mostly Deterministic)     │
│   - Nodes: Information units                                │
│   - Edges: Syntactic (AST), Coreference, Entity, Semantic  │
│   - Query Kuzu graph for entity co-occurrence               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Compute Multi-Dimensional Importance               │
│   ① Query-biased PageRank (Graph centrality)               │
│   ② Entity salience from Kuzu (Knowledge graph)            │
│   ③ Entropy-based salience (Information theory)            │
│   ④ Information structure (Topic/Focus/Background)         │
│   ⑤ RST nucleus detection (Essential vs optional)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Aggregate Importance Scores (Weighted Combination) │
│   Final = 0.30*PageRank + 0.25*EntitySalience +           │
│           0.20*Entropy + 0.15*InfoStructure + 0.10*RST    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Topic Clustering (Learned, 10M params)             │
│   - Cluster information units by semantic similarity        │
│   - Use both graph structure and embeddings                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Select Information Units (Deterministic Threshold) │
│   - 1-sentence: Top 3-4 units, importance > 0.75           │
│   - 1-paragraph: Top 8-10 units, importance > 0.55         │
│   - Multi-paragraph: Top 15+ units, importance > 0.40      │
│   - Organize by topic clusters                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: Construct Summary AST (Deterministic Fusion)       │
│   - Fuse selected information units using AST operations    │
│   - Apply RST structure (nucleus first, satellite optional) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 9: Deparse to Text (Deterministic)                    │
│   - AST → Esperanto text using grammar rules               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Output: Summary (1-sentence, 1-paragraph, or multi-para)   │
└─────────────────────────────────────────────────────────────┘
```

### Learned vs Deterministic Breakdown

| Component | Deterministic | Learned | Why This Balance |
|-----------|--------------|---------|------------------|
| AST parsing | 100% | 0 | Grammar rules explicit |
| Information unit extraction | 100% | 0 | AST structure provides boundaries |
| Syntactic edge detection | 100% | 0 | AST structure |
| Coreference resolution | 95% | 0 (5M optional) | Esperanto pronouns explicit |
| Entity edge detection | 100% | 0 | Kuzu graph lookup |
| Semantic edge detection | 0% | Embeddings | Need learned similarity |
| PageRank computation | 100% | 0 | Standard algorithm |
| Entity salience (Kuzu) | 100% | 0 | Graph queries |
| Entropy computation | 100% | 0 | Information theory formula |
| Info structure (topic/focus) | 90% | 0 (5M optional) | Heuristics from word order |
| RST nucleus detection | 80% | 0 (5M optional) | Discourse connectives |
| Importance aggregation | 100% | 0 | Weighted sum |
| Topic clustering | 20% | 10M | Learned clustering better |
| Unit selection | 100% | 0 | Threshold-based |
| AST fusion | 100% | 0 | Grammar rules |
| Deparsing | 100% | 0 | Grammar rules |
| **Total** | **85-90%** | **10M (optional 15M)** | **Mostly deterministic!** |

## Part 5: Implementation Roadmap

### Phase 1: Deterministic Foundation (Weeks 1-2)
1. Implement information unit extraction from ASTs
2. Build information graph with deterministic edges:
   - Syntactic (AST structure)
   - Coreference (Esperanto rules)
   - Entity (Kuzu lookup)
3. Implement query-biased PageRank
4. Implement entity salience from Kuzu
5. Implement entropy-based salience
6. Test on 20-30 sample queries

**Deliverable**: Working system with 0 learned parameters, ~75-80% quality

### Phase 2: Linguistic Features (Week 3)
7. Implement information structure detection:
   - Topic/focus identification (word order heuristics)
   - Given/new information (anaphora detection)
8. Implement RST nucleus detection:
   - Parse discourse connectives (ĉar, por, se, etc.)
   - Build RST tree (nucleus/satellite relations)
9. Aggregate importance scores (weighted combination)
10. Test improved system

**Deliverable**: System with linguistic grounding, ~80-85% quality

### Phase 3: Knowledge Graph Integration (Week 4)
11. Enhance Kuzu schema:
    - Add (Frazoteksto)-[:MENTIONS_ENTITY]->(Entity) edges
    - Precompute entity salience scores
12. Implement entity-boosted importance scoring
13. Add semantic edge detection (using existing embeddings)
14. Test with Kuzu integration

**Deliverable**: Kuzu-integrated system, ~85-90% quality

### Phase 4: Optional Learning (Weeks 5-6)
15. Collect evaluation dataset (100 queries, human summaries)
16. Measure: Where does deterministic approach fail?
17. If needed, train 10M topic clustering model
18. Compare: Deterministic vs Learned

**Deliverable**: Final system, target 90%+ quality

## Part 6: Example Walkthrough

### Input
```
Query: "Kiu fondis Esperanton?"

Retrieved sentences (5):
1. "Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887."
2. "Ludoviko Lazaro Zamenhof estis la kreinto de la lingvo."
3. "Li vivis en Rusio dum sia infanaĝo."
4. "La lingvo estis kreita por internacia komunikado."
5. "Esperanto havas regulan gramatikon."
```

### Step 1: Parse (Deterministic)
```
AST[0]: {subjekto: {kerno: "Zamenhof", apozicios: ["pola kuracisto", "el Bjalistoko"]}, verbo: "fondis", objekto: "Esperanton", aliaj: ["en 1887"]}
AST[1]: {subjekto: "Ludoviko Lazaro Zamenhof", verbo: "estis", objekto: "kreinto", ...}
...
```

### Step 2: Extract Information Units (Deterministic)
```
Units:
u1: "Zamenhof" (subjekto.kerno)
u2: "pola kuracisto" (subjekto.apozicio)
u3: "el Bjalistoko" (subjekto.apozicio)
u4: "fondis Esperanton" (verbo + objekto)
u5: "en 1887" (temporal modifier)
u6: "Ludoviko Lazaro Zamenhof" (subjekto)
u7: "estis kreinto de la lingvo" (verbo + objekto)
...
```

### Step 3: Build Information Graph (Mostly Deterministic)
```
Nodes: u1, u2, u3, u4, u5, u6, u7, ...

Edges:
(u1, u2): syntactic (same sentence, subject + appositive), weight=1.0
(u1, u4): syntactic (subject + verb), weight=1.0
(u1, u6): coreference ("Zamenhof" = "Ludoviko Lazaro Zamenhof"), weight=0.9
(u1, u7): entity (both mention "Zamenhof"), weight=0.8
(u4, u7): semantic (both about founding/creating), weight=0.75
...

Query Kuzu:
  "Zamenhof" appears in 500 sentences
  "Esperanton" appears in 800 sentences
  Co-occur in 450 sentences → strong connection
```

### Step 4: Compute Multi-Dimensional Importance

**① PageRank (Graph Centrality)**
```
Run PageRank with query-biased initialization:
  u1 ("Zamenhof"): 0.15 (high centrality, many connections)
  u4 ("fondis Esperanton"): 0.18 (highest, connects to many units)
  u6 ("Ludoviko Lazaro Zamenhof"): 0.12 (moderate centrality)
  u2 ("pola kuracisto"): 0.08 (fewer connections)
  u3 ("el Bjalistoko"): 0.05 (low centrality)
```

**② Entity Salience (Kuzu Graph)**
```
Query Kuzu for entity salience:
  "Zamenhof": frequency=500, centrality=50 → salience=0.85
  "Esperanton": frequency=800, centrality=80 → salience=0.92
  "Bjalistoko": frequency=20, centrality=5 → salience=0.25
  "kuracisto": frequency=200, centrality=30 → salience=0.60
```

**③ Entropy (Information Theory)**
```
Compute entropy for each unit:
  u1 ("Zamenhof"): P(Zamenhof | "kiu fondis") = 0.90 → entropy=0.32 → importance=0.76
  u4 ("fondis Esperanton"): P("fondis" | query) = 0.95 → entropy=0.21 → importance=0.83
  u3 ("el Bjalistoko"): P("Bjalistoko" | query) = 0.15 → entropy=2.10 → importance=0.32
```

**④ Information Structure (Linguistics)**
```
Identify topic/focus:
  u1 ("Zamenhof"): FOCUS (answers "kiu" question) → score=1.0
  u4 ("fondis Esperanton"): TOPIC (given from query context) → score=0.8
  u2 ("pola kuracisto"): BACKGROUND (descriptive detail) → score=0.6
  u3 ("el Bjalistoko"): BACKGROUND (locative detail) → score=0.4
```

**⑤ RST Structure (Discourse)**
```
Parse RST relations:
  Sentence 1: [Nucleus: "Zamenhof fondis Esperanton"] [Satellite: "en 1887" (temporal)]
  u4 ("fondis Esperanton"): NUCLEUS → score=1.0
  u5 ("en 1887"): SATELLITE → score=0.6
```

### Step 5: Aggregate Importance (Deterministic Weighted Sum)
```
Final importance = 0.30*PageRank + 0.25*EntitySalience + 0.20*Entropy + 0.15*InfoStructure + 0.10*RST

u1 ("Zamenhof"):
  = 0.30*(0.15) + 0.25*(0.85) + 0.20*(0.76) + 0.15*(1.0) + 0.10*(1.0)
  = 0.045 + 0.213 + 0.152 + 0.150 + 0.100
  = 0.66

u4 ("fondis Esperanton"):
  = 0.30*(0.18) + 0.25*(0.92) + 0.20*(0.83) + 0.15*(0.8) + 0.10*(1.0)
  = 0.054 + 0.230 + 0.166 + 0.120 + 0.100
  = 0.67

u6 ("Ludoviko Lazaro Zamenhof"):
  = 0.30*(0.12) + 0.25*(0.85) + 0.20*(0.76) + 0.15*(1.0) + 0.10*(0.8)
  = 0.036 + 0.213 + 0.152 + 0.150 + 0.080
  = 0.63

u2 ("pola kuracisto"):
  = 0.30*(0.08) + 0.25*(0.60) + 0.20*(0.65) + 0.15*(0.6) + 0.10*(0.8)
  = 0.024 + 0.150 + 0.130 + 0.090 + 0.080
  = 0.47

u3 ("el Bjalistoko"):
  = 0.30*(0.05) + 0.25*(0.25) + 0.20*(0.32) + 0.15*(0.4) + 0.10*(0.6)
  = 0.015 + 0.063 + 0.064 + 0.060 + 0.060
  = 0.26

Ranked: u4 (0.67) > u1 (0.66) > u6 (0.63) > u2 (0.47) > u3 (0.26) > ...
```

### Step 6: Topic Clustering (Optional 10M Learned)
```
Cluster information units:
  Topic 1: "Founder Identity" [u1, u6, u2]
  Topic 2: "Creation Event" [u4, u5]
```

### Step 7: Select Units (Deterministic Threshold)
```
For 1-sentence summary (importance > 0.60):
  Selected: u4, u1, u6, u2
  (Drop u3, u5 - below threshold)

For 3-sentence summary (importance > 0.40):
  Selected: u4, u1, u6, u2, u3, u5, ...
```

### Step 8: Construct Summary AST (Deterministic Fusion)
```
Fuse u6 + u2 + u4:
{
  subjekto: {kerno: "Ludoviko Lazaro Zamenhof", apozicio: "pola kuracisto"},
  verbo: "fondis",
  objekto: "Esperanton"
}
```

### Step 9: Deparse (Deterministic)
```
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton."
```

### Output
```
1-sentence summary:
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton."

Importance breakdown:
✓ u4 ("fondis Esperanton"): 0.67 - Main action (high PageRank, entity salience, entropy)
✓ u6 ("Ludoviko Lazaro Zamenhof"): 0.63 - Full name (focus, entity salience)
✓ u2 ("pola kuracisto"): 0.47 - Description (background but relevant)
✗ u3 ("el Bjalistoko"): 0.26 - Location (low importance)
✗ u5 ("en 1887"): (not shown) - Temporal (satellite, below threshold)
```

## Summary

**This design integrates**:
1. ✅ Linguistic theory (information structure, RST)
2. ✅ Graph-based algorithms (PageRank on information units)
3. ✅ Knowledge graph (Kuzu entity salience, co-occurrence)
4. ✅ Information theory (entropy-based salience)
5. ✅ Esperanto-specific features (case marking, word order, coreference)

**Learned parameters**: 10M (optional topic clustering only)

**Deterministic**: 85-90% of pipeline

**Unique to Klareco**: Combines AST structure + knowledge graph + linguistic theory in a way that's impossible for English or other languages!

## Sources

- [Information structure - Wikipedia](https://en.wikipedia.org/wiki/Information_structure)
- [Vallduví (2014) - Information Structure (PDF)](https://www.upf.edu/documents/2983731/3019795/2014-infostruct-cup-handbook.pdf)
- [Esperanto word order - Lingolia](https://esperanto.lingolia.com/en/grammar/sentence-structure/word-order)
- [Esperanto Grammar - Wikibooks](https://en.wikibooks.org/wiki/Esperanto:_A_Complete_and_Comprehensive_Grammar/Word_order)
- [Rhetorical Structure Theory - Wikipedia](https://en.wikipedia.org/wiki/Rhetorical_structure_theory)
- [Taboada & Mann (2006) - Applications of RST](https://journals.sagepub.com/doi/10.1177/1461445606064836)
- [Investigating Entropy for Extractive Document Summarization (arXiv)](https://arxiv.org/abs/2109.10886)
- [TextRank for Summarization (Medium)](https://medium.com/@yassineerraji/understanding-textrank-a-deep-dive-into-graph-based-text-summarization-and-keyword-extraction-905d1fb5d266)
- [LexRank: Graph-based Lexical Centrality (CMU)](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
- [PageRank and Centrality (Medium)](https://medium.com/@nassarhuda/pagerank-and-centrality-they-measure-importance-2458aa8b6eef)
- [GDELT Neural Entity Salience](https://blog.gdeltproject.org/neural-entity-extraction-disambiguation-sentiment-salience-knowledge-graph-linking-and-contextualization-via-web-ngrams-3-0/)
