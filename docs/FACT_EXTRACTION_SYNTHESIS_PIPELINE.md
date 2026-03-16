# Fact Extraction → Ranking → Synthesis Pipeline

## Complete Pipeline Overview

```
┌────────────────────────────────────────────────┐
│ Input: Query + Top 10-20 Retrieved Sentences  │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 1: Parse to ASTs (Deterministic)         │
│   - 16 Esperanto grammar rules                 │
│   - Already have parser                        │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 2: Extract Facts (Deterministic)         │
│   - Break ASTs into atomic facts               │
│   - (subject, predicate, object) triples       │
│   - Properties, possessions, modifiers         │
│   - 100% deterministic from AST structure      │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 3: Rank Fact Importance                  │
│   - Deterministic base scores (70%)            │
│   - Optional: 2M param model for semantic (30%)│
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 4: Cluster Facts by Topic (Deterministic)│
│   - Group by shared entities/roots             │
│   - Co-occurrence patterns                     │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 5: Select Facts (Deterministic Threshold)│
│   - Top N facts for target answer length      │
│   - 1 sentence: top 4-5 facts                  │
│   - 3-4 sentences: top 12-15 facts             │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Step 6: Synthesize Answer (Deterministic)     │
│   - Construct NEW ASTs from facts              │
│   - Combine related facts into clauses         │
│   - Order using grammar + discourse rules      │
│   - Deparse to text                            │
└────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────┐
│ Output: Synthesized Answer                     │
└────────────────────────────────────────────────┘
```

---

## Step 2: Fact Extraction (100% Deterministic)

### What Is a Fact?

**Fact = Atomic proposition that can be independently verified**

Examples:
- `(Zamenhof, fondis, Esperanton)` - predication
- `(Zamenhof, estas, kuracisto)` - categorization
- `(kuracisto, property, pola)` - property
- `(fondis, tempo, 1887)` - temporal modifier
- `(Zamenhof, vivis_en, Bjalistoko)` - locative

### Fact Extraction from AST

```python
class FactExtractor:
    """
    Extract atomic facts from AST structure.
    100% deterministic - uses AST structure only.
    """

    def extract_facts(self, ast, sentence_id):
        """
        Extract all facts from a single sentence AST.

        Returns: List of facts
        """
        facts = []

        # FACT TYPE 1: Main Predication (subject-verb-object)
        if 'verbo' in ast and 'subjekto' in ast:
            fact = {
                'type': 'predication',
                'subject': self._extract_root(ast['subjekto']),
                'subject_entity': self._extract_entity(ast['subjekto']),
                'predicate': self._extract_root(ast['verbo']),
                'object': self._extract_root(ast.get('objekto', {})),
                'object_entity': self._extract_entity(ast.get('objekto', {})),
                'sentence_id': sentence_id,
                'sentence_rank': ast.get('retrieval_rank', 0),
                'confidence': ast.get('parse_confidence', 1.0)
            }
            facts.append(fact)

        # FACT TYPE 2: Subject Properties
        if 'subjekto' in ast and 'priskriboj' in ast['subjekto']:
            for priskribo in ast['subjekto']['priskriboj']:
                fact = {
                    'type': 'property',
                    'entity': self._extract_root(ast['subjekto']),
                    'property': self._extract_root(priskribo),
                    'sentence_id': sentence_id,
                    'sentence_rank': ast.get('retrieval_rank', 0)
                }
                facts.append(fact)

        # FACT TYPE 3: Object Properties
        if 'objekto' in ast and 'priskriboj' in ast['objekto']:
            for priskribo in ast['objekto']['priskriboj']:
                fact = {
                    'type': 'property',
                    'entity': self._extract_root(ast['objekto']),
                    'property': self._extract_root(priskribo),
                    'sentence_id': sentence_id,
                    'sentence_rank': ast.get('retrieval_rank', 0)
                }
                facts.append(fact)

        # FACT TYPE 4: Possessions ("havas" constructions)
        if ast.get('verbo', {}).get('radiko') == 'hav':
            fact = {
                'type': 'possession',
                'possessor': self._extract_root(ast['subjekto']),
                'possessed': self._extract_root(ast['objekto']),
                'quantity': ast['objekto'].get('nombro'),
                'modifiers': self._extract_modifiers(ast['objekto']),
                'sentence_id': sentence_id,
                'sentence_rank': ast.get('retrieval_rank', 0)
            }
            facts.append(fact)

        # FACT TYPE 5: Temporal Modifiers
        for modifier in ast.get('aliaj', []):
            if modifier.get('type') == 'tempo':
                fact = {
                    'type': 'temporal',
                    'event': self._extract_root(ast['verbo']),
                    'time': modifier.get('valoro'),
                    'time_expression': modifier.get('teksto'),
                    'sentence_id': sentence_id,
                    'sentence_rank': ast.get('retrieval_rank', 0)
                }
                facts.append(fact)

        # FACT TYPE 6: Locative Modifiers
        for modifier in ast.get('aliaj', []):
            if modifier.get('type') == 'loko':
                fact = {
                    'type': 'locative',
                    'entity': self._extract_root(ast['subjekto']),
                    'location': modifier.get('valoro'),
                    'location_expression': modifier.get('teksto'),
                    'sentence_id': sentence_id,
                    'sentence_rank': ast.get('retrieval_rank', 0)
                }
                facts.append(fact)

        # FACT TYPE 7: Causal Relations ("ĉar" clauses)
        if self._has_causal_clause(ast):
            fact = {
                'type': 'cause',
                'effect': self._extract_main_clause(ast),
                'cause': self._extract_causal_clause(ast),
                'sentence_id': sentence_id,
                'sentence_rank': ast.get('retrieval_rank', 0)
            }
            facts.append(fact)

        # FACT TYPE 8: Purpose Relations ("por" clauses)
        if self._has_purpose_clause(ast):
            fact = {
                'type': 'purpose',
                'action': self._extract_main_clause(ast),
                'purpose': self._extract_purpose_clause(ast),
                'sentence_id': sentence_id,
                'sentence_rank': ast.get('retrieval_rank', 0)
            }
            facts.append(fact)

        return facts

    def _extract_root(self, node):
        """Extract root from AST node (deterministic)"""
        if not node:
            return None
        return node.get('radiko', node.get('kerno', {}).get('radiko'))

    def _extract_entity(self, node):
        """Extract entity name if it's a proper noun (deterministic)"""
        if not node:
            return None
        if node.get('proper_noun'):
            return node.get('vorto')  # Full word form for proper nouns
        return None

    def _extract_modifiers(self, node):
        """Extract all modifiers of a node (deterministic)"""
        if not node:
            return []
        return [self._extract_root(p) for p in node.get('priskriboj', [])]
```

### Example: Fact Extraction

**Input sentence**:
```
"Zamenhof, pola kuracisto el Bjalistoko, fondis Esperanton en 1887."
```

**AST** (simplified):
```python
{
    'subjekto': {
        'radiko': 'zamenhofo',
        'vorto': 'Zamenhof',
        'proper_noun': True,
        'apozicios': [
            {
                'radiko': 'kuracisto',
                'priskriboj': [{'radiko': 'pola'}]
            }
        ]
    },
    'verbo': {'radiko': 'fond', 'tempo': 'pasinto'},
    'objekto': {
        'radiko': 'esperanto',
        'vorto': 'Esperanton',
        'proper_noun': True
    },
    'aliaj': [
        {'type': 'loko', 'valoro': 'Bjalistoko', 'teksto': 'el Bjalistoko'},
        {'type': 'tempo', 'valoro': '1887', 'teksto': 'en 1887'}
    ]
}
```

**Extracted facts**:
```python
[
    {
        'type': 'predication',
        'subject': 'zamenhofo',
        'subject_entity': 'Zamenhof',
        'predicate': 'fond',
        'object': 'esperanto',
        'object_entity': 'Esperanton',
        'sentence_id': 's1',
        'sentence_rank': 1
    },
    {
        'type': 'property',
        'entity': 'kuracisto',
        'property': 'pola',
        'sentence_id': 's1',
        'sentence_rank': 1
    },
    {
        'type': 'property',  # From appositive
        'entity': 'zamenhofo',
        'property': 'kuracisto',
        'sentence_id': 's1',
        'sentence_rank': 1
    },
    {
        'type': 'locative',
        'entity': 'zamenhofo',
        'location': 'Bjalistoko',
        'location_expression': 'el Bjalistoko',
        'sentence_id': 's1',
        'sentence_rank': 1
    },
    {
        'type': 'temporal',
        'event': 'fond',
        'time': '1887',
        'time_expression': 'en 1887',
        'sentence_id': 's1',
        'sentence_rank': 1
    }
]
```

**5 atomic facts extracted from 1 sentence!**

---

## Step 3: Rank Fact Importance

### Deterministic Base Scoring (70% of quality)

```python
class DeterministicFactRanker:
    """
    Rank fact importance using deterministic features.
    No learned parameters.
    """

    def __init__(self, query_ast):
        self.query_ast = query_ast
        self.query_roots = extract_roots(query_ast)
        self.query_type = self._classify_query_type()

    def score_fact(self, fact, all_facts):
        """
        Score fact importance (0-1).
        100% deterministic.
        """
        score = 0.0

        # Factor 1: Query Root Overlap (25%)
        fact_roots = self._extract_fact_roots(fact)
        overlap = len(self.query_roots & fact_roots) / max(len(self.query_roots), 1)
        score += 0.25 * overlap

        # Factor 2: Answer Type Match (20%)
        if self._matches_answer_type(fact):
            score += 0.20

        # Factor 3: Fact Type Priority (15%)
        type_weights = {
            'predication': 1.0,   # Highest priority
            'property': 0.8,
            'possession': 0.7,
            'temporal': 0.6,
            'locative': 0.6,
            'cause': 0.9,
            'purpose': 0.9
        }
        score += 0.15 * type_weights.get(fact['type'], 0.5)

        # Factor 4: Sentence Rank (10%)
        # Facts from higher-ranked sentences are more important
        rank_score = 1.0 / (1.0 + fact.get('sentence_rank', 10))
        score += 0.10 * rank_score

        # Factor 5: Fact Centrality (15%)
        # How many other facts mention same entities?
        centrality = self._compute_centrality(fact, all_facts)
        score += 0.15 * centrality

        # Factor 6: Information Type (10%)
        # Definitions > Descriptions > Trivia
        if self._is_definition(fact):
            score += 0.10
        elif self._is_description(fact):
            score += 0.05

        # Factor 7: Redundancy Penalty (5%)
        # Penalize if information already covered by higher-ranked facts
        redundancy = self._compute_redundancy(fact, all_facts)
        score -= 0.05 * redundancy

        return min(max(score, 0.0), 1.0)

    def _classify_query_type(self):
        """Classify query from question word (deterministic)"""
        verbo_root = self.query_ast.get('verbo', {}).get('radiko', '')

        if verbo_root == 'kiu':
            return 'who'  # Expects person/entity
        elif verbo_root == 'kio':
            return 'what'  # Expects object/definition
        elif verbo_root == 'kiam':
            return 'when'  # Expects time
        elif verbo_root == 'kie':
            return 'where'  # Expects location
        elif verbo_root == 'kial':
            return 'why'  # Expects cause/purpose
        elif verbo_root == 'kiel':
            return 'how'  # Expects manner/process
        else:
            return 'general'

    def _matches_answer_type(self, fact):
        """Check if fact matches expected answer type (deterministic)"""

        if self.query_type == 'who':
            # Expect person entity in subject or object
            return (fact.get('subject_entity') or fact.get('object_entity'))

        elif self.query_type == 'what':
            # Expect definition or object
            return fact['type'] in ['predication', 'property']

        elif self.query_type == 'when':
            # Expect temporal fact
            return fact['type'] == 'temporal'

        elif self.query_type == 'where':
            # Expect locative fact
            return fact['type'] == 'locative'

        elif self.query_type == 'why':
            # Expect cause or purpose
            return fact['type'] in ['cause', 'purpose']

        return False

    def _compute_centrality(self, fact, all_facts):
        """
        How central is this fact? (how many other facts mention same entities?)
        Deterministic - just count shared entities.
        """
        fact_entities = set()
        if fact.get('subject'):
            fact_entities.add(fact['subject'])
        if fact.get('object'):
            fact_entities.add(fact['object'])
        if fact.get('entity'):
            fact_entities.add(fact['entity'])

        if not fact_entities:
            return 0.0

        # Count how many other facts share entities
        shared_count = 0
        for other_fact in all_facts:
            if other_fact == fact:
                continue

            other_entities = set()
            if other_fact.get('subject'):
                other_entities.add(other_fact['subject'])
            if other_fact.get('object'):
                other_entities.add(other_fact['object'])
            if other_fact.get('entity'):
                other_entities.add(other_fact['entity'])

            if fact_entities & other_entities:
                shared_count += 1

        # Normalize
        max_possible = len(all_facts) - 1
        return shared_count / max(max_possible, 1)

    def _is_definition(self, fact):
        """Is this a definition? (deterministic)"""
        return (fact['type'] == 'predication' and
                fact.get('predicate') == 'est')

    def _is_description(self, fact):
        """Is this a descriptive property? (deterministic)"""
        return fact['type'] in ['property', 'possession']

    def _compute_redundancy(self, fact, all_facts):
        """
        Is this fact redundant with higher-ranked facts?
        Deterministic root overlap check.
        """
        fact_roots = self._extract_fact_roots(fact)

        redundancy = 0.0
        for other_fact in all_facts:
            # Only check against higher-importance facts
            if other_fact.get('importance', 0) <= fact.get('importance', 0):
                continue

            other_roots = self._extract_fact_roots(other_fact)
            overlap = len(fact_roots & other_roots) / max(len(fact_roots), 1)

            if overlap > 0.7:  # High overlap = redundant
                redundancy = max(redundancy, overlap)

        return redundancy

    def _extract_fact_roots(self, fact):
        """Extract all roots mentioned in fact (deterministic)"""
        roots = set()
        for key in ['subject', 'predicate', 'object', 'entity', 'property', 'possessed']:
            if fact.get(key):
                roots.add(fact[key])
        return roots
```

### Optional: Semantic Enhancement Model (2M params, 30% quality boost)

**When deterministic scoring misses semantic nuance**:

```python
class SemanticFactRanker:
    """
    Optional: Add semantic scoring using embeddings.
    2M parameters.
    """

    def __init__(self, base_ranker, embedding_model):
        self.base_ranker = base_ranker
        self.embedding_model = embedding_model  # Existing 320K root embeddings
        self.adjustment_model = self._build_adjustment_model()  # 2M params

    def _build_adjustment_model(self):
        """
        Small model to adjust deterministic scores using semantic features.
        Input: Deterministic score + semantic features
        Output: Adjustment (-0.2 to +0.2)
        """
        return nn.Sequential(
            nn.Linear(10, 64),  # 10 features: det_score + 9 semantic features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output: -1 to +1, scale to -0.2 to +0.2
        )

    def score_fact_with_semantics(self, fact, all_facts):
        """
        Score fact using deterministic base + semantic adjustment.
        """
        # Get deterministic base score
        base_score = self.base_ranker.score_fact(fact, all_facts)

        # Extract semantic features
        semantic_features = self._extract_semantic_features(fact)

        # Combine into input tensor
        input_features = torch.tensor([base_score] + semantic_features)

        # Get adjustment from model
        adjustment = self.adjustment_model(input_features) * 0.2  # Scale to [-0.2, +0.2]

        # Final score
        final_score = base_score + adjustment.item()

        return min(max(final_score, 0.0), 1.0)

    def _extract_semantic_features(self, fact):
        """
        Extract semantic features using existing embeddings.
        0 new parameters (reuses 320K root embeddings).
        """
        features = []

        # Feature 1-3: Semantic similarity with query (using existing embeddings)
        fact_roots = self._extract_fact_roots(fact)
        query_roots = self.base_ranker.query_roots

        similarities = []
        for fact_root in fact_roots:
            for query_root in query_roots:
                fact_emb = self.embedding_model.get_embedding(fact_root)
                query_emb = self.embedding_model.get_embedding(query_root)
                sim = cosine_similarity(fact_emb, query_emb)
                similarities.append(sim)

        features.append(max(similarities) if similarities else 0.0)  # Max similarity
        features.append(np.mean(similarities) if similarities else 0.0)  # Avg similarity
        features.append(len(similarities))  # Number of related roots

        # Feature 4-6: Semantic coherence with other high-ranked facts
        high_ranked_facts = [f for f in all_facts if f.get('importance', 0) > 0.7]
        coherence_scores = []
        for other_fact in high_ranked_facts:
            if other_fact == fact:
                continue
            coherence = self._compute_semantic_coherence(fact, other_fact)
            coherence_scores.append(coherence)

        features.append(max(coherence_scores) if coherence_scores else 0.0)
        features.append(np.mean(coherence_scores) if coherence_scores else 0.0)
        features.append(len(coherence_scores))

        # Feature 7-9: Entity embedding centrality
        if fact.get('subject'):
            subj_emb = self.embedding_model.get_embedding(fact['subject'])
            # Measure how "central" this entity is in embedding space
            # (entities mentioned often have embeddings closer to centroid)
            features.append(np.linalg.norm(subj_emb))
        else:
            features.append(0.0)

        if fact.get('object'):
            obj_emb = self.embedding_model.get_embedding(fact['object'])
            features.append(np.linalg.norm(obj_emb))
        else:
            features.append(0.0)

        # Feature 9: Embedding-based answer type match
        features.append(self._semantic_answer_type_match(fact))

        return features[:9]  # Ensure exactly 9 features

    def _compute_semantic_coherence(self, fact1, fact2):
        """
        How semantically coherent are two facts? (using embeddings)
        """
        roots1 = self._extract_fact_roots(fact1)
        roots2 = self._extract_fact_roots(fact2)

        similarities = []
        for r1 in roots1:
            for r2 in roots2:
                emb1 = self.embedding_model.get_embedding(r1)
                emb2 = self.embedding_model.get_embedding(r2)
                sim = cosine_similarity(emb1, emb2)
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0
```

**Training data needed**: 5,000-8,000 (fact, query, gold_importance_score) examples

**When to use**:
- ✅ When deterministic scoring systematic errors (e.g., misses paraphrases)
- ✅ When semantic similarity matters (synonyms, related concepts)
- ❌ Not needed if deterministic scores are good enough

---

## Step 4: Cluster Facts by Topic (100% Deterministic)

```python
class FactClusterer:
    """
    Group facts by topic/theme.
    100% deterministic - uses shared entities and roots.
    """

    def cluster_facts(self, facts):
        """
        Group facts that should appear in same sentence/clause.
        Returns list of fact clusters.
        """

        clusters = []
        used_fact_ids = set()

        # Sort facts by importance (process most important first)
        sorted_facts = sorted(facts, key=lambda f: f.get('importance', 0), reverse=True)

        for fact in sorted_facts:
            if fact['id'] in used_fact_ids:
                continue

            # Start new cluster with this fact
            cluster = [fact]
            used_fact_ids.add(fact['id'])

            # Find related facts to add to cluster
            for other_fact in sorted_facts:
                if other_fact['id'] in used_fact_ids:
                    continue

                # Check if should be in same cluster
                if self._should_cluster_together(fact, other_fact, cluster):
                    cluster.append(other_fact)
                    used_fact_ids.add(other_fact['id'])

                # Limit cluster size (max 5 facts per cluster = 1 sentence)
                if len(cluster) >= 5:
                    break

            clusters.append(cluster)

        return clusters

    def _should_cluster_together(self, fact1, fact2, existing_cluster):
        """
        Decide if two facts should be in same cluster.
        Deterministic rules.
        """

        # Rule 1: Share main entity (subject or object)
        entities1 = self._extract_entities(fact1)
        entities2 = self._extract_entities(fact2)

        if entities1 & entities2:
            return True

        # Rule 2: One describes the other
        # E.g., fact1: (kato, estas, besto)
        #       fact2: (besto, property, malgranda)
        # → fact2 describes object of fact1
        if (fact1.get('object') == fact2.get('entity') or
            fact1.get('subject') == fact2.get('entity')):
            return True

        # Rule 3: Both properties of same entity
        if (fact1['type'] == 'property' and fact2['type'] == 'property' and
            fact1.get('entity') == fact2.get('entity')):
            return True

        # Rule 4: Temporal/locative modifiers of main predication in cluster
        for cluster_fact in existing_cluster:
            if cluster_fact['type'] == 'predication':
                if fact2['type'] in ['temporal', 'locative']:
                    # Check if fact2 modifies cluster_fact's predicate
                    if fact2.get('event') == cluster_fact.get('predicate'):
                        return True

        # Rule 5: Same sentence origin (likely related)
        if fact1.get('sentence_id') == fact2.get('sentence_id'):
            # But only if high importance (don't group trivia)
            if fact2.get('importance', 0) > 0.5:
                return True

        return False

    def _extract_entities(self, fact):
        """Extract all entities mentioned in fact (deterministic)"""
        entities = set()
        for key in ['subject', 'object', 'entity', 'possessor', 'possessed']:
            if fact.get(key):
                entities.add(fact[key])
        return entities
```

---

## Step 6: Synthesize Answer (100% Deterministic)

```python
class AnswerSynthesizer:
    """
    Construct NEW sentences from facts by building ASTs.
    100% deterministic - uses Esperanto grammar rules.
    """

    def synthesize(self, fact_clusters, target_length='short'):
        """
        Synthesize answer from fact clusters.

        target_length:
          'short': 1 sentence (for factoid queries)
          'medium': 2-3 sentences (for definitions)
          'long': 4-5 sentences (for "tell me about" queries)
        """

        sentences = []

        # Determine how many clusters to use
        if target_length == 'short':
            clusters_to_use = fact_clusters[:1]
        elif target_length == 'medium':
            clusters_to_use = fact_clusters[:2]
        else:  # long
            clusters_to_use = fact_clusters[:4]

        for cluster in clusters_to_use:
            # Construct AST from facts in cluster
            ast = self._construct_ast_from_facts(cluster)

            # Deparse AST to text
            sentence = deparse(ast)
            sentences.append(sentence)

        return sentences

    def _construct_ast_from_facts(self, facts):
        """
        Build NEW AST by synthesizing facts.
        This is the core innovation - deterministic AST construction!
        """

        # Step 1: Find main predication (highest priority)
        main_fact = self._find_main_predication(facts)

        if not main_fact:
            # No predication found, use highest-importance fact
            main_fact = max(facts, key=lambda f: f.get('importance', 0))

        # Step 2: Build base AST from main fact
        ast = self._build_base_ast(main_fact)

        # Step 3: Add properties to subject
        subject_properties = [f for f in facts
                             if f['type'] == 'property' and
                             f.get('entity') == main_fact.get('subject')]

        if subject_properties:
            ast['subjekto']['priskriboj'] = [
                {'radiko': f['property']} for f in subject_properties
            ]

        # Step 4: Add properties to object
        object_properties = [f for f in facts
                            if f['type'] == 'property' and
                            f.get('entity') == main_fact.get('object')]

        if object_properties:
            if 'objekto' not in ast:
                ast['objekto'] = {}
            ast['objekto']['priskriboj'] = [
                {'radiko': f['property']} for f in object_properties
            ]

        # Step 5: Add possessions as "kun" phrase
        possessions = [f for f in facts
                      if f['type'] == 'possession' and
                      f.get('possessor') == main_fact.get('subject')]

        if possessions:
            if 'objekto' not in ast:
                ast['objekto'] = {'radiko': None}  # Placeholder

            ast['objekto']['kun'] = []
            for poss in possessions:
                kun_item = {'radiko': poss['possessed']}
                if poss.get('quantity'):
                    kun_item['nombro'] = poss['quantity']
                if poss.get('modifiers'):
                    kun_item['priskriboj'] = [{'radiko': m} for m in poss['modifiers']]
                ast['objekto']['kun'].append(kun_item)

        # Step 6: Add temporal modifiers
        temporal_facts = [f for f in facts if f['type'] == 'temporal']
        if temporal_facts:
            ast['aliaj'] = ast.get('aliaj', [])
            for temp in temporal_facts:
                ast['aliaj'].append({
                    'type': 'tempo',
                    'teksto': temp.get('time_expression', temp['time'])
                })

        # Step 7: Add locative modifiers
        locative_facts = [f for f in facts if f['type'] == 'locative']
        if locative_facts:
            ast['aliaj'] = ast.get('aliaj', [])
            for loc in locative_facts:
                ast['aliaj'].append({
                    'type': 'loko',
                    'teksto': loc.get('location_expression', loc['location'])
                })

        return ast

    def _find_main_predication(self, facts):
        """
        Find most important predication fact to use as main clause.
        Prefer "estas" (definition) predications.
        """
        # First priority: "estas" predications
        estas_facts = [f for f in facts
                      if f['type'] == 'predication' and f.get('predicate') == 'est']

        if estas_facts:
            return max(estas_facts, key=lambda f: f.get('importance', 0))

        # Second priority: any predication
        predications = [f for f in facts if f['type'] == 'predication']

        if predications:
            return max(predications, key=lambda f: f.get('importance', 0))

        return None

    def _build_base_ast(self, fact):
        """Build base AST structure from main fact"""

        if fact['type'] == 'predication':
            return {
                'subjekto': {'radiko': fact['subject']},
                'verbo': {'radiko': fact['predicate']},
                'objekto': {'radiko': fact.get('object')} if fact.get('object') else None
            }

        elif fact['type'] == 'property':
            # Convert property to predication: "X estas Y"
            return {
                'subjekto': {'radiko': fact['entity']},
                'verbo': {'radiko': 'est'},
                'objekto': {'radiko': fact['property']}
            }

        else:
            # Generic structure
            return {
                'subjekto': {'radiko': fact.get('subject', fact.get('entity'))}
            }
```

---

## Complete Example: "Kiu fondis Esperanton?"

### Input (Top 10 retrieved sentences)
```
1. "Zamenhof fondis Esperanton." (rank: 1)
2. "Li estis pola kuracisto." (rank: 2)
3. "La lingvo estis kreita en 1887." (rank: 3)
4. "Ludoviko Lazaro Zamenhof naskigxis en Bjalistoko." (rank: 4)
5. "Zamenhof vivis en Pollando." (rank: 5)
6. "La kreinto parolis la rusan kaj polan lingvojn." (rank: 6)
7. "Esperanto estas planlingvo." (rank: 7)
8. "Zamenhof havis celon pri mondpaco." (rank: 8)
9. "Li estis okula kuracisto." (rank: 9)
10. "La lingvo havas regulan gramatikon." (rank: 10)
```

### Step 2: Extract Facts
```python
[
    {'type': 'predication', 'subject': 'zamenhofo', 'predicate': 'fond', 'object': 'esperanto', 'rank': 1},
    {'type': 'property', 'entity': 'zamenhofo', 'property': 'kuracisto', 'rank': 2},
    {'type': 'property', 'entity': 'kuracisto', 'property': 'pola', 'rank': 2},
    {'type': 'predication', 'subject': 'esperanto', 'predicate': 'kre', 'rank': 3},
    {'type': 'temporal', 'event': 'kre', 'time': '1887', 'rank': 3},
    {'type': 'locative', 'entity': 'zamenhofo', 'location': 'Bjalistoko', 'rank': 4},
    {'type': 'predication', 'subject': 'zamenhofo', 'predicate': 'naskigx', 'rank': 4},
    {'type': 'locative', 'entity': 'zamenhofo', 'location': 'Pollando', 'rank': 5},
    # ... more facts
]
```

### Step 3: Rank Facts
```python
# Deterministic scoring
facts_with_scores = [
    (fact1, 0.95),  # (Zamenhof, fondis, Esperanton) - answers "kiu", high rank
    (fact2, 0.82),  # (Zamenhof, property, kuracisto) - describes subject
    (fact5, 0.78),  # (fondis, tempo, 1887) - temporal context
    (fact3, 0.72),  # (kuracisto, property, pola) - additional detail
    (fact6, 0.68),  # (Zamenhof, naskigxis_en, Bjalistoko) - biographical
    # ... lower scores
]
```

### Step 4: Cluster Facts
```python
clusters = [
    [  # Cluster 1: Founding event + founder description
        (fact1, 0.95),  # fondis
        (fact2, 0.82),  # kuracisto
        (fact3, 0.72),  # pola
        (fact5, 0.78),  # en 1887
    ],
    [  # Cluster 2: Biographical details
        (fact6, 0.68),  # naskigxis en Bjalistoko
        (fact8, 0.65),  # vivis en Pollando
    ]
]
```

### Step 6: Synthesize
```python
# Target: 'short' (1 sentence for factoid query)

# Use Cluster 1 only
cluster = clusters[0]

# Construct AST:
ast = {
    'subjekto': {
        'radiko': 'zamenhofo',
        'priskriboj': [
            {'radiko': 'pola'},
            {'radiko': 'kuracisto'}
        ]
    },
    'verbo': {'radiko': 'fond'},
    'objekto': {'radiko': 'esperanto'},
    'aliaj': [
        {'type': 'tempo', 'teksto': 'en 1887'}
    ]
}

# Deparse:
sentence = deparse(ast)
```

### Output
```
"Zamenhof, pola kuracisto, fondis Esperanton en 1887."
```

**This is BETTER than top sentence alone** ("Zamenhof fondis Esperanton.") - adds useful context from multiple sources!

---

## Summary: Deterministic vs Learned

| Component | Deterministic? | Learned Params | Notes |
|-----------|---------------|----------------|-------|
| **Fact extraction** | 100% ✅ | 0 | From AST structure |
| **Base importance scoring** | 100% ✅ | 0 | Query overlap, answer type, centrality |
| **Semantic importance (optional)** | 0% | 2M | Adjustment model using embeddings |
| **Fact clustering** | 100% ✅ | 0 | Shared entities, roots |
| **Fact selection** | 100% ✅ | 0 | Threshold-based |
| **AST construction** | 100% ✅ | 0 | Grammar rules |
| **Deparsing** | 100% ✅ | 0 | Existing |
| **Total** | **~95%** | **0-2M** | Almost fully deterministic! |

## Recommendation

**Phase 1** (Weeks 1-2): Implement fully deterministic pipeline
- Extract facts from ASTs
- Rank using deterministic scoring
- Cluster using shared entities
- Synthesize using AST construction
- Test on 50-100 queries

**Target**: 85-90% quality with 0 learned parameters

**Phase 2** (Week 3): Evaluate
- Where does deterministic fail?
- Are failures systematic?

**Phase 3** (Week 4, optional): Add semantic model
- Train 2M adjustment model
- Only if deterministic has systematic gaps
- Measure improvement (target: +5-10% quality)

**This maximizes deterministic processing while leaving room for semantic enhancement if needed!**
