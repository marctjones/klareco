# Use Case Clarification: When Do We Need Fact-Based Synthesis?

## Your Question: "Kiu fondis Esperanton?" (Who created Esperanto?)

### RAG Approach (Simpler)
```
Query: "Kiu fondis Esperanton?"
↓
RAG Retrieval (embeddings + reranker)
↓
Top result: "Zamenhof fondis Esperanton en 1887." (score: 0.95)
↓
Output: "Zamenhof fondis Esperanton en 1887."
```

**Result**: ✅ Perfect answer in one sentence! Done!

### Kuzu-First Approach (Overkill?)
```
Query: "Kiu fondis Esperanton?"
↓
Query Kuzu for facts
↓
MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "esperanto"})
WHERE f.predicate IN ['fond', 'kre']
RETURN f.subject, f.object
↓
Results: (Zamenhof, fondis, Esperanto, en_1887)
↓
Synthesize back into sentence: "Zamenhof fondis Esperanton en 1887."
```

**Result**: ✅ Same answer... but with more steps?

**You're right - for this query type, RAG is sufficient!**

## When Does Fact-Based Synthesis Actually Help?

### Use Case 1: Summarizing Multiple Redundant Sentences ✓

**Scenario**: RAG returns 10 sentences with overlapping information

**Input from reranker**:
```
1. "Kato estas besto." (0.92)
2. "Katoj estas malgranda hejmaj bestoj." (0.90)
3. "La kato havas kvar piedojn." (0.88)
4. "Katoj havas voston." (0.85)
5. "Ili havas longan voston." (0.83)
6. "Katoj estas karnovoruloj." (0.82)
7. "Ili mangxas viandon." (0.80)
8. "Katoj cxasas musojn." (0.78)
9. "Ili ankaux cxasas birdojn." (0.76)
10. "Katoj dormas gxis 16 horojn tage." (0.74)
```

**Problem**: These are redundant and choppy. User wants a coherent paragraph, not 10 separate sentences.

**Task**: Summarize into 2-3 sentences

**Fact-based approach helps here**:
- Extract facts from all 10 sentences
- Identify redundancy (multiple sentences say "kato estas besto")
- Cluster related facts (physical features, behavior, diet)
- Synthesize NEW sentences combining information:

**Output**:
```
"Kato estas malgranda hejma besto kun kvar piedoj kaj longa vosto.
Ili estas karnovoruloj, kiuj cxasas musojn kaj birdojn kaj mangxas viandon.
Katoj dormas gxis 16 horojn tage."
```

**This is better than returning all 10 sentences!**

### Use Case 2: "Tell Me About X" (Biographical/Topical Summary) ✓

**Query**: "Rakontu al mi pri Zamenhof" (Tell me about Zamenhof)

**RAG returns 15 sentences**:
```
1. "Zamenhof naskigxis en Bjalistoko."
2. "Li estis pola kuracisto."
3. "Zamenhof fondis Esperanton en 1887."
4. "Li parolis la rusan, polan, kaj germanan lingvojn."
5. "Zamenhof studis medicinon en Moskvo."
6. "Li vivis en Varsovio post 1897."
7. "Zamenhof havis celon pri mondpaco."
8. "Li uzis la pseuximonon Doktoro Esperanto."
... (7 more sentences)
```

**Problem**: 15 disconnected sentences. User wants a coherent biographical paragraph.

**Fact-based synthesis helps**:
- Cluster facts by topic: Early life, Education, Esperanto creation, Later life, Goals
- Synthesize into 3-4 coherent sentences per topic

**Output**:
```
"Ludoviko Lazaro Zamenhof naskigxis en Bjalistoko, Pollando, kaj estis pola kuracisto.
Li studis medicinon en Moskvo kaj parolis la rusan, polan, kaj germanan lingvojn.
En 1887, Zamenhof fondis Esperanton sub la pseuximono 'Doktoro Esperanto', kun celo pri internacia komunikado kaj mondpaco.
Li vivis en Varsovio ekde 1897 kaj laboris kiel kuracisto dum sia vivo."
```

**Much better than 15 choppy sentences!**

### Use Case 3: Enrichment (RAG + Kuzu) ✓

**Query**: "Kiu fondis Esperanton?"

**RAG approach**:
```
→ "Zamenhof fondis Esperanton en 1887."
```

**But user asks follow-up**: "Diru pli" (Tell me more)

**Now query Kuzu for related facts about Zamenhof**:
```cypher
MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "Zamenhof"})
WHERE f.type IN ['property', 'action', 'biographical']
RETURN f
ORDER BY f.importance DESC
LIMIT 10
```

**Get additional facts**:
- (Zamenhof, estas, kuracisto)
- (Zamenhof, naskigxis_en, Bjalistoko)
- (Zamenhof, parolis, rusaj/polaj/germanaj lingvoj)

**Synthesize enriched answer**:
```
"Zamenhof fondis Esperanton en 1887. Li estis pola kuracisto el Bjalistoko, kiu parolis plurajn lingvojn."
```

**This adds context beyond the initial sentence!**

## When RAG Alone Is Sufficient

### Simple Factoid Questions ✓

**Questions with direct answers in single sentence**:
- "Kiu fondis Esperanton?" → "Zamenhof fondis Esperanton en 1887."
- "Kiam Esperanto estis kreita?" → "Esperanto estis kreita en 1887."
- "Kie naskigxis Zamenhof?" → "Zamenhof naskigxis en Bjalistoko."

**For these**: Just return top reranked sentence! Don't overthink it!

### Definition Questions with Good Coverage ✓

**Query**: "Kio estas kato?"

**If top sentence is comprehensive**:
```
"Kato estas malgranda hejma besto, kiu estas karnovora kaj cxasas musojn."
```

**Then just return it!** No need for synthesis.

**But if top sentences are fragmented**:
```
1. "Kato estas besto."
2. "Katoj estas malgranda."
3. "Ili havas kvar piedojn."
```

**Then synthesis helps** to combine into coherent answer.

## The Real Architecture

```
┌──────────────────────────────────────────────────┐
│ Query Analysis: What type of query?              │
└──────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────────┐   ┌──────────────────────┐
│ Simple Factoid   │   │ Summary/Biographical │
│ (Kiu/Kio/Kiam)  │   │ (Rakontu/Tell me)   │
└──────────────────┘   └──────────────────────┘
        ↓                       ↓
┌──────────────────┐   ┌──────────────────────┐
│ RAG Retrieval    │   │ RAG Retrieval        │
│ + Reranker       │   │ + Reranker           │
└──────────────────┘   └──────────────────────┘
        ↓                       ↓
┌──────────────────┐   ┌──────────────────────┐
│ Return top 1-3   │   │ Extract Facts        │
│ sentences        │   │ + Synthesize         │
└──────────────────┘   └──────────────────────┘
```

## Revised Recommendation

### For Your Example: "Kiu fondis Esperanton?"

**Answer**: RAG is perfect! Just return top reranked sentence.

```python
def answer_query(query):
    # Classify query type (deterministic from question word)
    query_type = classify_query_type(query)  # "kiu" → factoid

    # RAG retrieval
    sentences = rag_retrieval(query, top_k=20)

    if query_type == "factoid":
        # Simple: return top sentence(s)
        return sentences[0:1]  # Just top sentence!

    elif query_type == "summary" or query_type == "biographical":
        # Complex: synthesize from multiple sentences
        facts = extract_facts(sentences)
        clustered = cluster_facts(facts)
        synthesized = synthesize_sentences(clustered)
        return synthesized

    elif query_type == "definition":
        # Check if top sentence is comprehensive
        if is_comprehensive(sentences[0]):
            return [sentences[0]]
        else:
            # Synthesize from multiple sentences
            facts = extract_facts(sentences[0:5])
            synthesized = synthesize_sentences(facts)
            return synthesized
```

### When to Use Fact-Based Synthesis

✅ **YES** for:
- Summarization tasks ("Resumo de..." "Rakontu pri...")
- Biographical queries ("Tell me about X")
- When top sentences are fragmentary
- Comparison queries ("Compare X and Y")

❌ **NO** for:
- Simple factoid questions (Kiu/Kio/Kiam/Kie)
- When top sentence is already perfect answer
- When user wants specific quick answer

### When to Use Kuzu Direct Queries

✅ **YES** for:
- Structured queries ("How many X?", "List all Y")
- Aggregation ("What languages did X speak?")
- When need to compute over data
- Enrichment (add related facts)

❌ **NO** for:
- Simple retrieval (RAG already works)
- When answer exists in single sentence

## Bottom Line

**You're absolutely right!** For "Kiu fondis Esperanton?":
- ✅ RAG is sufficient (returns perfect answer)
- ❌ Fact extraction is overkill
- ❌ Kuzu-first is unnecessary

**Fact-based synthesis is only needed when**:
1. Summarizing multiple redundant sentences into coherent paragraph
2. Biographical/"tell me about" queries needing synthesis
3. Fragmentary answers needing combination

**For most factoid Q&A**: Reranker already gives the perfect answer! Don't overthink it!

## What We Should Build

**Priority 1**: Simple RAG pipeline
- Root embeddings (✓ exists)
- M1 selectional preferences (✓ exists)
- Reranker (needs training)
- Return top sentences

**Priority 2**: Add summarization when needed
- Detect query type (factoid vs summary)
- Extract facts only for summary queries
- Synthesize only when needed

**Priority 3**: Kuzu enrichment (optional)
- Add related facts from knowledge graph
- For follow-up queries
- For structured queries

**Don't build complex fact extraction for simple Q&A!**
