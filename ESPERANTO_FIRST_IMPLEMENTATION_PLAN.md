# Esperanto-First AI: Implementation Plan

**Vision**: Build a general-purpose AI that maximizes deterministic AST-based reasoning over learned weights, proving that explicit linguistic structure + small neural components can compete with massive LLMs.

**Philosophy**: The AST is the "consciousness" - all reasoning happens at the symbolic level in Esperanto. Neural components are minimal and only used where structure can't solve the problem.

---

## Core Hypothesis

**Traditional LLM Architecture**:
```
Raw Text → Tokenizer → Embeddings (50k+ vocab) → Transformer Layers (12-48)
         → Attention (learned dependencies) → Output Projection (50k vocab)
         = 110M - 175B parameters
```

**Klareco Architecture**:
```
Raw Text → Front Door (lang detect + translate) → Esperanto Text
        → Parser (deterministic, 0 params) → Rich AST (grammar + semantics)
        → Structural Filter (deterministic, 0 params) → Candidate sentences
        → Neural Reranker (15M params) → Best matches
        → AST-based Reasoner (deterministic + 5M params) → Response AST
        → Deparser (deterministic, 0 params) → Esperanto Text
        → Translation (MarianMT, 50M params) → Output
        = ~70M total parameters (vs 110M+ for traditional)
```

**Key Advantages**:
- **Explicit roles**: Subject/object/verb extracted deterministically, not learned
- **Compositional semantics**: Morpheme-level understanding, not token-level
- **Zero-shot structural reasoning**: AST operations handle many tasks without training
- **Interpretable**: Can inspect every decision in the AST trail
- **Data efficient**: Don't need to learn what grammar gives us for free

---

## System Architecture

### Current State (Phase 4 - Complete)

```
┌─────────────────────────────────────────────────────────────┐
│                    KLARECO PIPELINE v1                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input → Front Door → Parser → Gating → Orchestrator       │
│                          ↓                    ↓             │
│                         AST              Extractive         │
│                          ↓                  Expert          │
│                     Retriever                ↓              │
│                          ↓                Response          │
│                   [Structural              (text)           │
│                    + Neural]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Components:
✅ Parser: Morpheme-aware, 16 Esperanto rules
✅ Deparser: AST → text reconstruction
✅ Front Door: Language ID + translation
✅ Gating Network: Symbolic intent classification
✅ Orchestrator: Routes to experts
✅ Extractive Expert: Returns best retrieved sentence
✅ Retriever: Two-stage (structural + neural)
✅ Corpus v2: 35k high-quality sentences
✅ Index v3: Complete with AST metadata
```

### Target State (Phase 5-7)

```
┌───────────────────────────────────────────────────────────────────┐
│                    KLARECO PIPELINE v2                            │
│                  (AST-as-Consciousness)                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input → Front Door → Parser → Enricher → Reasoner → Deparser   │
│             ↓           ↓         ↓          ↓           ↓        │
│          Language    Grammar   Semantics  Inference  Esperanto   │
│           AST         AST       AST         AST        Text      │
│             ↓                                ↓                    │
│         Translate ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Output               │
│                                                                   │
│  Every module enriches the AST with metadata (all in Esperanto)  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

New Components:
⬜ Enricher: Adds semantic roles, animacy, coreference to AST
⬜ Reasoner: AST-to-AST transformations for inference
⬜ AST Trail: Full processing history in Esperanto terms
⬜ Proper Noun Dictionary: Dynamic learning + static list
⬜ Semantic Retrieval: Filters by semantic roles, not just structure
⬜ AST Templates: Common reasoning patterns as AST transforms
```

---

## Implementation Phases

### Phase 5: AST Enrichment (Foundation)
**Goal**: Add semantic and discourse layers to AST without breaking existing code

**Status**: Not started
**Duration**: 2-3 sessions
**Impact**: High - enables all future reasoning

#### 5.1: Proper Noun Dictionary (Week 1)

**Problem**: Parse rates artificially low because proper nouns marked as "failed"

**Solution**: Two-tier dictionary system

**Implementation**:

```python
# klareco/proper_nouns.py (NEW)

class ProperNounDictionary:
    """
    Maintains known proper nouns with metadata.

    Two sources:
    1. Static: Hand-curated common names (people, places)
    2. Dynamic: Learned from corpus (frequency-based)
    """

    def __init__(self, static_path: Path, dynamic_path: Path):
        self.static = self._load_static(static_path)
        self.dynamic = self._load_dynamic(dynamic_path)
        self.cache = {}  # In-memory cache for current session

    def is_proper_noun(self, word: str) -> bool:
        """Check if word is a known proper noun."""
        base = self._strip_esperanto_endings(word)
        return base in self.static or base in self.dynamic

    def get_metadata(self, word: str) -> dict:
        """Get proper noun metadata (category, frequency, source)."""
        base = self._strip_esperanto_endings(word)
        return self.static.get(base) or self.dynamic.get(base) or {}

    def add_to_session(self, word: str, category: str = "unknown"):
        """Add proper noun discovered in current session."""
        base = self._strip_esperanto_endings(word)
        self.cache[base] = {"category": category, "temporary": True}

    def _strip_esperanto_endings(self, word: str) -> str:
        """Remove -o, -on, -oj, -ojn endings."""
        if word.endswith(('ojn', 'oj', 'on', 'o')):
            # ... strip logic
        return word
```

**Static Dictionary** (`data/proper_nouns_static.json`):
```json
{
  "Frodo": {"category": "person", "source": "tolkien", "gender": "male"},
  "Bilbo": {"category": "person", "source": "tolkien", "gender": "male"},
  "Gandalf": {"category": "person", "source": "tolkien", "gender": "male"},
  "Saŭrono": {"category": "person", "source": "tolkien", "gender": "male"},
  "Ŝajro": {"category": "place", "source": "tolkien"},
  "Miterreno": {"category": "place", "source": "tolkien"},
  "Gondoro": {"category": "place", "source": "tolkien"},
  "Mordoro": {"category": "place", "source": "tolkien"},

  "Usono": {"category": "country", "source": "world"},
  "Ĉinio": {"category": "country", "source": "world"},
  "Parizo": {"category": "city", "source": "world"},
  "Londono": {"category": "city", "source": "world"},

  "Zamenhof": {"category": "person", "source": "esperanto", "gender": "male"},
  "Esperanto": {"category": "language", "source": "esperanto"}
}
```

**Dynamic Dictionary Builder** (`scripts/build_proper_noun_dict.py` - NEW):
```python
# Extract proper nouns from corpus
# Algorithm:
# 1. Find all capitalized words (excluding sentence-initial)
# 2. Count frequency
# 3. Check if has Esperanto endings (-o, -on, -oj, -ojn)
# 4. If frequency > 3, add to dictionary
# 5. Try to infer category from context

def extract_proper_nouns(corpus_path: Path) -> dict:
    proper_nouns = {}

    for entry in read_corpus(corpus_path):
        ast = entry['ast']
        # Find words marked as propra_nomo
        for word_ast in find_all_words(ast):
            if word_ast.get('vortspeco') == 'propra_nomo':
                base = strip_endings(word_ast['plena_vorto'])
                proper_nouns[base] = proper_nouns.get(base, 0) + 1

    # Filter by frequency and add metadata
    result = {}
    for word, freq in proper_nouns.items():
        if freq >= 3:  # Appears at least 3 times
            result[word] = {
                "frequency": freq,
                "category": infer_category(word),  # person/place/other
                "source": "corpus"
            }

    return result
```

**Parser Integration**:
```python
# In parser.py, modify categorize_unknown_word()

def categorize_unknown_word(word: str, error_msg: str = "") -> dict:
    # ... existing code ...

    # NEW: Check proper noun dictionary BEFORE categorizing
    if word[0].isupper():
        proper_noun_dict = get_proper_noun_dictionary()  # Singleton
        if proper_noun_dict.is_proper_noun(word):
            ast["parse_status"] = "success"  # ✨ Key change!
            ast["category"] = "proper_name_known"
            ast["metadata"] = proper_noun_dict.get_metadata(word)
            return ast

    # ... rest of existing logic ...
```

**Impact**:
- Parse rates increase from ~91% to ~95%+
- Proper nouns no longer filtered as "failed"
- Can distinguish known characters (Frodo) from typos (Frodoo)

**Testing**:
```python
# tests/test_proper_nouns.py
def test_proper_noun_recognition():
    ast = parse("Frodo kaj Bilbo loĝas en la Ŝajro.")
    stats = ast['parse_statistics']
    assert stats['success_rate'] >= 0.9  # Should be high now!

def test_proper_noun_endings():
    ast = parse("Mi vidas Frodon.")  # Accusative
    frodo = find_word(ast, "Frodon")
    assert frodo['parse_status'] == 'success'
    assert frodo['kazo'] == 'akuzativo'
```

#### 5.2: Semantic AST Enricher (Week 2)

**Goal**: Add semantic roles (agent, patient, instrument) to AST

**Why**: Enables semantic retrieval and reasoning beyond syntax

**Implementation**:

```python
# klareco/enricher.py (NEW)

class ASTEnricher:
    """
    Enriches grammatical AST with semantic and discourse information.

    Adds:
    - Semantic roles (aganto, paciento, instrumento, etc.)
    - Animacy (vivanta, nevivanta)
    - Definiteness (difina, nedifina)
    - Coreference IDs (links pronouns to referents)
    - Discourse relations (temo, fokuso, nova_informo)
    """

    def __init__(self, proper_noun_dict: ProperNounDictionary):
        self.proper_nouns = proper_noun_dict
        self.animacy_db = self._load_animacy_database()

    def enrich(self, ast: dict, discourse_context: list = None) -> dict:
        """
        Add semantic layers to grammatical AST.

        Args:
            ast: Grammatical AST from parser
            discourse_context: Previous sentences' ASTs for coreference

        Returns:
            Enriched AST with semantic and discourse fields
        """
        enriched = ast.copy()

        # Add semantic roles based on grammar
        enriched = self._add_semantic_roles(enriched)

        # Add animacy information
        enriched = self._add_animacy(enriched)

        # Resolve coreferences (pronouns → referents)
        if discourse_context:
            enriched = self._resolve_coreference(enriched, discourse_context)

        # Add discourse structure
        enriched = self._add_discourse_info(enriched, discourse_context)

        return enriched

    def _add_semantic_roles(self, ast: dict) -> dict:
        """
        Map grammatical roles to semantic roles.

        Rules (deterministic!):
        - Nominative subject + active verb → aganto (agent)
        - Accusative object + active verb → paciento (patient)
        - "per" prepositional phrase → instrumento (instrument)
        - "al" prepositional phrase → riceviĝanto (recipient)
        - Passive subject → paciento (patient)
        """
        subjekto = ast.get('subjekto')
        verbo = ast.get('verbo')
        objekto = ast.get('objekto')

        # Subject role depends on voice
        if subjekto:
            if self._is_passive_voice(verbo):
                subjekto['semantika_rolo'] = 'paciento'
            else:
                subjekto['semantika_rolo'] = 'aganto'

        # Object in active voice is patient
        if objekto and not self._is_passive_voice(verbo):
            objekto['semantika_rolo'] = 'paciento'

        # Scan for prepositional phrases
        for word in ast.get('aliaj', []):
            if word.get('vortspeco') == 'prepozicio':
                prep = word.get('radiko')
                # Next noun gets semantic role based on preposition
                if prep == 'per':
                    # "per martelo" → instrument
                    # Mark next substantive as instrumento
                    ...
                elif prep == 'al':
                    # "al Johano" → recipient
                    ...

        return ast

    def _add_animacy(self, ast: dict) -> dict:
        """
        Add animacy to nouns (vivanta vs nevivanta).

        Uses:
        - Animacy database (common words)
        - Proper noun metadata (persons are animate)
        - Heuristics (pronouns li/ŝi/oni are animate)
        """
        # ... implementation

    def _resolve_coreference(self, ast: dict, context: list) -> dict:
        """
        Link pronouns to their referents in context.

        Rules:
        - "li" (he) → masculine proper noun in context
        - "ŝi" (she) → feminine proper noun in context
        - "ĝi" (it) → neuter noun in context
        - "ili" (they) → plural noun in context
        """
        # ... implementation

    def _add_discourse_info(self, ast: dict, context: list) -> dict:
        """
        Add discourse-level structure.

        Identifies:
        - frazo_tipo: deklaro, demando, ordono, ekkrio
        - temo: What the sentence is about (often subject)
        - fokuso: New or emphasized information
        - rilato_al_antaŭa: Relation to previous sentence
        """
        ast['diskursa_analizo'] = {
            'frazo_tipo': self._classify_sentence_type(ast),
            'temo': self._identify_topic(ast),
            'fokuso': self._identify_focus(ast),
            'rilato_al_antaŭa': self._find_relation(ast, context) if context else None
        }
        return ast
```

**Animacy Database** (`data/animacy.json`):
```json
{
  "hund": "vivanta",
  "kat": "vivanta",
  "homo": "vivanta",
  "koro": "nevivanta",
  "libro": "nevivanta",
  "arbo": "vivanta",
  "ŝtono": "nevivanta",
  ...
}
```

**Example Enriched AST**:
```json
{
  "tipo": "frazo",
  "enhavo": "La hundo vidas la katon.",

  "subjekto": {
    "radiko": "hund",
    "vortspeco": "substantivo",
    "kazo": "nominativo",
    "nombro": "singularo",

    // NEW semantic fields
    "semantika_rolo": "aganto",
    "animateco": "vivanta",
    "difineco": "difina",
    "koreferenco_id": null
  },

  "verbo": {
    "radiko": "vid",
    "vortspeco": "verbo",
    "tempo": "prezenco",

    // NEW semantic fields
    "evento_tipo": "percepto",
    "aspekto": "durativa",
    "kaŭzeco": "rekta"
  },

  "objekto": {
    "radiko": "kat",
    "vortspeco": "substantivo",
    "kazo": "akuzativo",

    // NEW semantic fields
    "semantika_rolo": "paciento",
    "animateco": "vivanta",
    "difineco": "difina"
  },

  // NEW discourse-level metadata
  "diskursa_analizo": {
    "frazo_tipo": "deklaro",
    "temo": "hundo",
    "fokuso": "vidas_katon",
    "rilato_al_antaŭa": null
  }
}
```

**Testing**:
```python
def test_semantic_role_assignment():
    enricher = ASTEnricher(proper_noun_dict)
    ast = parse("La hundo vidas la katon.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['semantika_rolo'] == 'aganto'
    assert enriched['objekto']['semantika_rolo'] == 'paciento'

def test_animacy():
    enricher = ASTEnricher(proper_noun_dict)
    ast = parse("La libro kuŝas sur la tablo.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['animateco'] == 'nevivanta'
```

#### 5.3: AST Trail System (Week 2)

**Goal**: Track full processing history in AST

**Why**: Makes system interpretable, debuggable, explainable

**Implementation**:

```python
# klareco/ast_trail.py (NEW)

class ASTTrail:
    """
    Maintains full processing history of an AST.

    Each module adds its metadata:
    - Parser: Grammar annotations
    - Enricher: Semantic annotations
    - Retriever: Search results and scores
    - Reasoner: Inference steps
    - Generator: Response decisions

    All metadata in Esperanto!
    """

    def __init__(self, initial_ast: dict, original_text: str):
        self.stages = []
        self.current_ast = initial_ast
        self.original_text = original_text
        self.add_stage("parsado", initial_ast, {
            "originala_teksto": original_text
        })

    def add_stage(self, stage_name: str, updated_ast: dict, metadata: dict):
        """
        Record a processing stage.

        Args:
            stage_name: Name in Esperanto (parsado, enriĉigo, serĉado, etc.)
            updated_ast: AST after this stage
            metadata: Stage-specific info (also in Esperanto)
        """
        self.stages.append({
            "stadio": stage_name,
            "tempo": datetime.now().isoformat(),
            "ast": updated_ast,
            "metadatumoj": metadata
        })
        self.current_ast = updated_ast

    def get_ast(self) -> dict:
        """Get current AST."""
        return self.current_ast

    def get_history(self) -> list:
        """Get full processing history."""
        return self.stages

    def export(self) -> dict:
        """
        Export full trail as JSON.

        Useful for:
        - Debugging
        - Explanation generation
        - Model training (supervised learning from trails)
        """
        return {
            "originala_teksto": self.original_text,
            "procesaj_stadioj": self.stages,
            "fina_ast": self.current_ast
        }
```

**Usage in Pipeline**:
```python
# In orchestrator or main pipeline

def process_query(text: str) -> dict:
    # 1. Parse
    ast = parse(text)
    trail = ASTTrail(ast, text)

    # 2. Enrich
    enricher = ASTEnricher(proper_noun_dict)
    enriched_ast = enricher.enrich(ast)
    trail.add_stage("enriĉigo", enriched_ast, {
        "aldonitaj_kampoj": ["semantika_rolo", "animateco", "diskursa_analizo"]
    })

    # 3. Retrieve
    retriever = KlarecoRetriever(...)
    results = retriever.retrieve(text)
    trail.add_stage("serĉado", enriched_ast, {
        "retrovitaj_frazoj": len(results),
        "plej_bona_poento": results[0]['score'],
        "fontoj": [r['source'] for r in results]
    })

    # 4. Reason
    reasoner = ASTReasoner()
    response_ast = reasoner.generate_response(enriched_ast, results)
    trail.add_stage("rezonado", response_ast, {
        "rezona_tipo": "ekstrakta",
        "elektita_frazo": response_ast['enhavo']
    })

    # 5. Generate
    response_text = deparse(response_ast)
    trail.add_stage("generado", response_ast, {
        "eligita_teksto": response_text
    })

    return {
        "response": response_text,
        "trail": trail.export(),
        "confidence": calculate_confidence(trail)
    }
```

**Example Trail Output**:
```json
{
  "originala_teksto": "Kie estas Frodo?",
  "procesaj_stadioj": [
    {
      "stadio": "parsado",
      "tempo": "2025-11-27T10:30:00",
      "ast": { ... },
      "metadatumoj": {
        "parse_statistics": {
          "success_rate": 1.0,
          "esperanto_words": 3
        }
      }
    },
    {
      "stadio": "enriĉigo",
      "tempo": "2025-11-27T10:30:01",
      "ast": { ... },
      "metadatumoj": {
        "aldonitaj_kampoj": ["semantika_rolo", "animateco"],
        "frazo_tipo": "demando",
        "demando_tipo": "loko"
      }
    },
    {
      "stadio": "serĉado",
      "tempo": "2025-11-27T10:30:02",
      "ast": { ... },
      "metadatumoj": {
        "retrovitaj_frazoj": 5,
        "plej_bona_poento": 0.92,
        "fontoj": ["la_mastro_de_l_ringoj"]
      }
    },
    {
      "stadio": "rezonado",
      "tempo": "2025-11-27T10:30:03",
      "ast": { ... },
      "metadatumoj": {
        "rezona_tipo": "ekstrakta",
        "elektita_frazo": "Frodo estas en la Ŝajro."
      }
    }
  ],
  "fina_ast": { ... }
}
```

---

### Phase 6: AST-Based Reasoning (The Big One)
**Goal**: Implement reasoning as AST transformations, not learned weights

**Status**: Not started
**Duration**: 4-5 sessions
**Impact**: Revolutionary - this is where we prove the thesis

#### 6.1: AST Reasoning Patterns (Week 3)

**Core Idea**: Common reasoning tasks can be expressed as AST→AST transformations

**Example Reasoning Patterns**:

**1. Question Answering (Factoid)**
```
Input AST: "Kie estas Frodo?" (Where is Frodo?)
Pattern: demando(loko, Subjekto=Frodo)

Reasoning:
1. Extract focus: Frodo (propra_nomo)
2. Extract question type: "kie" → loko (location)
3. Retrieve: Find sentences with Frodo + location phrases
4. Extract location from best match
5. Construct response AST: "Frodo estas en [LOKO]."

Output AST: "Frodo estas en la Ŝajro."
```

**2. Comparison**
```
Input AST: "Ĉu Frodo estas pli aĝa ol Bilbo?" (Is Frodo older than Bilbo?)
Pattern: demando(komparo, Subjekto1=Frodo, Atributo=aĝa, Subjekto2=Bilbo)

Reasoning:
1. Retrieve facts about Frodo's age
2. Retrieve facts about Bilbo's age
3. Compare values (if available)
4. Construct yes/no response

Output AST: "Ne, Bilbo estas pli aĝa ol Frodo."
```

**3. Inference (Transitive)**
```
Retrieved ASTs:
  1. "Frodo estas hobito."
  2. "Hobitoj estas malaltaj."

Reasoning:
1. Identify relation: Frodo → hobito (is-a)
2. Identify property: hobito → malalta (has-property)
3. Apply transitivity: Frodo → malalta
4. Construct response

Output AST: "Frodo estas malalta, ĉar hobitoj estas malaltaj."
```

**Implementation**:

```python
# klareco/reasoner.py (NEW)

class ASTReasoner:
    """
    Performs reasoning through AST transformations.

    No learned weights - pure symbolic reasoning over structure!
    """

    def __init__(self, retriever, enricher):
        self.retriever = retriever
        self.enricher = enricher
        self.patterns = self._load_reasoning_patterns()

    def reason(self, query_ast: dict, discourse_context: list = None) -> dict:
        """
        Generate response AST through reasoning.

        Args:
            query_ast: Enriched query AST
            discourse_context: Previous conversation ASTs

        Returns:
            Response AST with reasoning trail
        """
        # 1. Classify reasoning type
        reasoning_type = self._classify_reasoning(query_ast)

        # 2. Select reasoning pattern
        pattern = self.patterns.get(reasoning_type)
        if not pattern:
            return self._fallback_extractive(query_ast)

        # 3. Execute reasoning pattern
        response_ast = pattern.execute(query_ast, self.retriever, discourse_context)

        # 4. Add reasoning metadata
        response_ast['rezona_metadatumoj'] = {
            'rezona_tipo': reasoning_type,
            'uzita_ŝablono': pattern.name,
            'certeco': pattern.confidence
        }

        return response_ast

    def _classify_reasoning(self, ast: dict) -> str:
        """
        Determine reasoning type from AST structure.

        Types:
        - ekstrakta: Direct extraction from single sentence
        - komparativa: Comparison between entities
        - transitiva: Inference through relations
        - kompendia: Summarization across multiple sentences
        - kaŭzeca: Causal reasoning (why X?)
        - kontraŭfakta: Counterfactual (what if X?)
        """
        diskurso = ast.get('diskursa_analizo', {})
        frazo_tipo = diskurso.get('frazo_tipo')

        if frazo_tipo != 'demando':
            return 'ekstrakta'  # Statements use extractive

        # Check question word to determine reasoning type
        question_word = self._find_question_word(ast)

        if question_word in ('kie', 'kiam', 'kio'):
            return 'ekstrakta'  # Where, when, what - direct facts
        elif question_word == 'kial':
            return 'kaŭzeca'  # Why - need causal reasoning
        elif question_word == 'kiel':
            # Check if comparison
            if self._has_comparison(ast):
                return 'komparativa'
            else:
                return 'ekstrakta'

        return 'ekstrakta'  # Default

    def _fallback_extractive(self, query_ast: dict) -> dict:
        """
        Fallback: Just return best retrieved sentence.
        """
        text = deparse(query_ast)
        results = self.retriever.retrieve(text, k=1)
        if results:
            # Parse retrieved sentence into AST
            response_text = results[0]['text']
            response_ast = parse(response_text)
            response_ast['rezona_metadatumoj'] = {
                'rezona_tipo': 'ekstrakta',
                'fonto': results[0].get('source'),
                'poento': results[0].get('score')
            }
            return response_ast
        else:
            # Generate "I don't know" AST
            return self._generate_unknown_response()


class ReasoningPattern:
    """Base class for reasoning patterns."""

    def __init__(self, name: str):
        self.name = name
        self.confidence = 0.0

    def execute(self, query_ast: dict, retriever, context: list) -> dict:
        raise NotImplementedError


class ExtractivePa (continues in next section...)
```

---

### Phase 7: Semantic Retrieval (Week 4)

**Goal**: Retrieve by semantic roles, not just keywords

**Current retrieval**:
```
Query: "Kiu vidas la katon?"
Matches: Any sentence with "vid" and "kat"
```

**Semantic retrieval**:
```
Query: "Kiu vidas la katon?" (Who sees the cat?)
  → semantika_signaturo: (aganto=?, ago=vidas, paciento=kato)

Matches only sentences where:
  - Cat is PATIENT (accusative, being seen)
  - Not where cat is AGENT (nominative, doing the seeing)
```

**Implementation**:

```python
# klareco/semantic_index.py (NEW)

class SemanticIndex:
    """
    Indexes sentences by semantic signatures.

    Signature format:
      (aganto, ago, paciento, [modifiers])

    Example:
      "La hundo vidas la katon."
      → (hundo, vidas, kato, [])
    """

    def __init__(self, corpus_path: Path):
        self.signatures = {}  # signature → [sentence_ids]
        self.sentences = {}   # sentence_id → full data
        self._build_index(corpus_path)

    def _build_index(self, corpus_path: Path):
        """Extract semantic signatures from enriched corpus."""
        for i, entry in enumerate(read_corpus(corpus_path)):
            ast = entry['ast']

            # Skip if not enriched
            if 'semantika_rolo' not in str(ast):
                continue

            # Extract signature
            sig = self._extract_signature(ast)

            # Add to index
            if sig not in self.signatures:
                self.signatures[sig] = []
            self.signatures[sig].append(i)
            self.sentences[i] = entry

    def _extract_signature(self, ast: dict) -> tuple:
        """
        Extract semantic signature from enriched AST.

        Returns:
            (aganto_root, verbo_root, paciento_root)
        """
        subjekto = ast.get('subjekto', {})
        verbo = ast.get('verbo', {})
        objekto = ast.get('objekto', {})

        # Get roots, handling vortgrupo
        aganto = self._get_root(subjekto) if subjekto.get('semantika_rolo') == 'aganto' else None
        ago = self._get_root(verbo)
        paciento = self._get_root(objekto) if objekto.get('semantika_rolo') == 'paciento' else None

        return (aganto, ago, paciento)

    def search(self, query_signature: tuple, k: int = 10) -> list:
        """
        Search by semantic signature.

        Args:
            query_signature: (aganto, ago, paciento) - use None for wildcards
            k: Number of results

        Returns:
            Matching sentences with scores
        """
        matches = []

        for sig, sentence_ids in self.signatures.items():
            score = self._match_signature(query_signature, sig)
            if score > 0:
                for sid in sentence_ids:
                    matches.append({
                        'sentence_id': sid,
                        'signature': sig,
                        'score': score,
                        'data': self.sentences[sid]
                    })

        # Sort by score and return top k
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:k]

    def _match_signature(self, query: tuple, candidate: tuple) -> float:
        """
        Score signature match.

        Rules:
        - Exact match: 1.0
        - Partial match: 0.5 per matching component
        - None (wildcard) matches anything
        """
        score = 0.0
        weights = [0.3, 0.4, 0.3]  # aganto, ago, paciento

        for i, (q, c) in enumerate(zip(query, candidate)):
            if q is None:  # Wildcard
                continue
            elif q == c:   # Exact match
                score += weights[i]

        return score
```

**Usage**:
```python
# Search for "Who sees the cat?"
# → aganto=?, ago=vidas, paciento=kat

semantic_index = SemanticIndex('data/corpus_v2_enriched.jsonl')

query_sig = (None, 'vid', 'kat')  # None = wildcard
results = semantic_index.search(query_sig, k=5)

# Returns only sentences where cat is being seen,
# not where cat is doing the seeing!
```

---

## Priority Implementation Order

Based on impact vs. effort:

### High Priority (Do First)

**1. Proper Noun Dictionary** ⭐⭐⭐
- **Impact**: High (fixes parse rates, improves quality immediately)
- **Effort**: Low (2-3 hours)
- **Dependencies**: None
- **Do this now!**

**2. AST Trail System** ⭐⭐⭐
- **Impact**: High (enables debugging, explanation, interpretability)
- **Effort**: Medium (1 day)
- **Dependencies**: None
- **Do after proper nouns**

**3. Semantic AST Enricher** ⭐⭐
- **Impact**: High (enables semantic reasoning)
- **Effort**: High (2-3 days)
- **Dependencies**: Proper nouns
- **Foundation for everything else**

### Medium Priority (Do Second)

**4. Reasoning Patterns** ⭐⭐
- **Impact**: Revolutionary (proves the thesis!)
- **Effort**: Very High (1-2 weeks)
- **Dependencies**: Enricher, trail system
- **Start with simple patterns**

**5. Semantic Index** ⭐
- **Impact**: Medium (better retrieval)
- **Effort**: Medium (2-3 days)
- **Dependencies**: Enricher
- **Can parallelize with reasoning**

### Low Priority (Future)

**6. Coreference Resolution** ⭐
- **Impact**: Medium (better context handling)
- **Effort**: High (1 week)
- **Dependencies**: Enricher
- **Nice to have, not essential**

**7. Discourse Relations** ⭐
- **Impact**: Low (mainly for multi-turn conversation)
- **Effort**: High (1 week)
- **Dependencies**: Enricher
- **Save for later**

---

## Testing Strategy

**Philosophy**: Every AST transformation must be testable and debuggable

### Unit Tests (Deterministic Components)

```python
# tests/test_enricher.py

def test_semantic_role_active_voice():
    """Agent in nominative, patient in accusative."""
    ast = parse("La hundo vidas la katon.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['semantika_rolo'] == 'aganto'
    assert enriched['objekto']['semantika_rolo'] == 'paciento'

def test_semantic_role_passive_voice():
    """Patient in nominative with passive verb."""
    ast = parse("La kato estas vidata.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['semantika_rolo'] == 'paciento'

def test_animacy_living():
    ast = parse("La hundo kuras.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['animateco'] == 'vivanta'

def test_animacy_nonliving():
    ast = parse("La ŝtono falas.")
    enriched = enricher.enrich(ast)

    assert enriched['subjekto']['animateco'] == 'nevivanta'
```

### Integration Tests (Full Pipeline)

```python
# tests/test_pipeline.py

def test_full_pipeline_factoid():
    """Test complete pipeline for factoid question."""
    query = "Kie estas Frodo?"

    # Parse
    ast = parse(query)
    assert ast['parse_statistics']['success_rate'] == 1.0

    # Enrich
    enriched = enricher.enrich(ast)
    assert enriched['diskursa_analizo']['frazo_tipo'] == 'demando'
    assert enriched['diskursa_analizo']['demando_tipo'] == 'loko'

    # Retrieve
    results = retriever.retrieve(query)
    assert len(results) > 0
    assert 'Frodo' in results[0]['text']

    # Reason
    response_ast = reasoner.reason(enriched)
    assert 'Ŝajro' in deparse(response_ast) or 'Mordoro' in deparse(response_ast)

def test_ast_trail_completeness():
    """Ensure trail captures all stages."""
    query = "Kio estas hobito?"
    trail = process_query_with_trail(query)

    stages = trail.get_history()
    stage_names = [s['stadio'] for s in stages]

    assert 'parsado' in stage_names
    assert 'enriĉigo' in stage_names
    assert 'serĉado' in stage_names
    assert 'rezonado' in stage_names
```

### Reasoning Tests (AST Transformations)

```python
# tests/test_reasoning.py

def test_extractive_reasoning():
    """Simple factoid extraction."""
    query_ast = parse("Kie estas Frodo?")
    enriched = enricher.enrich(query_ast)

    reasoner = ASTReasoner(retriever, enricher)
    response_ast = reasoner.reason(enriched)

    # Should find location in response
    response_text = deparse(response_ast)
    assert any(loc in response_text for loc in ['Ŝajro', 'Mordoro', 'Gondoro'])

def test_comparative_reasoning():
    """Comparison between entities."""
    query_ast = parse("Ĉu Frodo estas pli alta ol Gimli?")
    enriched = enricher.enrich(query_ast)

    response_ast = reasoner.reason(enriched)
    response_text = deparse(response_ast)

    # Should have yes/no answer
    assert 'jes' in response_text.lower() or 'ne' in response_text.lower()
```

---

## Success Metrics

**Phase 5 Success**: AST Enrichment
- ✅ Parse rate > 95% (with proper noun dictionary)
- ✅ 100% of sentences have semantic roles
- ✅ AST trail captures all processing stages
- ✅ Can export trail as JSON for inspection

**Phase 6 Success**: AST Reasoning
- ✅ Handles 5+ reasoning patterns (extractive, comparative, transitive, causal, summarization)
- ✅ 80%+ accuracy on factoid questions (vs. corpus ground truth)
- ✅ All reasoning explainable through AST trail
- ✅ Zero learned weights in reasoning patterns (purely symbolic)

**Phase 7 Success**: Semantic Retrieval
- ✅ Semantic index built from enriched corpus
- ✅ Retrieval considers semantic roles, not just keywords
- ✅ 90%+ precision on role-specific queries ("who did X to Y?")
- ✅ Combined structural + semantic + neural pipeline

---

## Long-Term Vision (Phase 8+)

### Learned Components (Minimal!)

**What NOT to learn**:
- ❌ Grammar parsing (we have that!)
- ❌ Role extraction (deterministic from case)
- ❌ Structural matching (symbolic)

**What TO learn** (small models only):
- ✅ Semantic similarity (15M param Tree-LSTM)
- ✅ Animacy/category prediction (5M param classifier)
- ✅ Response ranking (5M param scorer)
- ✅ Natural language generation (optional 50M param seq2seq for fluency)

**Total learned parameters**: ~75M (vs. 110M+ for BERT, 175B for GPT-3)

### Multi-Modal Extensions

Once AST-based reasoning is solid, extend to:
- **Vision**: Object detection → Esperanto names → AST
- **Audio**: Speech-to-text → Esperanto → AST
- **Structured data**: SQL/JSON → AST representation
- **Code**: Python AST → Esperanto AST → reasoning

The AST becomes the universal interface!

---

## Next Actions

**This Session**:
1. ✅ Run improved corpus builder
   ```bash
   python scripts/build_corpus_v2.py
   ```

**Next Session** (Priority Order):
1. Implement `scripts/build_proper_noun_dict.py` (extract from corpus)
2. Add `klareco/proper_nouns.py` (dictionary class)
3. Integrate proper nouns into parser
4. Test parse rate improvement
5. Rebuild corpus with improved parser

**Session After** (AST Enrichment):
1. Create `klareco/enricher.py` (semantic layer)
2. Add animacy database (`data/animacy.json`)
3. Implement semantic role assignment
4. Create `klareco/ast_trail.py` (trail system)
5. Test enrichment on sample sentences

**Future Sessions** (Reasoning):
1. Design reasoning pattern framework
2. Implement 3-5 core patterns (extractive, comparative, transitive)
3. Integrate into orchestrator
4. Build semantic index
5. Full pipeline integration and testing

---

## Philosophy Recap

**We're not building a smaller LLM. We're building a different architecture:**

- **LLM approach**: Learn everything, including what grammar gives us for free
- **Klareco approach**: Use grammar deterministically, learn only where structure can't help

**Key insight**: Esperanto's regularity means:
- 80% of NLP tasks are solvable with pure symbolic AST operations
- 15% need lightweight learned components (semantic similarity)
- 5% might need larger models (generation fluency)

**The AST is the consciousness** - every thought, inference, and decision happens in explicit Esperanto structures that we can inspect, debug, and understand.

This is interpretable AI through linguistic determinism. Let's build it! 🚀
