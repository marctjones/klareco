# Klareco Implementation Roadmap v2.0

**Updated**: 2025-11-29
**Vision**: Esperanto-first AI with AST-as-consciousness and compositional embeddings

**Core Thesis**: Maximize deterministic structural operations, minimize learned parameters, prove that explicit grammar beats learned weights.

---

## Quick Reference

### Current Status ✅
- Parser: Morpheme-aware, 16 Esperanto rules
- Corpus: 35,571 sentences, 91.8% parse rate
- Retriever: Two-stage (structural + neural Tree-LSTM)
- Embeddings: Compositional (320K params) - implementation complete
- Semantic Similarity: Training data generated from Tatoeba (159K pairs)

### Immediate Next Steps
1. ⏳ **Run Semantic Similarity Training** (`./scripts/run_semantic_training.sh`)
2. **Evaluate trained model** on held-out test set
3. **Integrate** semantic similarity into retrieval pipeline

### Near-Term
4. **AST Trail System** (3 days)
5. **Advanced reasoning patterns**

### Medium-Term
6. **AST-Based Reasoning** (4 weeks)

---

## Phase Priorities (Reordered for Maximum Impact)

### ✅ PHASE 5A: Semantic Similarity Training - COMPLETED

**Status**: Training data generated, training script ready

**Approach**: Use English as a similarity oracle, train only on Esperanto

The key insight: we can use English sentence embeddings to determine which Esperanto
sentences are semantically similar, but the model only ever sees Esperanto during training.
This preserves linguistic purity while leveraging cross-lingual resources.

**Data Pipeline**:
1. Downloaded Tatoeba EN-EO parallel corpus (271,265 sentence pairs)
2. Found Esperanto paraphrases (same English translation = same meaning)
3. Used `paraphrase-MiniLM-L6-v2` (80MB) to compute English similarity
4. Generated 159K training pairs containing ONLY Esperanto:
   - 42K paraphrases (similarity = 1.0)
   - 100K negatives (similarity < 0.3)
   - 17K medium similarity (0.3 - 0.8)

**Files Created**:
- `scripts/generate_similarity_pairs.py` - Generates training pairs
- `scripts/run_similarity_generation.sh` - Shell script for generation
- `scripts/train_semantic_similarity.py` - Training script
- `scripts/run_semantic_training.sh` - Shell script for training
- `data/similarity_pairs_train.jsonl` (143K pairs)
- `data/similarity_pairs_val.jsonl` (7.9K pairs)
- `data/similarity_pairs_test.jsonl` (7.9K pairs)

**Training Objective**:
- CosineSimilarityLoss: Predicted embedding similarity should match target similarity
- Model: TreeLSTMEncoder with compositional embeddings
- Key metric: Pearson correlation (baseline ~0, good >0.6, excellent >0.8)

**Initial Test Results** (500 samples, 1 epoch): val_corr = 0.56 ✅

**Next**: Run full training with `./scripts/run_semantic_training.sh`

---

### 🔧 PHASE 5B: Semantic Retrieval Integration - NEXT

**Why first?**: Easy to implement, impressive results, foundation for everything else

**Timeline**: 5 days
**Impact**: 🔥🔥🔥 (Users see it immediately!)
**Complexity**: ⭐ (Low - ~80 lines of code)
**Dependencies**: None

#### Day 1-2: Proper Noun Dictionary (6 hours)

**Goal**: Fix parse rates from 91.8% to 95%+

**Tasks**:
1. Extract proper nouns from corpus (2 hours)
   ```bash
   python scripts/build_proper_noun_dict.py \
     --corpus data/corpus_with_sources_v2.jsonl \
     --output data/proper_nouns_static.json \
     --min-frequency 3
   ```

2. Create `klareco/proper_nouns.py` (2 hours)
   ```python
   class ProperNounDictionary:
       def __init__(self, static_path, dynamic_path=None):
           self.static = load_json(static_path)
           self.dynamic = load_json(dynamic_path) if dynamic_path else {}

       def is_proper_noun(self, word: str) -> bool:
           base = strip_esperanto_endings(word)
           return base in self.static or base in self.dynamic
   ```

3. Integrate into parser (1 hour)
   ```python
   # In parser.py, categorize_unknown_word()
   if word[0].isupper():
       if proper_noun_dict.is_proper_noun(word):
           ast["parse_status"] = "success"  # ✅ No longer "failed"!
           ast["category"] = "proper_name_known"
           return ast
   ```

4. Test parse rate improvement (1 hour)
   ```python
   # Should see parse_rate jump from 0.918 to 0.95+
   ```

**Deliverables**:
- ✅ `data/proper_nouns_static.json` (500-1000 entries)
- ✅ `klareco/proper_nouns.py`
- ✅ `scripts/build_proper_noun_dict.py`
- ✅ Tests: `tests/test_proper_nouns.py`

#### Day 3-4: Semantic Signatures & Index (12 hours)

**Goal**: Build (agent, action, patient) index for role-based retrieval

**Tasks**:

1. Implement signature extraction (3 hours)
   ```python
   # klareco/semantic_signatures.py (NEW)

   def extract_signature(ast: dict) -> tuple:
       """Extract (agent, action, patient) from AST."""

       subjekto = ast.get('subjekto', {})
       verbo = ast.get('verbo', {})
       objekto = ast.get('objekto', {})

       # Handle both vorto and vortgrupo
       aganto = get_root(subjekto) if subjekto else None
       ago = get_root(verbo) if verbo else None
       paciento = get_root(objekto) if objekto else None

       return (aganto, ago, paciento)

   def get_root(node: dict) -> str:
       """Extract root from vorto or vortgrupo."""
       if not node:
           return None
       if node.get('tipo') == 'vorto':
           return node.get('radiko')
       elif node.get('tipo') == 'vortgrupo':
           kerno = node.get('kerno', {})
           return kerno.get('radiko')
       return None
   ```

2. Build semantic index from corpus (4 hours)
   ```python
   # scripts/build_semantic_index.py (NEW)

   def build_semantic_index(corpus_path: Path, output_path: Path):
       """Build signature → [sentence_ids] index."""

       signatures = {}  # (agent, action, patient) → [ids]
       metadata = []

       with open(corpus_path, 'r') as f:
           for i, line in enumerate(f):
               entry = json.loads(line)

               if 'ast' not in entry:
                   continue

               sig = extract_signature(entry['ast'])

               if sig not in signatures:
                   signatures[sig] = []
               signatures[sig].append(i)

               metadata.append({
                   'text': entry['text'],
                   'source': entry['source'],
                   'signature': sig
               })

       # Save index and metadata
       with open(output_path / 'signatures.json', 'w') as f:
           json.dump(signatures, f)

       with open(output_path / 'metadata.jsonl', 'w') as f:
           for meta in metadata:
               f.write(json.dumps(meta, ensure_ascii=False) + '\n')

       print(f"Built index with {len(signatures):,} unique signatures")
       print(f"Indexed {len(metadata):,} sentences")
   ```

3. Implement semantic search (3 hours)
   ```python
   # klareco/semantic_search.py (NEW)

   def search_semantic(
       query_sig: tuple,
       index: dict,
       metadata: list,
       k: int = 10
   ) -> list:
       """Search by semantic signature with wildcards."""

       matches = []

       for sig, sent_ids in index.items():
           score = match_signature(query_sig, sig)

           if score > 0:
               for sid in sent_ids:
                   matches.append({
                       'sentence_id': sid,
                       'signature': sig,
                       'score': score,
                       'text': metadata[sid]['text'],
                       'source': metadata[sid]['source']
                   })

       # Sort by score descending
       matches.sort(key=lambda x: x['score'], reverse=True)
       return matches[:k]

   def match_signature(query: tuple, candidate: tuple) -> float:
       """Score signature match (None = wildcard)."""
       score = 0.0
       weights = [0.3, 0.4, 0.3]  # agent, action, patient

       for i, (q, c) in enumerate(zip(query, candidate)):
           if q is None:  # Wildcard matches anything
               continue
           elif q == c:   # Exact match
               score += weights[i]

       return score
   ```

4. Testing (2 hours)
   ```python
   # tests/test_semantic_search.py

   def test_role_disambiguation():
       """Test that roles are correctly filtered."""

       # Query: "Who sees the cat?" → (?, vid, kat)
       query_sig = (None, 'vid', 'kat')
       results = search_semantic(query_sig, index, metadata, k=5)

       # All results should have 'kat' as PATIENT (object)
       for r in results:
           ast = parse(r['text'])
           objekto = ast.get('objekto', {})
           assert 'kat' in str(objekto), "Cat should be object, not subject!"
   ```

**Deliverables**:
- ✅ `klareco/semantic_signatures.py`
- ✅ `klareco/semantic_search.py`
- ✅ `scripts/build_semantic_index.py`
- ✅ `data/semantic_index/` (signatures + metadata)
- ✅ Tests: `tests/test_semantic_search.py`

#### Day 5: Integration & Demo (6 hours)

**Goal**: Add semantic filtering to retriever and create impressive demo

**Tasks**:

1. Integrate into retriever (3 hours)
   ```python
   # In klareco/rag/retriever.py

   class KlarecoRetriever:
       def __init__(self, ..., semantic_index_path=None):
           # ... existing code ...

           if semantic_index_path:
               self.semantic_index = load_semantic_index(semantic_index_path)
           else:
               self.semantic_index = None

       def retrieve(self, query: str, k=5, use_semantic=True):
           """Three-stage retrieval: structural → semantic → neural."""

           # Stage 1: Structural filtering (existing)
           query_ast = parse(query)
           canon_sig = canonicalize_sentence(query_ast)
           candidates = self._filter_structural(canon_sig)

           # Stage 1.5: Semantic filtering (NEW!)
           if use_semantic and self.semantic_index:
               semantic_sig = extract_signature(query_ast)
               candidates = self._filter_semantic(semantic_sig, candidates)
               logger.info(f"Semantic filter: {len(candidates)} candidates")

           # Stage 2: Neural reranking (existing)
           results = self._neural_rerank(query_ast, candidates, k)

           return results

       def _filter_semantic(self, query_sig: tuple, candidates: list) -> list:
           """Filter candidates by semantic signature."""

           # Get sentence IDs from semantic index
           semantic_matches = search_semantic(
               query_sig,
               self.semantic_index,
               self.metadata,
               k=len(candidates) * 2  # Get extra, filter later
           )

           semantic_ids = set(m['sentence_id'] for m in semantic_matches)

           # Keep only candidates in semantic matches
           filtered = [c for c in candidates if c['id'] in semantic_ids]

           return filtered
   ```

2. Create demo script (2 hours)
   ```python
   # scripts/demo_semantic_retrieval.py (NEW)

   def main():
       print("🔍 Semantic Retrieval Demo")
       print("=" * 60)

       retriever_old = KlarecoRetriever(..., use_semantic=False)
       retriever_new = KlarecoRetriever(..., use_semantic=True)

       test_queries = [
           ("Kiu vidas la katon?", "Who sees the cat?"),
           ("Kion vidas la hundo?", "What does the dog see?"),
           ("Kiu amas Frodon?", "Who loves Frodo?"),
       ]

       for eo_query, en_query in test_queries:
           print(f"\n📝 Query: {eo_query} ({en_query})")
           print("-" * 60)

           # Old retrieval (keyword-based)
           old_results = retriever_old.retrieve(eo_query, k=3)
           print("\n❌ OLD (Keyword-based):")
           for i, r in enumerate(old_results, 1):
               print(f"  {i}. [{r['score']:.2f}] {r['text'][:80]}...")

           # New retrieval (semantic role-based)
           new_results = retriever_new.retrieve(eo_query, k=3)
           print("\n✅ NEW (Semantic role-based):")
           for i, r in enumerate(new_results, 1):
               print(f"  {i}. [{r['score']:.2f}] {r['text'][:80]}...")

           # Explain the difference
           print("\n💡 Explanation:")
           query_ast = parse(eo_query)
           sig = extract_signature(query_ast)
           print(f"  Semantic signature: {sig}")
           print(f"  → Filters results to match these roles!")
   ```

3. Documentation (1 hour)
   - Add to `QUICK_WINS_ANALYSIS.md`
   - Update `README.md` with semantic retrieval example

**Deliverables**:
- ✅ Updated `klareco/rag/retriever.py`
- ✅ `scripts/demo_semantic_retrieval.py`
- ✅ Documentation updates
- ✅ Working demo showing role disambiguation!

**Week 1 Success Metrics**:
- ✅ Parse rate ≥ 95% (with proper nouns)
- ✅ Semantic index built (~20K unique signatures)
- ✅ Demo shows clear improvement over keyword search
- ✅ Role disambiguation working (cat as patient vs agent)

---

### 🎯 PHASE 5B: Compositional Embeddings (WEEK 2-3) - HIGH IMPACT

**Why second?**: 75% parameter reduction, better generalization, foundation for smaller models

**Timeline**: 10 days
**Impact**: 🔥🔥 (Smaller, faster models)
**Complexity**: ⭐⭐ (Medium - requires model retraining)
**Dependencies**: None (can parallelize with semantic retrieval)

#### Day 6-7: Root Vocabulary Extraction (2 days)

**Goal**: Extract all unique roots from corpus for compositional embeddings

**Tasks**:

1. Build root extractor (4 hours)
   ```python
   # scripts/extract_root_vocabulary.py (NEW)

   def extract_roots_from_corpus(corpus_path: Path) -> dict:
       """Extract all unique roots with frequency counts."""

       root_counts = {}

       with open(corpus_path, 'r') as f:
           for line in f:
               entry = json.loads(line)
               ast = entry.get('ast', {})

               # Extract roots from all words in AST
               for word_ast in extract_all_words(ast):
                   radiko = word_ast.get('radiko')
                   if radiko:
                       root_counts[radiko] = root_counts.get(radiko, 0) + 1

       # Sort by frequency
       sorted_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)

       # Create vocabulary (root → id)
       root_vocab = {root: i for i, (root, count) in enumerate(sorted_roots)}

       return root_vocab, root_counts

   def extract_all_words(ast: dict) -> list:
       """Recursively extract all word ASTs."""
       words = []

       if ast.get('tipo') == 'vorto':
           words.append(ast)
       elif ast.get('tipo') == 'vortgrupo':
           if ast.get('kerno'):
               words.append(ast['kerno'])
           for priskribo in ast.get('priskriboj', []):
               words.extend(extract_all_words(priskribo))
       elif ast.get('tipo') == 'frazo':
           for field in ['subjekto', 'verbo', 'objekto']:
               if ast.get(field):
                   words.extend(extract_all_words(ast[field]))
           for word in ast.get('aliaj', []):
               words.extend(extract_all_words(word))

       return words
   ```

2. Run extraction (1 hour)
   ```bash
   python scripts/extract_root_vocabulary.py \
     --corpus data/corpus_with_sources_v2.jsonl \
     --output data/root_vocabulary.json \
     --stats data/root_statistics.json
   ```

3. Analyze vocabulary (2 hours)
   ```python
   # Analyze root statistics
   # - Total unique roots
   # - Frequency distribution
   # - Coverage (90% of words from top N roots)
   ```

4. Create affix vocabularies (1 hour)
   ```python
   # data/affix_vocabulary.json
   {
     "prefixes": {
       "mal": 0, "re": 1, "ge": 2, "eks": 3, "ek": 4, "pra": 5, "for": 6
     },
     "suffixes": {
       "ul": 0, "ej": 1, "in": 2, "et": 3, "ad": 4, "ig": 5, "iĝ": 6,
       "ism": 7, "ist": 8, "ar": 9, "aĉ": 10, "aĵ": 11, "ebl": 12,
       "end": 13, "ec": 14, "eg": 15, "em": 16, "er": 17, "estr": 18,
       "id": 19, "il": 20, "ind": 21, "ing": 22, "uj": 23, "um": 24,
       "ant": 25, "int": 26, "ont": 27, "at": 28, "it": 29, "ot": 30
     }
   }
   ```

**Deliverables**:
- ✅ `data/root_vocabulary.json` (~5,000 roots)
- ✅ `data/root_statistics.json` (frequency analysis)
- ✅ `data/affix_vocabulary.json` (7 prefixes, 31 suffixes)
- ✅ `scripts/extract_root_vocabulary.py`

#### Day 8-10: Compositional Embedding Implementation (3 days)

**Goal**: Build compositional embedder that replaces whole-word embeddings

**Tasks**:

1. Implement compositional embedder (8 hours)
   ```python
   # klareco/embeddings/compositional.py (NEW)

   import torch
   import torch.nn as nn

   class CompositionalEmbedding(nn.Module):
       """
       Compositional word embeddings for Esperanto.

       Decomposes: word = root + prefix + suffix + ending + grammar

       Dimensions (96 total):
         - Root:    64 dims (learned)
         - Prefix:   8 dims (semi-learned)
         - Suffix:   8 dims (semi-learned)
         - Ending:   8 dims (programmatic)
         - Grammar:  8 dims (programmatic)

       Parameters: ~320K (vs 1.28M for whole-word)
       """

       def __init__(
           self,
           root_vocab: dict,
           prefix_vocab: dict,
           suffix_vocab: dict,
           root_dim: int = 64,
           affix_dim: int = 8,
           ending_dim: int = 8,
           grammar_dim: int = 8,
           freeze_affixes: bool = True
       ):
           super().__init__()

           self.root_vocab = root_vocab
           self.prefix_vocab = prefix_vocab
           self.suffix_vocab = suffix_vocab

           # Learned components
           self.root_embed = nn.Embedding(len(root_vocab), root_dim)
           self.prefix_embed = nn.Embedding(len(prefix_vocab), affix_dim)
           self.suffix_embed = nn.Embedding(len(suffix_vocab), affix_dim)

           # Initialize affixes with semantic priors
           self._initialize_affixes()

           # Optionally freeze affixes
           if freeze_affixes:
               self.prefix_embed.weight.requires_grad = False
               self.suffix_embed.weight.requires_grad = False

           # Dimensions
           self.root_dim = root_dim
           self.affix_dim = affix_dim
           self.ending_dim = ending_dim
           self.grammar_dim = grammar_dim
           self.output_dim = root_dim + affix_dim * 2 + ending_dim + grammar_dim

       def forward(self, word_ast: dict) -> torch.Tensor:
           """
           Compose word embedding from AST.

           Args:
               word_ast: Parsed word AST with radiko, prefikso, sufiksoj, etc.

           Returns:
               Composed embedding (96 dims)
           """
           device = self.root_embed.weight.device

           # 1. Root embedding (64 dims, learned)
           radiko = word_ast.get('radiko', '')
           if radiko in self.root_vocab:
               root_id = self.root_vocab[radiko]
               root_vec = self.root_embed(torch.tensor(root_id, device=device))
           else:
               # UNK root
               root_vec = torch.zeros(self.root_dim, device=device)

           # 2. Prefix embedding (8 dims, semi-learned)
           prefikso = word_ast.get('prefikso')
           if prefikso and prefikso in self.prefix_vocab:
               prefix_id = self.prefix_vocab[prefikso]
               prefix_vec = self.prefix_embed(torch.tensor(prefix_id, device=device))
           else:
               prefix_vec = torch.zeros(self.affix_dim, device=device)

           # 3. Suffix embeddings (8 dims, semi-learned, can have multiple)
           sufiksoj = word_ast.get('sufiksoj', [])
           if sufiksoj:
               suffix_vecs = []
               for suf in sufiksoj:
                   if suf in self.suffix_vocab:
                       suf_id = self.suffix_vocab[suf]
                       suffix_vecs.append(
                           self.suffix_embed(torch.tensor(suf_id, device=device))
                       )
               if suffix_vecs:
                   # Sum multiple suffixes
                   suffix_vec = torch.stack(suffix_vecs).sum(dim=0)
               else:
                   suffix_vec = torch.zeros(self.affix_dim, device=device)
           else:
               suffix_vec = torch.zeros(self.affix_dim, device=device)

           # 4. Ending encoding (8 dims, programmatic)
           ending_vec = self._encode_ending(word_ast, device)

           # 5. Grammar encoding (8 dims, programmatic)
           grammar_vec = self._encode_grammar(word_ast, device)

           # Concatenate all components
           return torch.cat([root_vec, prefix_vec, suffix_vec, ending_vec, grammar_vec])

       def _initialize_affixes(self):
           """Initialize affixes with semantic priors."""

           # Prefixes
           # "mal" = opposite (negative pattern)
           if 'mal' in self.prefix_vocab:
               self.prefix_embed.weight.data[self.prefix_vocab['mal']] = \
                   torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

           # "re" = again (repetition pattern)
           if 're' in self.prefix_vocab:
               self.prefix_embed.weight.data[self.prefix_vocab['re']] = \
                   torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

           # Suffixes
           # "ej" = place (location pattern)
           if 'ej' in self.suffix_vocab:
               self.suffix_embed.weight.data[self.suffix_vocab['ej']] = \
                   torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

           # "ul" = person (animate pattern)
           if 'ul' in self.suffix_vocab:
               self.suffix_embed.weight.data[self.suffix_vocab['ul']] = \
                   torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

           # "ig" = causative (action pattern)
           if 'ig' in self.suffix_vocab:
               self.suffix_embed.weight.data[self.suffix_vocab['ig']] = \
                   torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

           # "iĝ" = become (change-of-state pattern)
           if 'iĝ' in self.suffix_vocab:
               self.suffix_embed.weight.data[self.suffix_vocab['iĝ']] = \
                   torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

       def _encode_ending(self, word_ast: dict, device) -> torch.Tensor:
           """Programmatic encoding of grammatical ending (8 dims)."""

           vec = torch.zeros(8, device=device)
           vortspeco = word_ast.get('vortspeco', '')
           tempo = word_ast.get('tempo')

           # One-hot encoding of endings
           if vortspeco == 'substantivo':
               vec[0] = 1.0  # -o
           elif vortspeco == 'adjektivo':
               vec[1] = 1.0  # -a
           elif vortspeco == 'adverbo':
               vec[2] = 1.0  # -e
           elif vortspeco == 'verbo':
               if tempo == 'prezenco':
                   vec[3] = 1.0  # -as
               elif tempo == 'pasinteco':
                   vec[4] = 1.0  # -is
               elif tempo == 'futuro':
                   vec[5] = 1.0  # -os
               elif tempo == 'kondiĉa':
                   vec[6] = 1.0  # -us

           return vec

       def _encode_grammar(self, word_ast: dict, device) -> torch.Tensor:
           """Programmatic encoding of grammatical features (8 dims)."""

           vec = torch.zeros(8, device=device)

           # Dim 0: Case (1 = accusative, 0 = nominative)
           vec[0] = 1.0 if word_ast.get('kazo') == 'akuzativo' else 0.0

           # Dim 1: Number (1 = plural, 0 = singular)
           vec[1] = 1.0 if word_ast.get('nombro') == 'pluralo' else 0.0

           # Dims 2-3: Reserved for tense (already in ending)
           # Dims 4-7: Reserved for future features

           return vec
   ```

2. Create batch encoder for AST trees (4 hours)
   ```python
   # klareco/embeddings/batch_encoder.py (NEW)

   class BatchCompositionalEncoder:
       """Batched encoding of multiple words for efficiency."""

       def __init__(self, compositional_embedding):
           self.embedding = compositional_embedding

       def encode_sentence(self, sentence_ast: dict) -> torch.Tensor:
           """
           Encode all words in a sentence.

           Args:
               sentence_ast: Parsed sentence AST

           Returns:
               Tensor of word embeddings (num_words, 96)
           """
           words = extract_all_words(sentence_ast)
           embeddings = []

           for word_ast in words:
               emb = self.embedding(word_ast)
               embeddings.append(emb)

           if embeddings:
               return torch.stack(embeddings)
           else:
               # Empty sentence
               return torch.zeros(1, self.embedding.output_dim)
   ```

3. Testing (4 hours)
   ```python
   # tests/test_compositional_embeddings.py (NEW)

   def test_same_root_similar():
       """Words with same root should have similar embeddings."""

       comp_embed = CompositionalEmbedding(root_vocab, prefix_vocab, suffix_vocab)

       hundo_ast = parse("hundo")
       hundoj_ast = parse("hundoj")

       hundo_emb = comp_embed(hundo_ast['subjekto'])  # Extract word AST
       hundoj_emb = comp_embed(hundoj_ast['subjekto'])

       # Similarity should be high (same root, different grammar)
       sim = F.cosine_similarity(hundo_emb, hundoj_emb, dim=0)
       assert sim > 0.9, f"Same root should be similar, got {sim}"

   def test_opposite_prefix():
       """'mal' prefix should create different embeddings."""

       bona_ast = parse("La bona hundo.")
       malbona_ast = parse("La malbona hundo.")

       bona_emb = comp_embed(extract_word(bona_ast, "bona"))
       malbona_emb = comp_embed(extract_word(malbona_ast, "malbona"))

       # Should be dissimilar (opposite meanings)
       sim = F.cosine_similarity(bona_emb, malbona_emb, dim=0)
       assert sim < 0.5, f"Opposite words should differ, got {sim}"

   def test_programmatic_dimensions():
       """Grammar dimensions should be deterministic."""

       hundo_ast = parse("hundo")
       hundon_ast = parse("hundon")  # Accusative

       hundo_emb = comp_embed(extract_word(hundo_ast, "hundo"))
       hundon_emb = comp_embed(extract_word(hundon_ast, "hundon"))

       # First 80 dims should be identical (same root + affixes)
       assert torch.allclose(hundo_emb[:80], hundon_emb[:80])

       # Dim 80 (case) should differ
       assert hundo_emb[80] == 0.0  # Nominative
       assert hundon_emb[80] == 1.0  # Accusative

   def test_composition_generalization():
       """Should handle unseen word combinations."""

       # Create embedding for word never seen in training
       # "rehundejo" = re + hund + ej + o (re-dogification-place)
       fake_ast = {
           'radiko': 'hund',
           'prefikso': 're',
           'sufiksoj': ['ej'],
           'vortspeco': 'substantivo',
           'kazo': 'nominativo',
           'nombro': 'singularo'
       }

       emb = comp_embed(fake_ast)

       # Should produce valid embedding (no errors, reasonable values)
       assert emb.shape == (96,)
       assert not torch.isnan(emb).any()
   ```

**Deliverables**:
- ✅ `klareco/embeddings/compositional.py`
- ✅ `klareco/embeddings/batch_encoder.py`
- ✅ Tests: `tests/test_compositional_embeddings.py`
- ✅ All tests passing (compositional properties verified)

#### Day 11-13: Model Integration & Retraining (3 days)

**Goal**: Integrate compositional embeddings into Tree-LSTM and retrain

**Tasks**:

1. Update Tree-LSTM model (4 hours)
   ```python
   # In klareco/models/tree_lstm.py

   class TreeLSTMEncoder(nn.Module):
       def __init__(
           self,
           root_vocab: dict,
           prefix_vocab: dict,
           suffix_vocab: dict,
           hidden_dim: int = 128,
           output_dim: int = 128,
           use_compositional: bool = True  # NEW flag
       ):
           super().__init__()

           if use_compositional:
               # NEW: Compositional embeddings
               self.embed = CompositionalEmbedding(
                   root_vocab=root_vocab,
                   prefix_vocab=prefix_vocab,
                   suffix_vocab=suffix_vocab,
                   root_dim=64,
                   affix_dim=8,
                   ending_dim=8,
                   grammar_dim=8,
                   freeze_affixes=True
               )
               embed_dim = self.embed.output_dim  # 96
           else:
               # OLD: Whole-word embeddings
               vocab_size = len(root_vocab)  # Actually whole words
               embed_dim = 128
               self.embed = nn.Embedding(vocab_size, embed_dim)

           # Rest of model unchanged
           self.tree_lstm = ChildSumTreeLSTM(embed_dim, hidden_dim)
           self.output_proj = nn.Linear(hidden_dim, output_dim)
   ```

2. Update training script (4 hours)
   ```python
   # scripts/train_tree_lstm.py

   # Load vocabularies
   with open('data/root_vocabulary.json', 'r') as f:
       root_vocab = json.load(f)

   with open('data/affix_vocabulary.json', 'r') as f:
       affix_vocab = json.load(f)

   # Create model with compositional embeddings
   model = TreeLSTMEncoder(
       root_vocab=root_vocab,
       prefix_vocab=affix_vocab['prefixes'],
       suffix_vocab=affix_vocab['suffixes'],
       hidden_dim=128,
       output_dim=128,
       use_compositional=True  # Use new embeddings!
   )

   print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
   # Should see ~320K vs 1.28M before!
   ```

3. Run training (overnight, ~8-12 hours)
   ```bash
   python scripts/train_tree_lstm.py \
     --corpus data/corpus_with_sources_v2.jsonl \
     --output models/tree_lstm_compositional \
     --epochs 20 \
     --batch-size 32 \
     --learning-rate 0.001
   ```

4. Compare old vs new (4 hours)
   ```python
   # scripts/compare_embeddings.py (NEW)

   def compare_models():
       """Compare compositional vs whole-word embeddings."""

       old_model = load_model('models/tree_lstm/best_model.pt')
       new_model = load_model('models/tree_lstm_compositional/best_model.pt')

       # Test queries
       queries = [
           "Kie estas Frodo?",
           "Kiu vidas la katon?",
           "Kio estas hobito?"
       ]

       for query in queries:
           print(f"\nQuery: {query}")

           # Retrieve with both models
           old_results = retrieve_with_model(query, old_model)
           new_results = retrieve_with_model(query, new_model)

           # Compare results
           print(f"  OLD (1.28M params):")
           for r in old_results[:3]:
               print(f"    [{r['score']:.2f}] {r['text'][:60]}...")

           print(f"  NEW (320K params):")
           for r in new_results[:3]:
               print(f"    [{r['score']:.2f}] {r['text'][:60]}...")
   ```

**Deliverables**:
- ✅ Updated `klareco/models/tree_lstm.py`
- ✅ `models/tree_lstm_compositional/` (new checkpoint)
- ✅ `scripts/compare_embeddings.py`
- ✅ Comparison results (quality maintained, 75% fewer params!)

#### Day 14-15: Polish & Documentation (2 days)

**Goal**: Polish compositional embeddings system and document

**Tasks**:
1. Comprehensive testing (4 hours)
2. Documentation (4 hours)
   - Add to `COMPOSITIONAL_EMBEDDINGS.md`
   - Update `README.md`
3. Create visualization (4 hours)
   - Visualize root space (t-SNE/UMAP)
   - Show composition examples

**Week 2-3 Success Metrics**:
- ✅ 75% parameter reduction (1.28M → 320K)
- ✅ Retrieval quality maintained or improved
- ✅ Compositional generalization working (unseen word forms)
- ✅ Interpretable embeddings (can probe dimensions)

---

### 🔧 PHASE 5C: AST Trail System (WEEK 3) - ENABLER

**Why third?**: Enables debugging, explainability, and future reasoning

**Timeline**: 3 days
**Impact**: 🔥 (Foundation for transparency)
**Complexity**: ⭐ (Low - simple data structure)
**Dependencies**: None

(Continue with AST Trail implementation - same as before...)

---

### 🧠 PHASE 6: AST-Based Reasoning (WEEK 4-8) - REVOLUTIONARY

**Why fourth?**: Builds on semantic retrieval, proves the core thesis

(Continue with reasoning implementation - same as before...)

---

## Updated Priority Order Summary

### Weeks 1-3: Foundation (High ROI, Low Risk)

1. **Week 1**: Semantic Retrieval
   - Proper nouns (2 days)
   - Semantic signatures & index (2 days)
   - Integration & demo (1 day)

2. **Week 2-3**: Compositional Embeddings
   - Root vocabulary (2 days)
   - Implementation (3 days)
   - Integration & retraining (3 days)
   - Polish (2 days)

3. **Week 3**: AST Trail (parallel with embeddings)
   - Implementation (2 days)
   - Integration (1 day)

### Weeks 4-8: Reasoning (Transformational)

4. **Week 4-5**: Reasoning Patterns
5. **Week 6-7**: Advanced Patterns
6. **Week 8**: Integration & Evaluation

---

## Success Metrics

### End of Week 1
- ✅ Parse rate > 95%
- ✅ Semantic retrieval working
- ✅ Demo shows role disambiguation
- ✅ Users impressed!

### End of Week 3
- ✅ Compositional embeddings deployed
- ✅ 75% parameter reduction
- ✅ Quality maintained
- ✅ AST trail capturing all stages

### End of Week 8
- ✅ AST reasoning working
- ✅ 5+ reasoning patterns
- ✅ 80% accuracy on factoid questions
- ✅ All reasoning explainable via AST trail

---

## Next Actions

**RIGHT NOW**:
```bash
# Run improved corpus builder
python scripts/build_corpus_v2.py
```

**DAY 1 (Tomorrow)**:
```bash
# Extract proper nouns
python scripts/build_proper_noun_dict.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/proper_nouns_static.json \
  --min-frequency 3
```

**This is the path to Esperanto-first AI!** 🚀
