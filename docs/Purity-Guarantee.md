# Klareco Purity Guarantee

## Core Principle: Pure Esperanto AI

**Klareco is 100% Pure Esperanto** - no English contaminates the AI model at any stage.

English translations in demos are **DISPLAY ONLY** - like subtitles on a movie. The movie itself (Klareco) is in Esperanto.

---

## What Is Pure Esperanto

### ✅ Training Data (Pure Esperanto)
- **Corpus**: 4.2M Esperanto sentences from Gutenberg, Wikipedia, tier0 texts
- **Stage 1**: Trained on Esperanto roots only (10,819 roots)
- **M1**: Trained on Esperanto subject-verb-object triples
- **No English**: Zero English words in any training data

**Files**:
- `data/enhanced_corpus/corpus_with_tier0.jsonl` - Pure Esperanto
- `data/training/root_embeddings/` - Esperanto roots only
- `data/training/m1_semantic_full/` - Esperanto triples only

### ✅ AST Annotations (Pure Esperanto)
- **Field names**: subjekto, verbo, objekto, aliaj, radiko, vortspeco, kazo
- **Values**: All Esperanto (roots, endings, grammar markers)
- **No English**: Zero English in any AST node

**Example AST**:
```json
{
  "tipo": "frazo",
  "subjekto": {"tipo": "vorto", "radiko": "hund", "vortspeco": "substantivo"},
  "verbo": {"tipo": "vorto", "radiko": "manĝ", "tempo": "prezenco"},
  "objekto": {"tipo": "vorto", "radiko": "viand", "kazo": "akuzativo"}
}
```
**No English anywhere!**

### ✅ Model Architecture (Pure Esperanto)
- **Parser**: 16 Esperanto grammar rules (0 learned params)
- **Stage 1**: Root embeddings (692K params, Esperanto roots only)
- **M1**: Selectional preferences (838K params, Esperanto triples only)
- **Retriever**: Operates on Esperanto ASTs and roots

**Code**:
- `klareco/parser.py` - Pure Esperanto grammar rules
- `klareco/embeddings/compositional.py` - Esperanto morpheme embeddings
- `klareco/models/m1_selectional.py` - Esperanto triple scoring

### ✅ Processing Pipeline (Pure Esperanto)
```
Text (Esperanto)
  → Parser (Esperanto rules)
  → AST (Esperanto annotations)
  → Stage 1 (Esperanto embeddings)
  → M1 (Esperanto triples)
  → Retriever (Esperanto matching)
  → Answer (Esperanto)
```

**No English at any stage!**

---

## What Is NOT Part of Klareco

### ❌ Demo Display Layer (NOT part of model)

**Purpose**: Help users understand Esperanto output (like subtitles)

**Implementation**: Helsinki-NLP MarianMT (with fallback to dictionary)

**Files** (display only, NOT model):
- `scripts/demo_rag_with_m1.py:211-296` - Translation functions
- Uses `Helsinki-NLP/opus-mt-eo-en` model (Esperanto → English)
- Fallback to simple dictionary if model unavailable

**Usage**:
```python
# Pure Esperanto processing (THIS IS KLARECO)
answer = rag.answer("Kiu fondis Esperanton?")  # Pure EO
# answer['text'] = "Zamenhof fondis Esperanton."

# Display layer (NOT KLARECO - just UI)
if show_translations:
    translation = translate_to_english(answer['text'])  # MarianMT
    print(f"→ {translation}")  # "Zamenhof founded Esperanto."
```

**Translation Stack** (all display-only):
1. **Primary**: Helsinki-NLP MarianMT (`opus-mt-eo-en`)
   - Lazy-loaded (doesn't load unless translations requested)
   - Better quality than hardcoded dictionary
   - Handles full sentences, not just word-by-word

2. **Fallback**: Simple dictionary (~10 common words)
   - If MarianMT unavailable or fails
   - Basic word-by-word translation

**Analogy**: Like DVD subtitles
- The movie is in Esperanto (Klareco AI)
- Subtitles are in English (MarianMT for display)
- Subtitles don't change the movie

### ❌ External APIs (NOT used)

We do NOT use:
- ❌ Google Translate API
- ❌ Any external translation service
- ❌ Any cloud-based translation

We DO use (display-only):
- ✅ Helsinki-NLP MarianMT (local model, lazy-loaded, display-only)
- ✅ Fallback dictionary (simple hardcoded mappings)

---

## Verification Checklist

### Training Data Purity
- [ ] `data/enhanced_corpus/` - Check for English sentences
  ```bash
  grep -i "the\|is\|was\|are" data/enhanced_corpus/corpus_with_tier0.jsonl | head
  # Should find ZERO English sentences (only Esperanto)
  ```

- [ ] `data/training/root_embeddings/` - Check vocabulary
  ```python
  checkpoint = torch.load('models/root_embeddings_tier0/best_model.pt')
  roots = list(checkpoint['root_to_idx'].keys())
  print([r for r in roots if not is_esperanto_root(r)])  # Should be empty
  ```

- [ ] `data/training/m1_semantic_full/` - Check triples
  ```bash
  jq '.subject_root, .verb_root, .object_root' data/training/m1_semantic_full/train.jsonl | head
  # Should be all Esperanto roots
  ```

### AST Purity
- [ ] Parse sample sentences and check AST fields
  ```python
  from klareco.parser import parse
  ast = parse("Mi amas hundojn.")
  # Check: all field names are Esperanto (subjekto, verbo, objekto)
  # Check: all values are Esperanto (am, hund, etc.)
  ```

### Model I/O Purity
- [ ] Stage 1 inputs/outputs
  ```python
  # Input: Esperanto root → Output: embedding vector
  emb = stage1.get_embedding('hund')  # Pure Esperanto
  ```

- [ ] M1 inputs/outputs
  ```python
  # Input: (EO subj, EO verb, EO obj) → Output: score
  score = m1.score_triple('hund', 'manĝ', 'viand')  # Pure Esperanto
  ```

### Demo Layer Separation
- [ ] `translate_to_english()` is ONLY in demo scripts
  ```bash
  grep -r "translate_to_english" klareco/
  # Should return ZERO results (not in core code)

  grep -r "translate_to_english" scripts/
  # Should only find demo scripts (demo_rag_with_m1.py)
  ```

- [ ] No English in core Klareco modules
  ```bash
  grep -r "english\|translate" klareco/ --include="*.py" | grep -v "#"
  # Should return ZERO results (no English in processing)
  ```

---

## Why This Matters

### 1. Purity Enables Explainability
- Every prediction is traceable to Esperanto grammar rules or Esperanto corpus examples
- No hidden English "leakage" that would obscure the reasoning

### 2. Purity Enables Small Models
- By focusing learned parameters ONLY on Esperanto, we need fewer parameters
- English would dilute the semantic space and require larger models

### 3. Purity Enables Validation
- We can verify correctness against Fundamento (official Esperanto foundation)
- We can test on authoritative Esperanto literature (Zamenhof's works)
- No confusion from English contamination

### 4. Purity Is the Thesis
- "Traditional LLMs waste capacity learning grammar. By making grammar explicit through Esperanto's regularity, we can focus learned parameters on reasoning."
- This only works if we ACTUALLY keep it Pure Esperanto

---

## If You Add English...

**DON'T!** But if absolutely necessary:

### Rule 1: Display Only
English can ONLY appear in:
- Demo scripts (for user convenience)
- Documentation (to explain concepts)
- Test assertions (to describe expected behavior)

English MUST NOT appear in:
- Training data
- AST annotations
- Model inputs/outputs
- Core processing code

### Rule 2: Clearly Marked
If you add English display helpers:
```python
def translate_to_english(eo_text: str) -> str:
    """
    ⚠️  DISPLAY ONLY - NOT PART OF KLARECO MODEL
    ⚠️  This is like subtitles on a movie - doesn't change the movie
    """
```

### Rule 3: Flag to Disable
Always provide `--no-translate` flag:
```python
parser.add_argument('--no-translate', action='store_true',
                    help='Show pure Esperanto (no English translations)')
```

### Rule 4: Test Purity
Add test to verify no English contamination:
```python
def test_klareco_purity():
    """Verify no English in Klareco processing."""
    # Check corpus
    with open('data/enhanced_corpus/corpus_with_tier0.jsonl') as f:
        for line in f:
            doc = json.loads(line)
            assert not has_english(doc['text']), "English in corpus!"

    # Check AST
    ast = parse("Mi amas hundojn.")
    ast_str = json.dumps(ast)
    assert 'subject' not in ast_str, "English in AST!"
    assert 'subjekto' in ast_str, "Missing Esperanto in AST!"
```

---

## Current Status

✅ **Klareco core is 100% Pure Esperanto**
✅ **English translations are display-only in demo scripts**
✅ **No English in training data, ASTs, embeddings, or models**
✅ **`--no-translate` flag available to disable translations**

**Verification**: Run the tests above to confirm purity.

**Documentation updated**: 2026-01-19
