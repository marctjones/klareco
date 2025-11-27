# Corpus Improvement Plan

## Problem
Current corpus has fragmented sentences due to hard-wrapping at ~75 characters.
Example: `"ampleksa Prologo , en kiu li prezentis multajn informojn pri la hobitoj kaj"` (incomplete)

## Root Cause
`build_corpus_with_sources.py` treats each line as a sentence, but source texts are hard-wrapped.

## Solution: Two-Phase Corpus Building

### Phase 1: Sentence Boundary Detection (NEW)

Create `scripts/extract_sentences.py` that:

1. **Unwrap hard-wrapped text**
   - Concatenate lines that don't end with sentence terminators (. ! ?)
   - Preserve paragraph breaks (double newlines)

2. **Detect sentence boundaries**
   - Split on: `. ` `! ` `? ` (with space after)
   - Handle abbreviations: `D-ro` `S-ro` `k.t.p.` (don't split)
   - Handle ellipsis: `...` (don't split)

3. **Parse each sentence** (quality check)
   - Use `klareco.parser.parse()` to create AST
   - Track parse success rate
   - Flag sentences with low parse quality

4. **Store with metadata**
   ```json
   {
     "text": "La plej aĝa el tiuj, kaj la plej ŝatata de Bilbo, estis juna Frodo.",
     "source": "la_mastro_de_l_ringoj",
     "source_name": "La Mastro de l' Ringoj",
     "line": 1979,
     "ast": {...},
     "parse_success": true,
     "parse_rate": 0.85,
     "word_count": 13
   }
   ```

### Phase 2: Corpus Filtering & Quality Control

Before indexing:

1. **Filter by parse quality**
   - Keep only sentences with `parse_rate >= 0.7`
   - Skip metadata lines, headers, fragments

2. **Filter by completeness**
   - Check AST has subject/verb or verb/object
   - Skip one-word sentences unless they're complete (e.g., "Jes.")

3. **Filter by length**
   - Min: 3 words (current: 20 chars is too aggressive)
   - Max: 50 words (very long sentences may be errors)

4. **Deduplication**
   - Remove exact duplicates
   - Consider fuzzy deduplication (edit distance)

## Implementation Steps

### Step 1: Write Sentence Splitter
```python
# scripts/extract_sentences.py
def unwrap_text(lines: List[str]) -> List[str]:
    """Unwrap hard-wrapped paragraphs."""
    paragraphs = []
    current = []

    for line in lines:
        line = line.rstrip()
        if not line:  # Empty line = paragraph break
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append(' '.join(current))

    return paragraphs

def split_sentences(text: str) -> List[str]:
    """Split paragraph into sentences."""
    import re

    # Esperanto abbreviations that shouldn't split
    abbrevs = ['D-ro', 'S-ro', 'S-ino', 'k.t.p', 'k.a', 'n-ro', 'p.K']

    # Protect abbreviations
    for abbrev in abbrevs:
        text = text.replace(abbrev, abbrev.replace('.', '<DOT>'))

    # Split on sentence terminators with space
    sentences = re.split(r'([.!?])\s+', text)

    # Rejoin sentences with their terminators
    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1] in '.!?':
            result.append((sentences[i] + sentences[i+1]).strip())
            i += 2
        else:
            if sentences[i].strip():
                result.append(sentences[i].strip())
            i += 1

    # Restore abbreviations
    result = [s.replace('<DOT>', '.') for s in result]

    return [s for s in result if s]
```

### Step 2: Integrate AST Generation
```python
# In extract_sentences.py
from klareco.parser import parse

def extract_with_ast(cleaned_file: Path) -> List[Dict]:
    """Extract sentences with AST metadata."""
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    paragraphs = unwrap_text(lines)
    results = []

    for para_num, para in enumerate(paragraphs, 1):
        sentences = split_sentences(para)

        for sent_num, sent in enumerate(sentences, 1):
            if len(sent.split()) < 3:  # Skip very short
                continue

            # Parse sentence
            try:
                ast = parse(sent)
                parse_success = True
                parse_rate = ast.get('parse_statistics', {}).get('success_rate', 0.0)
            except Exception as e:
                ast = None
                parse_success = False
                parse_rate = 0.0

            results.append({
                'text': sent,
                'paragraph': para_num,
                'sentence_in_para': sent_num,
                'ast': ast,
                'parse_success': parse_success,
                'parse_rate': parse_rate,
                'word_count': len(sent.split())
            })

    return results
```

### Step 3: Update build_corpus_with_sources.py

Replace line-by-line processing with sentence extraction:

```python
# In build_corpus_with_sources.py
from scripts.extract_sentences import extract_with_ast

def build_corpus_v2(...):
    for filename, display_name in texts:
        file_path = cleaned_dir / filename

        # Extract sentences with AST
        sentences = extract_with_ast(file_path)

        # Filter by quality
        sentences = [s for s in sentences if s['parse_rate'] >= 0.7]

        # Write to corpus
        for i, sent_data in enumerate(sentences):
            entry = {
                'text': sent_data['text'],
                'source': filename.replace('cleaned_', '').replace('.txt', ''),
                'source_name': display_name,
                'paragraph': sent_data['paragraph'],
                'ast': sent_data['ast'],  # Store AST for indexing
                'parse_rate': sent_data['parse_rate']
            }
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

### Step 4: Update index_corpus.py

Use pre-computed ASTs instead of parsing again:

```python
# In index_corpus.py
def encode_sentence(self, corpus_entry: Dict) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Encode using pre-computed AST."""
    # Use AST from corpus if available
    ast = corpus_entry.get('ast')
    if not ast:
        # Fallback: parse on demand
        ast = parse(corpus_entry['text'])

    # Rest of encoding logic...
    graph_data = self.converter.ast_to_graph(ast)
    # ...
```

## Benefits

1. **Complete sentences**: "La plej aĝa el tiuj, kaj la plej ŝatata de Bilbo, estis juna Frodo." (instead of fragments)
2. **Better retrieval**: Full semantic context improves relevance
3. **Quality filtering**: Skip unparseable or malformed text
4. **Faster indexing**: Reuse pre-computed ASTs
5. **Debugging**: Inspect parse quality before indexing
6. **Smaller index**: Fewer low-quality entries

## Migration Plan

1. **Create new corpus**: `data/corpus_with_sources_v2.jsonl`
2. **Build with new script**: `python scripts/extract_sentences.py`
3. **Verify quality**: Check parse rates, inspect samples
4. **Rebuild index**: `python scripts/index_corpus.py --corpus data/corpus_with_sources_v2.jsonl --output data/corpus_index_v3`
5. **Compare results**: Query both indexes, measure improvement
6. **Deprecate old corpus**: Once v2 is validated

## Estimated Impact

- **Current**: 49K sentences, many fragments, parse rate unknown
- **Expected**: 30-40K complete sentences, parse rate >70%, much better quality
- **Retrieval improvement**: 2-3x better relevance scores for "Kiu estas X?" queries

## Next Steps

1. Implement `scripts/extract_sentences.py`
2. Add tests for sentence splitting edge cases
3. Run on small corpus (La Hobito) to validate
4. Build full corpus v2
5. Benchmark retrieval quality improvement
