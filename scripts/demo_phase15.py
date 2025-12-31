#!/usr/bin/env python3
"""
Phase 1.5 Demo: EnrichedAST, SemanticPipeline, Retriever, and Translation

Demonstrates the new abstractions built on top of the Stage 1 models:

1. EnrichedAST: Container that accumulates semantic meaning through the pipeline
   - Wraps parser output + learned embeddings
   - Tracks which stages have been applied
   - Serializable for storage

2. SemanticPipeline: Chains models where each reads/writes to EnrichedAST
   - Stage 0: Parser (deterministic)
   - Stage 1: Semantic Model (root + affix embeddings)
   - Stage 2: Grammatical Model (future)
   - Stage 3: Discourse Model (future)

3. Retriever: Semantic search using the pipeline
   - Fast query embedding via SemanticPipeline
   - FAISS index for similarity search
   - Lazy enrichment of results

4. Translation: Automatic EN↔EO translation
   - Uses Helsinki-NLP/opus-mt models
   - Language detection for auto-translation
   - Enables English speakers to query Esperanto corpus

Usage:
    python scripts/demo_phase15.py
    python scripts/demo_phase15.py --interactive
    python scripts/demo_phase15.py --translate "The dog runs fast"
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


# =============================================================================
# Translation Module
# =============================================================================

# Global translators (lazy loaded)
_translator_eo_en = None  # Esperanto → English
_translator_en_eo = None  # English → Esperanto
_translator_loading = False


def get_translator(direction: str = "eo-en"):
    """Lazy load translators. Direction: 'eo-en' or 'en-eo'."""
    global _translator_eo_en, _translator_en_eo, _translator_loading

    if direction == "eo-en":
        if _translator_eo_en is not None:
            return _translator_eo_en
    else:
        if _translator_en_eo is not None:
            return _translator_en_eo

    if _translator_loading:
        return None

    _translator_loading = True

    try:
        from transformers import MarianMTModel, MarianTokenizer

        if direction == "eo-en":
            print("  Loading EO→EN translation model...")
            model_name = "Helsinki-NLP/opus-mt-eo-en"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            _translator_eo_en = (model, tokenizer)
            print("  EO→EN model loaded!")
            _translator_loading = False
            return _translator_eo_en
        else:
            print("  Loading EN→EO translation model...")
            model_name = "Helsinki-NLP/opus-mt-en-eo"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            _translator_en_eo = (model, tokenizer)
            print("  EN→EO model loaded!")
            _translator_loading = False
            return _translator_en_eo

    except ImportError:
        print("  Warning: transformers not installed")
        print("  Install with: pip install transformers sentencepiece")
        _translator_loading = False
        return None
    except Exception as e:
        print(f"  Warning: Could not load translator: {e}")
        _translator_loading = False
        return None


def translate_to_english(text: str, max_length: int = 100) -> str:
    """Translate Esperanto text to English."""
    translator = get_translator("eo-en")

    if translator is None:
        return None

    model, tokenizer = translator

    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Translate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        return f"[translation error: {e}]"


def translate_to_esperanto(text: str, max_length: int = 100) -> str:
    """Translate English text to Esperanto."""
    translator = get_translator("en-eo")

    if translator is None:
        return None

    model, tokenizer = translator

    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Translate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        return f"[translation error: {e}]"


def detect_language(text: str) -> str:
    """Simple language detection - returns 'eo' or 'en'."""
    # Esperanto-specific characters
    eo_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
    if any(c in eo_chars for c in text):
        return 'eo'

    # Common English words that don't exist in Esperanto
    english_markers = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'been',
                       'what', 'where', 'when', 'how', 'why', 'which', 'who',
                       'this', 'that', 'these', 'those', 'with', 'from', 'about']

    words = text.lower().split()
    english_count = sum(1 for w in words if w.strip('.,!?') in english_markers)

    # If more than 15% of words are English markers, it's probably English
    if len(words) > 0 and english_count / len(words) > 0.15:
        return 'en'

    # Esperanto word endings
    eo_endings = ['oj', 'aj', 'on', 'an', 'as', 'is', 'os', 'us', 'ojn', 'ajn']
    eo_count = sum(1 for w in words if any(w.strip('.,!?').endswith(e) for e in eo_endings))

    if len(words) > 0 and eo_count / len(words) > 0.3:
        return 'eo'

    # Default to English if uncertain
    return 'en'


def auto_translate(text: str, target: str = 'eo') -> tuple:
    """
    Auto-detect language and translate if needed.

    Returns: (translated_text, source_lang, was_translated)
    """
    detected = detect_language(text)

    if detected == target:
        return text, detected, False

    if target == 'eo' and detected == 'en':
        translated = translate_to_esperanto(text)
        if translated:
            return translated, detected, True
    elif target == 'en' and detected == 'eo':
        translated = translate_to_english(text)
        if translated:
            return translated, detected, True

    return text, detected, False


def annotate_eo(text: str, max_len: int = 60) -> str:
    """
    Annotate Esperanto text with English translation in parentheses.
    Returns: "EO text (EN translation)"
    """
    if not text:
        return text

    # Try to translate
    en = translate_to_english(text)
    if en and en != text:
        # Truncate if too long
        if len(en) > max_len:
            en = en[:max_len-3] + "..."
        return f"{text} ({en})"
    return text


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def demo_enriched_ast():
    """Demonstrate the EnrichedAST container."""
    from klareco import EnrichedAST
    from klareco.parser import parse

    print_header("1. EnrichedAST: Structured Container for Pipeline Stages")

    # Example sentence with translation
    text = "La malbona hundo rapide kuris al la lernejo."
    en_text = translate_to_english(text) if get_translator("eo-en") else "The bad dog quickly ran to the school."
    print(f"\nInput: {text}")
    print(f"       ({en_text})")

    # Stage 0: Parse
    raw_ast = parse(text)
    enriched = EnrichedAST.from_parser_output(raw_ast, text)

    print_subheader("Stage 0: Parser Output")
    print(f"  tipo: {enriched.tipo}")
    print(f"  fraztipo: {enriched.fraztipo}")
    print(f"  negita: {enriched.negita}")
    print(f"  stages_applied: {enriched.stages_applied}")

    # Show parser statistics
    stats = enriched.parse_statistics
    if stats:
        print(f"  parse_rate: {stats.get('success_rate', 0)*100:.0f}%")

    # Show extracted words
    print_subheader("Content Words from AST")
    for word in enriched.get_content_words():
        root = word.get('radiko', '?')
        prefixes = word.get('prefiksoj', [])
        suffixes = word.get('sufiksoj', [])
        vortspeco = word.get('vortspeco', '?')

        parts = []
        if prefixes:
            parts.append(f"[{'+'.join(prefixes)}]")
        parts.append(root)
        if suffixes:
            parts.append(f"[{'+'.join(suffixes)}]")

        print(f"  {''.join(parts):<20} ({vortspeco})")

    # Show accessors
    print_subheader("AST Accessors")
    if enriched.subjekto:
        kerno = enriched.subjekto.get('kerno', {})
        print(f"  subjekto: {kerno.get('radiko', '?')}")
    if enriched.verbo:
        print(f"  verbo: {enriched.verbo.get('radiko', '?')} (tempo: {enriched.tempo})")
    if enriched.objekto:
        kerno = enriched.objekto.get('kerno', {})
        print(f"  objekto: {kerno.get('radiko', '?')}")

    return enriched


def demo_semantic_pipeline():
    """Demonstrate the SemanticPipeline."""
    from klareco import SemanticPipeline

    print_header("2. SemanticPipeline: Staged Model Processing")

    # Load pipeline
    print("\nLoading SemanticPipeline...")
    try:
        pipeline = SemanticPipeline.load()
        print("  Loaded successfully!")
    except Exception as e:
        print(f"  Error loading pipeline: {e}")
        print("  Make sure Stage 1 models are trained.")
        return None

    # Show model info
    semantic = pipeline.semantic
    print(f"\nSemanticModel:")
    print(f"  Roots: {len(semantic.root_to_idx):,}")
    print(f"  Embedding dim: {semantic.embedding_dim}")
    print(f"  Prefixes: {len(semantic.prefix_transforms)}")
    print(f"  Suffixes: {len(semantic.suffix_transforms)}")

    # Process some sentences with translations
    test_sentences = [
        ("La hundo manĝas la katon.", "The dog eats the cat."),
        ("La kato manĝas la hundon.", "The cat eats the dog."),
        ("La bona lernanto rapide lernas.", "The good student learns quickly."),
        ("La malbona vetero daŭris longe.", "The bad weather lasted long."),
    ]

    print_subheader("Processing Sentences")
    enriched_list = []
    for text, en_fallback in test_sentences:
        enriched = pipeline.for_retrieval(text)
        enriched_list.append((text, enriched))

        known = len(enriched.known_roots)
        unknown = len(enriched.unknown_roots)
        has_emb = enriched.sentence_embedding is not None

        # Get translation
        en = translate_to_english(text) if get_translator("eo-en") else en_fallback

        print(f"\n  '{text}'")
        print(f"   ({en})")
        print(f"    Stages: {sorted(enriched.stages_applied)}")
        print(f"    Known roots: {known}, Unknown: {unknown}")
        print(f"    Has embedding: {has_emb}")
        if enriched.unknown_roots:
            print(f"    Missing: {enriched.unknown_roots}")

    # Show similarity between sentences
    print_subheader("Sentence Similarities (Cosine)")
    for i, (text1, e1) in enumerate(enriched_list):
        for text2, e2 in enriched_list[i+1:]:
            if e1.sentence_embedding is not None and e2.sentence_embedding is not None:
                sim = F.cosine_similarity(
                    e1.sentence_embedding.unsqueeze(0),
                    e2.sentence_embedding.unsqueeze(0)
                ).item()
                t1_short = text1[:25] + "..." if len(text1) > 25 else text1
                t2_short = text2[:25] + "..." if len(text2) > 25 else text2
                print(f"  {t1_short:<28} ↔ {t2_short:<28}: {sim:.3f}")

    return pipeline


def demo_retriever():
    """Demonstrate the Retriever."""
    from klareco import Retriever

    print_header("3. Retriever: Semantic Search with Lazy Enrichment")

    # Check if index exists
    index_dir = Path("data/corpus_index_compositional")
    if not index_dir.exists():
        print(f"\n  Index not found at {index_dir}")
        print("  Run: ./scripts/run_compositional_indexing.sh")
        return None

    # Load retriever
    print("\nLoading Retriever...")
    try:
        retriever = Retriever.load()
        print(f"  {retriever}")
    except Exception as e:
        print(f"  Error: {e}")
        return None

    # Demo queries with translations
    queries = [
        ("La hundo kuras rapide.", "The dog runs fast."),
        ("Zamenhof kreis Esperanton.", "Zamenhof created Esperanto."),
        ("La lernejo estas granda.", "The school is big."),
    ]

    print_subheader("Search Results")
    for query, en_fallback in queries:
        en_query = translate_to_english(query) if get_translator("eo-en") else en_fallback
        print(f"\n  Query: '{query}'")
        print(f"         ({en_query})")
        results = retriever.search(query, top_k=3)

        if not results:
            print("    (No results)")
            continue

        for i, r in enumerate(results, 1):
            text = r.text[:50] + "..." if len(r.text) > 50 else r.text
            # Translate result
            en_result = translate_to_english(r.text) if get_translator("eo-en") else None
            print(f"    {i}. [{r.score:.3f}] {text}")
            if en_result:
                en_short = en_result[:45] + "..." if len(en_result) > 45 else en_result
                print(f"                   ({en_short})")

    # Demo lazy enrichment
    print_subheader("Lazy Enrichment")
    query = "La bona instruisto instruas."
    en_query = translate_to_english(query) if get_translator("eo-en") else "The good teacher teaches."
    print(f"\n  Query: '{query}'")
    print(f"         ({en_query})")

    results = retriever.search_and_enrich(query, top_k=2)
    for i, r in enumerate(results, 1):
        text = r.text[:50] + "..." if len(r.text) > 50 else r.text
        en_result = translate_to_english(r.text) if get_translator("eo-en") else None
        print(f"\n  Result {i}: [{r.score:.3f}] {text}")
        if en_result:
            en_short = en_result[:45] + "..." if len(en_result) > 45 else en_result
            print(f"              ({en_short})")

        if r.enriched_ast:
            ast = r.enriched_ast
            print(f"    Stages: {sorted(ast.stages_applied)}")
            print(f"    Known roots: {ast.known_roots}")
            if ast.sentence_embedding is not None:
                print(f"    Embedding: {ast.sentence_embedding.shape}")

    return retriever


def demo_serialization():
    """Demonstrate EnrichedAST serialization."""
    from klareco import EnrichedAST, SemanticPipeline
    import json

    print_header("4. Serialization: Save/Load EnrichedAST")

    try:
        pipeline = SemanticPipeline.load()
    except Exception:
        print("  (Skipped - models not available)")
        return

    text = "La rapida vulpo saltas super la laca hundo."
    en_text = translate_to_english(text) if get_translator("eo-en") else "The quick fox jumps over the lazy dog."
    enriched = pipeline.for_retrieval(text)

    print(f"\nInput: {text}")
    print(f"       ({en_text})")
    print(f"\nOriginal: {enriched}")

    # Serialize to dict
    data = enriched.to_dict()
    print(f"\nSerialized keys: {list(data.keys())}")

    # Convert to JSON and back
    json_str = json.dumps(data, ensure_ascii=False)
    print(f"JSON size: {len(json_str)} bytes")

    # Deserialize
    data2 = json.loads(json_str)
    enriched2 = EnrichedAST.from_dict(data2)

    print(f"\nRestored: {enriched2}")
    print(f"  Stages match: {enriched.stages_applied == enriched2.stages_applied}")

    # Compare embeddings
    if enriched.sentence_embedding is not None and enriched2.sentence_embedding is not None:
        sim = F.cosine_similarity(
            enriched.sentence_embedding.unsqueeze(0),
            enriched2.sentence_embedding.unsqueeze(0)
        ).item()
        print(f"  Embedding similarity: {sim:.6f}")


def demo_pipeline_hooks():
    """Demonstrate pipeline hooks for debugging."""
    from klareco import SemanticPipeline

    print_header("5. Pipeline Hooks: Stage Callbacks")

    try:
        pipeline = SemanticPipeline.load()
    except Exception:
        print("  (Skipped - models not available)")
        return

    # Add hooks
    def stage0_hook(ast):
        print(f"    [Hook] Stage 0: parsed '{ast.original_text[:30]}...'")

    def stage1_hook(ast):
        roots = len(ast.known_roots)
        print(f"    [Hook] Stage 1: embedded {roots} roots")

    pipeline.add_hook('stage0', stage0_hook)
    pipeline.add_hook('stage1', stage1_hook)

    text = "La instruisto instruas en la lernejo."
    en_text = translate_to_english(text) if get_translator("eo-en") else "The teacher teaches in the school."
    print(f"\nInput: {text}")
    print(f"       ({en_text})")
    print("\nProcessing with hooks enabled:")
    enriched = pipeline.for_retrieval(text)
    print(f"\n  Final: {enriched}")


def demo_translation():
    """Demonstrate automatic EN↔EO translation."""
    print_header("6. Translation: Automatic EN↔EO")

    print("\nLanguage Detection:")
    test_texts = [
        "The dog runs fast in the park.",
        "La hundo kuras rapide en la parko.",
        "Mi amas Esperanton.",
        "What is Esperanto?",
        "Zamenhof kreis la lingvon.",
    ]

    for text in test_texts:
        lang = detect_language(text)
        print(f"  [{lang.upper()}] {text}")

    print_subheader("English → Esperanto")
    english_sentences = [
        "The dog is big.",
        "I love learning languages.",
        "Where is the school?",
    ]

    for text in english_sentences:
        eo = translate_to_esperanto(text)
        if eo:
            print(f"  EN: {text}")
            print(f"  EO: {eo}\n")
        else:
            print(f"  (Translation not available)")
            break

    print_subheader("Esperanto → English")
    esperanto_sentences = [
        "La hundo estas granda.",
        "Mi amas lerni lingvojn.",
        "Kie estas la lernejo?",
    ]

    for text in esperanto_sentences:
        en = translate_to_english(text)
        if en:
            print(f"  EO: {text}")
            print(f"  EN: {en}\n")
        else:
            print(f"  (Translation not available)")
            break

    print_subheader("Auto-Translation for Search")
    print("  When searching, English queries are auto-translated to Esperanto:")
    queries = [
        "The teacher teaches students.",
        "Who created Esperanto?",
        "La instruisto instruas.",
    ]

    for query in queries:
        translated, source, was_translated = auto_translate(query, target='eo')
        if was_translated:
            print(f"  '{query}'")
            print(f"    → Detected: {source.upper()}, Translated: '{translated}'")
        else:
            print(f"  '{query}'")
            print(f"    → Detected: {source.upper()}, No translation needed")


def demo_grammatical_adjuster():
    """Demonstrate deterministic GrammaticalAdjuster (Stage 2)."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("7. GrammaticalAdjuster: Deterministic Grammatical Similarity")

    print("""
The GrammaticalAdjuster adjusts Stage 1 semantic similarity based on
grammatical features already present in the AST:

  - negita: negation (flips polarity)
  - tempo: tense (past/present/future)
  - fraztipo: sentence type (statement/question/command)
  - modo: mood (indicative/conditional)

This is Stage 2 with ZERO learned parameters - pure deterministic rules.
""")

    # Load pipeline
    try:
        pipeline = SemanticPipeline.load()
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return

    adjuster = GrammaticalAdjuster()

    print_subheader("Negation Detection")
    text1 = "La kato dormas."
    text2 = "La kato ne dormas."
    en1 = translate_to_english(text1) if get_translator("eo-en") else "(The cat sleeps.)"
    en2 = translate_to_english(text2) if get_translator("eo-en") else "(The cat doesn't sleep.)"

    print(f"\n  Sentence 1: {text1} ({en1})")
    print(f"  Sentence 2: {text2} ({en2})")

    enriched1 = pipeline.for_retrieval(text1)
    enriched2 = pipeline.for_retrieval(text2)

    # Calculate Stage 1 similarity
    import torch.nn.functional as F
    if enriched1.sentence_embedding is not None and enriched2.sentence_embedding is not None:
        semantic_sim = F.cosine_similarity(
            enriched1.sentence_embedding.unsqueeze(0),
            enriched2.sentence_embedding.unsqueeze(0)
        ).item()

        # Apply grammatical adjustment
        result = adjuster.adjust_with_explanation(enriched1, enriched2, semantic_sim)

        print(f"\n  Stage 1 (semantic): {semantic_sim:.3f}")
        print(f"  Stage 2 (adjusted): {result.adjusted_similarity:.3f}")
        print(f"  Adjustments: {result.adjustments}")
        print(f"\n  Interpretation: Same roots (kato, dorm-) but opposite meaning!")

    print_subheader("Tense Comparison")
    text3 = "Mi manĝas."  # I eat (present)
    text4 = "Mi manĝis."  # I ate (past)
    en3 = translate_to_english(text3) if get_translator("eo-en") else "(I eat.)"
    en4 = translate_to_english(text4) if get_translator("eo-en") else "(I ate.)"

    print(f"\n  Present: {text3} ({en3})")
    print(f"  Past: {text4} ({en4})")

    enriched3 = pipeline.for_retrieval(text3)
    enriched4 = pipeline.for_retrieval(text4)

    if enriched3.sentence_embedding is not None and enriched4.sentence_embedding is not None:
        semantic_sim = F.cosine_similarity(
            enriched3.sentence_embedding.unsqueeze(0),
            enriched4.sentence_embedding.unsqueeze(0)
        ).item()

        result = adjuster.adjust_with_explanation(enriched3, enriched4, semantic_sim)

        print(f"\n  Stage 1 (semantic): {semantic_sim:.3f}")
        print(f"  Stage 2 (adjusted): {result.adjusted_similarity:.3f}")
        print(f"  Adjustments: {result.adjustments}")

    print_subheader("Sentence Type (Statement vs Question)")
    text5 = "Vi parolas Esperanton."  # You speak Esperanto.
    text6 = "Ĉu vi parolas Esperanton?"  # Do you speak Esperanto?
    en5 = translate_to_english(text5) if get_translator("eo-en") else "(You speak Esperanto.)"
    en6 = translate_to_english(text6) if get_translator("eo-en") else "(Do you speak Esperanto?)"

    print(f"\n  Statement: {text5} ({en5})")
    print(f"  Question: {text6} ({en6})")

    enriched5 = pipeline.for_retrieval(text5)
    enriched6 = pipeline.for_retrieval(text6)

    if enriched5.sentence_embedding is not None and enriched6.sentence_embedding is not None:
        semantic_sim = F.cosine_similarity(
            enriched5.sentence_embedding.unsqueeze(0),
            enriched6.sentence_embedding.unsqueeze(0)
        ).item()

        result = adjuster.adjust_with_explanation(enriched5, enriched6, semantic_sim)

        print(f"\n  Stage 1 (semantic): {semantic_sim:.3f}")
        print(f"  Stage 2 (adjusted): {result.adjusted_similarity:.3f}")
        print(f"  Adjustments: {result.adjustments}")

    print_subheader("Default Adjustment Factors")
    print("\n  The adjuster uses these configurable factors:")
    for name, value in adjuster.adjustments.items():
        print(f"    {name}: {value}")


def demo_thought_decoder():
    """Demonstrate ThoughtDecoder for explainable AI."""
    from klareco import SemanticPipeline, ThoughtDecoder

    print_header("8. ThoughtDecoder: Making AI Thoughts Explainable")

    # Load pipeline
    try:
        pipeline = SemanticPipeline.load()
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return

    # Create decoder (without retriever for simpler demo)
    decoder = ThoughtDecoder(pipeline=pipeline)

    print_subheader("Decoding a Simple Sentence")

    # Test sentence
    text = "La hundo kuras rapide."  # The dog runs fast.
    en_text = translate_to_english(text) if translate_to_english(text) else "(The dog runs fast.)"
    print(f"\nInput: {text}")
    print(f"       ({en_text})")

    # Process through pipeline
    enriched = pipeline.for_retrieval(text)

    # Decode the thoughts
    decoded = decoder.decode(enriched, find_similar=False)
    print(f"\n{decoded}")

    print_subheader("Comparing Two Sentences")

    # Compare two sentences
    text1 = "La kato dormas."  # The cat sleeps.
    text2 = "La kato ne dormas."  # The cat doesn't sleep.
    en1 = translate_to_english(text1) if translate_to_english(text1) else "(The cat sleeps.)"
    en2 = translate_to_english(text2) if translate_to_english(text2) else "(The cat doesn't sleep.)"

    print(f"\nSentence 1: {text1} ({en1})")
    print(f"Sentence 2: {text2} ({en2})")

    enriched1 = pipeline.for_retrieval(text1)
    enriched2 = pipeline.for_retrieval(text2)

    comparison = decoder.compare_thoughts(enriched1, enriched2)

    print("\nComparison:")
    print(f"  Same type: {comparison['syntactic']['same_type']}")
    print(f"  Same negation: {comparison['syntactic']['same_negation']}")
    print(f"  Same subject: {comparison['syntactic']['same_subject']} ({comparison['syntactic']['subjects']})")
    print(f"  Same verb: {comparison['syntactic']['same_verb']} ({comparison['syntactic']['verbs']})")

    if 'embedding_similarity' in comparison.get('semantic', {}):
        sim = comparison['semantic']['embedding_similarity']
        interp = comparison['semantic']['interpretation']
        print(f"  Semantic similarity: {sim:.3f} ({interp})")

    print(f"  Shared concepts: {comparison['semantic'].get('shared_concepts', [])}")

    print_subheader("JSON Output (for API use)")
    text3 = "Ĉu vi komprenas Esperanton?"  # Do you understand Esperanto?
    en3 = translate_to_english(text3) if translate_to_english(text3) else "(Do you understand Esperanto?)"
    print(f"\nInput: {text3}")
    print(f"       ({en3})")

    enriched3 = pipeline.for_retrieval(text3)
    json_output = decoder.decode_to_json(enriched3, find_similar=False)
    print(f"\n{json_output[:500]}...")  # Truncate for display


def interactive_mode(enable_translation: bool = True):
    """Interactive demo mode with optional auto-translation."""
    from klareco import SemanticPipeline, Retriever

    print_header("Interactive Mode")

    # Load models
    try:
        pipeline = SemanticPipeline.load()
        print("  Pipeline loaded.")
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return

    retriever = None
    try:
        retriever = Retriever.load()
        print(f"  {retriever}")
    except Exception:
        print("  Retriever not available (no index)")

    translation_available = False
    if enable_translation:
        print("\n  Checking translation models...")
        # Try loading translators in background
        try:
            from transformers import MarianMTModel
            translation_available = True
            print("  Translation: available (EN↔EO)")
        except ImportError:
            print("  Translation: not available (install transformers)")

    print("\nCommands:")
    print("  <text>          - Embed and show EnrichedAST (auto-translates EN→EO)")
    print("  search <text>   - Search corpus (auto-translates EN→EO)")
    print("  translate <text>- Translate between EN↔EO")
    print("  en <text>       - Force translate to English")
    print("  eo <text>       - Force translate to Esperanto")
    print("  quit            - Exit")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGis revido!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Gis revido!")
            break

        # Translation commands
        if user_input.lower().startswith('translate '):
            text = user_input[10:]
            lang = detect_language(text)
            if lang == 'en':
                result = translate_to_esperanto(text)
                if result:
                    print(f"  EN: {text}")
                    print(f"  EO: {result}")
                else:
                    print("  (Translation not available)")
            else:
                result = translate_to_english(text)
                if result:
                    print(f"  EO: {text}")
                    print(f"  EN: {result}")
                else:
                    print("  (Translation not available)")
            print()
            continue

        if user_input.lower().startswith('en '):
            text = user_input[3:]
            result = translate_to_english(text)
            if result:
                print(f"  → EN: {result}")
            else:
                print("  (Translation not available)")
            print()
            continue

        if user_input.lower().startswith('eo '):
            text = user_input[3:]
            result = translate_to_esperanto(text)
            if result:
                print(f"  → EO: {result}")
            else:
                print("  (Translation not available)")
            print()
            continue

        # Search command with auto-translation
        if user_input.lower().startswith('search '):
            if retriever is None:
                print("  Retriever not available")
                continue

            query = user_input[7:]

            # Auto-translate if English
            if translation_available:
                eo_query, src_lang, was_translated = auto_translate(query, target='eo')
                if was_translated:
                    print(f"  [Translated EN→EO: '{eo_query}']")
                    query = eo_query

            results = retriever.search(query, top_k=5)
            if not results:
                print("  (No results)")
            else:
                print(f"\n  Results for: '{query}'")
                for i, r in enumerate(results, 1):
                    text = r.text[:60] + "..." if len(r.text) > 60 else r.text
                    print(f"  {i}. [{r.score:.3f}] {text}")

                    # Optionally translate results to English
                    if translation_available and src_lang == 'en':
                        en_text = translate_to_english(r.text)
                        if en_text:
                            en_short = en_text[:55] + "..." if len(en_text) > 55 else en_text
                            print(f"       EN: {en_short}")
            print()
            continue

        # Default: embed and show EnrichedAST (auto-translate if English)
        text_to_embed = user_input
        was_translated = False

        if translation_available:
            eo_text, src_lang, was_translated = auto_translate(user_input, target='eo')
            if was_translated:
                print(f"  [Translated EN→EO: '{eo_text}']")
                text_to_embed = eo_text

        enriched = pipeline.for_retrieval(text_to_embed)
        print(f"\n  {enriched}")
        print(f"  Known roots: {enriched.known_roots}")
        print(f"  Unknown roots: {enriched.unknown_roots}")
        if enriched.sentence_embedding is not None:
            emb = enriched.sentence_embedding
            print(f"  Embedding: shape={emb.shape}, norm={emb.norm():.3f}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1.5 Demo with Translation")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Run in interactive mode")
    parser.add_argument('--translate', '-t', type=str, metavar='TEXT',
                        help="Translate text (auto-detects direction)")
    parser.add_argument('--en', type=str, metavar='TEXT',
                        help="Translate Esperanto to English")
    parser.add_argument('--eo', type=str, metavar='TEXT',
                        help="Translate English to Esperanto")
    parser.add_argument('--no-translate', action='store_true',
                        help="Disable auto-translation in interactive mode")
    parser.add_argument('--skip-translation-demo', action='store_true',
                        help="Skip translation demo (faster startup)")
    args = parser.parse_args()

    print("=" * 70)
    print(" KLARECO PHASE 1.5 DEMO")
    print(" EnrichedAST, SemanticPipeline, Retriever, and Translation")
    print("=" * 70)
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Handle one-shot translation commands
    if args.translate:
        lang = detect_language(args.translate)
        if lang == 'en':
            result = translate_to_esperanto(args.translate)
            if result:
                print(f"\n  EN: {args.translate}")
                print(f"  EO: {result}")
            else:
                print("  Translation not available")
        else:
            result = translate_to_english(args.translate)
            if result:
                print(f"\n  EO: {args.translate}")
                print(f"  EN: {result}")
            else:
                print("  Translation not available")
        return

    if args.en:
        result = translate_to_english(args.en)
        if result:
            print(f"\n  EO: {args.en}")
            print(f"  EN: {result}")
        else:
            print("  Translation not available")
        return

    if args.eo:
        result = translate_to_esperanto(args.eo)
        if result:
            print(f"\n  EN: {args.eo}")
            print(f"  EO: {result}")
        else:
            print("  Translation not available")
        return

    if args.interactive:
        interactive_mode(enable_translation=not args.no_translate)
    else:
        # Run all demos
        demo_enriched_ast()
        demo_semantic_pipeline()
        demo_retriever()
        demo_serialization()
        demo_pipeline_hooks()

        if not args.skip_translation_demo:
            demo_translation()
            demo_grammatical_adjuster()
            demo_thought_decoder()

        print_header("Summary")
        print("""
Phase 1.5 Components:

1. EnrichedAST
   - Container for parser AST + learned embeddings
   - Immutable progression: each stage creates new instance
   - Fully serializable for storage

2. SemanticPipeline
   - Chains Stage 0 (parser) → Stage 1 (semantic) → Stage 2 (grammatical)
   - Convenience methods: for_retrieval(), for_qa(), for_analysis()
   - Supports hooks for debugging/visualization

3. Retriever
   - Uses SemanticPipeline for query embedding
   - FAISS index for fast similarity search
   - Lazy enrichment: enrich results on-demand

4. Translation
   - Automatic EN↔EO using Helsinki-NLP/opus-mt models
   - Language detection for auto-translation
   - English speakers can query Esperanto corpus seamlessly

5. GrammaticalAdjuster (Stage 2 - DETERMINISTIC)
   - Adjusts semantic similarity using AST grammatical annotations
   - Zero learned parameters - pure rules from parsed grammar
   - Handles: negation, tense, mood, sentence type
   - Example: "La kato dormas" vs "La kato ne dormas" → 1.0 → -0.8

6. ThoughtDecoder
   - Decodes EnrichedAST into human-readable explanations
   - Explains syntactic structure and semantic content
   - Compares sentences for similarity analysis
   - JSON output for API integration

Usage:
  from klareco import (EnrichedAST, SemanticPipeline, Retriever,
                       ThoughtDecoder, GrammaticalAdjuster)

  # Quick embedding
  pipeline = SemanticPipeline.load()
  enriched = pipeline.for_retrieval("La hundo kuras.")
  embedding = enriched.get_effective_embedding()

  # Search (works with English too via auto-translation)
  retriever = Retriever.load()
  results = retriever.search("Kio estas Esperanto?", top_k=10)

  # Grammatical adjustment (Stage 2)
  adjuster = GrammaticalAdjuster()
  adjusted_sim = adjuster.adjust(enriched1, enriched2, semantic_sim)

  # Decode thoughts (explainable AI)
  decoder = ThoughtDecoder(pipeline=pipeline)
  decoded = decoder.decode(enriched)
  print(decoded)  # Human-readable explanation

Translation CLI:
  python demo_phase15.py --translate "The dog runs fast"
  python demo_phase15.py --eo "Hello world"
  python demo_phase15.py --en "Saluton mondo"
  python demo_phase15.py -i   # Interactive with auto-translation
""")

        print("\nRun with --interactive for live testing with translation!")


if __name__ == '__main__':
    main()
