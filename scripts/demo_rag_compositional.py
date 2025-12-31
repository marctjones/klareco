#!/usr/bin/env python3
"""
Extended RAG Demo with Compositional Embeddings.

Demonstrates semantic search using the Stage 1 models:
- Root embeddings (11,121 roots × 64d)
- Affix transforms (low-rank matrices for prefixes/suffixes)

Unlike traditional word embeddings, this approach composes words morphologically:
  malbona = mal-(transform) → bon(root) → meaning
  lernejo = lern(root) → -ej(transform) → place of learning

This enables finding semantically similar sentences even with morphologically
complex Esperanto words.

Now includes English translation using Helsinki-NLP/opus-mt-eo-en model.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Global translators (lazy loaded)
_translator_eo_en = None  # Esperanto → English
_translator_en_eo = None  # English → Esperanto
_translator_loading = False

# Global model info (populated at load time)
_model_info = {}


def get_model_info(model_path: Path) -> dict:
    """Get model file info including modification time and version detection."""
    info = {
        'path': str(model_path),
        'name': model_path.name,
        'exists': model_path.exists(),
        'modified': None,
        'version': None,
        'size_kb': None,
    }

    if model_path.exists():
        stat = model_path.stat()
        info['modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        info['size_kb'] = stat.st_size // 1024

        # Detect version from path
        if '_v2' in str(model_path) or 'v2' in model_path.parent.name:
            info['version'] = 'V2'
        elif '_v3' in str(model_path) or 'v3' in model_path.parent.name:
            info['version'] = 'V3'
        else:
            info['version'] = 'V1'

    return info


def print_session_header(metadata: list, root_to_idx: dict, prefix_transforms: dict,
                        suffix_transforms: dict, root_model_path: Path,
                        affix_model_path: Path, index_dir: Path):
    """Print comprehensive session header with timestamp and model info."""
    global _model_info

    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    # Get model info
    root_info = get_model_info(root_model_path)
    affix_info = get_model_info(affix_model_path)

    # Get index info
    index_info = get_model_info(index_dir / "faiss_index.bin")

    # Store for reference
    _model_info = {
        'root': root_info,
        'affix': affix_info,
        'index': index_info,
        'timestamp': timestamp,
    }

    print(f"\n{'=' * 70}")
    print(f"Session: {timestamp}")
    print(f"{'=' * 70}")
    print(f"\nModels:")
    print(f"  Root embeddings:  {root_info['name']}")
    print(f"    Path: {root_info['path']}")
    print(f"    Modified: {root_info['modified']} | Size: {root_info['size_kb']}KB")
    print(f"    Stats: {len(root_to_idx):,} roots × 64d")
    print(f"  Affix transforms: {affix_info['name']} ({affix_info['version']})")
    print(f"    Path: {affix_info['path']}")
    print(f"    Modified: {affix_info['modified']} | Size: {affix_info['size_kb']}KB")
    print(f"    Stats: {len(prefix_transforms)} prefixes, {len(suffix_transforms)} suffixes")
    print(f"\nCorpus Index:")
    print(f"  Path: {index_dir}")
    print(f"  Sentences: {len(metadata):,}")
    print(f"  Index modified: {index_info['modified']}")


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
            print("Loading EO→EN translation model...")
            model_name = "Helsinki-NLP/opus-mt-eo-en"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            _translator_eo_en = (model, tokenizer)
            print("  EO→EN model loaded!")
            _translator_loading = False
            return _translator_eo_en
        else:
            print("Loading EN→EO translation model...")
            model_name = "Helsinki-NLP/opus-mt-en-eo"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            _translator_en_eo = (model, tokenizer)
            print("  EN→EO model loaded!")
            _translator_loading = False
            return _translator_en_eo

    except Exception as e:
        print(f"  Warning: Could not load translator: {e}")
        print("  Install with: pip install transformers sentencepiece")
        _translator_loading = False
        return None


def translate_to_english(text: str, max_length: int = 100) -> str:
    """Translate Esperanto text to English."""
    translator = get_translator("eo-en")

    if translator is None:
        return None

    model, tokenizer = translator

    try:
        import torch

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
        import torch

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

    # If more than 20% of words are English markers, it's probably English
    if len(words) > 0 and english_count / len(words) > 0.15:
        return 'en'

    # Esperanto word endings
    eo_endings = ['oj', 'aj', 'on', 'an', 'as', 'is', 'os', 'us', 'oj', 'ojn', 'ajn']
    eo_count = sum(1 for w in words if any(w.strip('.,!?').endswith(e) for e in eo_endings))

    if len(words) > 0 and eo_count / len(words) > 0.3:
        return 'eo'

    # Default to English if uncertain (user doesn't speak Esperanto)
    return 'en'


def translate_batch(texts: list, max_length: int = 100) -> list:
    """Translate a batch of texts for efficiency."""
    translator = get_translator()

    if translator is None:
        return [None] * len(texts)

    model, tokenizer = translator

    try:
        import torch

        # Tokenize batch
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                          max_length=512, padding=True)

        # Translate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        translations = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return translations
    except Exception as e:
        return [f"[translation error: {e}]"] * len(texts)


def load_index(index_dir: Path):
    """Load FAISS index and metadata."""
    import faiss

    print(f"Loading index from {index_dir}...")

    # Load embeddings
    embeddings = np.load(index_dir / "embeddings.npy")

    # Load FAISS index
    faiss_index = faiss.read_index(str(index_dir / "faiss_index.bin"))

    # Load metadata
    metadata = []
    with open(index_dir / "metadata.jsonl") as f:
        for line in f:
            metadata.append(json.loads(line))

    print(f"  Loaded {len(metadata):,} sentences")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    return embeddings, faiss_index, metadata


def load_root_only(root_model_path: Path):
    """Load only root embeddings (for testing baseline without affix transforms)."""
    import torch

    print("Loading root embeddings only...")

    root_checkpoint = torch.load(root_model_path, map_location='cpu', weights_only=False)
    root_to_idx = root_checkpoint['root_to_idx']
    embedding_dim = root_checkpoint['embedding_dim']

    state_dict = root_checkpoint['model_state_dict']
    root_embeddings = state_dict['embeddings.weight']

    print(f"  Root embeddings: {len(root_to_idx):,} roots x {embedding_dim}d")

    return root_embeddings, root_to_idx


def load_models(root_model_path: Path, affix_model_path: Path):
    """Load root embeddings and affix transforms."""
    import torch
    import torch.nn as nn

    print("Loading models...")

    # Load root embeddings
    root_checkpoint = torch.load(root_model_path, map_location='cpu', weights_only=False)
    root_to_idx = root_checkpoint['root_to_idx']
    embedding_dim = root_checkpoint['embedding_dim']
    vocab_size = root_checkpoint['vocab_size']

    # Extract embeddings from model_state_dict
    state_dict = root_checkpoint['model_state_dict']
    root_embeddings = state_dict['embeddings.weight']  # Shape: [vocab_size, embedding_dim]

    print(f"  Root embeddings: {len(root_to_idx):,} roots × {embedding_dim}d")

    # Load affix transforms
    affix_checkpoint = torch.load(affix_model_path, map_location='cpu', weights_only=False)
    affix_state_dict = affix_checkpoint['model_state_dict']
    rank = affix_checkpoint['rank']

    class LowRankTransform(nn.Module):
        def __init__(self, dim: int, rank: int = 4):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)

        def forward(self, x):
            return x + self.up(self.down(x))

    # Reconstruct transforms from state_dict
    prefix_transforms = {}
    suffix_transforms = {}

    for prefix in affix_checkpoint['prefixes']:
        t = LowRankTransform(embedding_dim, rank)
        t.down.weight.data = affix_state_dict[f'prefix_transforms.{prefix}.down.weight']
        t.up.weight.data = affix_state_dict[f'prefix_transforms.{prefix}.up.weight']
        t.eval()
        prefix_transforms[prefix] = t

    for suffix in affix_checkpoint['suffixes']:
        t = LowRankTransform(embedding_dim, rank)
        t.down.weight.data = affix_state_dict[f'suffix_transforms.{suffix}.down.weight']
        t.up.weight.data = affix_state_dict[f'suffix_transforms.{suffix}.up.weight']
        t.eval()
        suffix_transforms[suffix] = t

    print(f"  Affix transforms: {len(prefix_transforms)} prefixes, {len(suffix_transforms)} suffixes")

    return root_embeddings, root_to_idx, prefix_transforms, suffix_transforms


def embed_query(query: str, root_embeddings, root_to_idx, prefix_transforms, suffix_transforms):
    """Embed a query using compositional approach."""
    import torch

    # Simple morphological analysis for demo
    # In production, use the full parser
    from klareco.parser import parse

    ast = parse(query)

    def extract_words(node):
        """Extract word info from AST."""
        words = []
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                words.append({
                    'radiko': node.get('radiko', ''),
                    'prefiksoj': node.get('prefiksoj', []),
                    'sufiksoj': node.get('sufiksoj', []),
                    'text': node.get('teksto', '')
                })
            for v in node.values():
                words.extend(extract_words(v))
        elif isinstance(node, list):
            for item in node:
                words.extend(extract_words(item))
        return words

    words = extract_words(ast)

    # Embed each word and average
    word_embeddings = []

    for w in words:
        root = w['radiko']
        if not root or root not in root_to_idx:
            continue

        # Start with root embedding
        root_idx = root_to_idx[root]
        emb = root_embeddings[root_idx].clone()

        # Apply prefix transforms
        for prefix in w.get('prefiksoj', []):
            if prefix and prefix in prefix_transforms:
                emb = prefix_transforms[prefix](emb.unsqueeze(0)).squeeze(0)

        # Apply suffix transforms
        for suffix in w.get('sufiksoj', []):
            if suffix and suffix in suffix_transforms:
                emb = suffix_transforms[suffix](emb.unsqueeze(0)).squeeze(0)

        word_embeddings.append(emb.detach().numpy())

    if not word_embeddings:
        return None

    # Mean pooling
    query_embedding = np.mean(word_embeddings, axis=0)

    # Normalize for cosine similarity
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

    return query_embedding


def search(query: str, faiss_index, metadata, root_embeddings, root_to_idx,
           prefix_transforms, suffix_transforms, top_k: int = 10):
    """Search for similar sentences."""

    query_emb = embed_query(query, root_embeddings, root_to_idx,
                           prefix_transforms, suffix_transforms)

    if query_emb is None:
        return []

    # Search FAISS
    query_emb = query_emb.reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(metadata):
            result = metadata[idx].copy()
            result['score'] = float(score)
            results.append(result)

    return results


def demo_queries(faiss_index, metadata, root_embeddings, root_to_idx,
                prefix_transforms, suffix_transforms, translate: bool = True,
                root_model_path: Path = None, affix_model_path: Path = None,
                index_dir: Path = None):
    """Run demo queries to show RAG capabilities."""

    # Print comprehensive header
    if root_model_path and affix_model_path and index_dir:
        print_session_header(metadata, root_to_idx, prefix_transforms, suffix_transforms,
                           root_model_path, affix_model_path, index_dir)

    print("\n" + "=" * 70)
    print("KLARECO RAG DEMO - Compositional Embeddings")
    print("=" * 70)

    # Pre-load translator if needed
    if translate:
        get_translator()

    # Demo queries showcasing different knowledge domains and capabilities
    demo_cases = [
        # === WIKIPEDIA KNOWLEDGE ===
        {
            'category': 'Wikipedia: Geography',
            'query': 'Parizo estas la ĉefurbo de Francio.',
            'description': 'Paris is the capital of France'
        },
        {
            'category': 'Wikipedia: History',
            'query': 'La dua mondmilito finiĝis en 1945.',
            'description': 'World War II ended in 1945'
        },
        {
            'category': 'Wikipedia: Science',
            'query': 'La suno estas stelo en nia sunsistemo.',
            'description': 'The sun is a star in our solar system'
        },
        {
            'category': 'Wikipedia: Animals',
            'query': 'Leono estas granda karnomanganta besto.',
            'description': 'A lion is a large carnivorous animal'
        },
        # === EDGAR ALLAN POE ===
        {
            'category': 'Poe: The Raven',
            'query': 'La korvo diris neniam plu.',
            'description': 'The raven said nevermore (La Korvo)'
        },
        {
            'category': 'Poe: House of Usher',
            'query': 'La malnova domo estis malluma kaj terura.',
            'description': 'The old house was dark and terrible (Usher Domo)'
        },
        {
            'category': 'Poe: Pit and Pendulum',
            'query': 'La pendolo balanciĝis super li.',
            'description': 'The pendulum swung above him (Puto kaj Pendolo)'
        },
        # === CLASSIC LITERATURE ===
        {
            'category': 'Alice in Wonderland',
            'query': 'Alicio falis en profundan truon.',
            'description': 'Alice fell into a deep hole'
        },
        {
            'category': 'Wizard of Oz',
            'query': 'Doroteo iris al la lando de Oz.',
            'description': 'Dorothy went to the land of Oz'
        },
        {
            'category': 'The Time Machine',
            'query': 'Li vojaĝis tra la tempo per maŝino.',
            'description': 'He traveled through time by machine (H.G. Wells)'
        },
        {
            'category': 'Jekyll and Hyde',
            'query': 'Doktoro Jekyll transformiĝis en monstron.',
            'description': 'Dr. Jekyll transformed into a monster'
        },
        # === ESPERANTO HISTORY ===
        {
            'category': 'Zamenhof',
            'query': 'Zamenhof kreis Esperanton en Varsovio.',
            'description': 'Zamenhof created Esperanto in Warsaw'
        },
        {
            'category': 'Fundamento',
            'query': 'La Fundamento enhavas la bazajn regulojn.',
            'description': 'The Fundamento contains the basic rules'
        },
        # === MORPHOLOGY SHOWCASE ===
        {
            'category': 'Morphology: mal- prefix',
            'query': 'La malbona vetero daŭris dum semajno.',
            'description': 'mal- reverses meaning: bona→malbona (good→bad)'
        },
        {
            'category': 'Morphology: -ej suffix',
            'query': 'La infanoj lernas en la lernejo.',
            'description': '-ej means place: lerni→lernejo (learn→school)'
        },
        {
            'category': 'Morphology: -ist suffix',
            'query': 'La kuracisto helpas malsanulojn.',
            'description': '-ist means profession: kuraci→kuracisto (heal→doctor)'
        },
    ]

    for case in demo_cases:
        print(f"\n{'─' * 70}")
        query_text = case['query']

        # Translate query
        if translate:
            query_en = translate_to_english(query_text)
            print(f"[{case['category']}]")
            print(f"  EO: {query_text}")
            if query_en:
                print(f"  EN: {query_en}")
        else:
            print(f"[{case['category']}] {query_text}")

        print(f"  ({case['description']})")
        print()

        results = search(
            case['query'], faiss_index, metadata,
            root_embeddings, root_to_idx,
            prefix_transforms, suffix_transforms,
            top_k=5
        )

        if not results:
            print("  (No results - query words not in vocabulary)")
            continue

        # Collect texts for batch translation
        result_texts = [r['text'][:200] for r in results[:3]]  # Limit length for translation

        if translate:
            translations = translate_batch(result_texts)
        else:
            translations = [None] * len(result_texts)

        print("  Top matches:")
        for i, (r, trans) in enumerate(zip(results[:3], translations), 1):
            text = r['text']
            if len(text) > 70:
                text = text[:67] + "..."
            print(f"    {i}. [{r['score']:.3f}] {text}")
            if trans:
                if len(trans) > 70:
                    trans = trans[:67] + "..."
                print(f"       EN: {trans}")

    print(f"\n{'=' * 70}")


def interactive_mode(faiss_index, metadata, root_embeddings, root_to_idx,
                    prefix_transforms, suffix_transforms, translate: bool = True,
                    root_model_path: Path = None, affix_model_path: Path = None,
                    index_dir: Path = None):
    """Interactive query mode."""

    # Print comprehensive header
    if root_model_path and affix_model_path and index_dir:
        print_session_header(metadata, root_to_idx, prefix_transforms, suffix_transforms,
                           root_model_path, affix_model_path, index_dir)

    print("\n" + "=" * 70)
    print("KLARECO RAG - Interactive Mode")
    print("=" * 70)
    if translate:
        print("Translation: ENABLED")
        print("  - English queries will be translated to Esperanto for search")
        print("  - Esperanto results will be translated to English")
        # Pre-load both translators
        get_translator("eo-en")
    print("\nEnter queries in English OR Esperanto (or 'quit' to exit)")
    print()
    print("Example queries to try:")
    print("  Wikipedia:   'What is the capital of France?' or 'Tell me about lions'")
    print("  Poe:         'The raven said nevermore' or 'the dark old house'")
    print("  Literature:  'Alice fell down the hole' or 'Dorothy and the wizard'")
    print("  Esperanto:   'Zamenhof created Esperanto' or 'the Fundamento rules'")
    print("  Morphology:  'malbona' (bad), 'lernejo' (school), 'kuracisto' (doctor)")
    print()

    while True:
        try:
            query = input("Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! (Ĝis revido!)")
                break

            if not query:
                continue

            # Detect language and translate if needed
            search_query = query
            if translate:
                lang = detect_language(query)
                if lang == 'en':
                    # Translate English to Esperanto for search
                    eo_query = translate_to_esperanto(query)
                    if eo_query:
                        print(f"  Your query (EN): {query}")
                        print(f"  Searching (EO): {eo_query}")
                        search_query = eo_query
                    else:
                        print(f"  (Could not translate, searching as-is)")
                else:
                    # Already Esperanto - show English translation
                    en_query = translate_to_english(query)
                    if en_query:
                        print(f"  Query (EN): {en_query}")

            results = search(
                search_query, faiss_index, metadata,
                root_embeddings, root_to_idx,
                prefix_transforms, suffix_transforms,
                top_k=10
            )

            if not results:
                print("  (No results - query words not in vocabulary)\n")
                continue

            # Batch translate results
            result_texts = [r['text'][:200] for r in results[:5]]
            if translate:
                translations = translate_batch(result_texts)
            else:
                translations = [None] * len(result_texts)

            print(f"\nTop {min(5, len(results))} matches:")
            for i, (r, trans) in enumerate(zip(results[:5], translations), 1):
                text = r['text']
                if len(text) > 70:
                    text = text[:67] + "..."
                print(f"  {i}. [{r['score']:.3f}] {text}")
                if trans:
                    if len(trans) > 70:
                        trans = trans[:67] + "..."
                    print(f"     EN: {trans}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye! (Gis revido!)")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def show_morphology_demo(root_embeddings, root_to_idx, prefix_transforms, suffix_transforms):
    """Demonstrate morphological composition."""
    import torch

    print("\n" + "=" * 70)
    print("MORPHOLOGICAL COMPOSITION DEMO")
    print("=" * 70)
    print("\nShowing how affixes transform word embeddings:")

    # Demo words
    demos = [
        ('bon', [], [], 'bona (good)'),
        ('bon', ['mal'], [], 'malbona (bad) - mal- flips polarity'),
        ('lern', [], [], 'lerni (to learn)'),
        ('lern', [], ['ej'], 'lernejo (school) - -ej adds "place"'),
        ('lern', [], ['ant'], 'lernanto (student) - -ant adds "one who does"'),
        ('patr', [], [], 'patro (father)'),
        ('patr', [], ['in'], 'patrino (mother) - -in adds "female"'),
    ]

    print(f"\n{'Root':<8} {'Prefixes':<10} {'Suffixes':<10} {'Result':<30}")
    print("-" * 60)

    embeddings_list = []
    labels = []

    for root, prefixes, suffixes, description in demos:
        if root not in root_to_idx:
            continue

        # Get root embedding
        root_idx = root_to_idx[root]
        emb = root_embeddings[root_idx].clone()

        # Apply transforms
        for p in prefixes:
            if p in prefix_transforms:
                emb = prefix_transforms[p](emb.unsqueeze(0)).squeeze(0)
        for s in suffixes:
            if s in suffix_transforms:
                emb = suffix_transforms[s](emb.unsqueeze(0)).squeeze(0)

        embeddings_list.append(emb.detach().numpy())
        labels.append(description)

        prefix_str = ','.join(prefixes) if prefixes else '-'
        suffix_str = ','.join(suffixes) if suffixes else '-'
        print(f"{root:<8} {prefix_str:<10} {suffix_str:<10} {description:<30}")

    # Compute similarities
    if len(embeddings_list) >= 2:
        print("\nSimilarity matrix (cosine):")
        print(f"\n{'':>32}", end='')
        for i, label in enumerate(labels):
            short = label.split()[0][:8]
            print(f"{short:>10}", end='')
        print()

        for i, (emb_i, label_i) in enumerate(zip(embeddings_list, labels)):
            short = label_i.split()[0][:30]
            print(f"{short:>32}", end='')
            for j, emb_j in enumerate(embeddings_list):
                # Cosine similarity
                sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                print(f"{sim:>10.3f}", end='')
            print()

    print()


def main():
    parser = argparse.ArgumentParser(description="RAG Demo with Compositional Embeddings")
    parser.add_argument(
        '--index-dir',
        default='data/corpus_index_compositional',
        help='Path to FAISS index directory'
    )
    parser.add_argument(
        '--root-model',
        default='models/root_embeddings/best_model.pt',
        help='Path to root embeddings model'
    )
    parser.add_argument(
        '--affix-model',
        default='models/affix_transforms_v2/best_model.pt',
        help='Path to affix transforms model (use V2 for proper transform magnitudes)'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive query mode'
    )
    parser.add_argument(
        '--morphology',
        action='store_true',
        help='Show morphological composition demo'
    )
    parser.add_argument(
        '--no-translate',
        action='store_true',
        help='Disable English translations'
    )
    parser.add_argument(
        '--root-only',
        action='store_true',
        help='Use only root embeddings, skip affix transforms (for testing baseline)'
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Single query to run (if not interactive)'
    )

    args = parser.parse_args()
    translate = not args.no_translate

    # Convert to paths
    index_dir = Path(args.index_dir)
    root_model = Path(args.root_model)
    affix_model = Path(args.affix_model)

    # Check files exist
    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}")
        print("Run: ./scripts/run_compositional_indexing.sh")
        return 1

    if not root_model.exists():
        print(f"Error: Root model not found: {root_model}")
        print("Run: ./scripts/run_root_training.sh")
        return 1

    if not args.root_only and not affix_model.exists():
        print(f"Error: Affix model not found: {affix_model}")
        print("Run: ./scripts/run_affix_training.sh")
        return 1

    # Load everything
    embeddings, faiss_index, metadata = load_index(index_dir)

    if args.root_only:
        # Root-only mode: skip affix transforms
        print("ROOT-ONLY MODE: Skipping affix transforms")
        root_embeddings, root_to_idx = load_root_only(root_model)
        prefix_transforms = {}
        suffix_transforms = {}
    else:
        root_embeddings, root_to_idx, prefix_transforms, suffix_transforms = load_models(
            root_model, affix_model
        )

    if args.morphology:
        show_morphology_demo(root_embeddings, root_to_idx, prefix_transforms, suffix_transforms)
        return 0

    if args.interactive:
        interactive_mode(
            faiss_index, metadata, root_embeddings, root_to_idx,
            prefix_transforms, suffix_transforms, translate=translate,
            root_model_path=root_model, affix_model_path=affix_model,
            index_dir=index_dir
        )
    elif args.query:
        query = ' '.join(args.query)
        print(f"\nQuery (EO): {query}")

        # Translate query if enabled
        if translate:
            get_translator()  # Pre-load
            query_en = translate_to_english(query)
            if query_en:
                print(f"Query (EN): {query_en}")
        print()

        results = search(
            query, faiss_index, metadata,
            root_embeddings, root_to_idx,
            prefix_transforms, suffix_transforms,
            top_k=10
        )

        if not results:
            print("No results found.")
        else:
            # Batch translate results
            result_texts = [r['text'][:200] for r in results[:5]]
            if translate:
                translations = translate_batch(result_texts)
            else:
                translations = [None] * len(result_texts)

            print("Results:")
            for i, (r, trans) in enumerate(zip(results[:5], translations), 1):
                print(f"\n{i}. [{r['score']:.3f}]")
                print(f"   EO: {r['text']}")
                if trans:
                    print(f"   EN: {trans}")
    else:
        # Run demo
        demo_queries(
            faiss_index, metadata, root_embeddings, root_to_idx,
            prefix_transforms, suffix_transforms, translate=translate,
            root_model_path=root_model, affix_model_path=affix_model,
            index_dir=index_dir
        )

        # Also show morphology demo
        show_morphology_demo(root_embeddings, root_to_idx, prefix_transforms, suffix_transforms)

    return 0


if __name__ == '__main__':
    sys.exit(main())
