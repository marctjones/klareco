"""
ThoughtDecoder: Decode EnrichedAST internal representations into human-readable explanations.

This module provides tools to "read" the thoughts encoded in an EnrichedAST,
making the model's understanding transparent and explainable.

The decoder can:
1. Explain syntactic structure (from Stage 0 parser)
2. Describe semantic content (from Stage 1 embeddings)
3. Find semantically similar concepts (via embedding similarity)
4. Generate natural language explanations of understanding

Usage:
    from klareco import SemanticPipeline
    from klareco.thought_decoder import ThoughtDecoder

    pipeline = SemanticPipeline.load()
    decoder = ThoughtDecoder(pipeline)

    enriched = pipeline.for_retrieval("La hundo kuras rapide.")
    explanation = decoder.decode(enriched)
    print(explanation)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DecodedThought:
    """A structured explanation of what the model understood."""

    # Original text
    original_text: str

    # Syntactic analysis (Stage 0)
    sentence_type: str  # deklaro, demando, ordono
    is_negated: bool
    tense: Optional[str]  # pasinteco, prezenco, futuro

    # Semantic roles
    subject: Optional[str]
    verb: Optional[str]
    object: Optional[str]
    modifiers: List[str]

    # Morphological breakdown
    content_words: List[Dict[str, Any]]

    # Semantic understanding (Stage 1)
    known_concepts: List[str]  # Recognized roots
    unknown_concepts: List[str]  # Unknown roots
    embedding_quality: float  # 0-1, how well understood

    # Similar concepts (from embeddings)
    similar_sentences: List[Tuple[str, float]]  # (text, similarity)

    # Natural language explanation
    explanation_eo: str  # Esperanto explanation
    explanation_en: str  # English explanation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'original_text': self.original_text,
            'sentence_type': self.sentence_type,
            'is_negated': self.is_negated,
            'tense': self.tense,
            'subject': self.subject,
            'verb': self.verb,
            'object': self.object,
            'modifiers': self.modifiers,
            'content_words': self.content_words,
            'known_concepts': self.known_concepts,
            'unknown_concepts': self.unknown_concepts,
            'embedding_quality': self.embedding_quality,
            'similar_sentences': self.similar_sentences,
            'explanation_eo': self.explanation_eo,
            'explanation_en': self.explanation_en,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = []
        lines.append(f"Input: {self.original_text}")
        lines.append("")
        lines.append("=== Syntactic Understanding (Stage 0) ===")
        lines.append(f"  Type: {self.sentence_type}")
        lines.append(f"  Negated: {self.is_negated}")
        if self.tense:
            lines.append(f"  Tense: {self.tense}")
        if self.subject:
            lines.append(f"  Subject: {self.subject}")
        if self.verb:
            lines.append(f"  Verb: {self.verb}")
        if self.object:
            lines.append(f"  Object: {self.object}")
        if self.modifiers:
            lines.append(f"  Modifiers: {', '.join(self.modifiers)}")

        lines.append("")
        lines.append("=== Semantic Understanding (Stage 1) ===")
        lines.append(f"  Known concepts: {', '.join(self.known_concepts) or '(none)'}")
        if self.unknown_concepts:
            lines.append(f"  Unknown concepts: {', '.join(self.unknown_concepts)}")
        lines.append(f"  Embedding quality: {self.embedding_quality:.0%}")

        if self.similar_sentences:
            lines.append("")
            lines.append("=== Similar Sentences ===")
            for text, sim in self.similar_sentences[:3]:
                short_text = text[:50] + "..." if len(text) > 50 else text
                lines.append(f"  [{sim:.3f}] {short_text}")

        lines.append("")
        lines.append("=== Explanation ===")
        lines.append(f"  EO: {self.explanation_eo}")
        lines.append(f"  EN: {self.explanation_en}")

        return "\n".join(lines)


class ThoughtDecoder:
    """
    Decodes EnrichedAST internal representations into human-readable explanations.
    """

    # Tense translations
    TENSE_MAP = {
        'pasinteco': ('past', 'pasinteco'),
        'prezenco': ('present', 'prezenco'),
        'futuro': ('future', 'futuro'),
    }

    # Sentence type translations
    TYPE_MAP = {
        'deklaro': ('statement', 'deklara frazo'),
        'demando': ('question', 'demanda frazo'),
        'ordono': ('command', 'ordona frazo'),
    }

    # Mood translations
    MOOD_MAP = {
        'indikativo': ('indicative', 'indikativo'),
        'kondiĉa': ('conditional', 'kondiĉa modo'),
        'vola': ('imperative', 'vola modo'),
        'infinitivo': ('infinitive', 'infinitivo'),
    }

    def __init__(self, pipeline=None, retriever=None):
        """
        Initialize the decoder.

        Args:
            pipeline: SemanticPipeline for embedding similar sentences
            retriever: Retriever for finding similar sentences in corpus
        """
        self.pipeline = pipeline
        self.retriever = retriever
        self._translator = None

    def _get_translator(self):
        """Lazy load the EO->EN translator."""
        if self._translator is not None:
            return self._translator

        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-eo-en"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            self._translator = (model, tokenizer)
            return self._translator
        except Exception:
            return None

    def _translate_to_english(self, text: str) -> str:
        """Translate Esperanto to English."""
        translator = self._get_translator()
        if translator is None:
            return "(translation unavailable)"

        model, tokenizer = translator
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=100, num_beams=4)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"(translation error: {e})"

    def _extract_word_info(self, word_ast: Dict) -> Dict[str, Any]:
        """Extract information from a word AST."""
        if not isinstance(word_ast, dict) or word_ast.get('tipo') != 'vorto':
            return {}

        prefixes = word_ast.get('prefiksoj', [])
        if not prefixes:
            old_prefix = word_ast.get('prefikso')
            prefixes = [old_prefix] if old_prefix else []

        return {
            'root': word_ast.get('radiko', ''),
            'prefixes': prefixes,
            'suffixes': word_ast.get('sufiksoj', []),
            'pos': word_ast.get('vortspeco', ''),
            'tense': word_ast.get('tempo'),
            'mood': word_ast.get('modo'),
            'number': word_ast.get('nombro'),
            'case': word_ast.get('kazo'),
        }

    def _extract_phrase_root(self, phrase_ast: Optional[Dict]) -> Optional[str]:
        """Extract the root from a phrase AST (subjekto, objekto, etc.)."""
        if not phrase_ast:
            return None

        if phrase_ast.get('tipo') == 'vorto':
            return phrase_ast.get('radiko')

        kerno = phrase_ast.get('kerno', {})
        if kerno.get('tipo') == 'vorto':
            return kerno.get('radiko')

        return None

    def _build_word_string(self, word_info: Dict) -> str:
        """Build a displayable word string from word info."""
        parts = []
        if word_info.get('prefixes'):
            parts.append(f"[{'+'.join(word_info['prefixes'])}]")
        parts.append(word_info.get('root', '?'))
        if word_info.get('suffixes'):
            parts.append(f"[{'+'.join(word_info['suffixes'])}]")
        return ''.join(parts)

    def _generate_explanation(self, enriched) -> Tuple[str, str]:
        """Generate natural language explanations in EO and EN."""
        # Build Esperanto explanation
        parts_eo = []

        # Sentence type
        fraztipo = enriched.fraztipo
        if fraztipo == 'demando':
            parts_eo.append("Ĉi tio estas demando")
        elif fraztipo == 'ordono':
            parts_eo.append("Ĉi tio estas ordono")
        else:
            parts_eo.append("Ĉi tio estas deklaro")

        # Negation
        if enriched.negita:
            parts_eo.append("kun negacio")

        # Subject and verb
        subj = self._extract_phrase_root(enriched.subjekto)
        verb = enriched.verbo.get('radiko') if enriched.verbo else None
        obj = self._extract_phrase_root(enriched.objekto)

        if subj and verb:
            if obj:
                parts_eo.append(f"kie '{subj}' {verb}-as '{obj}'")
            else:
                parts_eo.append(f"kie '{subj}' {verb}-as")
        elif verb:
            parts_eo.append(f"kun verbo '{verb}'")

        # Tense
        tempo = enriched.tempo
        if tempo == 'pasinteco':
            parts_eo.append("en la pasinteco")
        elif tempo == 'futuro':
            parts_eo.append("en la estonteco")

        explanation_eo = " ".join(parts_eo) + "."

        # Translate to English
        explanation_en = self._translate_to_english(explanation_eo)

        return explanation_eo, explanation_en

    def decode(self, enriched, find_similar: bool = True, top_k: int = 3) -> DecodedThought:
        """
        Decode an EnrichedAST into a human-readable explanation.

        Args:
            enriched: The EnrichedAST to decode
            find_similar: Whether to search for similar sentences
            top_k: Number of similar sentences to find

        Returns:
            DecodedThought with full explanation
        """
        # Extract syntactic info
        sentence_type = enriched.fraztipo
        is_negated = enriched.negita
        tense = enriched.tempo

        # Extract semantic roles
        subject = self._extract_phrase_root(enriched.subjekto)
        verb_ast = enriched.verbo
        verb = verb_ast.get('radiko') if verb_ast else None
        obj = self._extract_phrase_root(enriched.objekto)

        # Extract modifiers
        modifiers = []
        for ali in enriched.aliaj:
            if isinstance(ali, dict):
                root = ali.get('radiko') or self._extract_phrase_root(ali)
                if root:
                    modifiers.append(root)

        # Extract content words
        content_words = []
        for word in enriched.get_content_words():
            info = self._extract_word_info(word)
            if info.get('root'):
                content_words.append({
                    'display': self._build_word_string(info),
                    'root': info['root'],
                    'pos': info['pos'],
                    'prefixes': info['prefixes'],
                    'suffixes': info['suffixes'],
                })

        # Semantic understanding
        known_concepts = list(enriched.known_roots)
        unknown_concepts = list(enriched.unknown_roots)

        # Calculate embedding quality
        total_roots = len(known_concepts) + len(unknown_concepts)
        embedding_quality = len(known_concepts) / total_roots if total_roots > 0 else 0.0

        # Find similar sentences
        similar_sentences = []
        if find_similar and self.retriever and enriched.sentence_embedding is not None:
            try:
                results = self.retriever.search(enriched.original_text, top_k=top_k)
                for r in results:
                    # Skip exact match
                    if r.text.strip() != enriched.original_text.strip():
                        similar_sentences.append((r.text, r.score))
            except Exception:
                pass

        # Generate explanations
        explanation_eo, explanation_en = self._generate_explanation(enriched)

        return DecodedThought(
            original_text=enriched.original_text,
            sentence_type=sentence_type,
            is_negated=is_negated,
            tense=tense,
            subject=subject,
            verb=verb,
            object=obj,
            modifiers=modifiers,
            content_words=content_words,
            known_concepts=known_concepts,
            unknown_concepts=unknown_concepts,
            embedding_quality=embedding_quality,
            similar_sentences=similar_sentences,
            explanation_eo=explanation_eo,
            explanation_en=explanation_en,
        )

    def decode_to_json(self, enriched, **kwargs) -> str:
        """Decode and return as JSON string."""
        decoded = self.decode(enriched, **kwargs)
        return json.dumps(decoded.to_dict(), ensure_ascii=False, indent=2)

    def compare_thoughts(self, enriched1, enriched2) -> Dict[str, Any]:
        """
        Compare two EnrichedASTs and explain similarities/differences.

        Returns a dict with comparison results.
        """
        if not TORCH_AVAILABLE:
            return {'error': 'torch not available'}

        result = {
            'text1': enriched1.original_text,
            'text2': enriched2.original_text,
            'syntactic': {},
            'semantic': {},
        }

        # Syntactic comparison
        result['syntactic']['same_type'] = enriched1.fraztipo == enriched2.fraztipo
        result['syntactic']['same_negation'] = enriched1.negita == enriched2.negita
        result['syntactic']['same_tense'] = enriched1.tempo == enriched2.tempo

        # Subject/verb/object comparison
        subj1 = self._extract_phrase_root(enriched1.subjekto)
        subj2 = self._extract_phrase_root(enriched2.subjekto)
        result['syntactic']['same_subject'] = subj1 == subj2
        result['syntactic']['subjects'] = (subj1, subj2)

        verb1 = enriched1.verbo.get('radiko') if enriched1.verbo else None
        verb2 = enriched2.verbo.get('radiko') if enriched2.verbo else None
        result['syntactic']['same_verb'] = verb1 == verb2
        result['syntactic']['verbs'] = (verb1, verb2)

        # Semantic comparison
        emb1 = enriched1.sentence_embedding
        emb2 = enriched2.sentence_embedding

        if emb1 is not None and emb2 is not None:
            similarity = F.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
            result['semantic']['embedding_similarity'] = similarity

            # Interpret similarity
            if similarity > 0.9:
                result['semantic']['interpretation'] = 'Nearly identical meaning'
            elif similarity > 0.7:
                result['semantic']['interpretation'] = 'Very similar meaning'
            elif similarity > 0.5:
                result['semantic']['interpretation'] = 'Related meaning'
            elif similarity > 0.2:
                result['semantic']['interpretation'] = 'Weakly related'
            elif similarity > -0.2:
                result['semantic']['interpretation'] = 'Unrelated'
            else:
                result['semantic']['interpretation'] = 'Opposite meaning'

        # Shared concepts
        shared = enriched1.known_roots & enriched2.known_roots
        only1 = enriched1.known_roots - enriched2.known_roots
        only2 = enriched2.known_roots - enriched1.known_roots

        result['semantic']['shared_concepts'] = list(shared)
        result['semantic']['only_in_first'] = list(only1)
        result['semantic']['only_in_second'] = list(only2)

        return result
