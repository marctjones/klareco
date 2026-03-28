"""
Semantic + Grammar Embeddings (190d = 120d semantic + 70d grammar)

This implements the hybrid embedding approach where:
- 120d semantic embedding is LEARNED from roots only
- 70d grammatical features are DETERMINISTIC (one-hot/multi-hot flags)

This maintains the core thesis: grammar is deterministic (from AST),
learned capacity focuses on semantics only.

VERSION: v2.1
COMPATIBLE WITH: v2.1 CompositionalEmbedding (uses root_embed only)
STAGE: Embeddings
"""

import numpy as np
import torch
from typing import Dict, List, Set


class SemanticPlusGrammarEmbedding:
    """
    Hybrid embedding combining learned semantic core with deterministic grammar flags.

    Output: 190d vector
    - [0:120] = Semantic embedding (learned from root only)
    - [120:190] = Grammar features (70d, deterministic)

    Grammar dimensions (70d total):
    - [120:125] = Word class (5d): noun, verb, adj, adv, other
    - [125:129] = Tense (4d): past, present, future, none
    - [129:132] = Mood (3d): infinitive, imperative, indicative
    - [132:133] = Number (1d): plural flag
    - [133:134] = Case (1d): accusative flag
    - [134:154] = Prefixes (20d): multi-hot for common prefixes
    - [154:184] = Suffixes (30d): multi-hot for common suffixes
    - [184:190] = Participle (6d): -int-, -ant-, -ont-, -it-, -at-, -ot-
    """

    # Grammar dimension boundaries
    WORD_CLASS_START = 120
    WORD_CLASS_END = 125
    TENSE_START = 125
    TENSE_END = 129
    MOOD_START = 129
    MOOD_END = 132
    NUMBER_START = 132
    NUMBER_END = 133
    CASE_START = 133
    CASE_END = 134
    PREFIX_START = 134
    PREFIX_END = 154
    SUFFIX_START = 154
    SUFFIX_END = 184
    PARTICIPLE_START = 184
    PARTICIPLE_END = 190

    # Total dimensions
    SEMANTIC_DIM = 120
    GRAMMAR_DIM = 70
    TOTAL_DIM = 190

    def __init__(self, compositional_embedding):
        """
        Args:
            compositional_embedding: CompositionalEmbedding model (v2.1)
                Must have root_embed layer and root_vocab.
        """
        self.comp_emb = compositional_embedding
        self.semantic_dim = self.SEMANTIC_DIM
        self.grammar_dim = self.GRAMMAR_DIM

        # Build prefix/suffix vocabularies for multi-hot encoding
        self.prefix_vocab = self._build_prefix_vocab()
        self.suffix_vocab = self._build_suffix_vocab()

    def _build_prefix_vocab(self) -> Dict[str, int]:
        """Build prefix vocabulary (top 20 prefixes)."""
        prefixes = [
            'mal', 'ge', 'pra', 'ek', 'dis', 'mis', 'bo', 'duon',
            'fi', 'ge', 'pra', 'vic', 'eks', 'ne', 'sen', 'kun',
            'inter', 'super', 'sub', 'trans'
        ]
        return {prefix: i for i, prefix in enumerate(prefixes[:20])}

    def _build_suffix_vocab(self) -> Dict[str, int]:
        """Build suffix vocabulary (top 30 suffixes)."""
        suffixes = [
            'ig', 'iĝ', 'ad', 'aĵ', 'an', 'ar', 'ebl', 'ec', 'eg',
            'ej', 'em', 'end', 'er', 'estr', 'et', 'id', 'ig', 'il',
            'in', 'ind', 'ing', 'ism', 'ist', 'obl', 'on', 'op', 'uj',
            'ul', 'um', 'ind'
        ]
        return {suffix: i for i, suffix in enumerate(suffixes[:30])}

    def embed_word(self, word_info: Dict) -> np.ndarray:
        """
        Embed a single word with semantic + grammar features.

        Args:
            word_info: Dict with keys:
                - root: str (required)
                - vortspeco: str (noun/verb/adj/adv)
                - tempo: str (pasinteco/estanteco/estonteco)
                - modo: str (infinitivo/imperativo/indicativo)
                - nombro: str (singularo/pluralo)
                - kazo: str (nominativo/akuzativo)
                - prefiksoj: List[str]
                - sufiksoj: List[str]

        Returns:
            190d numpy array: [120d semantic, 70d grammar]
        """
        # Semantic component (120d, learned)
        semantic_emb = self._embed_semantic(word_info['root'])

        # Grammar component (70d, deterministic)
        grammar_features = self._encode_grammar_features(word_info)

        # Concatenate
        full_emb = np.concatenate([semantic_emb, grammar_features])
        return full_emb.astype(np.float32)

    def _embed_semantic(self, root: str) -> np.ndarray:
        """
        Get semantic embedding for root (learned, 120d).

        Args:
            root: Root string (e.g., "hund", "esper")

        Returns:
            120d numpy array
        """
        if root not in self.comp_emb.root_vocab:
            # Unknown root → zero embedding
            return np.zeros(self.SEMANTIC_DIM, dtype=np.float32)

        root_idx = self.comp_emb.root_vocab[root]

        with torch.no_grad():
            root_tensor = torch.tensor([root_idx])
            emb = self.comp_emb.root_embed(root_tensor)
            semantic = emb.squeeze(0).detach().numpy()

        # Pad or truncate to 120d
        if semantic.shape[0] < self.SEMANTIC_DIM:
            semantic = np.pad(semantic, (0, self.SEMANTIC_DIM - semantic.shape[0]))
        elif semantic.shape[0] > self.SEMANTIC_DIM:
            semantic = semantic[:self.SEMANTIC_DIM]

        return semantic.astype(np.float32)

    def _encode_grammar_features(self, word_info: Dict) -> np.ndarray:
        """
        Encode deterministic grammar features (70d).

        Returns:
            70d numpy array with one-hot/multi-hot flags
        """
        features = np.zeros(self.GRAMMAR_DIM, dtype=np.float32)

        # Word class (5d one-hot)
        vortspeco = word_info.get('vortspeco', '')
        if vortspeco == 'substantivo':
            features[0] = 1.0  # is-noun
        elif vortspeco == 'verbo':
            features[1] = 1.0  # is-verb
        elif vortspeco == 'adjektivo':
            features[2] = 1.0  # is-adjective
        elif vortspeco == 'adverbo':
            features[3] = 1.0  # is-adverb
        else:
            features[4] = 1.0  # is-other

        # Tense (4d one-hot)
        tempo = word_info.get('tempo')
        if tempo == 'pasinteco':
            features[5] = 1.0  # past
        elif tempo == 'estanteco':
            features[6] = 1.0  # present
        elif tempo == 'estonteco':
            features[7] = 1.0  # future
        else:
            features[8] = 1.0  # no-tense (or N/A)

        # Mood (3d one-hot)
        modo = word_info.get('modo')
        if modo == 'infinitivo':
            features[9] = 1.0  # infinitive
        elif modo == 'imperativo':
            features[10] = 1.0  # imperative
        elif modo == 'indikativo':
            features[11] = 1.0  # indicative

        # Number (1d flag)
        if word_info.get('nombro') == 'pluralo':
            features[12] = 1.0  # plural

        # Case (1d flag)
        if word_info.get('kazo') == 'akuzativo':
            features[13] = 1.0  # accusative

        # Prefixes (20d multi-hot)
        prefixes = word_info.get('prefiksoj', []) or []
        for prefix in prefixes:
            if prefix in self.prefix_vocab:
                idx = self.prefix_vocab[prefix]
                features[14 + idx] = 1.0

        # Suffixes (30d multi-hot)
        suffixes = word_info.get('sufiksoj', []) or []
        for suffix in suffixes:
            if suffix in self.suffix_vocab:
                idx = self.suffix_vocab[suffix]
                features[34 + idx] = 1.0

        # Participle features (6d flags)
        # Check if any suffix is a participle marker
        participle_map = {
            'int': 0,  # active past participle
            'ant': 1,  # active present participle
            'ont': 2,  # active future participle
            'it': 3,   # passive past participle
            'at': 4,   # passive present participle
            'ot': 5,   # passive future participle
        }
        for suffix in suffixes:
            if suffix in participle_map:
                features[64 + participle_map[suffix]] = 1.0

        return features

    def embed_ast(self, ast: Dict) -> np.ndarray:
        """
        Embed entire AST using semantic + grammar features.

        Args:
            ast: Parsed AST dictionary

        Returns:
            190d numpy array (mean pooling over all words)
        """
        if not ast or 'parse_statistics' not in ast:
            return np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # Extract words with morphology
        words = self._extract_words_with_morphology(ast)

        if not words:
            return np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # Embed each word
        word_embeddings = []
        for word_info in words:
            word_emb = self.embed_word(word_info)
            word_embeddings.append(word_emb)

        # Mean pooling
        mean_emb = np.mean(word_embeddings, axis=0)
        return mean_emb.astype(np.float32)

    def _extract_words_with_morphology(self, ast: Dict) -> List[Dict]:
        """
        Extract words from AST with full morphological information.

        Returns:
            List of dicts with keys: root, vortspeco, tempo, modo, nombro, kazo,
                                     prefiksoj, sufiksoj
        """
        words = []

        def traverse(node):
            if not node or not isinstance(node, dict):
                return

            # Get word node
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '').lower()

                # Skip function words
                if root and root not in ['ki', 'kiu', 'kio', 'kie', 'kiam', 'kiel', 'la', 'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili']:
                    word_info = {
                        'root': root,
                        'vortspeco': node.get('vortspeco', ''),
                        'tempo': node.get('tempo'),
                        'modo': node.get('modo'),
                        'nombro': node.get('nombro', 'singularo'),
                        'kazo': node.get('kazo', 'nominativo'),
                        'prefiksoj': node.get('prefiksoj', []) or [],
                        'sufiksoj': node.get('sufiksoj', []) or [],
                    }
                    words.append(word_info)

            # Traverse structure
            for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                traverse(node.get(key))

            # Traverse lists
            for item in node.get('priskriboj', []) + node.get('aliaj', []):
                traverse(item)

        traverse(ast)
        return words

    def semantic_only(self, embedding: np.ndarray) -> np.ndarray:
        """
        Extract semantic component only (120d).

        Useful for semantic-only similarity queries.
        """
        return embedding[:self.SEMANTIC_DIM]

    def grammar_only(self, embedding: np.ndarray) -> np.ndarray:
        """
        Extract grammar component only (70d).

        Useful for grammar-specific filtering.
        """
        return embedding[self.SEMANTIC_DIM:]

    def semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute semantic similarity (ignoring grammar).

        Returns:
            Cosine similarity of semantic components only
        """
        sem1 = self.semantic_only(emb1)
        sem2 = self.semantic_only(emb2)

        # Cosine similarity
        norm1 = np.linalg.norm(sem1)
        norm2 = np.linalg.norm(sem2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(sem1, sem2) / (norm1 * norm2))

    def grammar_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute grammar similarity (ignoring semantics).

        Returns:
            Cosine similarity of grammar components only
        """
        gram1 = self.grammar_only(emb1)
        gram2 = self.grammar_only(emb2)

        # Cosine similarity
        norm1 = np.linalg.norm(gram1)
        norm2 = np.linalg.norm(gram2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(gram1, gram2) / (norm1 * norm2))
