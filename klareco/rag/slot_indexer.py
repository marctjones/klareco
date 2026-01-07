"""
Slot-based indexer that preserves AST structure for better retrieval.

Instead of averaging all word embeddings into a single sentence vector,
this indexer stores separate embeddings for each syntactic role (SUBJ, VERB, OBJ).
This enables role-aware matching and partial query support.

Example:
    Query: "Kiu kreis Esperanton?" (Who created Esperanto?)
    - SUBJ: kiu (question word - unknown)
    - VERB: kreis (created - must match)
    - OBJ: Esperanton (Esperanto - must match)

    Matches documents where VERB + OBJ match, returns subject as answer.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from klareco.parser import parse
from klareco.embeddings.hybrid_embeddings import HybridEmbeddings

logger = logging.getLogger(__name__)


class SlotBasedIndexer:
    """Build slot-based index preserving AST structure."""

    def __init__(
        self,
        root_model_path: Path,
        affix_model_path: Path,
        output_dir: Path,
        batch_size: int = 100,
        topical_model_path: Optional[Path] = None,
        use_hybrid: bool = False,
    ):
        self.root_model_path = root_model_path
        self.affix_model_path = affix_model_path
        self.topical_model_path = topical_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.use_hybrid = use_hybrid

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        self.device = torch.device('cpu')

        if use_hybrid and topical_model_path:
            # Load hybrid embeddings (linguistic + topical)
            self.hybrid_model = HybridEmbeddings.from_checkpoints(
                linguistic_checkpoint=root_model_path,
                topical_checkpoint=topical_model_path,
                pad_missing=True,
                default_mode='hybrid'
            )
            self.embedding_dim = 128  # 64d linguistic + 64d topical
            self.root_to_idx = self.hybrid_model.topical_model.root_to_idx  # Use larger topical vocab
            # For fallback embeddings
            self.avg_root_embedding = torch.zeros(self.embedding_dim)
        else:
            # Load linguistic embeddings only (legacy)
            self.hybrid_model = None
            self.root_emb, self.root_to_idx, self.embedding_dim = self._load_root_model()
            self.avg_root_embedding = self.root_emb.mean(dim=0)

        self.prefix_transforms, self.suffix_transforms = self._load_affix_transforms()

        # Function words (excluded from embeddings)
        self.function_words = {
            'la', 'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni', 'si',
            'mia', 'via', 'lia', 'ŝia', 'ĝia', 'nia', 'ilia', 'sia',
            'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sur', 'sub',
            'kaj', 'aŭ', 'sed', 'nek', 'ke', 'ĉar', 'se', 'dum', 'kvankam',
            'ne', 'ankaŭ', 'nur', 'eĉ', 'ja', 'jen', 'tre', 'pli', 'plej', 'tro',
        }

    def _load_root_model(self) -> Tuple[torch.Tensor, Dict[str, int], int]:
        """Load root embeddings from checkpoint."""
        logger.info(f"Loading root embeddings from {self.root_model_path}")
        checkpoint = torch.load(self.root_model_path, map_location='cpu', weights_only=False)
        embeddings = checkpoint['model_state_dict']['embeddings.weight']
        root_to_idx = checkpoint['root_to_idx']
        dim = embeddings.shape[1]
        logger.info(f"  Loaded {len(root_to_idx)} roots, {dim}d")
        return embeddings, root_to_idx, dim

    def _load_affix_transforms(self) -> Tuple[Dict, Dict]:
        """Load affix transformation matrices."""
        logger.info(f"Loading affix transforms from {self.affix_model_path}")
        checkpoint = torch.load(self.affix_model_path, map_location='cpu', weights_only=False)

        rank = checkpoint['rank']
        prefixes = checkpoint['prefixes']
        suffixes = checkpoint['suffixes']
        state_dict = checkpoint['model_state_dict']

        prefix_transforms = {}
        for p in prefixes:
            transform = self._create_low_rank_transform(rank, state_dict, f'prefix_transforms.{p}')
            prefix_transforms[p] = transform

        suffix_transforms = {}
        for s in suffixes:
            transform = self._create_low_rank_transform(rank, state_dict, f'suffix_transforms.{s}')
            suffix_transforms[s] = transform

        logger.info(f"  Loaded {len(prefixes)} prefix, {len(suffixes)} suffix transforms")
        return prefix_transforms, suffix_transforms

    def _create_low_rank_transform(self, rank: int, state_dict: Dict, prefix: str) -> torch.nn.Module:
        """Create a low-rank transform from checkpoint."""
        import torch.nn as nn

        class LowRankTransform(nn.Module):
            def __init__(self, dim: int, rank: int):
                super().__init__()
                self.down = nn.Linear(dim, rank, bias=False)
                self.up = nn.Linear(rank, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.up(self.down(x))

        transform = LowRankTransform(self.embedding_dim, rank)
        transform.down.weight.data = state_dict[f'{prefix}.down.weight']
        transform.up.weight.data = state_dict[f'{prefix}.up.weight']
        transform.eval()
        return transform

    def _char_hash_embedding(self, root: str) -> torch.Tensor:
        """Fallback embedding for unknown roots using character trigrams."""
        root_lower = root.lower()
        padded = f"^{root_lower}$"
        trigram_hashes = []
        for i in range(len(padded) - 2):
            trigram = padded[i:i+3]
            h = hash(trigram) % self.embedding_dim
            trigram_hashes.append(h)

        emb = torch.zeros(self.embedding_dim)
        for h in trigram_hashes:
            emb[h] += 1.0

        norm = torch.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Blend with average root embedding
        emb = 0.7 * emb + 0.3 * self.avg_root_embedding
        return emb

    def embed_word(self, root: str, prefixes: List[str], suffixes: List[str]) -> Optional[np.ndarray]:
        """Embed a word using compositional embeddings."""
        if root.lower() in self.function_words:
            return None

        # Get root embedding
        root_lower = root.lower()

        if self.use_hybrid and self.hybrid_model:
            # Use hybrid embeddings (linguistic + topical)
            emb_tensor = self.hybrid_model.get_root_embedding(root_lower, mode='hybrid')
            if emb_tensor is None:
                # Try without lowercase
                emb_tensor = self.hybrid_model.get_root_embedding(root, mode='hybrid')
            if emb_tensor is None:
                # Fallback to character hash
                emb = self._char_hash_embedding(root)
            else:
                emb = emb_tensor
        else:
            # Use linguistic embeddings only (legacy)
            if root_lower in self.root_to_idx:
                root_idx = self.root_to_idx[root_lower]
                emb = self.root_emb[root_idx].clone()
            elif root in self.root_to_idx:
                root_idx = self.root_to_idx[root]
                emb = self.root_emb[root_idx].clone()
            else:
                emb = self._char_hash_embedding(root)

        # Note: Affix transforms not yet supported for hybrid embeddings (128d)
        # TODO: Train new affix transforms for 128d embeddings
        if not self.use_hybrid:
            # Apply prefix transforms
            for p in prefixes:
                if p and p in self.prefix_transforms:
                    with torch.no_grad():
                        emb = self.prefix_transforms[p](emb.unsqueeze(0)).squeeze(0)

            # Apply suffix transforms
            for s in suffixes:
                if s and s in self.suffix_transforms:
                    with torch.no_grad():
                        emb = self.suffix_transforms[s](emb.unsqueeze(0)).squeeze(0)

        return emb.numpy()

    def _embed_ast_node(self, node) -> Optional[np.ndarray]:
        """Extract embedding from an AST node (word or phrase).

        Returns only the HEAD embedding. For compound words, modifiers are
        extracted separately via _extract_modifiers().
        """
        if node is None:
            return None

        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                # Single word
                root = node.get('radiko', '')
                prefixes = node.get('prefiksoj', [])
                if not prefixes:
                    p = node.get('prefikso')
                    if p:
                        prefixes = [p]
                suffixes = node.get('sufiksoj', [])
                return self.embed_word(root, prefixes, suffixes)

            elif node.get('tipo') == 'vortgrupo':
                # Phrase: use head word
                kerno = node.get('kerno')
                if kerno:
                    return self._embed_ast_node(kerno)

        return None

    def _extract_modifiers(self, node) -> List[np.ndarray]:
        """Extract modifier embeddings from compound words.

        For hyphenated compound words (Esperanto-klubo), extracts embeddings
        for each modifier component (left parts in the compound).

        Args:
            node: AST node (vorto or vortgrupo)

        Returns:
            List of modifier embeddings (empty if no compound)
        """
        if node is None:
            return []

        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                # Check if it's a compound word
                if node.get('estas_kunmetita') and node.get('kunmetajhoj'):
                    modifier_embeddings = []
                    for mod_ast in node.get('kunmetajhoj', []):
                        mod_emb = self._embed_ast_node(mod_ast)
                        if mod_emb is not None:
                            modifier_embeddings.append(mod_emb)
                    return modifier_embeddings

            elif node.get('tipo') == 'vortgrupo':
                # Phrase: extract from head word
                kerno = node.get('kerno')
                if kerno:
                    return self._extract_modifiers(kerno)

        return []

    def _is_compound(self, node) -> bool:
        """Check if a node is or contains a compound word."""
        if node is None:
            return False

        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                return bool(node.get('estas_kunmetita'))
            elif node.get('tipo') == 'vortgrupo':
                kerno = node.get('kerno')
                return self._is_compound(kerno) if kerno else False

        return False

    def extract_slots(self, ast: dict) -> Dict[str, any]:
        """Extract slot embeddings from AST.

        Returns a dictionary with:
        - SUBJ_HEAD: Subject head embedding (rightmost part of compound)
        - SUBJ_MOD: List of subject modifier embeddings (left parts of compound)
        - VERB: Verb embedding
        - OBJ_HEAD: Object head embedding
        - OBJ_MOD: List of object modifier embeddings

        For non-compound words, MOD lists are empty.
        This enables HEAD-weighted matching: a query for "Esperanto" should match
        "Esperanto-klubo" in OBJ_MOD position, not OBJ_HEAD.
        """
        slots = {
            'SUBJ_HEAD': None,
            'SUBJ_MOD': [],
            'VERB': None,
            'OBJ_HEAD': None,
            'OBJ_MOD': [],
        }

        if ast.get('tipo') == 'frazo':
            # Extract subject (HEAD + modifiers)
            if ast.get('subjekto'):
                slots['SUBJ_HEAD'] = self._embed_ast_node(ast['subjekto'])
                slots['SUBJ_MOD'] = self._extract_modifiers(ast['subjekto'])

            # Extract verb
            if ast.get('verbo'):
                slots['VERB'] = self._embed_ast_node(ast['verbo'])

            # Extract object (HEAD + modifiers)
            if ast.get('objekto'):
                slots['OBJ_HEAD'] = self._embed_ast_node(ast['objekto'])
                slots['OBJ_MOD'] = self._extract_modifiers(ast['objekto'])

        return slots

    def extract_features(self, ast: dict) -> Dict:
        """Extract grammatical features from AST."""
        return {
            'negita': ast.get('negita', False),
            'tempo': ast.get('tempo', 'prezenco'),
            'fraztipo': ast.get('fraztipo', 'deklaro'),
            'modo': ast.get('modo', 'indikativo'),
        }

    def _extract_word_info(self, word_node: Dict) -> Optional[Dict]:
        """Extract essential word information from a vorto node.

        Returns a compact dict with only fields needed for matching:
        - radiko: word root
        - vortspeco: part of speech
        - prefiksoj: list of prefixes (if any)
        - sufiksoj: list of suffixes (if any)
        - tempo: verb tense (only for verbs)
        - kazo: grammatical case (only for nouns)
        """
        if not word_node or word_node.get('tipo') != 'vorto':
            return None

        info = {
            'radiko': word_node.get('radiko', '').lower(),
            'vortspeco': word_node.get('vortspeco', 'nekonata'),
        }

        # Only include prefixes if present
        prefiksoj = word_node.get('prefiksoj', [])
        if not prefiksoj:
            p = word_node.get('prefikso')
            if p:
                prefiksoj = [p]
        if prefiksoj:
            info['prefiksoj'] = prefiksoj

        # Only include suffixes if present
        sufiksoj = word_node.get('sufiksoj', [])
        if sufiksoj:
            info['sufiksoj'] = sufiksoj

        # Only include tempo for verbs
        if word_node.get('vortspeco') == 'verbo' and word_node.get('tempo'):
            info['tempo'] = word_node.get('tempo')

        # Only include kazo for nouns/adjectives
        if word_node.get('kazo') and word_node.get('vortspeco') in ['substantivo', 'adjektivo', 'propra_nomo']:
            info['kazo'] = word_node.get('kazo')

        return info

    def _extract_words_from_phrase(self, phrase_node: Dict) -> List[Dict]:
        """Extract all word info from a phrase (vortgrupo or vorto).

        For vortgrupo: extracts kerno (head) + priskriboj (modifiers)
        For compound words: extracts both head and kunmetajhoj (compound parts)
        """
        if not phrase_node:
            return []

        words = []

        if phrase_node.get('tipo') == 'vorto':
            # Single word
            info = self._extract_word_info(phrase_node)
            if info:
                words.append(info)

            # Check for compound word parts
            if phrase_node.get('estas_kunmetita') and phrase_node.get('kunmetajhoj'):
                for part in phrase_node.get('kunmetajhoj', []):
                    part_info = self._extract_word_info(part)
                    if part_info:
                        words.append(part_info)

        elif phrase_node.get('tipo') == 'vortgrupo':
            # Phrase: extract head (kerno)
            kerno = phrase_node.get('kerno')
            if kerno:
                words.extend(self._extract_words_from_phrase(kerno))

            # Extract modifiers (priskriboj)
            for modifier in phrase_node.get('priskriboj', []):
                words.extend(self._extract_words_from_phrase(modifier))

        return words

    def _extract_prep_phrases(self, aliaj: List) -> List[Dict]:
        """Extract prepositional phrases from 'aliaj' list.

        Handles two parser output formats:
        1. Grouped: vortgrupo with preposition as kerno and object in priskriboj
        2. Flat: preposition and object as separate items in aliaj list

        Returns list of dicts with:
        - preposition: the preposition used
        - object_words: list of word info for the object of the preposition
        """
        if not aliaj:
            return []

        prep_phrases = []
        current_prep = None

        for item in aliaj:
            if not isinstance(item, dict):
                continue

            # Check for prepositional phrase structure
            # Format 1: vortgrupo with prep as kerno
            if item.get('tipo') == 'vortgrupo':
                kerno = item.get('kerno')
                if kerno and kerno.get('vortspeco') == 'prepozicio':
                    prep_phrase = {
                        'preposition': kerno.get('radiko', '').lower(),
                        'object_words': []
                    }
                    # Get objects from priskriboj
                    for modifier in item.get('priskriboj', []):
                        prep_phrase['object_words'].extend(
                            self._extract_words_from_phrase(modifier)
                        )
                    if prep_phrase['object_words']:
                        prep_phrases.append(prep_phrase)

            elif item.get('tipo') == 'vorto':
                # Format 2: Flat list - preposition followed by object(s)
                if item.get('vortspeco') == 'prepozicio':
                    # Save current prep if it had objects
                    if current_prep and current_prep['object_words']:
                        prep_phrases.append(current_prep)
                    # Start new prep phrase
                    current_prep = {
                        'preposition': item.get('radiko', '').lower(),
                        'object_words': []
                    }
                elif current_prep and item.get('vortspeco') in ['substantivo', 'propra_nomo', 'adjektivo']:
                    # This word is the object of the current preposition
                    word_info = self._extract_word_info(item)
                    if word_info:
                        current_prep['object_words'].append(word_info)

        # Don't forget the last prep phrase
        if current_prep and current_prep['object_words']:
            prep_phrases.append(current_prep)

        return prep_phrases

    def _extract_all_modifiers(self, ast: Dict) -> Dict[str, List]:
        """Extract all modifiers (adverbs, adjectives) from the sentence.

        Returns dict with:
        - adverbs: list of adverb word info
        - adjectives: list of adjective roots (strings for compactness)
        """
        modifiers = {
            'adverbs': [],
            'adjectives': []
        }

        def collect_modifiers(node):
            """Recursively collect modifiers from AST."""
            if not isinstance(node, dict):
                return

            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco')
                if vortspeco == 'adverbo':
                    info = self._extract_word_info(node)
                    if info:
                        modifiers['adverbs'].append(info)
                elif vortspeco == 'adjektivo':
                    radiko = node.get('radiko', '').lower()
                    if radiko and radiko not in modifiers['adjectives']:
                        modifiers['adjectives'].append(radiko)

            elif node.get('tipo') == 'vortgrupo':
                # Check kerno
                if node.get('kerno'):
                    collect_modifiers(node['kerno'])
                # Check priskriboj
                for mod in node.get('priskriboj', []):
                    collect_modifiers(mod)

            elif node.get('tipo') == 'frazo':
                # Traverse sentence structure
                if node.get('subjekto'):
                    collect_modifiers(node['subjekto'])
                if node.get('verbo'):
                    collect_modifiers(node['verbo'])
                if node.get('objekto'):
                    collect_modifiers(node['objekto'])
                for item in node.get('aliaj', []):
                    collect_modifiers(item)

        collect_modifiers(ast)
        return modifiers

    def extract_minimal_ast(self, ast: Dict) -> Dict:
        """Extract minimal AST fields for enhanced matching.

        This is a compact representation of the AST that captures:
        - Word-level morphology (root, prefixes, suffixes)
        - Syntactic roles (subject, verb, object words)
        - Prepositional phrases
        - Modifiers (adverbs, adjectives)

        The minimal AST enables:
        - Modifier matching (query/doc adjective similarity)
        - Prepositional phrase matching (location queries)
        - Entity type boosting (proper noun detection)
        - Morphological pattern matching (shared prefixes/suffixes)

        Returns:
            Dict with subjekto_words, verbo_words, objekto_words,
            prep_phrases, and modifiers.
        """
        minimal = {
            'subjekto_words': [],
            'verbo_words': [],
            'objekto_words': [],
            'prep_phrases': [],
            'modifiers': {'adverbs': [], 'adjectives': []}
        }

        if ast.get('tipo') != 'frazo':
            return minimal

        # Extract subject words
        if ast.get('subjekto'):
            minimal['subjekto_words'] = self._extract_words_from_phrase(ast['subjekto'])

        # Extract verb words
        if ast.get('verbo'):
            verb_info = self._extract_word_info(ast['verbo'])
            if verb_info:
                minimal['verbo_words'] = [verb_info]

        # Extract object words
        if ast.get('objekto'):
            minimal['objekto_words'] = self._extract_words_from_phrase(ast['objekto'])

        # Extract prepositional phrases from aliaj
        if ast.get('aliaj'):
            minimal['prep_phrases'] = self._extract_prep_phrases(ast['aliaj'])

        # Extract all modifiers
        minimal['modifiers'] = self._extract_all_modifiers(ast)

        return minimal

    def index_sentence(
        self,
        text: str,
        source_info: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Index a single sentence with slot-based structure.

        Returns:
            Dictionary with slots, features, full embedding, and metadata.

        Slots structure:
            - SUBJ_HEAD: Subject head embedding (or None)
            - SUBJ_MOD: List of subject modifier embeddings (or empty list)
            - VERB: Verb embedding (or None)
            - OBJ_HEAD: Object head embedding (or None)
            - OBJ_MOD: List of object modifier embeddings (or empty list)
        """
        try:
            ast = parse(text)
        except Exception as e:
            logger.debug(f"Parse failed: {text[:50]}... - {e}")
            return None

        # Extract slots (now includes HEAD and MOD for compounds)
        slots = self.extract_slots(ast)

        # Extract features
        features = self.extract_features(ast)

        # Extract minimal AST for enhanced matching
        minimal_ast = self.extract_minimal_ast(ast)

        # Compute full embedding (fallback) from HEAD embeddings only
        word_embeddings = []
        for key in ['SUBJ_HEAD', 'VERB', 'OBJ_HEAD']:
            if slots.get(key) is not None:
                word_embeddings.append(slots[key])

        if not word_embeddings:
            return None

        full_emb = np.mean(word_embeddings, axis=0)
        norm = np.linalg.norm(full_emb)
        if norm > 0:
            full_emb = full_emb / norm

        # Build index entry with proper serialization
        serialized_slots = {}
        for k, v in slots.items():
            if v is None:
                serialized_slots[k] = None
            elif isinstance(v, list):
                # List of modifier embeddings
                serialized_slots[k] = [emb.tolist() for emb in v]
            elif isinstance(v, np.ndarray):
                serialized_slots[k] = v.tolist()
            else:
                serialized_slots[k] = v

        entry = {
            'text': text,
            'slots': serialized_slots,
            'features': features,
            'full_embedding': full_emb.tolist(),
            'minimal_ast': minimal_ast,
        }

        # Add source info if provided
        if source_info:
            entry['source'] = source_info

        return entry

    def build_index(
        self,
        corpus_path: Path,
        checkpoint_interval: int = 5000,
        resume: bool = True,
    ):
        """
        Build slot-based index from corpus.

        Args:
            corpus_path: Path to corpus JSONL file
            checkpoint_interval: Save checkpoint every N sentences
            resume: Resume from checkpoint if exists
        """
        output_file = self.output_dir / "slot_index.jsonl"
        checkpoint_file = self.output_dir / "checkpoint.json"

        # Resume handling
        start_idx = 0
        if resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get('processed', 0)
            logger.info(f"Resuming from sentence {start_idx}")

        # Open output file
        mode = 'a' if start_idx > 0 else 'w'
        out_f = open(output_file, mode)

        stats = {
            'processed': start_idx,
            'successful': 0,
            'failed': 0,
        }

        try:
            with open(corpus_path) as f:
                # Skip to resume point
                for _ in range(start_idx):
                    next(f)

                # Process sentences
                for i, line in enumerate(f, start=start_idx):
                    data = json.loads(line)
                    text = data.get('text', '')
                    source = data.get('source', {})

                    entry = self.index_sentence(text, source)

                    if entry:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1

                    stats['processed'] += 1

                    # Progress logging
                    if stats['processed'] % 1000 == 0:
                        pct = 100.0  # Don't know total
                        logger.info(
                            f"Processed {stats['processed']:,} | "
                            f"Success: {stats['successful']:,} | "
                            f"Failed: {stats['failed']:,}"
                        )

                    # Checkpoint
                    if stats['processed'] % checkpoint_interval == 0:
                        with open(checkpoint_file, 'w') as cf:
                            json.dump(stats, cf)

        finally:
            out_f.close()

        # Save final stats
        with open(checkpoint_file, 'w') as cf:
            json.dump(stats, cf)

        logger.info(f"Indexing complete!")
        logger.info(f"  Total: {stats['processed']:,}")
        logger.info(f"  Successful: {stats['successful']:,}")
        logger.info(f"  Failed: {stats['failed']:,}")
        logger.info(f"  Output: {output_file}")

        return stats
