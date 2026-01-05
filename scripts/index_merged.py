#!/usr/bin/env python3
"""
Build merged FAISS index with source-tier weighting.

Combines:
- Curated facts (50+ sentences, tier 0): Essential Esperanto facts for Q&A
- Authoritative corpus (18K sentences, tiers 1-3): Fundamento, ReVo, curated sources
- General corpus (4.3M sentences): Wikipedia, Gutenberg, etc.

Source tiers (lower = more authoritative):
- Tier 0: Curated Esperanto facts (highest authority, +25% boost)
- Tier 1: Fundamento de Esperanto (+15% boost)
- Tier 2: ReVo dictionary definitions (+10% boost)
- Tier 3: Curated authoritative texts (+5% boost)
- Tier 5: General Wikipedia/Gutenberg (no boost)
- Tier 6: Other sources (no boost)

Curated facts are indexed FIRST for highest priority in retrieval.

Usage:
    python scripts/index_merged_corpus.py
    python scripts/index_merged_corpus.py --fresh
    python scripts/index_merged_corpus.py --authoritative-only  # Just curated + authoritative
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Function words (grammatical, not semantic) - excluded from embeddings
FUNCTION_WORDS = {
    'la', 'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni', 'si',
    'mia', 'via', 'lia', 'ŝia', 'ĝia', 'nia', 'ilia', 'sia',
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sur', 'sub',
    'tra', 'trans', 'ĉe', 'ĉi', 'ĉirkaŭ', 'ekster', 'inter', 'kontraŭ',
    'antaŭ', 'post', 'super', 'apud', 'preter', 'malgraŭ', 'krom', 'laŭ',
    'anstataŭ', 'ĝis', 'sen', 'pro', 'spite',
    'kaj', 'aŭ', 'sed', 'nek', 'ke', 'ĉar', 'se', 'dum', 'kvankam',
    'tamen', 'do', 'tial', 'ĉu',
    'kio', 'kiu', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial', 'kies',
    'tio', 'tiu', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial', 'ties',
    'io', 'iu', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial', 'ies',
    'ĉio', 'ĉiu', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial', 'ĉies',
    'nenio', 'neniu', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial', 'nenies',
    'ne', 'ankaŭ', 'nur', 'eĉ', 'ja', 'jen', 'tre', 'pli', 'plej', 'tro',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class LowRankTransform(nn.Module):
    """Low-rank transformation for affixes: x + up(down(x))"""
    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class MergedCorpusIndexer:
    """Index merged corpus with source tier information."""

    def __init__(
        self,
        root_model_path: Path,
        affix_model_path: Path,
        output_dir: Path,
        batch_size: int = 100,
    ):
        self.root_model_path = root_model_path
        self.affix_model_path = affix_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.checkpoint_path = self.output_dir / "indexing_checkpoint.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.failed_path = self.output_dir / "failed_sentences.jsonl"
        self.index_path = self.output_dir / "faiss_index.bin"
        self.log_path = self.output_dir / "indexing.log"

        # Load models
        self.device = torch.device('cpu')
        self.root_emb, self.root_to_idx, self.embedding_dim = self._load_root_model()
        self.prefix_transforms, self.suffix_transforms = self._load_affix_transforms()
        self.avg_root_embedding = self.root_emb.mean(dim=0)

        # Stats
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "by_tier": {},
        }

    def _load_root_model(self) -> Tuple[torch.Tensor, Dict[str, int], int]:
        logger.info(f"Loading root embeddings from {self.root_model_path}")
        checkpoint = torch.load(self.root_model_path, map_location='cpu', weights_only=False)
        embeddings = checkpoint['model_state_dict']['embeddings.weight']
        root_to_idx = checkpoint['root_to_idx']
        dim = embeddings.shape[1]
        logger.info(f"  Loaded {len(root_to_idx)} roots, {dim}d")
        return embeddings, root_to_idx, dim

    def _load_affix_transforms(self) -> Tuple[Dict[str, LowRankTransform], Dict[str, LowRankTransform]]:
        logger.info(f"Loading affix transforms from {self.affix_model_path}")
        checkpoint = torch.load(self.affix_model_path, map_location='cpu', weights_only=False)

        rank = checkpoint['rank']
        prefixes = checkpoint['prefixes']
        suffixes = checkpoint['suffixes']
        state_dict = checkpoint['model_state_dict']

        prefix_transforms = {}
        for p in prefixes:
            transform = LowRankTransform(self.embedding_dim, rank)
            transform.down.weight.data = state_dict[f'prefix_transforms.{p}.down.weight']
            transform.up.weight.data = state_dict[f'prefix_transforms.{p}.up.weight']
            transform.eval()
            prefix_transforms[p] = transform

        suffix_transforms = {}
        for s in suffixes:
            transform = LowRankTransform(self.embedding_dim, rank)
            transform.down.weight.data = state_dict[f'suffix_transforms.{s}.down.weight']
            transform.up.weight.data = state_dict[f'suffix_transforms.{s}.up.weight']
            transform.eval()
            suffix_transforms[s] = transform

        logger.info(f"  Loaded {len(prefixes)} prefix, {len(suffixes)} suffix transforms")
        return prefix_transforms, suffix_transforms

    def _char_hash_embedding(self, root: str) -> torch.Tensor:
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
        emb = 0.7 * emb + 0.3 * self.avg_root_embedding
        return emb

    def embed_word(self, root: str, prefixes: List[str], suffixes: List[str]) -> Optional[np.ndarray]:
        if root.lower() in FUNCTION_WORDS:
            return None

        root_lower = root.lower()
        if root_lower not in self.root_to_idx:
            if root not in self.root_to_idx:
                emb = self._char_hash_embedding(root)
            else:
                root_lower = root
                root_idx = self.root_to_idx[root_lower]
                emb = self.root_emb[root_idx].clone()
        else:
            root_idx = self.root_to_idx[root_lower]
            emb = self.root_emb[root_idx].clone()

        for p in prefixes:
            if p and p in self.prefix_transforms:
                with torch.no_grad():
                    emb = self.prefix_transforms[p](emb.unsqueeze(0)).squeeze(0)
        for s in suffixes:
            if s and s in self.suffix_transforms:
                with torch.no_grad():
                    emb = self.suffix_transforms[s](emb.unsqueeze(0)).squeeze(0)

        return emb.numpy()

    def embed_sentence(self, text: str, source_info: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Dict]:
        tier = source_info.get('tier', 6) if source_info else 6
        source_name = source_info.get('name', 'unknown') if source_info else 'unknown'

        metadata = {
            "text": text,
            "tier": tier,
            "source": source_name,
            "words_total": 0,
            "words_embedded": 0,
        }

        try:
            ast = parse(text)
        except Exception as e:
            metadata["parse_error"] = str(e)
            return None, metadata

        word_embeddings = []

        def extract_words(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    metadata["words_total"] += 1
                    root = node.get('radiko', '')
                    prefixes = node.get('prefiksoj', [])
                    if not prefixes:
                        p = node.get('prefikso')
                        if p:
                            prefixes = [p]
                    suffixes = node.get('sufiksoj', [])
                    emb = self.embed_word(root, prefixes, suffixes)
                    if emb is not None:
                        word_embeddings.append(emb)
                        metadata["words_embedded"] += 1
                for v in node.values():
                    extract_words(v)
            elif isinstance(node, list):
                for item in node:
                    extract_words(item)

        extract_words(ast)

        if not word_embeddings:
            return None, metadata

        sentence_emb = np.mean(word_embeddings, axis=0)
        norm = np.linalg.norm(sentence_emb)
        if norm > 0:
            sentence_emb = sentence_emb / norm

        return sentence_emb, metadata

    def load_corpus_entries(self, corpus_path: Path, default_tier: int = 6) -> List[Dict]:
        """Load corpus entries with source information."""
        entries = []
        with open(corpus_path) as f:
            for line in f:
                data = json.loads(line)
                text = data.get('text', '')
                if not text:
                    continue

                # Extract source info if available
                source = data.get('source', {})
                if isinstance(source, dict):
                    tier = source.get('tier', default_tier)
                    name = source.get('name', corpus_path.stem)
                else:
                    tier = default_tier
                    name = corpus_path.stem

                entries.append({
                    'text': text,
                    'source': {'tier': tier, 'name': name}
                })
        return entries

    def index_merged(
        self,
        authoritative_path: Path,
        general_path: Optional[Path] = None,
        facts_path: Optional[Path] = None,
        resume: bool = True,
        authoritative_only: bool = False,
    ):
        """Index merged corpus with curated facts and authoritative sentences first."""
        # Setup file logging
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("=" * 60)
        logger.info("Merged Corpus Indexing with Source Tiers")
        logger.info("=" * 60)

        # Load curated facts (tier 0) - highest priority
        facts_entries = []
        if facts_path and facts_path.exists():
            logger.info(f"Loading curated facts: {facts_path}")
            facts_entries = self.load_corpus_entries(facts_path, default_tier=0)
            logger.info(f"  Loaded {len(facts_entries)} curated facts (tier 0)")

        # Load authoritative corpus (tiers 1-3)
        logger.info(f"Loading authoritative corpus: {authoritative_path}")
        authoritative_entries = self.load_corpus_entries(authoritative_path, default_tier=1)
        logger.info(f"  Loaded {len(authoritative_entries)} authoritative sentences")

        # Load general corpus if provided
        general_entries = []
        if not authoritative_only and general_path and general_path.exists():
            logger.info(f"Loading general corpus: {general_path}")
            general_entries = self.load_corpus_entries(general_path, default_tier=5)
            logger.info(f"  Loaded {len(general_entries)} general sentences")

        # Merge: facts first, then authoritative, then general
        all_entries = facts_entries + authoritative_entries + general_entries
        total_sentences = len(all_entries)
        logger.info(f"Total sentences to index: {total_sentences}")
        logger.info(f"  Curated facts (tier 0): {len(facts_entries)}")
        logger.info(f"  Authoritative (tiers 1-3): {len(authoritative_entries)}")
        logger.info(f"  General (tier 5+): {len(general_entries)}")

        # Resume handling
        start_idx = 0
        all_embeddings = []
        if resume and self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                self.stats = json.load(f)
            start_idx = self.stats.get("processed", 0)
            if start_idx > 0 and self.embeddings_path.exists():
                all_embeddings = list(np.load(self.embeddings_path))
                logger.info(f"Resuming from sentence {start_idx}, {len(all_embeddings)} embeddings loaded")

        if start_idx == 0:
            self.stats = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "total_sentences": total_sentences,
                "facts_count": len(facts_entries),
                "authoritative_count": len(authoritative_entries),
                "general_count": len(general_entries),
                "by_tier": {},
            }

        # Open output files
        metadata_mode = 'a' if start_idx > 0 else 'w'
        metadata_file = open(self.metadata_path, metadata_mode)
        failed_file = open(self.failed_path, metadata_mode)

        try:
            start_time = time.time()
            last_log_pct = 0

            for i in range(start_idx, total_sentences, self.batch_size):
                batch = all_entries[i:i + self.batch_size]
                batch_embeddings = []

                for entry in batch:
                    text = entry['text']
                    source = entry['source']
                    emb, meta = self.embed_sentence(text, source)

                    tier = meta.get('tier', 6)
                    tier_key = f"tier_{tier}"
                    if tier_key not in self.stats["by_tier"]:
                        self.stats["by_tier"][tier_key] = {"success": 0, "failed": 0}

                    if emb is not None:
                        batch_embeddings.append(emb)
                        meta["index"] = len(all_embeddings) + len(batch_embeddings) - 1
                        metadata_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        self.stats["successful"] += 1
                        self.stats["by_tier"][tier_key]["success"] += 1
                    else:
                        failed_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        self.stats["failed"] += 1
                        self.stats["by_tier"][tier_key]["failed"] += 1

                    self.stats["processed"] += 1

                all_embeddings.extend(batch_embeddings)

                # Progress update
                if (i + len(batch)) % 1000 == 0 or i + len(batch) == total_sentences:
                    elapsed = time.time() - start_time
                    rate = (self.stats["processed"] - start_idx) / elapsed if elapsed > 0 else 0
                    remaining = (total_sentences - self.stats["processed"]) / rate if rate > 0 else 0
                    pct = 100 * self.stats["processed"] / total_sentences

                    msg = (f"Progress: {self.stats['processed']:,}/{total_sentences:,} ({pct:.1f}%) "
                           f"| {self.stats['successful']:,} OK | {rate:.0f}/s | ETA: {remaining/60:.1f}m")

                    current_pct = int(pct // 10) * 10
                    if current_pct > last_log_pct:
                        print(f"\r{msg:<100}")
                        logger.info(msg)
                        last_log_pct = current_pct
                    else:
                        print(f"\r{msg:<100}", end='', flush=True)

                # Checkpoint every 5000
                if self.stats["processed"] % 5000 == 0:
                    self._save_checkpoint()
                    np.save(self.embeddings_path, np.array(all_embeddings, dtype=np.float32))

        finally:
            metadata_file.close()
            failed_file.close()

        print()  # Newline after progress

        # Save final embeddings
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        np.save(self.embeddings_path, embeddings_array)
        logger.info(f"Saved embeddings: {embeddings_array.shape}")

        # Build FAISS index
        self._build_faiss_index(embeddings_array)

        # Save final checkpoint
        self._save_checkpoint()

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Indexing Complete")
        logger.info("=" * 60)
        logger.info(f"Total: {total_sentences}")
        logger.info(f"Successful: {self.stats['successful']} ({100*self.stats['successful']/total_sentences:.1f}%)")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info("")
        logger.info("By tier:")
        for tier_key, counts in sorted(self.stats["by_tier"].items()):
            total_tier = counts["success"] + counts["failed"]
            logger.info(f"  {tier_key}: {counts['success']}/{total_tier} successful")

    def _save_checkpoint(self):
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.stats, f)
        temp_path.rename(self.checkpoint_path)

    def _build_faiss_index(self, embeddings: np.ndarray):
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, skipping index build")
            return

        logger.info(f"Building FAISS index for {len(embeddings)} embeddings...")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        logger.info(f"Saved FAISS index: {self.index_path}")


def main():
    parser = argparse.ArgumentParser(description='Build merged FAISS index with source tiers')
    parser.add_argument('--facts', type=Path,
                        default=Path('data/corpus/esperanto_facts.jsonl'),
                        help='Path to curated facts (tier 0, highest priority)')
    parser.add_argument('--authoritative', type=Path,
                        default=Path('data/corpus/authoritative_corpus.jsonl'),
                        help='Path to authoritative corpus (tiers 1-3)')
    parser.add_argument('--general', type=Path,
                        default=Path('data/corpus/unified_corpus.jsonl'),
                        help='Path to general corpus (tier 5+)')
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_transforms_v2/best_model.pt'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/indexes/merged'))
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--fresh', action='store_true', help='Start fresh')
    parser.add_argument('--authoritative-only', action='store_true',
                        help='Only index authoritative corpus (faster for testing)')

    args = parser.parse_args()

    # Validate inputs
    if not args.authoritative.exists():
        logger.error(f"Authoritative corpus not found: {args.authoritative}")
        sys.exit(1)
    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        sys.exit(1)
    if not args.affix_model.exists():
        logger.error(f"Affix model not found: {args.affix_model}")
        sys.exit(1)

    indexer = MergedCorpusIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    indexer.index_merged(
        authoritative_path=args.authoritative,
        general_path=args.general,
        facts_path=args.facts,
        resume=not args.fresh,
        authoritative_only=args.authoritative_only,
    )


if __name__ == '__main__':
    main()
