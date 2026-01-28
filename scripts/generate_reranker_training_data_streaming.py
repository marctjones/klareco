#!/usr/bin/env python3
"""
Generate training data for query-document reranker (STREAMING VERSION).

Memory-efficient streaming implementation with checkpoint support.
Uses constant memory regardless of corpus size.

Features:
- Streams through corpus (never loads all into RAM)
- Checkpoint support (resume from crashes)
- Reservoir sampling for memory efficiency
- Three-pass algorithm:
  1. Count patterns
  2. Generate synthetic pairs
  3. Mine real Q&A pairs

Usage:
    # Generate with checkpoints
    python scripts/generate_reranker_training_data_streaming.py \
        --strategy both \
        --output data/training/reranker/

    # Resume from checkpoint
    python scripts/generate_reranker_training_data_streaming.py \
        --resume

    # Start fresh (ignore checkpoint)
    python scripts/generate_reranker_training_data_streaming.py \
        --fresh
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReservoirSampler:
    """
    Reservoir sampling for memory-efficient random sampling.

    Maintains a fixed-size sample while streaming through data.
    """

    def __init__(self, k: int):
        """
        Initialize reservoir sampler.

        Args:
            k: Maximum samples to keep
        """
        self.k = k
        self.reservoir = []
        self.n = 0  # Total items seen

    def add(self, item):
        """Add item to reservoir with proper probability."""
        self.n += 1

        if len(self.reservoir) < self.k:
            # Reservoir not full yet
            self.reservoir.append(item)
        else:
            # Replace with decreasing probability
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.reservoir[j] = item

    def sample(self) -> List:
        """Get current sample."""
        return self.reservoir


class StreamingDataGenerator:
    """
    Memory-efficient streaming data generator with checkpoints.
    """

    # Template definitions (same as before)
    SYNTHETIC_TEMPLATES = {
        'definition': [
            ('kio_estas', "Kio estas {subject}?", 1.0),
            ('cu_estas', "Ĉu {subject} estas {predicate}?", 0.9),
            ('kia_estas', "Kia estas {subject}?", 0.8),
        ],
        'factual_who': [
            ('kiu_verbo', "Kiu {verb} {object}?", 1.0),
        ],
        'factual_what': [
            ('kion_verbo', "Kion {subject} {verb}?", 1.0),
        ],
        'location': [
            ('kie_estas', "Kie estas {subject}?", 1.0),
        ],
        'temporal': [
            ('kiam_verbo', "Kiam {verb} {subject}?", 1.0),
        ],
        'causal': [
            ('kial_verbo', "Kial {subject} {verb}?", 1.0),
        ],
    }

    MINING_PATTERNS = {
        'definition': {
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() == 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kio estas {subject}?",
        },
        'who_action': {
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() != 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kiu {verb} {object}?",
        },
        'what_action': {
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() != 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kion {subject} {verb}?",
        },
        'location': {
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() == 'est'
                and any(
                    isinstance(m, dict) and m.get('radiko', '').lower() in ['en', 'ĉe', 'sur', 'apud']
                    for m in ast.get('aliaj', [])
                )
            ),
            'query_template': "Kie estas {subject}?",
        },
    }

    def __init__(
        self,
        corpus_path: Path,
        output_dir: Path,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initialize streaming generator.

        Args:
            corpus_path: Path to corpus JSONL
            output_dir: Output directory for datasets
            checkpoint_path: Path to checkpoint file
        """
        self.corpus_path = corpus_path
        self.output_dir = Path(output_dir)
        self.checkpoint_path = checkpoint_path or (self.output_dir / 'checkpoint.json')

        # State
        self.pattern_counts = defaultdict(int)
        self.total_sentences = 0
        self.synthetic_pairs = []
        self.mined_pairs = []
        self.phase = 'pattern_counting'

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self):
        """Save checkpoint (atomic write)."""
        checkpoint = {
            'phase': self.phase,
            'pattern_counts': dict(self.pattern_counts),
            'total_sentences': self.total_sentences,
            'synthetic_pairs_count': len(self.synthetic_pairs),
            'mined_pairs_count': len(self.mined_pairs),
            'timestamp': datetime.now().isoformat(),
        }

        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            temp_path.rename(self.checkpoint_path)
            logger.info(f"  Checkpoint saved: {self.phase}")
        except Exception as e:
            logger.error(f"  Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if exists.

        Returns:
            True if checkpoint was loaded
        """
        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.phase = checkpoint['phase']
            self.pattern_counts = defaultdict(int, checkpoint['pattern_counts'])
            self.total_sentences = checkpoint['total_sentences']

            logger.info(f"Loaded checkpoint from: {checkpoint['timestamp']}")
            logger.info(f"  Phase: {self.phase}")
            logger.info(f"  Total sentences processed: {self.total_sentences:,}")
            logger.info(f"  Synthetic pairs: {checkpoint['synthetic_pairs_count']:,}")
            logger.info(f"  Mined pairs: {checkpoint['mined_pairs_count']:,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _detect_synthetic_patterns(self, ast: Dict) -> Set[str]:
        """Detect patterns for synthetic generation."""
        patterns = set()

        verb = ast.get('verbo', {})
        verb_root = verb.get('radiko', '').lower()
        subjekto = ast.get('subjekto')
        objekto = ast.get('objekto')

        # Definition pattern
        if verb_root == 'est' and subjekto and objekto:
            obj_kazo = None
            if objekto.get('tipo') == 'vortgrupo':
                obj_kazo = objekto.get('kerno', {}).get('kazo')
            elif objekto.get('tipo') == 'vorto':
                obj_kazo = objekto.get('kazo')

            if obj_kazo != 'akuzativo':
                patterns.add('definition')

        # Factual patterns
        if verb_root != 'est' and subjekto and objekto:
            patterns.add('factual_who')
            patterns.add('factual_what')

        # Location pattern
        if verb_root == 'est':
            for modifier in ast.get('aliaj', []):
                if isinstance(modifier, dict):
                    mod_root = modifier.get('radiko', '').lower()
                    if mod_root in ['en', 'ĉe', 'sur', 'sub', 'apud']:
                        patterns.add('location')
                        break

        # Temporal pattern
        for modifier in ast.get('aliaj', []):
            if isinstance(modifier, dict):
                mod_root = modifier.get('radiko', '').lower()
                if mod_root in ['jar', 'monat', 'tag', 'hodiaŭ', 'hieraŭ', 'morgaŭ']:
                    patterns.add('temporal')
                    break

        # Causal pattern
        for modifier in ast.get('aliaj', []):
            if isinstance(modifier, dict):
                if modifier.get('radiko', '').lower() == 'ĉar':
                    patterns.add('causal')
                    break

        return patterns

    def _extract_slots(self, ast: Dict) -> Dict[str, str]:
        """Extract slots from AST for template filling."""
        slots = {}

        # Extract subject
        subjekto = ast.get('subjekto')
        if subjekto:
            slots['subject'] = self._extract_text(subjekto)

        # Extract verb
        verb = ast.get('verbo', {})
        if verb.get('radiko'):
            verb_root = verb['radiko']
            slots['verb'] = verb_root + 'is'

        # Extract object
        objekto = ast.get('objekto')
        if objekto:
            text = self._extract_text(objekto)
            slots['object'] = text
            # For definitions
            if verb.get('radiko', '').lower() == 'est':
                slots['predicate'] = text

        return slots

    def _extract_text(self, node: Dict) -> str:
        """Extract text from AST node."""
        if not node or not isinstance(node, dict):
            return ""

        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            vortspeco = node.get('vortspeco', 'substantivo')

            if vortspeco == 'substantivo':
                return root + 'o'
            elif vortspeco == 'adjektivo':
                return root + 'a'
            elif vortspeco == 'verbo':
                return root + 'i'
            return root

        elif node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno', {})
            text = self._extract_text(kerno)

            # Add adjectives
            priskriboj = node.get('priskriboj', [])
            if priskriboj:
                adj_texts = [self._extract_text(p) for p in priskriboj if isinstance(p, dict)]
                if adj_texts:
                    text = ' '.join(adj_texts) + ' ' + text

            return text

        return ""

    def phase1_count_patterns(self):
        """Phase 1: Stream through corpus and count patterns."""
        logger.info("=" * 60)
        logger.info("Phase 1: Counting Patterns")
        logger.info("=" * 60)

        self.phase = 'pattern_counting'
        self.pattern_counts.clear()
        self.total_sentences = 0

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line)
                    text = doc['text']

                    # Parse and detect patterns
                    try:
                        ast = parse(text)
                        patterns = self._detect_synthetic_patterns(ast)

                        for pattern in patterns:
                            self.pattern_counts[pattern] += 1

                        self.total_sentences += 1

                        if self.total_sentences % 100000 == 0:
                            logger.info(f"  Processed {self.total_sentences:,} sentences...")
                            self.save_checkpoint()

                    except Exception:
                        continue

                except json.JSONDecodeError:
                    continue

        logger.info(f"Processed {self.total_sentences:,} total sentences")
        logger.info("Pattern distribution:")
        for pattern, count in sorted(self.pattern_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {pattern}: {count:,} ({count/self.total_sentences*100:.1f}%)")

        self.save_checkpoint()

    def phase2_generate_synthetic(self, num_samples: int = 30000):
        """Phase 2: Generate synthetic pairs using reservoir sampling."""
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Synthetic Query Generation")
        logger.info("=" * 60)

        self.phase = 'synthetic_generation'

        # Calculate samples per pattern
        patterns = list(self.SYNTHETIC_TEMPLATES.keys())
        samples_per_pattern = num_samples // len(patterns)

        # Create reservoir samplers for each pattern
        reservoirs = {
            pattern: ReservoirSampler(samples_per_pattern)
            for pattern in patterns
        }

        # Stream through corpus and fill reservoirs
        logger.info(f"Collecting {num_samples:,} samples using reservoir sampling...")

        sentences_seen = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc['text']
                    doc_id = doc.get('doc_id')

                    try:
                        ast = parse(text)
                        patterns = self._detect_synthetic_patterns(ast)

                        # Add to reservoirs
                        for pattern in patterns:
                            if pattern in reservoirs:
                                reservoirs[pattern].add({
                                    'text': text,
                                    'ast': ast,
                                    'doc_id': doc_id,
                                })

                        sentences_seen += 1

                        if sentences_seen % 100000 == 0:
                            logger.info(f"  Streamed {sentences_seen:,} sentences...")

                    except Exception:
                        continue

                except json.JSONDecodeError:
                    continue

        logger.info(f"Reservoir sampling complete ({sentences_seen:,} sentences)")

        # Generate pairs from reservoirs
        logger.info("Generating query-document pairs...")

        self.synthetic_pairs = []

        for pattern, reservoir in reservoirs.items():
            samples = reservoir.sample()
            logger.info(f"  {pattern}: {len(samples):,} samples")

            templates = self.SYNTHETIC_TEMPLATES.get(pattern, [])

            for sentence in samples:
                slots = self._extract_slots(sentence['ast'])

                for template_name, template, relevance in templates:
                    try:
                        query_text = template.format(**slots)

                        pair = {
                            'query': query_text,
                            'doc_text': sentence['text'],
                            'doc_id': sentence.get('doc_id'),
                            'relevance': relevance,
                            'strategy': f'synthetic_{pattern}',
                            'template': template_name,
                            'is_positive': True,
                        }
                        self.synthetic_pairs.append(pair)

                    except KeyError:
                        continue

        logger.info(f"Generated {len(self.synthetic_pairs):,} positive synthetic pairs")

        # Generate negatives
        num_negatives = int(len(self.synthetic_pairs) * 2.0)
        logger.info(f"Generating {num_negatives:,} negative pairs...")

        negatives = self._generate_negatives_synthetic(num_negatives, reservoirs)
        self.synthetic_pairs.extend(negatives)

        logger.info(f"Total synthetic pairs: {len(self.synthetic_pairs):,}")

        self.save_checkpoint()

    def _generate_negatives_synthetic(
        self,
        num_negatives: int,
        reservoirs: Dict[str, ReservoirSampler],
    ) -> List[Dict]:
        """Generate negative pairs for synthetic data."""
        negatives = []

        # Collect all samples
        all_samples = []
        for reservoir in reservoirs.values():
            all_samples.extend(reservoir.sample())

        if not all_samples or not self.synthetic_pairs:
            return negatives

        # 40% hard, 40% medium, 20% easy
        num_hard = int(num_negatives * 0.4)
        num_medium = int(num_negatives * 0.4)
        num_easy = num_negatives - num_hard - num_medium

        # Hard negatives
        for _ in range(num_hard):
            pos = random.choice(self.synthetic_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.2,
                'strategy': 'negative_hard',
                'is_positive': False,
                'negative_type': 'hard',
            })

        # Medium negatives
        for _ in range(num_medium):
            pos = random.choice(self.synthetic_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.15,
                'strategy': 'negative_medium',
                'is_positive': False,
                'negative_type': 'medium',
            })

        # Easy negatives
        for _ in range(num_easy):
            pos = random.choice(self.synthetic_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.05,
                'strategy': 'negative_easy',
                'is_positive': False,
                'negative_type': 'easy',
            })

        return negatives

    def phase3_mine_patterns(self, num_samples: int = 20000):
        """Phase 3: Mine real Q&A pairs using reservoir sampling."""
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: Pattern Mining")
        logger.info("=" * 60)

        self.phase = 'pattern_mining'

        # Calculate samples per pattern
        patterns = list(self.MINING_PATTERNS.keys())
        samples_per_pattern = num_samples // len(patterns)

        # Create reservoir samplers
        reservoirs = {
            pattern: ReservoirSampler(samples_per_pattern)
            for pattern in patterns
        }

        # Stream through corpus and fill reservoirs
        logger.info(f"Collecting {num_samples:,} samples using reservoir sampling...")

        sentences_seen = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc['text']
                    doc_id = doc.get('doc_id')

                    try:
                        ast = parse(text)

                        # Check which mining patterns match
                        for pattern_name, pattern_spec in self.MINING_PATTERNS.items():
                            if pattern_spec['detector'](ast):
                                reservoirs[pattern_name].add({
                                    'text': text,
                                    'ast': ast,
                                    'doc_id': doc_id,
                                    'pattern': pattern_name,
                                })

                        sentences_seen += 1

                        if sentences_seen % 100000 == 0:
                            logger.info(f"  Streamed {sentences_seen:,} sentences...")

                    except Exception:
                        continue

                except json.JSONDecodeError:
                    continue

        logger.info(f"Reservoir sampling complete ({sentences_seen:,} sentences)")

        # Generate pairs from reservoirs
        logger.info("Mining query-document pairs...")

        self.mined_pairs = []

        for pattern_name, reservoir in reservoirs.items():
            samples = reservoir.sample()
            logger.info(f"  {pattern_name}: {len(samples):,} samples")

            pattern_spec = self.MINING_PATTERNS[pattern_name]
            query_template = pattern_spec['query_template']

            for sentence in samples:
                slots = self._extract_slots(sentence['ast'])

                try:
                    query_text = query_template.format(**slots)

                    pair = {
                        'query': query_text,
                        'doc_text': sentence['text'],
                        'doc_id': sentence.get('doc_id'),
                        'relevance': 1.0,
                        'strategy': f'mined_{pattern_name}',
                        'pattern': pattern_name,
                        'is_positive': True,
                    }
                    self.mined_pairs.append(pair)

                except KeyError:
                    continue

        logger.info(f"Mined {len(self.mined_pairs):,} positive pairs")

        # Generate negatives
        num_negatives = int(len(self.mined_pairs) * 2.0)
        logger.info(f"Generating {num_negatives:,} negative pairs...")

        negatives = self._generate_negatives_mined(num_negatives, reservoirs)
        self.mined_pairs.extend(negatives)

        logger.info(f"Total mined pairs: {len(self.mined_pairs):,}")

        self.save_checkpoint()

    def _generate_negatives_mined(
        self,
        num_negatives: int,
        reservoirs: Dict[str, ReservoirSampler],
    ) -> List[Dict]:
        """Generate negative pairs for mined data."""
        negatives = []

        # Collect all samples
        all_samples = []
        for reservoir in reservoirs.values():
            all_samples.extend(reservoir.sample())

        if not all_samples or not self.mined_pairs:
            return negatives

        # 50% hard, 30% medium, 20% easy
        num_hard = int(num_negatives * 0.5)
        num_medium = int(num_negatives * 0.3)
        num_easy = num_negatives - num_hard - num_medium

        # Hard negatives
        for _ in range(num_hard):
            pos = random.choice(self.mined_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.25,
                'strategy': 'mined_negative_hard',
                'is_positive': False,
                'negative_type': 'hard',
            })

        # Medium negatives
        for _ in range(num_medium):
            pos = random.choice(self.mined_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.2,
                'strategy': 'mined_negative_medium',
                'is_positive': False,
                'negative_type': 'medium',
            })

        # Easy negatives
        for _ in range(num_easy):
            pos = random.choice(self.mined_pairs)
            neg_sample = random.choice(all_samples)

            negatives.append({
                'query': pos['query'],
                'doc_text': neg_sample['text'],
                'doc_id': neg_sample.get('doc_id'),
                'relevance': 0.05,
                'strategy': 'mined_negative_easy',
                'is_positive': False,
                'negative_type': 'easy',
            })

        return negatives

    def phase4_save_datasets(self):
        """Phase 4: Split and save datasets."""
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: Saving Datasets")
        logger.info("=" * 60)

        self.phase = 'saving'

        # Save synthetic
        if self.synthetic_pairs:
            logger.info("Saving synthetic dataset...")
            synthetic_dir = self.output_dir / 'synthetic'
            self._save_split(self.synthetic_pairs, synthetic_dir)

        # Save mined
        if self.mined_pairs:
            logger.info("\nSaving mined dataset...")
            mined_dir = self.output_dir / 'mined'
            self._save_split(self.mined_pairs, mined_dir)

        # Merge and save combined
        if self.synthetic_pairs and self.mined_pairs:
            logger.info("\nMerging datasets...")
            combined = self.synthetic_pairs + self.mined_pairs
            random.shuffle(combined)

            combined_dir = self.output_dir / 'combined'
            self._save_split(combined, combined_dir)

            logger.info(f"  Combined: {len(self.synthetic_pairs):,} synthetic + {len(self.mined_pairs):,} mined = {len(combined):,} total")

        self.phase = 'complete'
        self.save_checkpoint()

    def _save_split(self, pairs: List[Dict], output_dir: Path):
        """Split and save train/val/test."""
        random.shuffle(pairs)

        n = len(pairs)
        train_size = int(n * 0.8)
        val_size = int(n * 0.1)

        train = pairs[:train_size]
        val = pairs[train_size:train_size + val_size]
        test = pairs[train_size + val_size:]

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            output_file = output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in split_data:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')

            logger.info(f"  {split_name}: {len(split_data):,} pairs → {output_file}")

    def generate(
        self,
        strategy: str = 'both',
        num_synthetic: int = 30000,
        num_mined: int = 20000,
    ):
        """
        Run full generation pipeline.

        Args:
            strategy: 'synthetic', 'pattern_mining', or 'both'
            num_synthetic: Number of synthetic samples
            num_mined: Number of mined samples
        """
        logger.info("=" * 60)
        logger.info("Streaming Reranker Data Generation")
        logger.info("=" * 60)
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Corpus: {self.corpus_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")

        # Phase 1: Count patterns (always needed)
        if self.phase == 'pattern_counting' or self.phase == 'complete':
            self.phase1_count_patterns()

        # Phase 2: Synthetic generation
        if strategy in ['synthetic', 'both']:
            if self.phase in ['pattern_counting', 'synthetic_generation']:
                self.phase2_generate_synthetic(num_synthetic)

        # Phase 3: Pattern mining
        if strategy in ['pattern_mining', 'both']:
            if self.phase in ['pattern_counting', 'synthetic_generation', 'pattern_mining']:
                self.phase3_mine_patterns(num_mined)

        # Phase 4: Save datasets
        self.phase4_save_datasets()

        logger.info("\n" + "=" * 60)
        logger.info("Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate reranker training data (streaming version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['synthetic', 'pattern_mining', 'both'],
        default='both',
        help='Data generation strategy'
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Input corpus file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/reranker/'),
        help='Output directory'
    )
    parser.add_argument(
        '--num-synthetic',
        type=int,
        default=30000,
        help='Number of synthetic samples'
    )
    parser.add_argument(
        '--num-mined',
        type=int,
        default=20000,
        help='Number of mined samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Check corpus exists
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    # Initialize generator
    generator = StreamingDataGenerator(
        corpus_path=args.corpus,
        output_dir=args.output,
    )

    # Load checkpoint if requested
    if args.resume and not args.fresh:
        if generator.load_checkpoint():
            logger.info("Resuming from checkpoint\n")
        else:
            logger.info("No checkpoint found, starting fresh\n")
    elif args.fresh:
        logger.info("Starting fresh (ignoring checkpoint)\n")

    # Generate
    generator.generate(
        strategy=args.strategy,
        num_synthetic=args.num_synthetic,
        num_mined=args.num_mined,
    )


if __name__ == '__main__':
    main()
