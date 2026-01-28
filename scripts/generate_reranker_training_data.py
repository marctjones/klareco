#!/usr/bin/env python3
"""
Generate training data for query-document reranker.

Supports multiple strategies:
1. Synthetic query generation (template-based)
2. Pattern mining (extract real Q&A pairs)
3. BM25 pseudo-labeling (use current retriever)

Usage:
    # Synthetic queries (fast, high quality)
    python scripts/generate_reranker_training_data.py \
        --strategy synthetic \
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
        --output data/training/reranker/synthetic/ \
        --num-samples 30000

    # Pattern mining (slower, highest quality)
    python scripts/generate_reranker_training_data.py \
        --strategy pattern_mining \
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
        --output data/training/reranker/mined/ \
        --num-samples 20000

    # Both strategies
    python scripts/generate_reranker_training_data.py \
        --strategy both \
        --output data/training/reranker/combined/
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


class SyntheticQueryGenerator:
    """
    Generate synthetic queries from corpus sentences using templates.

    Strategy: Given sentence, generate plausible questions it would answer.
    """

    # Question templates by pattern type
    TEMPLATES = {
        'definition': [
            # "X estas Y" → "Kio estas X?"
            ('kio_estas', "Kio estas {subject}?", 1.0),
            ('cu_estas', "Ĉu {subject} estas {predicate}?", 0.9),
            ('kia_estas', "Kia estas {subject}?", 0.8),
        ],
        'factual_who': [
            # "S V-is O" → "Kiu V-is O?"
            ('kiu_verbo', "Kiu {verb} {object}?", 1.0),
            ('kiu_verbo_kiam', "Kiu {verb} {object} en {time}?", 0.95),
        ],
        'factual_what': [
            # "S V-is O" → "Kion S V-is?"
            ('kion_verbo', "Kion {subject} {verb}?", 1.0),
        ],
        'location': [
            # "X estas en Y" → "Kie estas X?"
            ('kie_estas', "Kie estas {subject}?", 1.0),
            ('en_kiu_lando', "En kiu lando estas {subject}?", 0.95),
        ],
        'temporal': [
            # "X V-is en DATO" → "Kiam V-is X?"
            ('kiam_verbo', "Kiam {verb} {subject}?", 1.0),
            ('en_kiu_jaro', "En kiu jaro {verb} {subject}?", 0.95),
        ],
        'causal': [
            # "X V-as ĉar Y" → "Kial X V-as?"
            ('kial_verbo', "Kial {subject} {verb}?", 1.0),
        ],
    }

    def __init__(self, corpus_path: Path):
        """Initialize generator with corpus."""
        self.corpus_path = corpus_path
        self.sentences = []
        self.by_pattern = defaultdict(list)

    def load_corpus(self, max_sentences: Optional[int] = None):
        """Load and index corpus by sentence patterns."""
        logger.info(f"Loading corpus from {self.corpus_path}...")

        count = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_sentences and count >= max_sentences:
                    break

                try:
                    doc = json.loads(line)
                    sentence = {
                        'text': doc['text'],
                        'doc_id': doc.get('doc_id'),
                        'source': doc.get('source', {}),
                    }

                    # Parse and detect patterns
                    try:
                        ast = parse(doc['text'])
                        sentence['ast'] = ast

                        # Classify by pattern
                        patterns = self._detect_patterns(ast)
                        for pattern in patterns:
                            self.by_pattern[pattern].append(sentence)

                        self.sentences.append(sentence)
                        count += 1

                        if count % 10000 == 0:
                            logger.info(f"  Loaded {count:,} sentences...")

                    except Exception as e:
                        # Skip unparseable sentences
                        continue

                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.sentences):,} sentences")
        logger.info(f"Pattern distribution:")
        for pattern, sents in sorted(self.by_pattern.items(), key=lambda x: -len(x[1])):
            logger.info(f"  {pattern}: {len(sents):,}")

    def _detect_patterns(self, ast: Dict) -> Set[str]:
        """
        Detect which patterns this sentence matches.

        Returns:
            Set of pattern names (e.g., 'definition', 'factual_who', 'location')
        """
        patterns = set()

        verb = ast.get('verbo', {})
        verb_root = verb.get('radiko', '').lower()

        subjekto = ast.get('subjekto')
        objekto = ast.get('objekto')

        # Pattern: "X estas Y" (definition)
        if verb_root == 'est' and subjekto and objekto:
            # Check if object is predicate nominative (not accusative)
            if objekto.get('tipo') == 'vortgrupo':
                obj_kerno = objekto.get('kerno', {})
                if obj_kerno.get('kazo') != 'akuzativo':
                    patterns.add('definition')
            elif objekto.get('tipo') == 'vorto':
                if objekto.get('kazo') != 'akuzativo':
                    patterns.add('definition')

        # Pattern: "S V-is O" (factual action)
        if verb_root != 'est' and subjekto and objekto:
            patterns.add('factual_who')
            patterns.add('factual_what')

        # Pattern: "X estas en/ĉe Y" (location)
        if verb_root == 'est':
            for modifier in ast.get('aliaj', []):
                if isinstance(modifier, dict):
                    mod_root = modifier.get('radiko', '').lower()
                    if mod_root in ['en', 'ĉe', 'sur', 'sub', 'apud']:
                        patterns.add('location')
                        break

        # Pattern: temporal (has time expression)
        for modifier in ast.get('aliaj', []):
            if isinstance(modifier, dict):
                mod_root = modifier.get('radiko', '').lower()
                # Check for years, months, temporal markers
                if mod_root in ['jar', 'monat', 'tag', 'hodiaŭ', 'hieraŭ', 'morgaŭ']:
                    patterns.add('temporal')
                    break

        # Pattern: causal (has "ĉar")
        for modifier in ast.get('aliaj', []):
            if isinstance(modifier, dict):
                if modifier.get('radiko', '').lower() == 'ĉar':
                    patterns.add('causal')
                    break

        return patterns

    def _extract_slots(self, ast: Dict) -> Dict[str, str]:
        """
        Extract slots from AST for template filling.

        Returns:
            Dictionary with slots like 'subject', 'verb', 'object', 'predicate', 'time'
        """
        slots = {}

        # Extract subject
        subjekto = ast.get('subjekto')
        if subjekto:
            slots['subject'] = self._extract_text(subjekto)

        # Extract verb
        verb = ast.get('verbo', {})
        if verb.get('radiko'):
            # Convert to appropriate form for question
            verb_root = verb['radiko']
            # For questions, often use present tense
            slots['verb'] = verb_root + 'is'  # Simple past for "Kiu fondis..."

        # Extract object
        objekto = ast.get('objekto')
        if objekto:
            slots['object'] = self._extract_text(objekto)
            # Also check if it's predicate nominative (for definitions)
            if verb.get('radiko', '').lower() == 'est':
                slots['predicate'] = self._extract_text(objekto)

        # Extract time expressions
        for modifier in ast.get('aliaj', []):
            if isinstance(modifier, dict):
                text = modifier.get('radiko', '')
                # Check if it looks like a year
                if text.isdigit() and 1000 <= int(text) <= 2100:
                    slots['time'] = text
                    break

        return slots

    def _extract_text(self, node: Dict) -> str:
        """Extract readable text from AST node."""
        if not node or not isinstance(node, dict):
            return ""

        if node.get('tipo') == 'vorto':
            return node.get('radiko', '')

        elif node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno', {})
            text = kerno.get('radiko', '')

            # Add adjectives
            for prisk in node.get('priskriboj', []):
                if isinstance(prisk, dict):
                    adj = prisk.get('radiko', '')
                    if adj:
                        text = f"{adj} {text}"

            return text

        return ""

    def generate_pairs(
        self,
        num_samples: int = 30000,
        negative_ratio: float = 2.0,
    ) -> List[Dict]:
        """
        Generate query-document pairs.

        Args:
            num_samples: Target number of positive samples
            negative_ratio: Negatives per positive (default: 2.0)

        Returns:
            List of training examples
        """
        logger.info(f"Generating {num_samples:,} training pairs...")
        logger.info(f"  Negative ratio: {negative_ratio}")

        pairs = []

        # Generate positives from each pattern type
        patterns_to_generate = list(self.TEMPLATES.keys())
        samples_per_pattern = num_samples // len(patterns_to_generate)

        for pattern_type in patterns_to_generate:
            if pattern_type not in self.by_pattern:
                logger.warning(f"  No sentences for pattern: {pattern_type}")
                continue

            candidates = self.by_pattern[pattern_type]
            if not candidates:
                continue

            logger.info(f"  Generating from pattern '{pattern_type}' ({len(candidates):,} candidates)...")

            # Sample sentences
            sample_size = min(samples_per_pattern, len(candidates))
            sampled = random.sample(candidates, sample_size)

            for sentence in sampled:
                # Generate queries using templates
                slots = self._extract_slots(sentence['ast'])

                for template_name, template, relevance in self.TEMPLATES[pattern_type]:
                    try:
                        query_text = template.format(**slots)

                        # Create positive pair
                        pair = {
                            'query': query_text,
                            'doc_text': sentence['text'],
                            'doc_id': sentence.get('doc_id'),
                            'relevance': relevance,
                            'strategy': f'synthetic_{pattern_type}',
                            'template': template_name,
                            'is_positive': True,
                        }
                        pairs.append(pair)

                    except KeyError:
                        # Template needs slots we don't have
                        continue

        logger.info(f"  Generated {len(pairs):,} positive pairs")

        # Generate negative examples
        num_negatives = int(len(pairs) * negative_ratio)
        logger.info(f"  Generating {num_negatives:,} negative pairs...")

        negatives = self._generate_negatives(pairs, num_negatives)
        pairs.extend(negatives)

        logger.info(f"Total pairs generated: {len(pairs):,}")
        logger.info(f"  Positive: {sum(1 for p in pairs if p['is_positive']):,}")
        logger.info(f"  Negative: {sum(1 for p in pairs if not p['is_positive']):,}")

        return pairs

    def _generate_negatives(
        self,
        positive_pairs: List[Dict],
        num_negatives: int,
    ) -> List[Dict]:
        """
        Generate negative examples for training.

        Types of negatives:
        1. Hard negatives: Same pattern type, different entity
        2. Medium negatives: Different pattern, related topic
        3. Easy negatives: Random sentences
        """
        negatives = []

        # 40% hard, 40% medium, 20% easy
        num_hard = int(num_negatives * 0.4)
        num_medium = int(num_negatives * 0.4)
        num_easy = num_negatives - num_hard - num_medium

        # Hard negatives: Same query, different document of same pattern
        for _ in range(num_hard):
            if not positive_pairs:
                break

            pos = random.choice(positive_pairs)
            pattern = pos['strategy'].replace('synthetic_', '')

            if pattern in self.by_pattern and len(self.by_pattern[pattern]) > 1:
                # Find different sentence with same pattern
                different_sent = random.choice(self.by_pattern[pattern])
                while different_sent['text'] == pos['doc_text']:
                    different_sent = random.choice(self.by_pattern[pattern])

                negative = {
                    'query': pos['query'],
                    'doc_text': different_sent['text'],
                    'doc_id': different_sent.get('doc_id'),
                    'relevance': 0.2,
                    'strategy': f"negative_hard_{pattern}",
                    'is_positive': False,
                    'negative_type': 'hard_same_pattern',
                }
                negatives.append(negative)

        # Medium negatives: Same topic (shared root), different pattern
        for _ in range(num_medium):
            if not positive_pairs or not self.sentences:
                break

            pos = random.choice(positive_pairs)

            # Try to find sentence with shared root but different pattern
            random_sent = random.choice(self.sentences)

            negative = {
                'query': pos['query'],
                'doc_text': random_sent['text'],
                'doc_id': random_sent.get('doc_id'),
                'relevance': 0.15,
                'strategy': 'negative_medium',
                'is_positive': False,
                'negative_type': 'medium_related',
            }
            negatives.append(negative)

        # Easy negatives: Completely random
        for _ in range(num_easy):
            if not positive_pairs or not self.sentences:
                break

            pos = random.choice(positive_pairs)
            random_sent = random.choice(self.sentences)

            negative = {
                'query': pos['query'],
                'doc_text': random_sent['text'],
                'doc_id': random_sent.get('doc_id'),
                'relevance': 0.05,
                'strategy': 'negative_easy',
                'is_positive': False,
                'negative_type': 'easy_random',
            }
            negatives.append(negative)

        return negatives


class PatternMiningGenerator:
    """
    Mine query-document pairs from corpus by finding sentences that answer questions.

    Strategy: Search for sentences matching answer patterns, then generate the questions
    they would naturally answer. This produces higher quality but fewer pairs than synthetic.
    """

    # Answer patterns we can mine
    ANSWER_PATTERNS = {
        'definition': {
            'pattern': 'X estas Y',
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() == 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kio estas {subject}?",
        },
        'who_action': {
            'pattern': 'S V-is O',
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() != 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kiu {verb} {object}?",
        },
        'what_action': {
            'pattern': 'S V-is O',
            'detector': lambda ast: (
                ast.get('verbo', {}).get('radiko', '').lower() != 'est'
                and ast.get('subjekto') is not None
                and ast.get('objekto') is not None
            ),
            'query_template': "Kion {subject} {verb}?",
        },
        'location': {
            'pattern': 'X estas en/ĉe Y',
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

    def __init__(self, corpus_path: Path):
        """Initialize pattern miner with corpus."""
        self.corpus_path = corpus_path
        self.sentences = []
        self.by_pattern = defaultdict(list)

    def load_corpus(self, max_sentences: Optional[int] = None):
        """Load corpus and classify sentences by answer patterns."""
        logger.info(f"Loading corpus for pattern mining from {self.corpus_path}...")

        count = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_sentences and count >= max_sentences:
                    break

                try:
                    doc = json.loads(line)
                    sentence = {
                        'text': doc['text'],
                        'doc_id': doc.get('doc_id'),
                        'source': doc.get('source', {}),
                    }

                    # Parse sentence
                    try:
                        ast = parse(doc['text'])
                        sentence['ast'] = ast

                        # Detect which answer patterns this matches
                        for pattern_name, pattern_spec in self.ANSWER_PATTERNS.items():
                            if pattern_spec['detector'](ast):
                                self.by_pattern[pattern_name].append(sentence)

                        self.sentences.append(sentence)
                        count += 1

                        if count % 10000 == 0:
                            logger.info(f"  Loaded {count:,} sentences...")

                    except Exception:
                        # Skip unparseable
                        continue

                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.sentences):,} sentences")
        logger.info(f"Answer pattern distribution:")
        for pattern, sents in sorted(self.by_pattern.items(), key=lambda x: -len(x[1])):
            logger.info(f"  {pattern}: {len(sents):,}")

    def mine_pairs(
        self,
        num_samples: int = 20000,
        negative_ratio: float = 2.0,
    ) -> List[Dict]:
        """
        Mine query-document pairs from corpus.

        Args:
            num_samples: Target number of positive samples
            negative_ratio: Negatives per positive

        Returns:
            List of training examples
        """
        logger.info(f"Mining {num_samples:,} training pairs from corpus...")
        logger.info(f"  Negative ratio: {negative_ratio}")

        pairs = []

        # Mine from each pattern type
        patterns_to_mine = list(self.ANSWER_PATTERNS.keys())
        samples_per_pattern = num_samples // len(patterns_to_mine)

        for pattern_name in patterns_to_mine:
            if pattern_name not in self.by_pattern:
                logger.warning(f"  No sentences for pattern: {pattern_name}")
                continue

            candidates = self.by_pattern[pattern_name]
            if not candidates:
                continue

            logger.info(f"  Mining from pattern '{pattern_name}' ({len(candidates):,} candidates)...")

            # Sample sentences that match this answer pattern
            sample_size = min(samples_per_pattern, len(candidates))
            sampled = random.sample(candidates, sample_size)

            pattern_spec = self.ANSWER_PATTERNS[pattern_name]
            query_template = pattern_spec['query_template']

            for sentence in sampled:
                # Extract slots from sentence
                slots = self._extract_slots(sentence['ast'])

                try:
                    # Generate the question this sentence answers
                    query_text = query_template.format(**slots)

                    # Create positive pair (high relevance - this is a real answer!)
                    pair = {
                        'query': query_text,
                        'doc_text': sentence['text'],
                        'doc_id': sentence.get('doc_id'),
                        'relevance': 1.0,  # Real answer → perfect relevance
                        'strategy': f'mined_{pattern_name}',
                        'pattern': pattern_name,
                        'is_positive': True,
                    }
                    pairs.append(pair)

                except KeyError:
                    # Missing required slot
                    continue

        logger.info(f"  Mined {len(pairs):,} positive pairs")

        # Generate negative examples
        num_negatives = int(len(pairs) * negative_ratio)
        logger.info(f"  Generating {num_negatives:,} negative pairs...")

        negatives = self._generate_negatives(pairs, num_negatives)
        pairs.extend(negatives)

        logger.info(f"Total pairs mined: {len(pairs):,}")
        logger.info(f"  Positive: {sum(1 for p in pairs if p['is_positive']):,}")
        logger.info(f"  Negative: {sum(1 for p in pairs if not p['is_positive']):,}")

        return pairs

    def _extract_slots(self, ast: Dict) -> Dict[str, str]:
        """
        Extract slots from AST for query generation.

        Returns:
            Dict with 'subject', 'verb', 'object', etc.
        """
        slots = {}

        # Extract subject
        subjekto = ast.get('subjekto')
        if subjekto:
            slots['subject'] = self._extract_text(subjekto)

        # Extract verb (convert to past tense for questions)
        verb = ast.get('verbo', {})
        if verb.get('radiko'):
            verb_root = verb['radiko']
            # For "Kiu fondis..." type questions, use past tense
            slots['verb'] = verb_root + 'is'

        # Extract object
        objekto = ast.get('objekto')
        if objekto:
            slots['object'] = self._extract_text(objekto)

        return slots

    def _extract_text(self, node: Dict) -> str:
        """Extract readable text from AST node."""
        if not node or not isinstance(node, dict):
            return ""

        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '')
            # Reconstruct with appropriate ending
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

    def _generate_negatives(
        self,
        positive_pairs: List[Dict],
        num_negatives: int,
    ) -> List[Dict]:
        """
        Generate negative examples.

        For pattern mining, we focus on hard negatives:
        - Same topic (shared roots) but wrong answer pattern
        - Same pattern type but different entity
        """
        negatives = []

        # 50% hard (same topic, wrong pattern)
        # 30% medium (same pattern, different entity)
        # 20% easy (random)
        num_hard = int(num_negatives * 0.5)
        num_medium = int(num_negatives * 0.3)
        num_easy = num_negatives - num_hard - num_medium

        # Hard negatives: Same topic roots, but wrong answer type
        for _ in range(num_hard):
            if not positive_pairs or not self.sentences:
                break

            pos = random.choice(positive_pairs)

            # Find sentence with shared root but different pattern
            # (This teaches: "mentions the topic but doesn't answer the question")
            random_sent = random.choice(self.sentences)

            negative = {
                'query': pos['query'],
                'doc_text': random_sent['text'],
                'doc_id': random_sent.get('doc_id'),
                'relevance': 0.25,  # Some relevance (mentions topic)
                'strategy': 'mined_negative_hard',
                'is_positive': False,
                'negative_type': 'hard_wrong_pattern',
            }
            negatives.append(negative)

        # Medium negatives: Same pattern type, different entity
        for _ in range(num_medium):
            if not positive_pairs:
                break

            pos = random.choice(positive_pairs)
            pattern = pos['pattern']

            if pattern in self.by_pattern and len(self.by_pattern[pattern]) > 1:
                # Find different sentence with same pattern
                different_sent = random.choice(self.by_pattern[pattern])
                while different_sent['text'] == pos['doc_text']:
                    different_sent = random.choice(self.by_pattern[pattern])

                negative = {
                    'query': pos['query'],
                    'doc_text': different_sent['text'],
                    'doc_id': different_sent.get('doc_id'),
                    'relevance': 0.2,
                    'strategy': 'mined_negative_medium',
                    'is_positive': False,
                    'negative_type': 'medium_same_pattern',
                }
                negatives.append(negative)

        # Easy negatives: Completely random
        for _ in range(num_easy):
            if not positive_pairs or not self.sentences:
                break

            pos = random.choice(positive_pairs)
            random_sent = random.choice(self.sentences)

            negative = {
                'query': pos['query'],
                'doc_text': random_sent['text'],
                'doc_id': random_sent.get('doc_id'),
                'relevance': 0.05,
                'strategy': 'mined_negative_easy',
                'is_positive': False,
                'negative_type': 'easy_random',
            }
            negatives.append(negative)

        return negatives


def split_dataset(
    pairs: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split pairs into train/val/test sets.

    Stratified by relevance and query pattern.
    """
    random.shuffle(pairs)

    n = len(pairs)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = pairs[:train_size]
    val = pairs[train_size:train_size + val_size]
    test = pairs[train_size + val_size:]

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train):,} ({len(train)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val):,} ({len(val)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test):,} ({len(test)/n*100:.1f}%)")

    return train, val, test


def save_dataset(pairs: List[Dict], output_path: Path, split_name: str):
    """Save pairs to JSONL file."""
    output_file = output_path / f"{split_name}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    logger.info(f"  Saved {len(pairs):,} pairs to {output_file}")


def merge_datasets(
    synthetic_dir: Path,
    mined_dir: Path,
    output_dir: Path,
):
    """
    Merge synthetic and mined datasets.

    Args:
        synthetic_dir: Directory with synthetic data
        mined_dir: Directory with mined data
        output_dir: Output directory for merged data
    """
    logger.info("\n" + "=" * 60)
    logger.info("Merging Synthetic + Mined Datasets")
    logger.info("=" * 60)

    for split_name in ['train', 'val', 'test']:
        synthetic_file = synthetic_dir / f"{split_name}.jsonl"
        mined_file = mined_dir / f"{split_name}.jsonl"

        if not synthetic_file.exists() or not mined_file.exists():
            logger.warning(f"  Skipping {split_name}: Missing files")
            continue

        # Load both
        synthetic_pairs = []
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            for line in f:
                synthetic_pairs.append(json.loads(line))

        mined_pairs = []
        with open(mined_file, 'r', encoding='utf-8') as f:
            for line in f:
                mined_pairs.append(json.loads(line))

        # Merge and shuffle
        merged = synthetic_pairs + mined_pairs
        random.shuffle(merged)

        # Save merged
        output_file = output_dir / f"{split_name}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in merged:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        logger.info(f"  {split_name}: {len(synthetic_pairs):,} synthetic + {len(mined_pairs):,} mined = {len(merged):,} total")

    logger.info(f"\nMerged data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate reranker training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['synthetic', 'pattern_mining', 'both'],
        default='synthetic',
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
        default=Path('data/training/reranker/synthetic/'),
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=30000,
        help='Number of positive samples to generate'
    )
    parser.add_argument(
        '--max-corpus-sentences',
        type=int,
        default=None,
        help='Max sentences to load from corpus (for testing)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Reranker Training Data Generation")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target samples: {args.num_samples:,}")
    logger.info("")

    # Check corpus exists
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    # Generate based on strategy
    if args.strategy in ['synthetic', 'both']:
        logger.info("=" * 60)
        logger.info("Strategy 1: Synthetic Query Generation")
        logger.info("=" * 60)

        generator = SyntheticQueryGenerator(args.corpus)
        generator.load_corpus(max_sentences=args.max_corpus_sentences)
        pairs = generator.generate_pairs(num_samples=args.num_samples)

        # Split and save
        train, val, test = split_dataset(pairs)

        output_dir = args.output if args.strategy == 'synthetic' else args.output / 'synthetic'
        save_dataset(train, output_dir, 'train')
        save_dataset(val, output_dir, 'val')
        save_dataset(test, output_dir, 'test')

        logger.info(f"\nSynthetic data saved to {output_dir}")

    if args.strategy in ['pattern_mining', 'both']:
        logger.info("\n" + "=" * 60)
        logger.info("Strategy 2: Pattern Mining")
        logger.info("=" * 60)

        miner = PatternMiningGenerator(args.corpus)
        miner.load_corpus(max_sentences=args.max_corpus_sentences)
        mined_pairs = miner.mine_pairs(num_samples=args.num_samples)

        # Split and save
        train, val, test = split_dataset(mined_pairs)

        output_dir = args.output if args.strategy == 'pattern_mining' else args.output / 'mined'
        save_dataset(train, output_dir, 'train')
        save_dataset(val, output_dir, 'val')
        save_dataset(test, output_dir, 'test')

        logger.info(f"\nMined data saved to {output_dir}")

    # Merge datasets if using 'both' strategy
    if args.strategy == 'both':
        synthetic_dir = args.output / 'synthetic'
        mined_dir = args.output / 'mined'
        combined_dir = args.output / 'combined'

        merge_datasets(synthetic_dir, mined_dir, combined_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Generation Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
