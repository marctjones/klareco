"""
Training Data Generation for Entity Type Classifier.

Generates training data using three strategies:
1. Corpus auto-labeling: Use deterministic features on existing corpus (~70% coverage)
2. Test set extraction: Extract labeled examples from test questions
3. Synthetic generation: Create examples from root vocabulary + affixes

Philosophy: Maximize reuse of existing deterministic knowledge.
"""

from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from collections import defaultdict

from .deterministic import DeterministicFeatureExtractor
from .enricher import ASTSemanticEnricher
from .taxonomy import PersonType, LocationType, TimeType, ThingType

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generate training data for entity type classifier.

    Combines deterministic auto-labeling with test set extraction and
    synthetic generation to create a comprehensive training dataset.
    """

    def __init__(self):
        """Initialize training data generator."""
        self.enricher = ASTSemanticEnricher(model=None)
        self.deterministic_extractor = DeterministicFeatureExtractor()

        logger.info("TrainingDataGenerator initialized")

    def auto_label_corpus(
        self,
        corpus_path: Path,
        output_path: Path,
        min_confidence: float = 0.70
    ) -> Dict:
        """
        Auto-label corpus using deterministic features.

        Args:
            corpus_path: Path to corpus JSONL (unified_corpus.jsonl)
            output_path: Path to save enriched corpus
            min_confidence: Minimum confidence to include (default 0.70)

        Returns:
            Statistics:
            {
                'total_words': int,
                'enriched': int,
                'high_confidence': int,  # >= 0.90
                'medium_confidence': int,  # 0.70-0.90
                'low_confidence': int,  # < 0.70 (excluded)
                'deterministic_coverage': float
            }
        """
        logger.info(f"Auto-labeling corpus from {corpus_path}")

        total_words = 0
        enriched = 0
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(corpus_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                try:
                    sentence_data = json.loads(line)
                    ast = sentence_data.get('ast')

                    if not ast:
                        continue

                    # Extract words from sentence AST
                    words = self._extract_words_from_ast(ast)

                    for word_ast in words:
                        total_words += 1

                        # Enrich with deterministic features
                        enriched_ast = self.enricher.enrich(word_ast)
                        annotation = enriched_ast.get('semantic_annotation', {})
                        final = annotation.get('final_classification', {})
                        confidence = final.get('confidence', 0.0)

                        # Filter by confidence
                        if confidence < min_confidence:
                            low_confidence += 1
                            continue

                        # Track confidence distribution
                        if confidence >= 0.90:
                            high_confidence += 1
                        else:
                            medium_confidence += 1

                        enriched += 1

                        # Save enriched AST
                        f_out.write(json.dumps(enriched_ast, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line[:100]}")
                    continue

        stats = {
            'total_words': total_words,
            'enriched': enriched,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'deterministic_coverage': (high_confidence / total_words * 100) if total_words > 0 else 0
        }

        logger.info(f"Auto-labeled {enriched}/{total_words} words ({enriched/total_words*100:.1f}%)")
        logger.info(f"High confidence (>=0.90): {high_confidence} ({high_confidence/total_words*100:.1f}%)")
        logger.info(f"Medium confidence (0.70-0.90): {medium_confidence}")
        logger.info(f"Excluded low confidence (<0.70): {low_confidence}")

        return stats

    def _extract_words_from_ast(self, ast: Dict) -> List[Dict]:
        """Extract individual word ASTs from sentence AST."""
        words = []

        if ast.get('tipo') == 'vorto':
            return [ast]

        # Handle vortgrupo
        if ast.get('tipo') == 'vortgrupo':
            kerno = ast.get('kerno')
            if kerno:
                words.append(kerno)
            priskriboj = ast.get('priskriboj', [])
            for priskribo in priskriboj:
                words.extend(self._extract_words_from_ast(priskribo))

        # Handle frazo
        if ast.get('tipo') == 'frazo':
            for role in ['subjekto', 'verbo', 'objekto']:
                component = ast.get(role)
                if component:
                    words.extend(self._extract_words_from_ast(component))
            aliaj = ast.get('aliaj', [])
            for other in aliaj:
                words.extend(self._extract_words_from_ast(other))

        return words

    def extract_from_test_set(
        self,
        test_set_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Extract labeled examples from test set.

        Args:
            test_set_path: Path to test set JSONL
            output_path: Path to save training examples

        Returns:
            Statistics:
            {
                'total_questions': int,
                'examples_extracted': int,
                'person_examples': int,
                'location_examples': int,
                'time_examples': int,
                'other_examples': int
            }
        """
        logger.info(f"Extracting examples from test set: {test_set_path}")

        total_questions = 0
        examples_extracted = 0
        type_counts = defaultdict(int)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(test_set_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                try:
                    question_data = json.loads(line)
                    total_questions += 1

                    # Extract entities from question
                    question = question_data.get('question', '')
                    expected_answer = question_data.get('expected_answer', '')
                    question_type = question_data.get('question_type', '')

                    # Parse question to AST (if we have parser access)
                    # For now, create simplified examples
                    examples = self._extract_examples_from_question(
                        question,
                        expected_answer,
                        question_type
                    )

                    for example in examples:
                        examples_extracted += 1
                        tier2_type = example.get('label', {}).get('tier2_type')
                        if tier2_type:
                            type_counts[tier2_type] += 1

                        f_out.write(json.dumps(example, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line[:100]}")
                    continue

        stats = {
            'total_questions': total_questions,
            'examples_extracted': examples_extracted,
            **{f'{k}_examples': v for k, v in type_counts.items()}
        }

        logger.info(f"Extracted {examples_extracted} examples from {total_questions} questions")
        logger.info(f"Type distribution: {dict(type_counts)}")

        return stats

    def _extract_examples_from_question(
        self,
        question: str,
        expected_answer: str,
        question_type: str
    ) -> List[Dict]:
        """Extract training examples from question/answer pair."""
        examples = []

        # Map question types to entity types
        type_mapping = {
            'WHO': 'person',
            'WHERE': 'location',
            'WHEN': 'time_point',
            'WHAT': 'thing',
        }

        tier2_type = type_mapping.get(question_type)
        if not tier2_type:
            return examples

        # Create example from expected answer
        # Note: This is simplified - in practice, would parse to AST
        if expected_answer and expected_answer != '[full document]':
            example = {
                'word_ast': {
                    'tipo': 'vorto',
                    'teksto': expected_answer,
                    'vortspeco': 'substantivo',  # Simplified
                },
                'label': {
                    'tier2_type': tier2_type,
                    'tier3_type': None,  # Would need manual annotation
                    'source': 'test_set'
                }
            }
            examples.append(example)

        return examples

    def generate_synthetic_examples(
        self,
        root_vocab_path: Path,
        output_path: Path,
        max_per_affix: int = 100
    ) -> Dict:
        """
        Generate synthetic training examples from root vocabulary + affixes.

        Args:
            root_vocab_path: Path to root vocabulary JSON
            output_path: Path to save synthetic examples
            max_per_affix: Maximum examples per affix type

        Returns:
            Statistics:
            {
                'total_roots': int,
                'examples_generated': int,
                'by_affix': {affix: count, ...}
            }
        """
        logger.info(f"Generating synthetic examples from {root_vocab_path}")

        # Load root vocabulary
        with open(root_vocab_path, 'r') as f:
            root_vocab = json.load(f)

        total_roots = len(root_vocab)
        examples_generated = 0
        affix_counts = defaultdict(int)

        # Affix templates with known semantic types
        affix_templates = [
            ('ist', PersonType.PERSON_PROFESSION, 0.95),
            ('ul', PersonType.PERSON_ROLE, 0.95),
            ('in', PersonType.PERSON_ROLE, 0.90),
            ('an', PersonType.PERSON_ROLE, 0.90),
            ('ej', LocationType.PLACE_INSTITUTION, 0.90),
            ('il', ThingType.THING_TOOL, 0.90),
            ('aĵ', ThingType.THING_CONCRETE, 0.85),
            ('ar', ThingType.THING_COLLECTION, 0.90),
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f_out:
            for affix, tier3_type, confidence in affix_templates:
                count = 0

                for root_entry in root_vocab[:max_per_affix]:
                    root = root_entry.get('root', root_entry) if isinstance(root_entry, dict) else root_entry

                    # Generate word with affix
                    word = f"{root}{affix}o"

                    # Create AST
                    word_ast = {
                        'tipo': 'vorto',
                        'vortspeco': 'substantivo',
                        'radiko': root,
                        'sufiksoj': [affix],
                        'teksto': word,
                        'kazo': 'nominativo',
                        'nombro': 'singularo'
                    }

                    # Create training example
                    example = {
                        'word_ast': word_ast,
                        'label': {
                            'tier3_type': tier3_type.value,
                            'confidence': confidence,
                            'source': 'synthetic'
                        }
                    }

                    f_out.write(json.dumps(example, ensure_ascii=False) + '\n')

                    count += 1
                    examples_generated += 1
                    affix_counts[affix] = count

        stats = {
            'total_roots': total_roots,
            'examples_generated': examples_generated,
            'by_affix': dict(affix_counts)
        }

        logger.info(f"Generated {examples_generated} synthetic examples")
        logger.info(f"By affix: {dict(affix_counts)}")

        return stats

    def create_training_dataset(
        self,
        enriched_corpus_path: Path,
        test_examples_path: Path,
        synthetic_examples_path: Path,
        output_path: Path,
        validation_split: float = 0.15
    ) -> Dict:
        """
        Combine all data sources into training/validation datasets.

        Args:
            enriched_corpus_path: Auto-labeled corpus
            test_examples_path: Test set examples
            synthetic_examples_path: Synthetic examples
            output_path: Directory to save train.jsonl and val.jsonl
            validation_split: Fraction for validation (default 0.15)

        Returns:
            Statistics:
            {
                'total_examples': int,
                'train_examples': int,
                'val_examples': int,
                'source_distribution': {...}
            }
        """
        logger.info("Creating training dataset from all sources")

        all_examples = []

        # Load from all sources
        sources = [
            (enriched_corpus_path, 'corpus'),
            (test_examples_path, 'test_set'),
            (synthetic_examples_path, 'synthetic')
        ]

        source_counts = defaultdict(int)

        for source_path, source_name in sources:
            if not source_path.exists():
                logger.warning(f"Source not found: {source_path}")
                continue

            with open(source_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        example = json.loads(line)
                        # Normalize format
                        training_example = self._normalize_example(example, source_name)
                        if training_example:
                            all_examples.append(training_example)
                            source_counts[source_name] += 1
                    except json.JSONDecodeError:
                        continue

        # Shuffle and split
        import random
        random.shuffle(all_examples)

        val_size = int(len(all_examples) * validation_split)
        train_examples = all_examples[val_size:]
        val_examples = all_examples[:val_size]

        # Save splits
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / 'train.jsonl'
        val_path = output_path / 'val.jsonl'

        with open(train_path, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        with open(val_path, 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        stats = {
            'total_examples': len(all_examples),
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'source_distribution': dict(source_counts)
        }

        logger.info(f"Created training dataset: {len(train_examples)} train, {len(val_examples)} val")
        logger.info(f"Source distribution: {dict(source_counts)}")

        return stats

    def _normalize_example(self, example: Dict, source: str) -> Optional[Dict]:
        """
        Normalize example to consistent format.

        Format:
        {
            'word_ast': {...},
            'context_ast': {...},  # Optional
            'deterministic_priors': {...},  # From deterministic extractor
            'label': {
                'tier3_type': 'person_profession',
                'confidence': 0.95,
                'source': 'corpus' | 'test_set' | 'synthetic'
            }
        }
        """
        # If already enriched (from corpus auto-labeling)
        if 'semantic_annotation' in example:
            annotation = example['semantic_annotation']
            det_features = annotation.get('deterministic_features', {})
            final = annotation.get('final_classification', {})

            # Only use high-confidence examples for training
            if final.get('confidence', 0) < 0.70:
                return None

            return {
                'word_ast': {k: v for k, v in example.items() if k != 'semantic_annotation'},
                'deterministic_priors': det_features,
                'label': {
                    'tier3_type': final.get('tier3_type'),
                    'confidence': final.get('confidence'),
                    'source': source
                }
            }

        # If from test set or synthetic (already has label)
        if 'label' in example:
            word_ast = example.get('word_ast', {})

            # Extract deterministic priors
            det_features = self.deterministic_extractor.extract(word_ast)

            return {
                'word_ast': word_ast,
                'deterministic_priors': {
                    'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                    'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                    'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                    'confidence': det_features['confidence']
                },
                'label': example['label']
            }

        return None
