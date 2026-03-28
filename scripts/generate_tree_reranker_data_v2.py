#!/usr/bin/env python3
"""
Generate High-Quality Training Data for TreeMatchReranker (v2 - Improved Negatives)

VERSION: v2.2
COMPATIBLE WITH: v2.1 database, Whoosh index
DEPENDENCIES: Whoosh retriever, AST parser
STAGE: Data Generation

Description:
    Generates high-quality training examples with SMARTER hard negatives.

Improved Negative Mining Strategy:
    1. Type-mismatched negatives (WHO question → place/thing documents)
    2. Topic-related but wrong fact (same domain, different answer)
    3. Syntactically similar but semantically wrong (same verb, different subject)
    4. Random from different topics (completely unrelated)

Key Differences from v1:
    - v1: Hard negatives from BM25 ranks 6-20 (too relevant!)
    - v2: Smarter hard negatives based on entity type and syntax mismatch

Output:
    - JSONL with (query, doc, label, query_ast, doc_ast, negative_type)
    - 20 examples per question
    - Better training signal for semantic attention

Usage:
    python scripts/generate_tree_reranker_data_v2.py \\
        --questions data/test_sets/train_questions_150.jsonl \\
        --output data/training/tree_reranker_train_v2.jsonl \\
        --examples-per-question 20

Last Updated: 2026-03-26
Author: Claude + Marc
Related Issues: #704
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.importance_scorer import classify_question_type, QuestionType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedTreeRerankerDataGenerator:
    """Generate high-quality training data with smarter hard negatives."""

    def __init__(
        self,
        whoosh_index_dir: Path,
        kuzu_db_path: Path,
        random_seed: int = 42
    ):
        """Initialize data generator."""
        self.retriever = WhooshRetriever(
            whoosh_index_dir=whoosh_index_dir,
            kuzu_db_path=kuzu_db_path
        )
        random.seed(random_seed)

        # Cache for entity-type specific documents
        self.entity_type_cache = {
            'PERSON': [],
            'PLACE': [],
            'THING': [],
            'TIME': []
        }

        logger.info("ImprovedTreeRerankerDataGenerator initialized")
        logger.info(f"  Whoosh index: {whoosh_index_dir}")
        logger.info(f"  Kuzu database: {kuzu_db_path}")

    def generate_from_questions(
        self,
        questions: List[Dict],
        examples_per_question: int = 20
    ) -> List[Dict]:
        """Generate training examples from question list."""
        all_examples = []
        stats = defaultdict(int)

        for i, q in enumerate(questions):
            question_text = q.get('question', '')
            expected_keywords = q.get('expected_keywords', [])
            question_type = classify_question_type(question_text)

            logger.info(f"\n[{i+1}/{len(questions)}] {question_text}")
            logger.info(f"  Type: {question_type.value}")

            # Generate examples for this question
            examples = self._generate_for_question(
                question_text,
                expected_keywords,
                question_type,
                examples_per_question
            )

            all_examples.extend(examples)

            # Update stats
            stats['total_questions'] += 1
            stats['total_examples'] += len(examples)
            stats[f'type_{question_type.value}'] += 1

            logger.info(f"  Generated: {len(examples)} examples")

        # Log final stats
        logger.info(f"\n{'='*60}")
        logger.info("Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total questions: {stats['total_questions']}")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Examples per question: {stats['total_examples'] / stats['total_questions']:.1f}")

        return all_examples

    def _generate_for_question(
        self,
        question: str,
        expected_keywords: List[str],
        question_type: QuestionType,
        num_examples: int
    ) -> List[Dict]:
        """Generate training examples for single question with smarter negatives."""

        # Parse question to AST
        try:
            query_ast = parse(question)
        except Exception as e:
            logger.error(f"  Failed to parse: {e}")
            return []

        # Extract roots for retrieval
        roots = self._extract_roots_from_ast(query_ast)
        if not roots:
            logger.warning(f"  No roots extracted")
            return []

        # Retrieve candidates with Whoosh BM25
        try:
            candidates = self.retriever.retrieve(
                query_roots=roots,
                question_type=question_type.value,
                query_entity=None,
                top_k=100  # Get more for better sampling
            )
        except Exception as e:
            logger.error(f"  Retrieval failed: {e}")
            return []

        if not candidates:
            logger.warning(f"  No candidates retrieved")
            return []

        logger.info(f"  Retrieved: {len(candidates)} candidates")

        # Generate examples with improved strategy
        examples = []

        # === POSITIVES: Top-3 (very confident) ===
        num_positives = min(3, len(candidates))
        for doc in candidates[:num_positives]:
            example = self._create_example(question, query_ast, doc, label=1.0)
            if example:
                examples.append(example)

        logger.info(f"  Positives: {len([e for e in examples if e['label'] == 1.0])}")

        # === HARD NEGATIVES (IMPROVED STRATEGY) ===

        # Strategy 1: Type-mismatched (WHO → place, WHERE → person, etc.)
        type_mismatch_negs = self._get_type_mismatched_negatives(
            question_type, candidates, num=5
        )
        for doc, neg_type in type_mismatch_negs:
            example = self._create_example(question, query_ast, doc, label=0.0, negative_type=neg_type)
            if example:
                examples.append(example)

        logger.info(f"  Type-mismatched: {len(type_mismatch_negs)}")

        # Strategy 2: Syntax-similar but semantic-wrong (same verb, different subject)
        syntax_similar_negs = self._get_syntax_similar_negatives(
            query_ast, candidates, num=4
        )
        for doc, neg_type in syntax_similar_negs:
            example = self._create_example(question, query_ast, doc, label=0.0, negative_type=neg_type)
            if example:
                examples.append(example)

        logger.info(f"  Syntax-similar: {len(syntax_similar_negs)}")

        # Strategy 3: Topic-related but wrong answer (BM25 ranks 20-40)
        topic_related_negs = []
        if len(candidates) > 20:
            topic_related_negs = [(doc, 'topic_related') for doc in candidates[20:24]]
            for doc, neg_type in topic_related_negs:
                example = self._create_example(question, query_ast, doc, label=0.0, negative_type=neg_type)
                if example:
                    examples.append(example)

        logger.info(f"  Topic-related: {len(topic_related_negs)}")

        # Strategy 4: Random unrelated (completely different topics)
        random_negs = []
        for _ in range(3):
            doc = self._sample_random_document()
            if doc:
                random_negs.append((doc, 'random'))
                example = self._create_example(question, query_ast, doc, label=0.0, negative_type='random')
                if example:
                    examples.append(example)

        logger.info(f"  Random: {len(random_negs)}")

        # Balance to ~50/50 positive/negative
        examples = self._balance_examples(examples)

        # Limit to requested number
        if len(examples) > num_examples:
            examples = random.sample(examples, num_examples)

        return examples

    def _get_type_mismatched_negatives(
        self,
        question_type: QuestionType,
        candidates: List[Dict],
        num: int
    ) -> List[Tuple[Dict, str]]:
        """Get negatives with wrong entity type for question."""

        # Define what entity types SHOULDN'T match each question type
        mismatch_map = {
            QuestionType.WHO: ['PLACE', 'THING'],      # WHO → place/thing docs
            QuestionType.WHERE: ['PERSON'],             # WHERE → person docs
            QuestionType.WHEN: ['PERSON', 'PLACE'],    # WHEN → person/place docs
            QuestionType.WHAT: ['TIME'],                # WHAT → time docs
        }

        wrong_types = mismatch_map.get(question_type, [])
        if not wrong_types:
            return []

        # Find documents with wrong entity types
        mismatched = []
        for doc in candidates:
            doc_text = doc.get('text', '')
            detected_type = self._detect_entity_type(doc_text)

            if detected_type in wrong_types:
                mismatched.append((doc, f'type_mismatch_{detected_type.lower()}'))

            if len(mismatched) >= num:
                break

        return mismatched[:num]

    def _get_syntax_similar_negatives(
        self,
        query_ast: Dict,
        candidates: List[Dict],
        num: int
    ) -> List[Tuple[Dict, str]]:
        """Get negatives with same syntactic structure but different semantics."""

        # Extract query verb
        query_verb = self._get_verb_root(query_ast)
        if not query_verb:
            return []

        # Find documents with same verb but different subject/object
        syntax_similar = []
        for doc in candidates[10:]:  # Skip top-10 (likely too relevant)
            doc_ast = doc.get('ast')
            if not doc_ast:
                continue

            doc_verb = self._get_verb_root(doc_ast)
            if doc_verb == query_verb:
                # Same verb, different context
                syntax_similar.append((doc, 'syntax_similar'))

            if len(syntax_similar) >= num:
                break

        return syntax_similar[:num]

    def _detect_entity_type(self, text: str) -> str:
        """Simple entity type detection based on keywords."""
        text_lower = text.lower()

        # Person indicators
        person_words = ['li', 'ŝi', 'persono', 'viro', 'virino', 'homo', 'prezidanto', 'aŭtoro', 'verkisto']
        if any(word in text_lower for word in person_words):
            return 'PERSON'

        # Place indicators
        place_words = ['urbo', 'lando', 'regiono', 'kie', 'tie', 'loko', 'situas']
        if any(word in text_lower for word in place_words):
            return 'PLACE'

        # Time indicators
        time_words = ['jaro', 'monato', 'tago', 'kiam', 'tiam', 'tempo', 'epoko']
        if any(word in text_lower for word in time_words):
            return 'TIME'

        return 'THING'

    def _get_verb_root(self, ast: Dict) -> Optional[str]:
        """Extract verb root from AST."""
        if not isinstance(ast, dict):
            return None

        # Check if this node is a verb
        if ast.get('tipo') == 'vorto' and ast.get('vortspeco') == 'verbo':
            return ast.get('radiko', '').lower()

        # Recursively search for verb
        verb_node = ast.get('verbo')
        if verb_node:
            return self._get_verb_root(verb_node)

        # Search in other branches
        for key in ['kerno', 'subjekto', 'objekto']:
            node = ast.get(key)
            if node:
                verb = self._get_verb_root(node)
                if verb:
                    return verb

        return None

    def _extract_roots_from_ast(self, ast: Dict) -> List[str]:
        """Extract all roots from AST for retrieval."""
        roots = []

        def traverse(node):
            if not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '').lower()
                if root:
                    roots.append(root)

            for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                if key in node:
                    traverse(node[key])
            for item in node.get('priskriboj', []) + node.get('aliaj', []):
                traverse(item)

        traverse(ast)
        return roots

    def _sample_random_document(self) -> Optional[Dict]:
        """Sample a random document from corpus."""
        try:
            # Use Whoosh to get random documents
            random_results = self.retriever.retrieve(
                query_roots=['la'],  # Very common word
                question_type='what',
                query_entity=None,
                top_k=100
            )
            if random_results:
                return random.choice(random_results)
        except:
            pass
        return None

    def _create_example(
        self,
        question: str,
        query_ast: Dict,
        doc: Dict,
        label: float,
        negative_type: Optional[str] = None
    ) -> Optional[Dict]:
        """Create training example."""

        # Parse document if needed
        doc_ast = doc.get('ast')
        if not doc_ast:
            doc_text = doc.get('text', '')
            try:
                doc_ast = parse(doc_text)
            except:
                return None

        example = {
            'query': question,
            'document': doc.get('text', ''),
            'label': label,
            'query_ast': query_ast,
            'doc_ast': doc_ast,
        }

        if negative_type:
            example['negative_type'] = negative_type
        elif label == 0.0:
            example['negative_type'] = 'hard'

        return example

    def _balance_examples(self, examples: List[Dict]) -> List[Dict]:
        """Balance positive and negative examples to ~50/50."""
        positives = [e for e in examples if e['label'] == 1.0]
        negatives = [e for e in examples if e['label'] == 0.0]

        if not positives or not negatives:
            return examples

        # Make counts equal
        target = min(len(positives), len(negatives))
        target = max(target, 5)  # At least 5 of each

        if len(positives) > target:
            positives = random.sample(positives, target)
        if len(negatives) > target:
            negatives = random.sample(negatives, target)

        return positives + negatives


def main():
    parser = argparse.ArgumentParser(description='Generate improved TreeMatchReranker training data')
    parser.add_argument('--questions', type=Path, required=True,
                       help='Path to questions JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Path to output JSONL file')
    parser.add_argument('--whoosh-index', type=Path,
                       default=Path('data/indexes/whoosh_fts'),
                       help='Path to Whoosh index')
    parser.add_argument('--kuzu-db', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kuzu database')
    parser.add_argument('--examples-per-question', type=int, default=20,
                       help='Number of examples per question')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load questions
    logger.info(f"Loading questions from {args.questions}...")
    questions = []
    with open(args.questions) as f:
        for line in f:
            questions.append(json.loads(line))
    logger.info(f"Loaded {len(questions)} questions")

    # Initialize generator
    generator = ImprovedTreeRerankerDataGenerator(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.kuzu_db,
        random_seed=args.seed
    )

    # Generate examples
    examples = generator.generate_from_questions(
        questions=questions,
        examples_per_question=args.examples_per_question
    )

    # Save to JSONL
    logger.info(f"\nSaving {len(examples)} examples to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    logger.info("Done!")


if __name__ == '__main__':
    main()
