#!/usr/bin/env python3
"""
Extractive Answer Generator - Generate Coherent Answers from Retrieved Sentences

Main orchestrator for extractive question answering:
1. Extract facts from retrieved sentence ASTs
2. Score fact importance (question-aware)
3. Select top facts
4. Plan discourse structure
5. Generate coherent answer paragraph

Design Philosophy:
- Deterministic-first (80% deterministic, 20% learned)
- Explainable (score breakdown for each fact)
- AST-native (work with structures, not text)
- Compositional (Fact → AST → Sentence using deparser)

Example Usage:
    generator = ExtractiveAnswerGenerator()
    answer = generator.generate(
        sentences=retrieved_sentences,
        query="Kio estas Esperanto?",
        question_type=QuestionType.WHAT,
        query_entity="esperant"
    )
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from klareco.rag.fact_extractor import FactExtractor, Fact
from klareco.rag.importance_scorer import (
    FactImportanceScorer, ScoreBreakdown, QuestionType, classify_question_type
)
from klareco.rag.discourse_planner import DiscoursePlanner, DiscoursePlan


@dataclass
class Answer:
    """Generated answer with metadata."""
    text: str                           # Final answer paragraph
    facts_used: List[Fact]              # Facts included in answer
    score_breakdowns: List[ScoreBreakdown]  # Score for each fact
    discourse_plan: DiscoursePlan       # Discourse structure
    num_facts_extracted: int            # Total facts extracted
    num_facts_selected: int             # Facts selected for answer


class ExtractiveAnswerGenerator:
    """Generate coherent extractive answers from retrieved sentences."""

    def __init__(self):
        self.fact_extractor = FactExtractor()
        self.importance_scorer = FactImportanceScorer()
        self.discourse_planner = DiscoursePlanner()

    def generate(self,
                 sentences: List[Dict],
                 query: str,
                 question_type: Optional[QuestionType] = None,
                 query_entity: Optional[str] = None,
                 max_facts: int = 4) -> Answer:
        """
        Generate answer from retrieved sentences.

        Args:
            sentences: Retrieved sentences with 'text' and 'ast' fields
            query: Original query text
            question_type: Type of question (auto-detected if None)
            query_entity: Entity being queried about
            max_facts: Maximum facts to include in answer

        Returns:
            Answer object with text and metadata
        """
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = classify_question_type(query)

        # Step 1: Extract facts from all sentences
        all_facts = []
        for i, sent in enumerate(sentences):
            ast = sent.get('ast')
            text = sent.get('text')
            metadata = sent.get('metadata', {})

            if not ast:
                continue

            # Ensure metadata is a dict (might be string from database)
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            elif not isinstance(metadata, dict):
                metadata = {}

            # Add sentence position to metadata
            metadata['sentence_position'] = i

            # Extract facts
            facts = self.fact_extractor.extract(ast, source_sentence=text)

            # Store metadata with facts
            for fact in facts:
                all_facts.append((fact, metadata))

        # Step 2: Score fact importance
        scored_facts = []
        for fact, metadata in all_facts:
            score_breakdown = self.importance_scorer.score(
                fact, question_type, query_entity, metadata
            )
            scored_facts.append((fact, score_breakdown))

        # Step 3: Sort by importance score
        scored_facts.sort(key=lambda x: x[1].final_score, reverse=True)

        # Step 4: Select top facts
        top_facts = [fact for fact, _ in scored_facts[:max_facts]]
        top_scores = [score for _, score in scored_facts[:max_facts]]

        if not top_facts:
            return Answer(
                text="Mi ne trovis respondon.",  # "I didn't find an answer"
                facts_used=[],
                score_breakdowns=[],
                discourse_plan=DiscoursePlan(facts=[], relations=[], markers=[]),
                num_facts_extracted=len(all_facts),
                num_facts_selected=0
            )

        # Step 5: Plan discourse structure
        discourse_plan = self.discourse_planner.plan(top_facts, max_facts=max_facts)

        # Step 6: Generate answer text
        answer_text = self._generate_text(discourse_plan)

        return Answer(
            text=answer_text,
            facts_used=discourse_plan.facts,
            score_breakdowns=top_scores,
            discourse_plan=discourse_plan,
            num_facts_extracted=len(all_facts),
            num_facts_selected=len(top_facts)
        )

    def _generate_text(self, discourse_plan: DiscoursePlan) -> str:
        """
        Generate answer text from discourse plan.

        For now, uses source sentences directly (extractive).
        Future: Convert facts → AST → sentences using deparser (abstractive).
        """
        sentences = []

        for i, (fact, marker) in enumerate(zip(discourse_plan.facts,
                                               discourse_plan.markers)):
            # Use source sentence for now (extractive)
            sent = fact.source_sentence

            if not sent:
                continue

            # Add discourse marker if present (not for first sentence)
            if marker and i > 0:
                # Lowercase first letter of sentence when adding marker
                sent_lower = sent[0].lower() + sent[1:] if sent else sent
                sent = f"{marker}, {sent_lower}"

            sentences.append(sent)

        # Join into paragraph
        return ' '.join(sentences)


def demo_extractive_qa():
    """Demo extractive QA on example query."""
    from klareco.parser import parse

    # Example retrieved sentences (with ASTs)
    sentences = [
        {
            'text': 'Esperanto estas internacia planlingvo.',
            'ast': parse('Esperanto estas internacia planlingvo.'),
            'metadata': {'source': 'wikipedia', 'doc_title': 'Esperanto'}
        },
        {
            'text': 'Zamenhof kreis Esperanton en 1887.',
            'ast': parse('Zamenhof kreis Esperanton en 1887.'),
            'metadata': {'source': 'wikipedia', 'doc_title': 'Esperanto'}
        },
        {
            'text': 'Hodiaŭ Esperanto havas milionojn da parolantoj.',
            'ast': parse('Hodiaŭ Esperanto havas milionojn da parolantoj.'),
            'metadata': {'source': 'wikipedia', 'doc_title': 'Esperanto'}
        }
    ]

    # Generate answer
    generator = ExtractiveAnswerGenerator()
    answer = generator.generate(
        sentences=sentences,
        query='Kio estas Esperanto?',
        question_type=QuestionType.WHAT,
        query_entity='esperant',
        max_facts=3
    )

    # Print results
    print("=" * 70)
    print("EXTRACTIVE ANSWER GENERATION DEMO")
    print("=" * 70)
    print()
    print(f"Query: Kio estas Esperanto?")
    print(f"Question Type: {QuestionType.WHAT.value}")
    print()
    print(f"Answer:")
    print(f"  {answer.text}")
    print()
    print(f"Metadata:")
    print(f"  Facts extracted: {answer.num_facts_extracted}")
    print(f"  Facts selected: {answer.num_facts_selected}")
    print()
    print(f"Score Breakdowns:")
    for i, (fact, score) in enumerate(zip(answer.facts_used, answer.score_breakdowns), 1):
        print(f"  {i}. {fact}")
        print(f"     {score}")
    print()


if __name__ == '__main__':
    demo_extractive_qa()
