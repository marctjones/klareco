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
from pathlib import Path
import logging

import torch

from klareco.rag.fact_extractor import FactExtractor, Fact, RelationType
from klareco.rag.importance_scorer import (
    FactImportanceScorer, ScoreBreakdown, QuestionType, classify_question_type
)
from klareco.rag.discourse_planner import DiscoursePlanner, DiscoursePlan
from klareco.rag.answer_extractor import ASTAnswerExtractor
from klareco.models.reranker import ASTReranker
from klareco.models.m1_inference import M1Inference
from klareco.embeddings.compositional import CompositionalEmbedding

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation to source document (Issue #674)."""
    id: int                    # Citation number [1], [2], etc.
    sentence_id: str           # Database sentence ID
    sentence_text: str         # Full sentence text
    doc_title: str             # Article/document title
    doc_source: str            # wikipedia, gutenberg, etc.
    doc_metadata: Dict         # Full document metadata


@dataclass
class Answer:
    """Generated answer with metadata."""
    text: str                           # Final answer paragraph (with citation markers)
    facts_used: List[Fact]              # Facts included in answer
    score_breakdowns: List[ScoreBreakdown]  # Score for each fact
    discourse_plan: DiscoursePlan       # Discourse structure
    num_facts_extracted: int            # Total facts extracted
    num_facts_selected: int             # Facts selected for answer
    citations: List[Citation]           # Citations to source sentences (Issue #674)


class ExtractiveAnswerGenerator:
    """Generate coherent extractive answers from retrieved sentences."""

    def __init__(
        self,
        reranker_path: Optional[Path] = None,
        m1_model_path: Optional[Path] = None,
        embedding_path: Optional[Path] = None,
        use_reranker: bool = True,
        use_m1: bool = True,
        m1_threshold: float = 0.3,
        use_ast_extraction: bool = False,
        multi_sentence_question_types: Optional[Dict[QuestionType, bool]] = None
    ):
        """
        Initialize answer generator with optional neural models.

        Args:
            reranker_path: Path to neural reranker model
            m1_model_path: Path to M1 selectional preference model
            embedding_path: Path to root embeddings (needed for reranker and M1)
            use_reranker: Enable neural reranking (default: True)
            use_m1: Enable M1 plausibility filtering (default: True)
            m1_threshold: Minimum plausibility score for M1 filtering (default: 0.3)
            use_ast_extraction: Enable ASTAnswerExtractor cascade (default: False - always use multi-sentence)
            multi_sentence_question_types: Dict[QuestionType, bool] controlling whether
                                          each question type should use discourse planning.
                                          Default: all True (multi-sentence for all types)
        """
        self.fact_extractor = FactExtractor()
        self.importance_scorer = FactImportanceScorer()
        self.discourse_planner = DiscoursePlanner()

        # Initialize ASTAnswerExtractor for cascade (Phase 3)
        self.use_ast_extraction = use_ast_extraction
        self.ast_extractor = ASTAnswerExtractor() if use_ast_extraction else None

        # Multi-sentence configuration (default: all True)
        self.multi_sentence_config = multi_sentence_question_types or {
            QuestionType.WHO: True,
            QuestionType.WHAT: True,
            QuestionType.WHERE: True,
            QuestionType.WHEN: True,
            QuestionType.WHY: True,
            QuestionType.HOW: True,
            QuestionType.OTHER: True,
        }

        # Set default paths if not provided
        if reranker_path is None:
            reranker_path = Path('models/reranker/best_model.pt')
        if m1_model_path is None:
            m1_model_path = Path('models/m1_selectional/best_model.pt')
        if embedding_path is None:
            # Reranker and M1 were trained with 64D embeddings, not 128D phase1_fast
            embedding_path = Path('models/root_embeddings/best_model.pt')

        # Load neural reranker
        self.reranker = None
        self.use_reranker = use_reranker
        if use_reranker and reranker_path.exists():
            try:
                logger.info(f"Loading neural reranker from {reranker_path}")

                # Load compositional embedding checkpoint
                emb_checkpoint = torch.load(embedding_path, map_location='cpu', weights_only=False)

                # Check if this is a full CompositionalEmbedding checkpoint or skip-gram
                if 'root_vocab' in emb_checkpoint:
                    # Full compositional embedding checkpoint
                    comp_embedding = CompositionalEmbedding(
                        root_vocab=emb_checkpoint['root_vocab'],
                        prefix_vocab=emb_checkpoint['prefix_vocab'],
                        suffix_vocab=emb_checkpoint['suffix_vocab'],
                        embed_dim=emb_checkpoint.get('embed_dim', 128),
                    )
                    comp_embedding.load_state_dict(emb_checkpoint['model_state_dict'])
                else:
                    # Skip-gram checkpoint - create minimal CompositionalEmbedding
                    root_to_idx = emb_checkpoint.get('root_to_idx') or emb_checkpoint.get('vocab')
                    prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
                    suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

                    comp_embedding = CompositionalEmbedding(
                        root_vocab=root_to_idx,
                        prefix_vocab=prefix_vocab,
                        suffix_vocab=suffix_vocab,
                        embed_dim=emb_checkpoint.get('embedding_dim', 64),
                    )

                    # Load root embeddings from checkpoint
                    # Try 'embeddings' first (direct tensor), then check model_state_dict
                    if 'embeddings' in emb_checkpoint:
                        comp_embedding.root_embed.weight.data = emb_checkpoint['embeddings']
                    elif 'embeddings.weight' in emb_checkpoint.get('model_state_dict', {}):
                        comp_embedding.root_embed.weight.data = emb_checkpoint['model_state_dict']['embeddings.weight']
                    elif 'target_embeddings.weight' in emb_checkpoint.get('model_state_dict', {}):
                        # Skip-gram has target and context - use target embeddings
                        comp_embedding.root_embed.weight.data = emb_checkpoint['model_state_dict']['target_embeddings.weight']

                comp_embedding.eval()

                # Load reranker using classmethod
                self.reranker = ASTReranker.load(reranker_path, comp_embedding)
                self.reranker.eval()

                logger.info("✓ Neural reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                self.reranker = None
                self.use_reranker = False

        # Load M1 selectional preference model
        self.m1_filter = None
        self.use_m1 = use_m1
        self.m1_threshold = m1_threshold
        if use_m1 and m1_model_path.exists():
            try:
                logger.info(f"Loading M1 selectional preference from {m1_model_path}")
                self.m1_filter = M1Inference(
                    model_path=m1_model_path,
                    comp_model_path=embedding_path,
                    device='cpu'
                )
                logger.info("✓ M1 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load M1: {e}. Continuing without M1 filtering.")
                self.m1_filter = None
                self.use_m1 = False

    def _filter_facts_by_question_type(self, facts_with_metadata, question_type):
        """
        Filter facts by question type (Issue #684 - Quick Win +10% relevance).

        Removes facts that are unlikely to answer the question type.
        For example, WHEN questions should keep only facts with temporal info.

        Args:
            facts_with_metadata: List of (fact, metadata) tuples
            question_type: QuestionType enum

        Returns:
            Filtered list of (fact, metadata) tuples
        """
        from klareco.rag.fact_extractor import RelationType

        filtered = []

        for fact, metadata in facts_with_metadata:
            keep = True

            if question_type == QuestionType.WHEN:
                # WHEN questions: Only keep facts with temporal modifiers
                if 'time' not in fact.modifiers:
                    # Exception: Relations that often have temporal info
                    if fact.relation not in [RelationType.CREATED_BY, RelationType.PUBLISHED,
                                            RelationType.BORN, RelationType.DIED]:
                        keep = False

            elif question_type == QuestionType.WHERE:
                # WHERE questions: Only keep facts with location info
                if 'location' not in fact.modifiers and 'location' not in fact.arguments:
                    # Exception: Relations that specify location
                    if fact.relation not in [RelationType.LOCATED_AT, RelationType.BORN]:
                        keep = False

            elif question_type == QuestionType.WHO:
                # WHO questions: Only keep facts with agent/person info
                if 'agent' not in fact.arguments:
                    # Exception: Relations that identify people
                    if fact.relation not in [RelationType.CREATED_BY, RelationType.FOUNDED]:
                        # Still allow IS-A facts (they might define who someone is)
                        if fact.relation != RelationType.IS_A:
                            keep = False

            # For WHAT, HOW, WHY, OTHER: Don't filter (too broad)

            if keep:
                filtered.append((fact, metadata))

        # If filtering removed everything, return original (failsafe)
        if not filtered:
            return facts_with_metadata

        return filtered

    def _rerank_sentences(self, sentences: List[Dict], query_ast: Dict) -> List[Dict]:
        """
        Rerank sentences using neural reranker.

        Args:
            sentences: List of sentence dicts with 'text' and 'ast' fields
            query_ast: Parsed query AST

        Returns:
            Reranked list of sentences
        """
        if not self.use_reranker or self.reranker is None:
            return sentences

        if not sentences:
            return sentences

        try:
            # Extract ASTs
            doc_asts = [s.get('ast') for s in sentences if s.get('ast')]

            if not doc_asts:
                return sentences

            # Score all sentences
            with torch.no_grad():
                scores = self.reranker.score_batch(query_ast, doc_asts)

            # Sort by score (descending)
            scored_sentences = list(zip(sentences, scores))
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            # Return reranked sentences
            return [s for s, _ in scored_sentences]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return sentences

    def _filter_by_m1_plausibility(self, facts_with_metadata: List) -> List:
        """
        Filter facts by M1 selectional preference (plausibility).

        Removes facts with implausible subject-verb-object combinations.

        Args:
            facts_with_metadata: List of (fact, metadata) tuples

        Returns:
            Filtered list of (fact, metadata) tuples
        """
        if not self.use_m1 or self.m1_filter is None:
            return facts_with_metadata

        if not facts_with_metadata:
            return facts_with_metadata

        filtered = []

        for fact, metadata in facts_with_metadata:
            # Extract subject-verb-object from fact
            subject = fact.entity  # Main entity is subject
            verb = None
            obj = None

            # Get verb from relation type
            relation_to_verb = {
                RelationType.IS_A: 'est',
                RelationType.HAS: 'hav',
                RelationType.CREATED_BY: 'kre',
                RelationType.LOCATED_AT: 'lok',
                RelationType.BORN: 'nask',
                RelationType.DIED: 'mort',
                RelationType.PUBLISHED: 'publik',
                RelationType.FOUNDED: 'fond',
            }
            verb = relation_to_verb.get(fact.relation)

            # Get object from arguments
            if 'type' in fact.arguments:
                obj = fact.arguments['type']
            elif 'agent' in fact.arguments:
                obj = fact.arguments['agent']
            elif 'property' in fact.arguments:
                obj = fact.arguments['property']
            elif 'object' in fact.arguments:
                obj = fact.arguments['object']

            # If we can't extract a complete triple, keep the fact (conservative)
            if not subject or not verb or not obj:
                filtered.append((fact, metadata))
                continue

            try:
                # Score plausibility
                score = self.m1_filter.score_triple(subject, verb, obj)

                # Keep if plausible
                if score >= self.m1_threshold:
                    filtered.append((fact, metadata))
                else:
                    logger.debug(f"M1 filtered out: ({subject}, {verb}, {obj}) = {score:.3f}")

            except Exception as e:
                # If scoring fails, keep the fact (conservative)
                logger.debug(f"M1 scoring failed for ({subject}, {verb}, {obj}): {e}")
                filtered.append((fact, metadata))

        # If filtering removed everything, return original (failsafe)
        if not filtered:
            logger.warning("M1 filtering removed all facts. Returning original.")
            return facts_with_metadata

        # Log filtering statistics
        num_before = len(facts_with_metadata)
        num_after = len(filtered)
        num_removed = num_before - num_after
        if num_removed > 0:
            logger.info(f"M1 filtering removed {num_removed}/{num_before} facts ({100*num_removed/num_before:.1f}%)")

        return filtered

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

        # === OPTIONAL: Try direct AST extraction first (cascade) ===
        # NOTE: Default is use_ast_extraction=False to ensure multi-sentence answers for all question types
        # Enable with use_ast_extraction=True for faster single-span answers on simple questions
        if self.use_ast_extraction and self.ast_extractor is not None:
            from klareco.parser import parse
            query_ast = parse(query)

            # Prepare documents for ASTAnswerExtractor format
            # It expects: List[Tuple[score, doc, stats]]
            ranked_docs = []
            for i, s in enumerate(sentences[:20]):  # Only try top 20 for speed
                score = s.get('score', 1.0 / (i + 1))  # Use retrieval score or rank-based
                ranked_docs.append((score, s, {}))

            # Try extracting answer using AST pattern matching
            ast_answer = self.ast_extractor.extract_answer_from_multiple_docs(
                query_ast, ranked_docs, top_n=10
            )

            if ast_answer and ast_answer['confidence'] >= 0.7:
                # Check if this question type should use multi-sentence answers
                use_discourse = self.multi_sentence_config.get(question_type, True)

                if not use_discourse:
                    # Fast path: return direct single-span answer
                    # Extract citation from source document
                    doc_rank = ast_answer['aggregation_stats']['doc_ranks'][0]
                    source_doc = ranked_docs[doc_rank - 1][1]  # Convert to 0-indexed

                    citation = Citation(
                        id=1,
                        sentence_id=source_doc.get('id', 'unknown'),
                        sentence_text=ast_answer['text'],
                        doc_title=source_doc.get('doc_title', 'Unknown'),
                        doc_source=source_doc.get('metadata', {}).get('source', 'Unknown') if isinstance(source_doc.get('metadata'), dict) else 'Unknown',
                        doc_metadata=source_doc.get('metadata', {}) if isinstance(source_doc.get('metadata'), dict) else {}
                    )

                    logger.info(f"AST extraction successful (confidence={ast_answer['confidence']:.2f}), returning single-span answer")

                    return Answer(
                        text=ast_answer['text'],
                        facts_used=[],
                        score_breakdowns=[],
                        discourse_plan=DiscoursePlan(facts=[], relations=[], markers=[]),
                        num_facts_extracted=1,
                        num_facts_selected=1,
                        citations=[citation],
                    )

                # Multi-sentence enabled: continue to discourse planning with AST answer as seed
                logger.info(f"AST extraction successful (confidence={ast_answer['confidence']:.2f}), continuing to discourse planning")

        # Step 0: Rerank sentences with neural reranker (if enabled)
        if self.use_reranker and self.reranker is not None:
            from klareco.parser import parse
            query_ast = parse(query)
            sentences = self._rerank_sentences(sentences, query_ast)
            logger.info(f"Neural reranker reordered {len(sentences)} sentences")

        # Step 1: Extract facts from all sentences
        all_facts = []
        for i, sent in enumerate(sentences):
            ast = sent.get('ast')
            text = sent.get('text')
            sent_id = sent.get('id')           # Database sentence ID
            doc_title = sent.get('doc_title')  # Document title
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

            # Attach citation info to each fact (Issue #674)
            for fact in facts:
                fact.sentence_id = sent_id
                fact.doc_title = doc_title or 'Unknown'
                fact.doc_metadata = metadata
                all_facts.append((fact, metadata))

        # Step 2: Filter facts by question type (Issue #684 - Quick Win +10% relevance)
        filtered_facts = self._filter_facts_by_question_type(all_facts, question_type)
        num_facts_before_filter = len(all_facts)
        num_facts_after_filter = len(filtered_facts)

        # Step 2.5: Filter by M1 selectional preference (plausibility)
        if self.use_m1 and self.m1_filter is not None:
            filtered_facts = self._filter_by_m1_plausibility(filtered_facts)

        # Step 3: Score fact importance
        scored_facts = []
        for fact, metadata in filtered_facts:
            score_breakdown = self.importance_scorer.score(
                fact, question_type, query_entity, metadata
            )
            scored_facts.append((fact, score_breakdown))

        # Step 4: Sort by importance score
        scored_facts.sort(key=lambda x: x[1].final_score, reverse=True)

        # Step 5: Select top facts
        top_facts = [fact for fact, _ in scored_facts[:max_facts]]
        top_scores = [score for _, score in scored_facts[:max_facts]]

        if not top_facts:
            return Answer(
                text="Mi ne trovis respondon.",  # "I didn't find an answer"
                facts_used=[],
                score_breakdowns=[],
                discourse_plan=DiscoursePlan(facts=[], relations=[], markers=[]),
                num_facts_extracted=len(all_facts),
                num_facts_selected=0,
                citations=[]
            )

        # Step 6: Plan discourse structure
        discourse_plan = self.discourse_planner.plan(top_facts, max_facts=max_facts)

        # Step 7: Generate answer text with citations (Issue #674)
        answer_text, citations = self._generate_text(discourse_plan)

        return Answer(
            text=answer_text,
            facts_used=discourse_plan.facts,
            score_breakdowns=top_scores,
            discourse_plan=discourse_plan,
            num_facts_extracted=len(all_facts),
            num_facts_selected=len(top_facts),
            citations=citations
        )

    def _generate_text(self, discourse_plan: DiscoursePlan) -> tuple:
        """
        Generate answer text from discourse plan with citations (Issue #674).

        For now, uses source sentences directly (extractive).
        Future: Convert facts → AST → sentences using deparser (abstractive).

        Returns:
            (answer_text, citations): Answer with citation markers and citation list
        """
        sentences = []
        citations = []
        citation_map = {}  # sentence_id -> citation_number

        for i, (fact, marker) in enumerate(zip(discourse_plan.facts,
                                               discourse_plan.markers)):
            # Use source sentence for now (extractive)
            sent = fact.source_sentence

            if not sent:
                continue

            # Get or create citation number for this sentence
            sent_id = fact.sentence_id or f"unknown_{i}"
            if sent_id not in citation_map:
                citation_num = len(citations) + 1
                citation_map[sent_id] = citation_num

                # Extract source from metadata (check multiple fields)
                doc_source = 'unknown'
                if fact.doc_metadata:
                    # Check common metadata fields for source
                    if 'source' in fact.doc_metadata:
                        doc_source = fact.doc_metadata['source']
                    elif 'wikipedia' in str(fact.doc_metadata).lower():
                        doc_source = 'wikipedia'
                    elif fact.doc_title and 'wikipedia' in fact.doc_title.lower():
                        doc_source = 'wikipedia'
                    else:
                        doc_source = 'database'

                # Add to citations list
                citations.append(Citation(
                    id=citation_num,
                    sentence_id=sent_id,
                    sentence_text=fact.source_sentence,
                    doc_title=fact.doc_title or 'Unknown',
                    doc_source=doc_source,
                    doc_metadata=fact.doc_metadata or {}
                ))
            else:
                citation_num = citation_map[sent_id]

            # Add citation marker to fact
            fact.citation_id = citation_num

            # Add citation marker to sentence
            sent_with_citation = f"{sent} [{citation_num}]"

            # Add discourse marker if present (not for first sentence)
            if marker and i > 0:
                # Lowercase first letter of sentence when adding marker
                sent_lower = sent_with_citation[0].lower() + sent_with_citation[1:] if sent_with_citation else sent_with_citation
                sent_with_citation = f"{marker}, {sent_lower}"

            sentences.append(sent_with_citation)

        # Join into paragraph
        answer_text = ' '.join(sentences)

        return answer_text, citations


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
