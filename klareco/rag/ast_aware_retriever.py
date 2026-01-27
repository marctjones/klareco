"""
AST-Aware Retriever using Kuzu Graph Database.

Features:
1. Root-based inverted index lookup (O(1))
2. Transitive synonym expansion via graph traversal
3. Hypernym chain traversal
4. Grammar-aware scoring
5. Question type classification for slot weighting
6. Sentence context retrieval (adjacent sentences)
7. Role-aware pattern matching

This is a DETERMINISTIC retriever that leverages Klareco's unique advantage:
fully parsed AST annotations for both queries and corpus.

MEMORY EFFICIENT:
- Kuzu-backed root lookups (~50MB RAM instead of 8GB)
- Memory-mapped document access
- No HNSW index needed (root index provides fast lookup)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from klareco.parser import parse
from klareco.rag.question_classifier import QuestionClassifier, QuestionType
from klareco.rag.entity_recognizer import EntityRecognizer
from klareco.rag.kuzu_inverted_index import KuzuInvertedIndex, FallbackMode, RetrievalStats
from klareco.entity_classifier import EntityClassifier, EntityType
from klareco.utils.ast_utils import extract_word_structure

logger = logging.getLogger(__name__)


class ASTAwareRetriever:
    """
    AST-aware retriever using Kuzu graph database backend.

    PURE DETERMINISTIC by default - no embeddings loaded.
    A/B testing showed deterministic lookup has equal recall with lower latency.

    Memory efficient: Uses Kuzu-backed root lookups and mmap document access.
    No HNSW index needed - root inverted index provides O(1) root lookup.

    Features:
    - Transitive synonym expansion (2+ hops)
    - Hypernym chain traversal
    - Sentence context retrieval (adjacent sentences)
    - Role-aware pattern matching

    To enable embedding fallback (opt-in):
        retriever = ASTAwareRetriever(fallback_mode=FallbackMode.EMBEDDING)
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        fallback_mode: FallbackMode = FallbackMode.NONE,
        m1_model = None,  # Optional M1Inference instance
    ):
        """
        Initialize AST-aware retriever.

        Args:
            index_path: Path to Kuzu index directory.
                        Defaults to data/indexes/kuzu_index.
            fallback_mode: Fallback mode (default: NONE for pure deterministic)
            m1_model: Optional M1Inference model for query expansion filtering.
                     If provided, enables semantic plausibility filtering of
                     synonym expansions BEFORE search (M1's intended purpose).
        """
        # Set default path
        if index_path is None:
            index_path = Path("data/indexes/kuzu_index")
        self.index_path = Path(index_path)

        # Store M1 model for query expansion
        self.m1 = m1_model

        logger.info("Initializing AST-aware retriever (Kuzu backend)...")

        # Initialize question classifier
        self.question_classifier = QuestionClassifier()
        logger.info("  ✓ QuestionClassifier initialized")

        # Initialize entity recognizer
        self.entity_recognizer = EntityRecognizer()
        logger.info("  ✓ EntityRecognizer initialized")

        # Initialize entity classifier
        self.entity_classifier = EntityClassifier()
        logger.info("  ✓ EntityClassifier initialized")

        # Initialize Kuzu inverted index
        # Note: Embedding fallback was removed - pure deterministic lookup
        # has equal recall with lower latency (see A/B test in issue #246)
        self.root_index = KuzuInvertedIndex(
            index_path=self.index_path,
            fallback_mode=fallback_mode,
        )

        if self.root_index.total_docs > 0:
            logger.info(f"  ✓ KuzuInvertedIndex: {self.root_index.total_docs:,} docs, "
                       f"{self.root_index.total_roots:,} roots")
        else:
            logger.warning(f"  ! KuzuInvertedIndex not loaded from {self.index_path}")

        if self.m1:
            logger.info("  ✓ M1 model enabled for query expansion filtering")

        logger.info("AST-aware retriever initialized")

    def expand_query_with_m1(
        self,
        query_ast: Dict,
        min_plausibility: float = 0.5,
        max_synonyms: int = 10,
    ) -> List[Dict]:
        """
        Expand query with verb synonym substitution, filtered by M1 plausibility.

        This is M1's INTENDED PURPOSE: filter query expansions BEFORE search
        to avoid retrieving nonsense documents.

        Example:
            Query: "Kiu manĝas insektojn?" (Who eats insects?)
            Verb synonyms: manĝ, konsum, absorb, nutr, devorar
            M1 filtering:
              ✓ manĝ (0.95) - plausible
              ✓ konsum (0.87) - plausible
              ✗ absorb (0.12) - implausible (liquids absorb, not eat)
              ✓ nutr (0.82) - plausible
            → Search with: [manĝ, konsum, nutr] only

        Args:
            query_ast: Parsed query AST
            min_plausibility: Minimum M1 score to keep expansion (0.0-1.0)
            max_synonyms: Maximum number of synonyms to consider per verb

        Returns:
            List of query expansion dicts with:
            - 'query_ast': Modified query AST with synonym verb
            - 'verb_root': Synonym verb root
            - 'm1_score': Plausibility score
            - 'is_original': True if this is the original query
        """
        if not self.m1:
            # No M1 model - return original query only
            return [{
                'query_ast': query_ast,
                'verb_root': query_ast.get('verbo', {}).get('radiko', ''),
                'm1_score': 1.0,
                'is_original': True
            }]

        # Extract S-V-O from query
        if not all(k in query_ast for k in ['subjekto', 'verbo', 'objekto']):
            # Incomplete query - no expansion
            logger.warning("Query missing S-V-O structure - skipping expansion")
            return [{
                'query_ast': query_ast,
                'verb_root': query_ast.get('verbo', {}).get('radiko', ''),
                'm1_score': 1.0,
                'is_original': True
            }]

        subj = query_ast['subjekto']
        verb = query_ast['verbo']
        obj = query_ast['objekto']

        # Extract word structures (with case normalization)
        # Handle vortgrupo (extract kerno)
        def get_word(node):
            if node.get('tipo') == 'vortgrupo':
                return node.get('kerno', {})
            return node

        subj_word = get_word(subj)
        verb_word = get_word(verb)
        obj_word = get_word(obj)

        # Extract morphological structures
        subj_struct = extract_word_structure(subj_word, strip_case=True)
        verb_struct = extract_word_structure(verb_word, strip_case=True)
        obj_struct = extract_word_structure(obj_word, strip_case=True)

        original_verb_root = verb_struct['root']

        # Get synonyms for verb (most impactful for expansion)
        # Use graph-based transitive synonym expansion
        verb_synonyms = self.get_synonyms_transitive(original_verb_root, max_hops=2)

        # Add original verb
        verb_synonyms = {original_verb_root} | verb_synonyms

        # Limit to max_synonyms
        verb_synonyms = list(verb_synonyms)[:max_synonyms]

        logger.info(f"Query expansion: '{original_verb_root}' → {len(verb_synonyms)} synonyms")

        # Score all verb replacements with M1
        candidates = []
        for syn_verb in verb_synonyms:
            # Create modified verb structure
            syn_verb_struct = {**verb_struct, 'root': syn_verb}

            # Score with M1
            try:
                score = self.m1.score_triple_full(subj_struct, syn_verb_struct, obj_struct)
            except Exception as e:
                logger.warning(f"M1 scoring failed for '{syn_verb}': {e}")
                score = 0.0

            is_original = (syn_verb == original_verb_root)

            if score >= min_plausibility or is_original:
                # Keep if plausible OR if it's the original query
                candidates.append({
                    'verb_root': syn_verb,
                    'm1_score': score,
                    'is_original': is_original
                })

                status = "✓" if score >= min_plausibility else "○"
                logger.info(f"  {status} {syn_verb}: {score:.3f} {'(original)' if is_original else ''}")
            else:
                logger.info(f"  ✗ {syn_verb}: {score:.3f} (implausible - filtered)")

        # Sort by plausibility (original query always included regardless of score)
        candidates.sort(key=lambda x: x['m1_score'], reverse=True)

        # Create query variants with substituted verbs
        expansions = []
        for candidate in candidates:
            # Create modified query AST (shallow copy with verb replacement)
            modified_ast = {**query_ast}
            modified_verb = {**verb_word, 'radiko': candidate['verb_root']}
            modified_ast['verbo'] = modified_verb

            expansions.append({
                'query_ast': modified_ast,
                'verb_root': candidate['verb_root'],
                'm1_score': candidate['m1_score'],
                'is_original': candidate['is_original']
            })

        logger.info(f"Query expansion: {len(expansions)} plausible variants (min_plausibility={min_plausibility})")

        return expansions if expansions else [{
            'query_ast': query_ast,
            'verb_root': original_verb_root,
            'm1_score': 1.0,
            'is_original': True
        }]

    def search(
        self,
        query: str,
        top_k: int = 10,
        fallback_mode: Optional[FallbackMode] = None,
        filter_by_entity_type: bool = False,
        use_m1_expansion: bool = True,
        m1_min_plausibility: float = 0.5,
    ) -> List[Tuple[float, Dict, RetrievalStats]]:
        """
        Search for relevant documents using AST-aware root-based retrieval.

        With M1 model enabled (via constructor), this performs query expansion
        with plausibility filtering BEFORE search (M1's intended purpose).

        Args:
            query: Query string (Esperanto)
            top_k: Number of results to return
            fallback_mode: Override default fallback mode for this search
            filter_by_entity_type: If True, filter results by entity type match
            use_m1_expansion: If True and M1 available, expand query with
                             plausibility filtering (default: True)
            m1_min_plausibility: Minimum M1 score for keeping expansions (0.0-1.0)

        Returns:
            List of (score, document, stats) tuples sorted by relevance
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Classify question for grammar-aware scoring
        classification = self.question_classifier.classify(query, query_ast)
        question_type = classification['question_type']

        logger.info(f"Query: \"{query}\"")
        logger.info(f"  Question type: {question_type.value}")

        # M1-based query expansion (if enabled)
        if use_m1_expansion and self.m1:
            logger.info("M1 query expansion enabled")
            expansions = self.expand_query_with_m1(
                query_ast,
                min_plausibility=m1_min_plausibility,
                max_synonyms=10
            )

            # Search with each plausible expansion
            all_results = {}  # doc_id -> (score, doc, m1_weight)
            combined_stats = None

            for expansion in expansions:
                exp_ast = expansion['query_ast']
                m1_score = expansion['m1_score']
                is_original = expansion['is_original']

                # Search with expanded query
                mode = fallback_mode if fallback_mode is not None else self.root_index.fallback_mode
                exp_results, exp_stats = self.root_index.search(
                    query_ast=exp_ast,
                    max_results=top_k * 2,  # Get more for merging
                    fallback_mode=mode,
                )

                # Store stats from first expansion (they're similar)
                if combined_stats is None:
                    combined_stats = exp_stats

                # Weight results by M1 score
                for result in exp_results:
                    doc_id = result.doc_id
                    weighted_score = result.score * m1_score

                    # Take max score across expansions (best match wins)
                    if doc_id not in all_results or weighted_score > all_results[doc_id][0]:
                        doc = self.root_index.get_document(doc_id)
                        if doc:
                            all_results[doc_id] = (weighted_score, doc, m1_score)

            # Convert to SearchResult-like objects for ranking
            class SearchResult:
                def __init__(self, doc_id, score):
                    self.doc_id = doc_id
                    self.score = score

            results = [SearchResult(doc_id, score) for doc_id, (score, _, _) in all_results.items()]
            results.sort(key=lambda r: r.score, reverse=True)
            stats = combined_stats

            logger.info(f"  M1 expansions: {len(expansions)} variants")
            logger.info(f"  Merged results: {len(results)} unique documents")

        else:
            # Original search without M1 expansion
            mode = fallback_mode if fallback_mode is not None else self.root_index.fallback_mode
            results, stats = self.root_index.search(
                query_ast=query_ast,
                max_results=top_k,
                fallback_mode=mode,
            )

        # Log stats
        logger.info(f"  Roots found: {len(stats.roots_found_in_index)}")
        if stats.roots_not_found:
            logger.info(f"  Roots not found: {stats.roots_not_found}")
        if stats.graph_expansions:
            logger.info(f"  Graph expansions: {stats.graph_expansions}")
            # Show ConceptNet expansions if available
            if hasattr(stats, 'conceptnet_expansions') and stats.conceptnet_expansions:
                logger.info(f"  ConceptNet: {stats.conceptnet_expansions[:5]}...")
        if stats.embedding_synonyms:
            logger.info(f"  Synonyms (Embedding): {stats.embedding_synonyms[:3]}...")
        logger.info(f"  Results: {len(results)}")

        # Apply entity-aware scoring (boost docs with query entities)
        results = self._apply_entity_boost(query_ast, results)
        logger.info(f"  After entity boost: {len(results)}")

        # Apply document quality filtering (penalize low-quality docs)
        results = self._apply_document_quality_filter(results)
        logger.info(f"  After quality filter: {len(results)}")

        # Apply role-based ranking to improve query disambiguation
        results = self._apply_root_role_ranking(query_ast, results)
        logger.info(f"  After role ranking: {len(results)}")

        # Apply entity type filtering if requested
        if filter_by_entity_type:
            results = self._filter_by_entity_type(query_ast, results)
            logger.info(f"  After entity type filtering: {len(results)}")

        # Convert to expected format (score, doc, stats)
        output = []
        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if doc:
                # Add doc_id to the document for benchmarking/evaluation
                doc['doc_id'] = result.doc_id
                output.append((result.score, doc, stats))

        return output

    def _get_grammar_hints(
        self,
        question_type: QuestionType,
        query_ast: Dict,
    ) -> Dict[str, str]:
        """
        Get grammar hints based on question type for scoring.

        Different question types expect answers with different grammatical features:
        - WHO: Look for subject role (nominative)
        - WHEN: Look for temporal markers (pasinteco, prezenco, futuro)
        - WHERE: Look for locative expressions
        """
        hints = {}

        # Role hints based on question type
        if question_type == QuestionType.WHO:
            hints['preferred_role'] = 'subjekto'
        elif question_type == QuestionType.WHAT:
            # For "Kio estas X?" - look for predicate nominative
            verb = query_ast.get('verbo', {})
            if verb.get('radiko') == 'est':
                hints['preferred_role'] = 'subjekto'  # Definition questions
            else:
                hints['preferred_role'] = 'objekto'
        elif question_type == QuestionType.WHERE:
            hints['preferred_role'] = 'aliaj'  # Location in modifiers
        elif question_type == QuestionType.WHEN:
            hints['preferred_role'] = 'aliaj'  # Time in modifiers
            # Also look for tense matches
            verb = query_ast.get('verbo', {})
            if verb.get('tempo'):
                hints['tempo'] = verb['tempo']

        return hints

    def _apply_entity_boost(self, query_ast: Dict, results: List) -> List:
        """
        Apply entity-aware scoring based on question type and expected answer entities.

        Strategy:
        1. WHO questions → Boost documents with PERSON entities
        2. WHERE questions → Boost documents with PLACE entities
        3. WHEN questions → Boost documents with TIME entities (dates, years)
        4. For all questions → Boost documents with SPECIFIC query entities (but not overly common ones)

        This is more effective than just boosting query entities, because:
        - In Esperanto corpus, "Esperanto" appears everywhere (not discriminative)
        - WHO questions need documents with person names (Zamenhof, etc.)
        - WHERE questions need documents with place names
        - WHEN questions need documents with dates

        Scoring:
        - Document contains expected entity type for question: 2.5x boost
        - Document contains specific query entity: 1.5x boost (lower weight)
        - Document contains both: 2.5x × 1.5x = 3.75x combined boost

        Args:
            query_ast: Parsed query AST
            results: List of SearchResult objects from root_index.search()

        Returns:
            List of SearchResult objects with adjusted scores
        """
        # Classify question to determine expected answer type
        classification = self.question_classifier.classify("", query_ast)
        question_type = classification['question_type']

        # Extract entities from query (to check for specific entities)
        query_entities = self.entity_recognizer.recognize_entities(query_ast)

        # Determine expected entity type based on question type
        from klareco.rag.entity_recognizer import EntityType
        from klareco.rag.question_classifier import QuestionType

        expected_entity_type = None
        if question_type == QuestionType.WHO:
            expected_entity_type = EntityType.PERSON
        elif question_type == QuestionType.WHERE:
            expected_entity_type = EntityType.PLACE
        elif question_type == QuestionType.WHEN:
            expected_entity_type = EntityType.TIME

        if not expected_entity_type and not query_entities:
            # No entity-based boosting needed
            return results

        logger.info(f"  Applying entity-aware boost:")
        if expected_entity_type:
            logger.info(f"    Expected entity type: {expected_entity_type.value}")
        if query_entities:
            entity_summary = [f"{e.text} ({e.entity_type.value})" for e in query_entities]
            logger.info(f"    Query entities: {', '.join(entity_summary)}")

        # Apply entity boost
        boosted_results = []
        type_matches = 0
        entity_matches = 0

        ENTITY_TYPE_BOOST = 2.5  # Boost for expected entity type
        SPECIFIC_ENTITY_BOOST = 1.5  # Smaller boost for specific entity match

        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if not doc:
                continue

            doc_text = doc.get('text', '')
            doc_text_lower = doc_text.lower()
            original_score = result.score
            boost_multiplier = 1.0
            reasons = []

            # Check if document contains expected entity type
            if expected_entity_type:
                # Parse document on-the-fly to extract entities
                # Note: This is slower but necessary since docs don't have pre-parsed ASTs
                try:
                    from klareco.parser import parse
                    doc_ast = parse(doc_text)
                    doc_entities = self.entity_recognizer.recognize_entities(doc_ast)
                    has_expected_type = any(e.entity_type == expected_entity_type for e in doc_entities)

                    if has_expected_type:
                        boost_multiplier *= ENTITY_TYPE_BOOST
                        type_matches += 1
                        reasons.append(f"contains {expected_entity_type.value}")
                except Exception as e:
                    # Parsing failed - skip entity type check for this doc
                    logger.debug(f"Failed to parse document {result.doc_id}: {e}")

            # Check if document contains specific query entities
            # But SKIP overly common entities like "Esperanto" in Esperanto corpus
            for entity in query_entities:
                # Skip if entity is the word "Esperanto" (too common in this corpus)
                if entity.root.lower() in {'esperant', 'esper'}:
                    continue

                entity_text = entity.text.lower()
                entity_root = entity.root.lower() if entity.root else None

                # Check for exact text or root match
                if entity_text in doc_text_lower or (entity_root and entity_root in doc_text_lower):
                    boost_multiplier *= SPECIFIC_ENTITY_BOOST
                    entity_matches += 1
                    reasons.append(f"contains {entity.text}")
                    break  # Only count once per document

            # Apply boost if any reason found
            if boost_multiplier > 1.0:
                result.score *= boost_multiplier
                logger.info(f"    ✓ {', '.join(reasons)} → "
                           f"score: {original_score:.3f} × {boost_multiplier:.2f} = {result.score:.3f}")

            boosted_results.append(result)

        # Re-sort by adjusted scores
        boosted_results.sort(key=lambda r: r.score, reverse=True)

        logger.info(f"  Entity boost summary: {type_matches} with expected type, {entity_matches} with specific entities")

        return boosted_results

    def _apply_document_quality_filter(self, results: List) -> List:
        """
        Apply document quality filtering to penalize low-quality documents.

        Problem: Some documents are indices, tables, or lists that match many queries
        but don't contain meaningful answers.

        Example: A 3429-word document with 736 clauses (likely an index):
        ```
        "Dato...............................53, 367, 384, 385
        N-finaĵo...................................384, 385..."
        ```

        This ranks #1 for many queries because it contains many keywords,
        but it's not actually an answer.

        Solution: Detect and penalize low-quality documents based on:
        1. Excessive length (>1000 words → likely index/appendix)
        2. Extreme clause count (>100 clauses → likely list/table)
        3. High punctuation density (>0.15 → likely table/reference)
        4. Very short documents (<10 words → likely fragment)

        Scoring:
        - Normal document: 1.0x (no penalty)
        - Long document (>1000 words): 0.5x penalty
        - Extreme clauses (>100): 0.5x penalty
        - High punctuation (>0.15): 0.7x penalty
        - Very short (<10 words): 0.6x penalty
        - Penalties are multiplicative

        Args:
            results: List of SearchResult objects

        Returns:
            List of SearchResult objects with adjusted scores
        """
        filtered_results = []
        penalized_count = 0

        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if not doc:
                continue

            doc_text = doc.get('text', '')
            original_score = result.score
            penalty_multiplier = 1.0
            reasons = []

            # Count words
            word_count = len(doc_text.split())

            # Count clauses (rough estimate using punctuation)
            clause_markers = doc_text.count(',') + doc_text.count(';') + doc_text.count(':')
            estimated_clauses = 1 + clause_markers

            # Calculate punctuation density
            punctuation_count = sum(1 for c in doc_text if c in ',.;:!?-–—')
            punctuation_density = punctuation_count / len(doc_text) if doc_text else 0

            # Apply penalties (graduated for extreme cases)

            # 1. Very short documents (fragments)
            if word_count < 10:
                penalty_multiplier *= 0.6
                reasons.append(f"very short ({word_count} words)")

            # 2. Length penalties (graduated)
            if word_count > 2000:
                penalty_multiplier *= 0.2
                reasons.append(f"extremely long ({word_count} words)")
            elif word_count > 1000:
                penalty_multiplier *= 0.5
                reasons.append(f"very long ({word_count} words)")

            # 3. Clause count penalties (graduated)
            if estimated_clauses > 500:
                penalty_multiplier *= 0.1
                reasons.append(f"extremely high clauses ({estimated_clauses})")
            elif estimated_clauses > 100:
                penalty_multiplier *= 0.5
                reasons.append(f"high clauses ({estimated_clauses})")

            # 4. Punctuation density penalties (graduated)
            if punctuation_density > 0.20:
                penalty_multiplier *= 0.5
                reasons.append(f"very high punctuation ({punctuation_density:.2%})")
            elif punctuation_density > 0.15:
                penalty_multiplier *= 0.7
                reasons.append(f"high punctuation ({punctuation_density:.2%})")

            # 5. Index/table document detection (combined signals)
            # Index documents have: long length + extreme clauses + high punctuation
            if word_count > 1000 and estimated_clauses > 200 and punctuation_density > 0.15:
                penalty_multiplier *= 0.1
                reasons.append("likely index/table document")

            # Apply penalty if any reason found
            if penalty_multiplier < 1.0:
                result.score *= penalty_multiplier
                penalized_count += 1
                logger.info(f"    ⚠ Low quality: {', '.join(reasons)} → "
                           f"score: {original_score:.3f} × {penalty_multiplier:.2f} = {result.score:.3f}")

            filtered_results.append(result)

        # Re-sort by adjusted scores
        filtered_results.sort(key=lambda r: r.score, reverse=True)

        if penalized_count > 0:
            logger.info(f"  Quality filter: {penalized_count} documents penalized")

        return filtered_results

    def _apply_root_role_ranking(self, query_ast: Dict, results: List) -> List:
        """
        Apply role-based ranking to improve query disambiguation.

        Problem: Query "Kiu fondis Esperanton?" (Who founded Esperanto?)
        - "esperant" is the HEAD (main concept)
        - Should prefer results where "esperant" is also HEAD
        - Should downrank results where "esperant" is just a MODIFIER

        Examples:
        - "Zamenhof fondis Esperanton" → HEAD match (good)
        - "Esperanto-movado kreskis" → HEAD match if "esperant" is head (good)
        - "Schmidt fondis Esperanto-klubon" → MODIFIER match (downrank)

        Solution: Extract query root, check role in result compounds, apply penalty.

        Scoring:
        - Query root is result HEAD (radiko): 1.0x (no penalty)
        - Query root is result MODIFIER (kunmetajhoj): 0.3x (penalty)

        Args:
            query_ast: Parsed query AST
            results: List of SearchResult objects from root_index.search()

        Returns:
            List of SearchResult objects with adjusted scores
        """
        query_obj = query_ast.get('objekto')

        # Check if query has an object (some queries don't, like "Kio estas X?")
        if not query_obj:
            return results

        # Get the core word from vortgrupo
        if query_obj.get('tipo') == 'vortgrupo':
            query_kerno = query_obj.get('kerno', {})
        else:
            query_kerno = query_obj

        # Extract query root
        query_root = query_kerno.get('radiko', '').lower()
        if not query_root:
            # No root to match on - return as-is
            return results

        # Check if query is a proper noun or substantivo (where disambiguation matters)
        query_vortspeco = query_kerno.get('vortspeco', '')
        if query_vortspeco not in ['propra_nomo', 'substantivo']:
            # Not a noun - no role ranking needed
            return results

        # Apply role-based scoring
        logger.info(f"  Applying role-based ranking for query root: '{query_root}'")

        ranked_results = []
        head_matches = 0
        modifier_matches = 0
        neutral_matches = 0

        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if not doc:
                continue

            result_ast = doc.get('ast', {})
            result_obj = result_ast.get('objekto', {})

            # Get the core word from result vortgrupo
            if result_obj.get('tipo') == 'vortgrupo':
                result_kerno = result_obj.get('kerno', {})
            else:
                result_kerno = result_obj

            # Check role of query root in result
            result_head = result_kerno.get('radiko', '').lower()
            result_modifiers = result_kerno.get('kunmetajhoj', [])
            modifier_roots = [m.get('radiko', '').lower() for m in result_modifiers if isinstance(m, dict)]

            original_score = result.score

            # Calculate role match score
            if result_head == query_root:
                # Query root is HEAD in result → perfect match
                role_score = 1.0
                head_matches += 1
                logger.info(f"    ✓ HEAD match: '{query_root}' in '{result_kerno.get('plena_vorto', '')}' (score: {original_score:.3f} × 1.0 = {original_score:.3f})")
            elif query_root in modifier_roots:
                # Query root is MODIFIER in result → penalty
                role_score = 0.3
                modifier_matches += 1
                new_score = original_score * role_score
                logger.info(f"    ⚠ MODIFIER match: '{query_root}' in '{result_kerno.get('plena_vorto', '')}' (score: {original_score:.3f} × 0.3 = {new_score:.3f})")
            else:
                # Query root not in object at all (matched on subject/verb) → neutral
                role_score = 1.0
                neutral_matches += 1

            # Apply role score
            result.score *= role_score
            ranked_results.append(result)

        # Re-sort by adjusted scores
        ranked_results.sort(key=lambda r: r.score, reverse=True)

        logger.info(f"  Role ranking summary: {head_matches} HEAD, {modifier_matches} MODIFIER (penalized), {neutral_matches} neutral")

        return ranked_results

    def _filter_by_entity_type(self, query_ast: Dict, results: List) -> List:
        """
        Filter results by entity type match.

        Problem: Query "Kiu fondis Esperanton?" asks about founding a LANGUAGE
        - Should match: "Zamenhof fondis Esperanton" (LANGUAGE)
        - Should NOT match: "Schmidt fondis Esperanto-klubon" (ORGANIZATION)

        Solution: Classify query and result entities, filter by type match.

        Args:
            query_ast: Parsed query AST
            results: List of SearchResult objects from root_index.search()

        Returns:
            Filtered list of SearchResult objects
        """
        query_obj = query_ast.get('objekto')

        # Check if query has an object
        if not query_obj:
            return results

        # Get the core word from vortgrupo
        if query_obj.get('tipo') == 'vortgrupo':
            query_kerno = query_obj.get('kerno', {})
        else:
            query_kerno = query_obj

        # Classify query entity type
        query_entity_type = self.entity_classifier.classify(query_kerno)

        # If query entity type is UNKNOWN, don't filter
        if query_entity_type == EntityType.UNKNOWN:
            logger.info(f"  Query entity type UNKNOWN - skipping entity type filtering")
            return results

        logger.info(f"  Query entity type: {query_entity_type}")

        # Filter results by entity type
        filtered_results = []
        type_matches = 0
        type_mismatches = 0

        for result in results:
            doc = self.root_index.get_document(result.doc_id)
            if not doc:
                continue

            result_ast = doc.get('ast', {})
            result_obj = result_ast.get('objekto', {})

            # Get the core word from result vortgrupo
            if result_obj.get('tipo') == 'vortgrupo':
                result_kerno = result_obj.get('kerno', {})
            else:
                result_kerno = result_obj

            # Classify result entity type
            result_entity_type = self.entity_classifier.classify(result_kerno)

            # Check if types match
            if result_entity_type == query_entity_type:
                type_matches += 1
                filtered_results.append(result)
                logger.debug(f"    ✓ Match: {query_entity_type} == {result_entity_type} ({result_kerno.get('plena_vorto', '')})")
            elif result_entity_type == EntityType.UNKNOWN:
                # Keep UNKNOWN results (might be correct, just can't classify)
                filtered_results.append(result)
            else:
                type_mismatches += 1
                logger.info(f"    ✗ Filtered: {query_entity_type} != {result_entity_type} ({result_kerno.get('plena_vorto', '')})")

        logger.info(f"  Entity type filter: {type_matches} matches, {type_mismatches} filtered out")

        return filtered_results

    def search_simple(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """
        Simplified search returning just (score, document) tuples.

        Convenience wrapper around search() that drops stats.
        """
        results = self.search(query, top_k)
        return [(score, doc) for score, doc, _ in results]

    def explain_retrieval(self, query: str, doc_id: int) -> Dict:
        """
        Explain why a document was retrieved for a query.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            Explanation dict with roots, synonyms, and scoring details
        """
        # Parse query
        query_ast = parse(query)

        # Classify
        classification = self.question_classifier.classify(query, query_ast)

        # Get document
        doc = self.root_index.get_document(doc_id)
        if not doc:
            return {'error': f'Document {doc_id} not found'}

        # Extract query roots
        query_roots = self._extract_roots(query_ast)

        # Get synonyms for each root (using Kuzu graph traversal)
        synonyms = {}
        for root in query_roots:
            syns = self.root_index.get_synonyms_transitive(root, max_hops=2)
            if syns:
                synonyms[root] = list(syns)

        # Check which roots are in the document
        doc_text = doc.get('text', '').lower()
        roots_in_doc = [r for r in query_roots if r in doc_text]
        synonyms_in_doc = []
        for root, syns in synonyms.items():
            for syn in syns:
                if syn in doc_text:
                    synonyms_in_doc.append(f"{root}→{syn}")

        # Add hypernym chains for query roots
        hypernym_chains = {}
        for root in query_roots:
            chain = self.root_index.get_hypernym_chain(root, max_depth=3)
            if chain:
                hypernym_chains[root] = chain

        result = {
            'query': query,
            'document': doc.get('text', ''),
            'doc_id': doc_id,
            'question_type': classification['question_type'].value,
            'query_roots': query_roots,
            'roots_found_in_doc': roots_in_doc,
            'synonyms': synonyms,
            'synonyms_found_in_doc': synonyms_in_doc,
        }

        if hypernym_chains:
            result['hypernym_chains'] = hypernym_chains

        return result

    def get_sentence_context(
        self,
        doc_id: int,
        window: int = 2,
    ) -> List[Dict]:
        """
        Get surrounding sentences for a document/sentence.

        Args:
            doc_id: Sentence/document ID
            window: Number of sentences before/after to retrieve

        Returns:
            List of context sentences with distance info
        """
        return self.root_index.get_sentence_context(doc_id, window)

    def search_by_role(
        self,
        verb: Optional[str] = None,
        subj: Optional[str] = None,
        obj: Optional[str] = None,
        expand_synonyms: bool = True,
        max_results: int = 100,
    ) -> List[Tuple[int, str]]:
        """
        Search for sentences matching specific grammatical role patterns.

        Example:
            search_by_role(verb="fond", obj="esperant")
            → Find sentences where "fond" is the VERB and "esperant" is the OBJECT

        Args:
            verb: Root that should appear as verb
            subj: Root that should appear as subject
            obj: Root that should appear as object
            expand_synonyms: Whether to expand roots with synonyms
            max_results: Maximum results to return

        Returns:
            List of (doc_id, text) tuples
        """
        return self.root_index.search_by_role(
            verb=verb,
            subj=subj,
            obj=obj,
            expand_synonyms=expand_synonyms,
            max_results=max_results,
        )

    def get_hypernym_chain(self, root: str, max_depth: int = 5) -> List[str]:
        """
        Get hypernym chain for a root word.

        Example: "hundo" → ["besto", "vivaĵo", "aĵo"]

        Args:
            root: Root word to look up
            max_depth: Maximum chain depth

        Returns:
            List of hypernyms from specific to general
        """
        return self.root_index.get_hypernym_chain(root, max_depth)

    def get_synonyms_transitive(self, root: str, max_hops: int = 2) -> set:
        """
        Get transitive synonyms (synonyms of synonyms).

        Args:
            root: Root word to look up
            max_hops: Maximum synonym chain length

        Returns:
            Set of transitive synonyms
        """
        return self.root_index.get_synonyms_transitive(root, max_hops)

    def _extract_roots(self, ast: Dict) -> List[str]:
        """Extract content word roots from AST."""
        roots = []
        skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

        def extract(node):
            if not node or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco', '')
                if vortspeco not in skip_vortspeco:
                    root = node.get('radiko', '')
                    if root and len(root) >= 2:
                        roots.append(root.lower())
            elif node.get('tipo') == 'vortgrupo':
                extract(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract(p)
            elif node.get('tipo') == 'frazo':
                extract(node.get('subjekto'))
                extract(node.get('verbo'))
                extract(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract(a)

        extract(ast)
        return roots

    def close(self):
        """Clean up resources."""
        if self.root_index:
            self.root_index.close()

    def __del__(self):
        self.close()
