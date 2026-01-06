---
id: 53
title: Implement multi-strategy AST-aware retriever
state: open
created: '2026-01-05T15:48:57.102148Z'
labels:
- enhancement
- retrieval
- 'priority: high'
priority: high
---
## Objective
Build the integrated retriever that combines all AST-aware strategies (pattern matching, entity search, semantic roles, embeddings) for robust question answering.

## Architecture

### Multi-Strategy Parallel Retrieval

```python
class ASTAwareRetriever:
    """
    Intelligent retriever that uses AST structure and multiple strategies.
    
    Combines:
    - AST pattern matching (Task #50)
    - Entity-centric search (Task #52)
    - Semantic role matching
    - Embedding-based fallback (existing)
    """
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        question_classifier: QuestionClassifier,  # Task #49
        pattern_matcher: ASTPatternMatcher,       # Task #50
        semantic_db: SemanticRelationDB,          # Task #51
        entity_recognizer: EntityRecognizer,      # Task #52
    ):
        self.index_path = index_path
        self.indexer = indexer
        self.question_classifier = question_classifier
        self.pattern_matcher = pattern_matcher
        self.semantic_db = semantic_db
        self.entity_recognizer = entity_recognizer
        
        # Load corpus index
        self._load_corpus_index()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy_weights: Dict[str, float] = None
    ) -> List[Tuple[float, Dict]]:
        """
        Multi-strategy retrieval with AST awareness.
        
        Returns:
            List of (score, document) tuples ranked by relevance
        """
        # Parse query
        query_ast = parse(query)
        
        # Classify question type
        q_type = self.question_classifier.classify(query_ast)
        logger.info(f"Question type: {q_type.category}, seeking: {q_type.seeking}")
        
        # Run parallel retrieval strategies
        candidates = {}  # doc_id -> {scores: {...}, doc: ...}
        
        # Strategy 1: AST Pattern Matching
        logger.debug("Running pattern matching...")
        pattern_matches = self._pattern_search(query_ast, q_type)
        self._merge_candidates(candidates, pattern_matches, 'pattern')
        
        # Strategy 2: Entity-Centric Search
        logger.debug("Running entity search...")
        entity_matches = self._entity_search(query_ast, q_type)
        self._merge_candidates(candidates, entity_matches, 'entity')
        
        # Strategy 3: Semantic Role Search
        logger.debug("Running semantic role search...")
        role_matches = self._semantic_role_search(query_ast, q_type)
        self._merge_candidates(candidates, role_matches, 'role')
        
        # Strategy 4: Embedding-Based Fallback
        logger.debug("Running embedding search...")
        embedding_matches = self._embedding_search(query_ast, top_n=1000)
        self._merge_candidates(candidates, embedding_matches, 'embedding')
        
        # Rerank with AST-aware scoring
        scored = []
        for doc_id, data in candidates.items():
            final_score = self._ast_aware_score(
                query_ast, q_type, data['doc'], data['scores']
            )
            scored.append((final_score, data['doc']))
        
        # Sort and return top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
    
    def _pattern_search(self, query_ast: Dict, q_type: QuestionType) -> List[Tuple[int, float]]:
        """Search using AST pattern matching."""
        # Generate patterns for this question
        patterns = self.pattern_matcher.create_patterns_for_question(query_ast)
        
        matches = []
        for doc_id, doc_ast in enumerate(self.corpus_asts):
            for pattern in patterns:
                match = self.pattern_matcher.match(pattern, doc_ast)
                if match and match.matched:
                    # Score based on pattern confidence
                    score = match.confidence
                    matches.append((doc_id, score))
                    break  # Found a match, no need to try other patterns
        
        return matches
    
    def _entity_search(self, query_ast: Dict, q_type: QuestionType) -> List[Tuple[int, float]]:
        """Find documents with matching entities and correct entity types."""
        # Extract query entities
        query_entities = self.entity_recognizer.extract_entities(query_ast)
        
        # What entity type are we seeking?
        seeking_type = q_type.seeking  # PERSON, PLACE, TIME, etc.
        
        matches = []
        for doc_id, doc_ast in enumerate(self.corpus_asts):
            doc_entities = self.entity_recognizer.extract_entities(doc_ast)
            
            # Check entity overlap
            query_entity_texts = {e.text.lower() for e in query_entities}
            doc_entity_texts = {e.text.lower() for e in doc_entities}
            overlap = query_entity_texts & doc_entity_texts
            
            # Check if doc has entity of sought type
            has_sought_type = any(e.type == seeking_type for e in doc_entities)
            
            # Score
            score = 0.0
            score += len(overlap) * 1.5  # Entity overlap
            score += 2.0 if has_sought_type else 0.0  # Has sought entity type
            
            if score > 0:
                matches.append((doc_id, score))
        
        return matches
    
    def _semantic_role_search(self, query_ast: Dict, q_type: QuestionType) -> List[Tuple[int, float]]:
        """Match based on semantic roles (agent, action, theme)."""
        # Extract semantic roles from query
        q_action = self._get_main_action(query_ast)
        q_theme = self._get_theme(query_ast)
        
        if not q_action:
            return []
        
        # Get synonyms for action
        action_variants = [q_action] + self.semantic_db.get_synonyms(q_action)
        
        matches = []
        for doc_id, doc_ast in enumerate(self.corpus_asts):
            d_action = self._get_main_action(doc_ast)
            d_theme = self._get_theme(doc_ast)
            
            score = 0.0
            
            # Action match (or synonym)
            if d_action in action_variants:
                score += 1.5
            
            # Theme match
            if q_theme and d_theme and q_theme == d_theme:
                score += 1.5
            
            if score > 0:
                matches.append((doc_id, score))
        
        return matches
    
    def _ast_aware_score(
        self,
        query_ast: Dict,
        q_type: QuestionType,
        doc_ast: Dict,
        strategy_scores: Dict[str, float]
    ) -> float:
        """Combine strategy scores with AST-aware features."""
        # Base score from strategies
        score = 0.0
        weights = {
            'pattern': 2.0,    # Highest weight - structural match
            'entity': 1.5,     # High weight - entity evidence
            'role': 1.0,       # Medium weight - semantic match
            'embedding': 0.5   # Lowest weight - fallback
        }
        
        for strategy, weight in weights.items():
            if strategy in strategy_scores:
                score += weight * strategy_scores[strategy]
        
        # Bonus features
        
        # Source tier bonus (Wikipedia > books > etc.)
        tier = doc_ast.get('source', {}).get('tier', 0)
        score += tier * 0.1
        
        # Parse quality bonus
        parse_rate = doc_ast.get('parse_rate', 0.0)
        score += parse_rate * 0.2
        
        # Question-type specific bonuses
        if q_type.category == 'WHO':
            # Boost docs with person entities
            if self.entity_recognizer.has_entity_type(doc_ast, 'PERSON'):
                score += 1.0
        
        elif q_type.category == 'WHEN':
            # Boost docs with temporal expressions
            if self.entity_recognizer.has_entity_type(doc_ast, 'TIME'):
                score += 1.0
        
        return score
```

## Question-Type Specific Strategies

```python
def _pattern_search(self, query_ast, q_type):
    """Customize pattern matching based on question type."""
    
    if q_type.category == 'WHO':
        # For WHO questions, also accept:
        # - Appositive patterns (X, aŭtoro de Y)
        # - Title patterns (X estas Y)
        # - Biographical fragments
        pass
    
    elif q_type.intent == 'definition':
        # For "Kio estas X?" questions:
        # - Look for "X estas Y" patterns
        # - Look for definitional sentences
        pass
    
    elif q_type.category == 'WHEN':
        # For WHEN questions:
        # - Look for dates/years near key entities
        # - Temporal prepositions (en, dum, post, antaŭ)
        pass
```

## Deliverable
- `klareco/rag/ast_aware_retriever.py`
- Integration tests with benchmark questions
- Comparison script: AST-aware vs current slot-based retriever
- Documentation of strategy weights and tuning

## Success Criteria
```python
retriever = ASTAwareRetriever(...)

# Should now find the answer
results = retriever.search("Kiu fondis Esperanton?")
assert any("ZAMENHOF" in doc['text'].upper() for score, doc in results[:5])

# Should match fragments without verbs
results = retriever.search("Kiu verkis La Espero?")
# Should find "Zamenhof, aŭtoro de La Espero"

# Should handle temporal questions
results = retriever.search("Kiam aperis la Fundamento?")
# Should find sentences with "Fundamento" + "1905"
```

## Evaluation Plan
Run on 50 benchmark questions:
- Measure: % questions with answer in top-1, top-5, top-10
- Compare to baseline (current slot-based retriever)
- Target: >60% accuracy in top-10 (vs current 10-12%)

## Dependencies
- **REQUIRES**: Task #49 (question classifier)
- **REQUIRES**: Task #50 (AST pattern matcher)
- **REQUIRES**: Task #51 (semantic relation DB) - at least Phase 1
- **REQUIRES**: Task #52 (entity recognizer)

## Effort
~10 hours (integration + testing + tuning)
