"""
Whoosh-based Full-Text Search Retriever

Fast keyword-based sentence retrieval using Whoosh FTS index.
Replaces slow Kuzu scanning with efficient BM25-ranked search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import kuzu
from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

from klareco.parser import parse

logger = logging.getLogger(__name__)


def expand_esperanto_root(root: str, question_type: Optional[str] = None) -> List[str]:
    """
    Expand an Esperanto root to word forms, optimized by question type.

    SMART EXPANSION OPTIMIZATION:
    Instead of generating all 15 forms, prioritize forms based on question type.
    This reduces query size and improves precision.

    Args:
        root: Esperanto root (e.g., "fond")
        question_type: Optional question type (who/what/when/where) for prioritization

    Returns:
        List of expanded forms (4-6 forms for typed questions, 15 for unknown)
    """
    forms = [root]  # Always include bare root

    # TEMPORARILY DISABLED: Always use full expansion to test if smart expansion is causing issues
    # If no question type, use full expansion (backward compatibility)
    if True:  # question_type is None or question_type.lower() not in ['who', 'what', 'when', 'where', 'how', 'why']:
        # Full expansion (~15 forms)
        forms.extend([root + 'as', root + 'is', root + 'i'])
        forms.extend([root + 'anta', root + 'inta', root + 'ita'])
        forms.extend([root + 'o', root + 'on'])
        forms.extend([root + 'into', root + 'anto', root + 'onto'])
        forms.append(root + 'a')
        return forms

    # Smart expansion based on question type
    question_type = question_type.lower()

    if question_type == 'who':
        # For WHO questions, focus on agent nouns and past tense
        forms.extend([
            root + 'is',    # past tense (most common in Wikipedia)
            root + 'into',  # agent noun (the one who X'd)
            root + 'anto',  # present agent
        ])

    elif question_type == 'what':
        # For WHAT questions, focus on nouns and adjectives
        forms.extend([
            root + 'o',     # noun
            root + 'a',     # adjective
            root + 'i',     # infinitive (for definitions)
            root + 'aĵo',   # thing characterized by root
        ])

    elif question_type == 'when':
        # For WHEN questions, focus on verb forms
        forms.extend([
            root + 'is',    # past tense
            root + 'as',    # present tense
            root + 'os',    # future tense
            root + 'inta',  # past participle
        ])

    elif question_type in ['where', 'how', 'why']:
        # For other questions, use medium expansion
        forms.extend([
            root + 'is',    # past tense
            root + 'o',     # noun
            root + 'a',     # adjective
            root + 'i',     # infinitive
        ])

    return forms  # 4-6 forms per root (3-4x smaller than full expansion)


class WhooshRetriever:
    """
    Retrieve sentences using Whoosh full-text search.

    Combines Whoosh FTS (for fast keyword retrieval) with Kuzu (for AST metadata).
    """

    def __init__(
        self,
        whoosh_index_dir: Path,
        kuzu_db_path: Path
    ):
        """
        Initialize retriever.

        Args:
            whoosh_index_dir: Path to Whoosh index directory
            kuzu_db_path: Path to Kuzu database (for AST retrieval)
        """
        self.whoosh_index_dir = Path(whoosh_index_dir)
        self.kuzu_db_path = Path(kuzu_db_path)

        # Open Whoosh index
        logger.info(f"Loading Whoosh index from {whoosh_index_dir}")
        self.ix = open_dir(str(whoosh_index_dir))

        # Connect to Kuzu for AST retrieval
        logger.info(f"Connecting to Kuzu database at {kuzu_db_path}")
        self.kuzu_db = kuzu.Database(str(kuzu_db_path))
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)

    def retrieve(
        self,
        query_roots: List[str],
        top_k: int = 20,
        retrieval_limit: int = 200,
        question_type: Optional[str] = None,
        query_entity: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve sentences matching query roots with AST-aware filtering.

        OPTIMIZATIONS APPLIED:
        - AND queries for proper names: Require proper name + word forms (10-50x speedup!)
        - Reduced retrieval limit: 200 instead of 1000 (1.5x speedup)
        - Lazy parsing: Parse ASTs only for top 50 candidates after BM25 filtering
        - AST cache: parse() function uses LRU cache to avoid re-parsing duplicates

        Args:
            query_roots: List of root words to search for
            top_k: Number of top results to return
            retrieval_limit: Maximum candidates to retrieve from Whoosh
            question_type: Question type (who/what/where/when) for AST filtering
            query_entity: Entity being asked about (e.g., "esperant" for "Esperanton")

        Returns:
            List of sentence dicts with 'text', 'ast', 'id', 'score', 'matching_roots'
        """
        if not query_roots:
            return []

        logger.info(f"Query roots received: {query_roots}")

        # AND QUERY OPTIMIZATION: Separate proper names from common words
        # Proper names (capitalized) should be required via AND, not expanded
        # Common words should be expanded to word forms
        proper_names = []
        common_words = []

        for root in query_roots:
            # Detect proper names: capitalized AND length > 2 (avoid "Mi", "Li", etc.)
            if root and len(root) > 2 and root[0].isupper():
                proper_names.append(root.lower())  # Lowercase for case-insensitive search
            else:
                common_words.append(root)

        # Build Whoosh query based on what we have
        if proper_names and common_words:
            # BEST CASE: Have both proper names and common words
            # Query: (name1 OR name2) AND (expanded_word_forms)
            # Example: "Lincoln AND (estis OR estinta OR estanto)"

            name_part = ' OR '.join(proper_names)

            word_forms = []
            for word in common_words:
                word_forms.extend(expand_esperanto_root(word, question_type=None))
            word_part = ' OR '.join(word_forms)

            query_str = f"({name_part}) AND ({word_part})"
            logger.info(f"AND query: {len(proper_names)} names + {len(word_forms)} forms")

        elif proper_names:
            # Only proper names - OR them together
            # Example: "Lincoln OR Roosevelt"
            query_str = ' OR '.join(proper_names)
            logger.info(f"Proper names only: {len(proper_names)} names")

        else:
            # Only common words - expand all (current behavior)
            all_forms = []
            for root in query_roots:
                all_forms.extend(expand_esperanto_root(root, question_type=None))
            query_str = ' OR '.join(all_forms)
            logger.info(f"Common words only: {len(all_forms)} forms")

        logger.debug(f"Whoosh query: {query_str[:150]}...")

        # Search Whoosh index
        with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
            query = QueryParser("text_lower", self.ix.schema, group=OrGroup).parse(query_str)
            results = searcher.search(query, limit=retrieval_limit)

            logger.info(f"Whoosh found {len(results)} matching sentences")
            id_to_score = {hit['id']: hit.score for hit in results}
            sentence_ids = list(id_to_score.keys())

        if not sentence_ids:
            return []

        # Fetch ASTs from Kuzu for matched sentences
        # Build query with IN clause (IDs must be integers, not strings)
        ids_str = ','.join(str(sid) for sid in sentence_ids)
        kuzu_query = f"""
            MATCH (ft:Frazoteksto)
            WHERE ft.id IN [{ids_str}]
            RETURN ft.id AS id, ft.teksto AS text
        """

        result = self.kuzu_conn.execute(kuzu_query)

        # Build sentence dict WITHOUT parsing ASTs yet (lazy parsing optimization)
        documents = []
        while result.has_next():
            row = result.get_next()
            sentence_id = str(row[0])
            text = row[1]

            if not text:
                continue

            # Count matching roots (for metadata)
            text_lower = text.lower()
            matching_roots = [r for r in query_roots if r in text_lower]

            # Get Whoosh BM25 score
            bm25_score = id_to_score.get(sentence_id, 0.0)

            documents.append({
                'text': text,
                'ast': None,  # Lazy parsing - parse later
                'id': sentence_id,
                'score': bm25_score,  # Use actual BM25 score from Whoosh
                'matching_roots': matching_roots,
                'num_matches': len(matching_roots),
                'doc_title': '',  # Not available in current schema
                'metadata': ''
            })

        # Meta-content filtering: penalize dictionary definitions, word lists, disambiguation pages
        # This addresses the problem where BM25 retrieves meta-content instead of factual sentences
        documents = self._filter_meta_content(documents)
        penalized_count = sum(1 for d in documents if d.get('meta_penalty', 1.0) < 1.0)
        if penalized_count > 0:
            logger.info(f"Meta-content penalty applied to {penalized_count}/{len(documents)} documents")

        # Sort by score (BM25 + meta-content penalty) BEFORE parsing
        documents.sort(key=lambda d: d['score'], reverse=True)

        # Lazy parsing: Parse ASTs only for top candidates (default: top 50)
        # This reduces parsing from 200 sentences to 50 (4x speedup)
        parse_limit = min(50, len(documents))
        for i in range(parse_limit):
            if documents[i]['ast'] is None:
                documents[i]['ast'] = parse(documents[i]['text'])

        # AST-aware filtering: boost documents where entity appears in correct grammatical role
        # Only applies to parsed documents (top 50)
        if question_type and query_entity:
            logger.info(f"Applying AST filtering: question_type={question_type}, entity={query_entity}")
            documents = self._apply_ast_filtering(documents[:parse_limit], question_type, query_entity)

            # Log boost statistics
            boosted_count = sum(1 for d in documents if d.get('ast_boost', 1.0) > 1.0)
            if boosted_count > 0:
                logger.info(f"AST boosted {boosted_count}/{len(documents)} documents")

        # Final sort by score (BM25 + AST boost)
        documents.sort(key=lambda d: d['score'], reverse=True)

        return documents[:top_k]

    def _filter_meta_content(self, documents: List[Dict]) -> List[Dict]:
        """
        Penalize meta-content (dictionaries, word lists, disambiguation pages).

        PROBLEM: BM25 retrieves meta-content with high keyword frequency instead of
        factual sentences:
        - "La rusa adjektivo учредительный signifas ankaŭ fonda..." (dictionary)
        - "Kre aŭ KRE havas plurajn signifojn: *kre estas kodo..." (disambiguation)
        - "*Esperant' *Sociolekta Triopo..." (word list)

        SOLUTION: Detect meta-content patterns and apply score penalty.

        Args:
            documents: Retrieved documents

        Returns:
            Documents with meta-content penalties applied
        """
        meta_content_patterns = [
            # Dictionary definitions (foreign language explanations)
            ('учредительный', 0.2),  # Cyrillic (Russian dictionary entries)
            ('signifo', 0.4),         # "meaning" - often in dictionary entries
            ('signifas', 0.4),        # "means" - dictionary definitions

            # Disambiguation pages
            ('havas plurajn signifojn', 0.1),  # "has multiple meanings"
            ('estas kodo de lingvo', 0.1),      # "is a language code"
            ('laŭ normo iso', 0.2),             # "according to ISO standard"

            # Word lists / bullet lists
            ('* ', 0.5),  # Lines starting with bullet points
            ('*Esperant', 0.3),  # Word lists with asterisks

            # Meta-linguistic content
            ('adjektivo', 0.4),   # "adjective" - grammar terminology
            ('substantivo', 0.4), # "noun" - grammar terminology
            ('verbo estas', 0.4), # Verb conjugation tables

            # Generic fragments / truncated sentences
            (' ... ', 0.5),  # Ellipsis indicating truncation
            ('(UTC)', 0.3),  # Timestamps from talk pages

            # ReVo dictionary structure
            ('|| Vidu la artikolon:', 0.3),  # ReVo cross-references
            ('Tabulo omaĝe al', 0.3),        # Monument/plaque descriptions
        ]

        for doc in documents:
            text = doc['text']
            penalty = 1.0  # No penalty by default

            # Check each pattern
            for pattern, pattern_penalty in meta_content_patterns:
                if pattern in text:
                    penalty = min(penalty, pattern_penalty)
                    # Take the MOST severe penalty (lowest multiplier)

            # Apply penalty to score
            if penalty < 1.0:
                doc['score'] *= penalty
                doc['meta_penalty'] = penalty

        return documents

    def _apply_ast_filtering(
        self,
        documents: List[Dict],
        question_type: str,
        query_entity: str
    ) -> List[Dict]:
        """
        Apply AST-aware filtering by checking grammatical roles.

        For "Kiu kreis Esperanton?" - boost sentences where "esperant" is OBJECT (accusative)
        For "Kio estas Esperanto?" - boost sentences where "esperant" is SUBJECT
        For "Kie estas Pollando?" - boost sentences where "pol" is SUBJECT

        Args:
            documents: Retrieved documents with ASTs
            question_type: Question type (who/what/where/when)
            query_entity: Entity root to check (e.g., "esperant")

        Returns:
            Documents with AST-boosted scores
        """
        question_type_lower = question_type.lower()

        for doc in documents:
            ast = doc.get('ast')
            if not ast or ast.get('tipo') != 'frazo':
                continue

            # Check if entity appears in correct grammatical role
            boost_multiplier = 1.0

            # WHO questions: "Kiu kreis Esperanton?" → esperant should be OBJECT (accusative)
            if question_type_lower == 'who':
                if self._entity_in_object(ast, query_entity):
                    boost_multiplier = 3.0  # Strong boost for correct grammatical role
                    logger.debug(f"AST boost: {query_entity} in OBJECT position")

            # WHAT questions: "Kio estas Esperanto?" → esperant should be SUBJECT
            elif question_type_lower == 'what':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.5
                    logger.debug(f"AST boost: {query_entity} in SUBJECT position")
                elif self._entity_in_object(ast, query_entity):
                    boost_multiplier = 1.5  # Moderate boost for object position

            # WHERE questions: "Kie estas Pollando?" → pol should be SUBJECT
            elif question_type_lower == 'where':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.5
                elif self._entity_in_location_modifier(ast, query_entity):
                    boost_multiplier = 2.0  # "en Pollando" is also valid

            # WHEN questions: entity could be subject or object
            elif question_type_lower == 'when':
                if self._entity_in_subject(ast, query_entity):
                    boost_multiplier = 2.0
                elif self._entity_in_object(ast, query_entity):
                    boost_multiplier = 2.0

            # Apply boost to BM25 score
            doc['score'] *= boost_multiplier
            if boost_multiplier > 1.0:
                doc['ast_boost'] = boost_multiplier

        return documents

    def _entity_in_subject(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears in subject position."""
        subjekto = ast.get('subjekto')
        if not subjekto:
            return False

        return self._contains_root(subjekto, entity_root)

    def _entity_in_object(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears in object position (accusative case)."""
        objekto = ast.get('objekto')
        if not objekto:
            return False

        # Check if object contains the root AND has accusative case (-n)
        if not self._contains_root(objekto, entity_root):
            return False

        # Verify accusative case
        return self._has_accusative_case(objekto)

    def _entity_in_location_modifier(self, ast: Dict, entity_root: str) -> bool:
        """Check if entity appears as location modifier (en X, al X, etc.)."""
        aliaj = ast.get('aliaj', [])
        for alia in aliaj:
            if self._contains_root(alia, entity_root):
                # Check if it's a prepositional phrase with location preposition
                if isinstance(alia, dict):
                    if alia.get('tipo') == 'vorto':
                        radiko = alia.get('radiko', '')
                        if radiko in ['en', 'al', 'de', 'ĉe']:  # Location prepositions
                            return True
        return False

    def _contains_root(self, node: Dict, root: str) -> bool:
        """Recursively check if AST node contains a specific root."""
        if not isinstance(node, dict):
            return False

        # Check if this node has the root
        if node.get('radiko', '').lower() == root.lower():
            return True

        # Recursively check vortgrupo structure
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno and self._contains_root(kerno, root):
                return True

            priskriboj = node.get('priskriboj', [])
            for priskribo in priskriboj:
                if self._contains_root(priskribo, root):
                    return True

        return False

    def _has_accusative_case(self, node: Dict) -> bool:
        """Check if node has accusative case (-n ending)."""
        if not isinstance(node, dict):
            return False

        # Check direct accusative
        if node.get('kazo') == 'akuzativo':
            return True

        # Check vortgrupo kerno
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno and self._has_accusative_case(kerno):
                return True

        return False
