"""
Retriever - Query Kuzu for Relevant Sentences

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Summarization - Phase 0

Description:
    Retrieves relevant sentences from Kuzu database based on query.
    Uses simple keyword matching for Phase 0 (deterministic).

Retrieval Strategy (Phase 0):
    1. Extract keywords from query (roots)
    2. Query Kuzu for sentences containing those roots
    3. Return top-k sentences with metadata

Future (Phase 1-2):
    - Semantic similarity search
    - Embedding-based retrieval
    - Learned reranking

Usage:
    from klareco.summarization import Retriever

    retriever = Retriever(db_path='data/indexes/v2.1_kuzu_index_full')
    sentences = retriever.retrieve(query="Rakontu pri Zamenhof", top_k=20)

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: klareco/rag/retriever.py (for advanced retrieval)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import re

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

from klareco.utils.kuzu_open import open_kuzu


@dataclass
class RetrievedSentence:
    """A sentence retrieved from corpus."""
    sentence_id: str  # Unique ID in database
    text: str  # Sentence text
    article_title: Optional[str] = None
    metadata: Dict[str, Any] = None
    relevance_score: float = 1.0  # Relevance to query (for ranking)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Retriever:
    """
    Retrieve relevant sentences from Kuzu database.

    Phase 0: Simple keyword-based retrieval (deterministic)
    """

    def __init__(self, db_path: str):
        """
        Initialize retriever with database connection.

        Args:
            db_path: Path to Kuzu database directory
        """
        self.db_path = db_path
        self.db = open_kuzu(db_path)
        self.conn = kuzu.Connection(self.db)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        subject: Optional[str] = None
    ) -> List[RetrievedSentence]:
        """
        Retrieve relevant sentences for query.

        Args:
            query: Query text (Esperanto)
            top_k: Maximum sentences to return
            subject: Optional subject filter (person name, concept, etc.)

        Returns:
            List of RetrievedSentence objects
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query, subject)

        if not keywords:
            print("Warning: No keywords extracted from query")
            return []

        # Query database for sentences containing keywords
        sentences = self._query_sentences(keywords, top_k * 2)  # Get extra for filtering

        # Rank by relevance
        ranked = self._rank_sentences(sentences, keywords)

        # Return top-k
        return ranked[:top_k]

    def _extract_keywords(self, query: str, subject: Optional[str]) -> Set[str]:
        """
        Extract keywords from query for retrieval.

        Simple heuristic: remove question words and common words.

        Args:
            query: Query text
            subject: Optional subject

        Returns:
            Set of keyword roots
        """
        keywords = set()

        # Remove question words
        question_words = {
            'kiu', 'kio', 'kiam', 'kie', 'kial', 'kiel', 'kiom', 'kies',
            'ĉu', 'ĉiuj', 'ĉio', 'ĉiu'
        }

        # Remove common function words
        function_words = {
            'la', 'de', 'al', 'en', 'sur', 'sub', 'estas', 'estis',
            'kaj', 'aŭ', 'sed', 'ĉar', 'se', 'da', 'el', 'pri'
        }

        # Split and clean
        words = re.findall(r'\b[a-zĉĝĥĵŝŭA-ZĈĜĤĴŜŬ]+\b', query.lower())

        for word in words:
            # Skip question/function words
            if word in question_words or word in function_words:
                continue

            # Try to extract root (simple heuristic)
            root = self._extract_root_simple(word)
            if root and len(root) >= 3:
                keywords.add(root)

        # Add subject if provided
        if subject:
            # Handle capitalized names
            subject_lower = subject.lower()
            keywords.add(subject_lower)

        return keywords

    def _extract_root_simple(self, word: str) -> Optional[str]:
        """
        Extract root from Esperanto word (simple heuristic).

        Removes common endings: -o, -a, -e, -i, -is, -as, -os, -us, -u, -on, -an, -ojn, etc.

        Args:
            word: Esperanto word

        Returns:
            Root (or None if too short)
        """
        # Remove common endings
        if word.endswith('ojn'):
            return word[:-3]
        elif word.endswith(('on', 'an', 'en', 'is', 'as', 'os', 'us', 'oj', 'aj')):
            return word[:-2]
        elif word.endswith(('o', 'a', 'e', 'i', 'u', 'n', 'j')):
            return word[:-1]
        else:
            return word

    def _query_sentences(self, keywords: Set[str], limit: int) -> List[Dict[str, Any]]:
        """
        Query Kuzu for sentences containing keywords.

        Args:
            keywords: Set of keyword roots
            limit: Maximum sentences to retrieve

        Returns:
            List of sentence dictionaries with metadata
        """
        sentences = []

        # Query v2.1 schema: Radiko → Vorto → AST → Frazoteksto → Dokumento
        # This traverses the AST graph to find sentences containing the root

        for keyword in keywords:
            try:
                # Query for sentences mentioning this root through AST structure
                query_str = f"""
                    MATCH (r:Radiko {{radiko: '{keyword}'}})<-[:HAVAS_RADIKON]-(v:Vorto)
                    MATCH (v)<-[:HAVAS_VERBON|HAVAS_SUBJEKTON_VORTO|HAVAS_OBJEKTON_VORTO|HAVAS_ALIAJN|HAVAS_KERNON|HAVAS_PRISKRIBON*1..3]-(f:Frazo)
                    MATCH (f)<-[:AST_HAVAS_FRAZON]-(ast:AST)
                    MATCH (ast)<-[:FRAZOTEKSTO_HAVAS_AST {{estas_nuna: true}}]-(ft:Frazoteksto)
                    OPTIONAL MATCH (ft)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)
                    RETURN DISTINCT ft.teksto as text,
                           ft.id as sentence_id,
                           d.titolo as document_title
                    LIMIT {limit}
                """

                result = self.conn.execute(query_str)

                while result.has_next():
                    row = result.get_next()
                    sentence_text = row[0]
                    sentence_id = row[1]
                    document_title = row[2]

                    if sentence_text and sentence_id:
                        sentences.append({
                            'sentence_id': str(sentence_id),
                            'text': sentence_text,
                            'article_title': document_title,
                            'matched_keyword': keyword
                        })

            except Exception as e:
                print(f"Warning: Query failed for keyword '{keyword}': {e}")
                continue

        # Deduplicate by sentence_id
        seen = set()
        unique_sentences = []
        for sent in sentences:
            if sent['sentence_id'] not in seen:
                seen.add(sent['sentence_id'])
                unique_sentences.append(sent)

        return unique_sentences

    def _rank_sentences(
        self,
        sentences: List[Dict[str, Any]],
        keywords: Set[str]
    ) -> List[RetrievedSentence]:
        """
        Rank sentences by relevance to keywords.

        Simple scoring: count how many keywords appear in sentence.

        Args:
            sentences: Retrieved sentences
            keywords: Query keywords

        Returns:
            Ranked list of RetrievedSentence objects
        """
        scored = []

        for sent_dict in sentences:
            text = sent_dict['text'].lower()
            score = 0.0

            # Count keyword matches
            for keyword in keywords:
                if keyword in text:
                    score += 1.0

            # Boost if multiple keywords
            if score > 1:
                score *= 1.2

            retrieved = RetrievedSentence(
                sentence_id=sent_dict['sentence_id'],
                text=sent_dict['text'],
                article_title=sent_dict.get('article_title'),
                metadata={'matched_keyword': sent_dict.get('matched_keyword')},
                relevance_score=score
            )
            scored.append(retrieved)

        # Sort by score (descending)
        scored.sort(key=lambda s: s.relevance_score, reverse=True)

        return scored

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics from database."""
        try:
            # Count total sentences (v2.1 schema uses Frazoteksto)
            result = self.conn.execute("MATCH (ft:Frazoteksto) RETURN COUNT(*) as total")
            total_sentences = result.get_next()[0] if result.has_next() else 0

            # Count total documents (v2.1 schema uses Dokumento)
            result = self.conn.execute("MATCH (d:Dokumento) RETURN COUNT(*) as total")
            total_articles = result.get_next()[0] if result.has_next() else 0

            # Count total roots
            result = self.conn.execute("MATCH (r:Radiko) RETURN COUNT(*) as total")
            total_roots = result.get_next()[0] if result.has_next() else 0

            return {
                'total_sentences': total_sentences,
                'total_articles': total_articles,
                'total_roots': total_roots
            }
        except Exception as e:
            print(f"Warning: Failed to get statistics: {e}")
            return {}
