"""
Fact Extractor - Extract Facts from AST-Parsed Sentences

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema, klareco.parser
STAGE: Summarization - Phase 0

Description:
    Extracts facts from AST-parsed Esperanto sentences.
    Uses deterministic rules to identify subject-predicate-object triples.

Fact Format:
    {
        'predicate': 'fond',           # Main verb root
        'subject': 'Zamenhof',         # Subject phrase
        'object': 'Esperanton',        # Object phrase
        'subject_root': 'hom',         # Subject head root
        'object_root': 'nom',          # Object head root
        'temporal_marker': False,      # Has temporal info?
        'spatial_marker': False,       # Has spatial info?
        'source_id': 'sent_123',       # Source sentence ID
        'source_text': '...'           # Original sentence
    }

Usage:
    from klareco.summarization import FactExtractor

    extractor = FactExtractor()
    facts = extractor.extract_facts(sentences)

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: klareco/parser.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from klareco.parser import parse as parse_esperanto
from .retriever import RetrievedSentence


class FactExtractor:
    """
    Extract facts from AST-parsed sentences.

    Uses deterministic rules based on Esperanto grammar.
    """

    def __init__(self):
        """Initialize fact extractor."""
        pass  # No parser instance needed, using functional API

        # Temporal markers (words indicating time)
        self.temporal_keywords = {
            'kiam', 'tiam', 'nun', 'antaŭe', 'poste',
            'hodiaŭ', 'hieraŭ', 'morgaŭ', 'en', 'dum', 'post', 'antaŭ'
        }

        # Spatial markers (words indicating location)
        self.spatial_keywords = {
            'kie', 'tie', 'ĉi tie', 'for', 'proksime',
            'en', 'sur', 'sub', 'apud', 'ĉe', 'trans'
        }

    def extract_facts(
        self,
        sentences: List[RetrievedSentence]
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from sentences.

        Args:
            sentences: Retrieved sentences

        Returns:
            List of fact dictionaries
        """
        facts = []

        for sentence in sentences:
            try:
                sentence_facts = self._extract_from_sentence(sentence)
                facts.extend(sentence_facts)
            except Exception as e:
                print(f"Warning: Failed to extract facts from sentence '{sentence.sentence_id}': {e}")
                continue

        return facts

    def _extract_from_sentence(
        self,
        sentence: RetrievedSentence
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from a single sentence.

        Args:
            sentence: Retrieved sentence

        Returns:
            List of facts extracted from this sentence
        """
        # Parse sentence to AST
        parse_result = parse_esperanto(sentence.text)

        if not parse_result or parse_result.get('tipo') != 'frazo':
            return []

        ast = parse_result

        # Extract main clause fact (subject-verb-object)
        facts = []

        # Get verb (predicate)
        verb_node = ast.get('verbo')
        if not verb_node or verb_node.get('tipo') != 'vorto':
            return []

        predicate = verb_node.get('radiko', '')
        if not predicate:
            return []

        # Get subject
        subject_node = ast.get('subjekto')
        subject, subject_root = self._extract_phrase(subject_node)

        # Get object
        object_node = ast.get('objekto')
        obj, object_root = self._extract_phrase(object_node)

        # Check for temporal/spatial markers
        temporal = self._has_temporal_marker(ast)
        spatial = self._has_spatial_marker(ast)

        # Phase 1: Build minimal AST for clean fact sentence (subject-verb-object only)
        minimal_ast = {
            'tipo': 'frazo',
            'subjekto': subject_node,
            'verbo': verb_node,
            'objekto': object_node,
            'aliaj': [],  # No extra modifiers for clean facts
            'fraztipo': ast.get('fraztipo', 'deklaro'),
            'negita': ast.get('negita', False)
        }

        # Build fact
        fact = {
            'predicate': predicate,
            'subject': subject or '',
            'object': obj or '',
            'subject_root': subject_root or '',
            'object_root': object_root or '',
            'temporal_marker': temporal,
            'spatial_marker': spatial,
            'source_id': sentence.sentence_id,
            'source_text': sentence.text,
            'ast': minimal_ast,  # Phase 1: Minimal AST for clean deparser output
            'full_ast': ast  # Preserve full AST for reference
        }

        facts.append(fact)

        return facts

    def _extract_phrase(self, node: Optional[Dict]) -> tuple:
        """
        Extract text and head root from phrase node.

        Args:
            node: AST node (vortgrupo or vorto)

        Returns:
            (phrase_text, head_root)
        """
        if not node:
            return None, None

        if node.get('tipo') == 'vorto':
            # Single word - parser uses 'plena_vorto' not 'vorto'
            text = node.get('plena_vorto', '')
            root = node.get('radiko', '')
            return text, root

        elif node.get('tipo') == 'vortgrupo':
            # Phrase - get head word (kerno)
            kerno = node.get('kerno')
            if kerno and kerno.get('tipo') == 'vorto':
                text = kerno.get('plena_vorto', '')
                root = kerno.get('radiko', '')

                # Add modifiers (simplified)
                priskriboj = node.get('priskriboj', [])
                if priskriboj:
                    # Just use head for now (full phrase would be more complex)
                    pass

                return text, root

        return None, None

    def _has_temporal_marker(self, ast: Dict) -> bool:
        """Check if AST contains temporal markers."""
        return self._contains_keywords(ast, self.temporal_keywords)

    def _has_spatial_marker(self, ast: Dict) -> bool:
        """Check if AST contains spatial markers."""
        return self._contains_keywords(ast, self.spatial_keywords)

    def _contains_keywords(self, node: Dict, keywords: set) -> bool:
        """
        Recursively check if AST node contains any keywords.

        Args:
            node: AST node
            keywords: Set of keywords to look for

        Returns:
            True if any keyword found
        """
        if not isinstance(node, dict):
            return False

        # Check if this node is a word with matching root
        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '').lower()
            word = node.get('plena_vorto', '').lower()  # Parser uses plena_vorto
            if root in keywords or word in keywords:
                return True

        # Recursively check children
        for key, value in node.items():
            if isinstance(value, dict):
                if self._contains_keywords(value, keywords):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if self._contains_keywords(item, keywords):
                            return True

        return False

    def get_statistics(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get fact extraction statistics."""
        if not facts:
            return {
                'total_facts': 0,
                'with_subject': 0,
                'with_object': 0,
                'temporal': 0,
                'spatial': 0,
                'unique_predicates': 0
            }

        predicates = set()
        with_subject = 0
        with_object = 0
        temporal = 0
        spatial = 0

        for fact in facts:
            if fact.get('predicate'):
                predicates.add(fact['predicate'])
            if fact.get('subject'):
                with_subject += 1
            if fact.get('object'):
                with_object += 1
            if fact.get('temporal_marker'):
                temporal += 1
            if fact.get('spatial_marker'):
                spatial += 1

        return {
            'total_facts': len(facts),
            'with_subject': with_subject,
            'with_object': with_object,
            'temporal': temporal,
            'spatial': spatial,
            'unique_predicates': len(predicates)
        }
