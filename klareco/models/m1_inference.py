"""
M1 Selectional Preference Inference Wrapper

Provides efficient batch inference for M1 selectional preference model.
Used for query expansion, answer validation, and plausibility filtering.

Usage:
    from klareco.models.m1_inference import M1Inference

    m1 = M1Inference()

    # Score single triple
    score = m1.score_triple('hundo', 'manĝas', 'viando')  # Returns float

    # Score multiple triples
    scores = m1.score_triples([
        ('hundo', 'manĝas', 'viando'),
        ('tablo', 'pensas', 'ideo')
    ])  # Returns list of floats

    # Filter candidates by plausibility
    plausible = m1.filter_plausible([
        ('subj1', 'verb', 'obj1'),
        ('subj2', 'verb', 'obj2')
    ], threshold=0.5)  # Returns list of plausible triples
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn

from klareco.models.m1_selectional import M1SelectionalPreference
from klareco.embeddings.compositional import CompositionalEmbedding


class M1Inference:
    """
    M1 Selectional Preference Inference Wrapper.

    Provides efficient batch scoring of subject-verb-object triples.
    """

    def __init__(self, model_path: Optional[Path] = None,
                 comp_model_path: Optional[Path] = None,
                 device: str = 'cpu'):
        """
        Initialize M1 inference with compositional embeddings.

        Args:
            model_path: Path to M1 model checkpoint (default: models/m1_compositional/best_model.pt)
            comp_model_path: Path to CompositionalEmbedding model (default: from M1 checkpoint or tier0)
            device: Device to run on ('cpu' or 'cuda')
        """
        if model_path is None:
            model_path = Path('models/m1_compositional/best_model.pt')

        self.device = device

        # Load M1 checkpoint
        m1_checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Load CompositionalEmbedding
        if comp_model_path is None:
            comp_model_path = Path(m1_checkpoint.get('comp_model_path',
                                   'models/root_embeddings_tier0/best_model.pt'))

        comp_checkpoint = torch.load(comp_model_path, map_location=device, weights_only=False)

        # Check if this is a new-format CompositionalEmbedding or old Stage 1 checkpoint
        if 'root_vocab' in comp_checkpoint:
            # New format - direct load
            self.comp_emb = CompositionalEmbedding(
                root_vocab=comp_checkpoint['root_vocab'],
                prefix_vocab=comp_checkpoint['prefix_vocab'],
                suffix_vocab=comp_checkpoint['suffix_vocab'],
                embed_dim=comp_checkpoint['embed_dim'],
                composition_method=comp_checkpoint.get('composition_method', 'sum'),
            )
            self.comp_emb.load_state_dict(comp_checkpoint['model_state_dict'])
        else:
            # Old Stage 1 format - convert to CompositionalEmbedding
            root_vocab = comp_checkpoint['root_to_idx']
            embed_dim = comp_checkpoint['embedding_dim']

            # Build standard Esperanto affix vocabularies
            prefix_vocab = {
                '<NONE>': 0, 'mal': 1, 're': 2, 'dis': 3, 'ge': 4,
                'pra': 5, 'bo': 6, 'ek': 7
            }
            suffix_vocab = {
                '<NONE>': 0, 'aĉ': 1, 'ad': 2, 'aĵ': 3, 'an': 4, 'ar': 5,
                'ebl': 6, 'ec': 7, 'eg': 8, 'ej': 9, 'em': 10, 'end': 11,
                'er': 12, 'estr': 13, 'et': 14, 'id': 15, 'ig': 16, 'iĝ': 17,
                'il': 18, 'in': 19, 'ind': 20, 'ing': 21, 'ism': 22, 'ist': 23,
                'obl': 24, 'on': 25, 'op': 26, 'uj': 27, 'ul': 28, 'um': 29
            }

            self.comp_emb = CompositionalEmbedding(
                root_vocab=root_vocab,
                prefix_vocab=prefix_vocab,
                suffix_vocab=suffix_vocab,
                embed_dim=embed_dim,
                composition_method='sum',
            )

            # Load root embeddings, initialize others randomly
            state_dict = self.comp_emb.state_dict()
            state_dict['root_embed.weight'] = comp_checkpoint['model_state_dict']['embeddings.weight']
            self.comp_emb.load_state_dict(state_dict)

        self.comp_emb.eval()
        self.comp_emb = self.comp_emb.to(device)

        # Load M1 model
        self.model = M1SelectionalPreference(
            embedding_dim=m1_checkpoint['embedding_dim'],
            hidden_dim=m1_checkpoint['hidden_dim']
        )
        self.model.load_state_dict(m1_checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(device)

    def score_triple(self, subject: str, verb: str, obj: str) -> float:
        """
        Score triple using root strings only (backward compatible).

        Assumes default morphology:
        - Nouns: nominative singular ending 'o'
        - Verbs: present tense 'as'

        For full morphological control, use score_triple_full().

        Args:
            subject: Subject root
            verb: Verb root
            obj: Object root

        Returns:
            Plausibility score [0-1]
        """
        return self.score_triple_full(
            subject={'root': subject, 'prefixes': [], 'suffixes': [], 'ending': 'o'},
            verb={'root': verb, 'prefixes': [], 'suffixes': [], 'ending': 'as'},
            obj={'root': obj, 'prefixes': [], 'suffixes': [], 'ending': 'o'}
        )

    def score_triple_full(self, subject: Dict, verb: Dict, obj: Dict) -> float:
        """
        Score triple using full word structures.

        This is the primary interface for RAG query expansion and
        plausibility filtering with morphological awareness.

        Args:
            subject: {'root': str, 'prefixes': List, 'suffixes': List, 'ending': str}
            verb: {...}
            obj: {...}

        Returns:
            Plausibility score [0-1]

        Example:
            >>> m1.score_triple_full(
            ...     subject={'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'o'},
            ...     verb={'root': 'manĝ', 'prefixes': [], 'suffixes': [], 'ending': 'as'},
            ...     obj={'root': 'nutraĵ', 'prefixes': [], 'suffixes': [], 'ending': 'o'}
            ... )
            0.923
        """
        with torch.no_grad():
            # Encode with CompositionalEmbedding
            subj_emb = self.comp_emb.encode_word(
                root=subject['root'],
                prefixes=subject['prefixes'],
                suffixes=subject['suffixes'],
                ending=subject['ending']
            ).unsqueeze(0).to(self.device)

            verb_emb = self.comp_emb.encode_word(
                root=verb['root'],
                prefixes=verb['prefixes'],
                suffixes=verb['suffixes'],
                ending=verb['ending']
            ).unsqueeze(0).to(self.device)

            obj_emb = self.comp_emb.encode_word(
                root=obj['root'],
                prefixes=obj['prefixes'],
                suffixes=obj['suffixes'],
                ending=obj['ending']
            ).unsqueeze(0).to(self.device)

            # Score
            outputs = self.model(subj_emb, verb_emb, obj_emb)
            return outputs['triple_score'].item()

    def score_triples(self, triples: List[Tuple[str, str, str]]) -> List[float]:
        """
        Score multiple triples in batch (backward compatible).

        Assumes default morphology for all roots.

        Args:
            triples: List of (subject, verb, object) root tuples

        Returns:
            List of plausibility scores [0-1]
        """
        if not triples:
            return []

        # Convert to full structures with default morphology
        full_triples = [
            (
                {'root': subj, 'prefixes': [], 'suffixes': [], 'ending': 'o'},
                {'root': verb, 'prefixes': [], 'suffixes': [], 'ending': 'as'},
                {'root': obj, 'prefixes': [], 'suffixes': [], 'ending': 'o'}
            )
            for subj, verb, obj in triples
        ]

        return self.score_triples_full(full_triples)

    def score_triples_full(self, triples: List[Tuple[Dict, Dict, Dict]]) -> List[float]:
        """
        Batch score triples with full word structures.

        This is the primary interface for batch scoring with morphological awareness.

        Args:
            triples: List of (subject_struct, verb_struct, object_struct) tuples

        Returns:
            List of plausibility scores

        Example:
            >>> triples = [
            ...     (
            ...         {'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'o'},
            ...         {'root': 'manĝ', 'prefixes': [], 'suffixes': [], 'ending': 'as'},
            ...         {'root': 'nutraĵ', 'prefixes': [], 'suffixes': [], 'ending': 'o'}
            ...     ),
            ...     (
            ...         {'root': 'nutraĵ', 'prefixes': [], 'suffixes': [], 'ending': 'o'},
            ...         {'root': 'manĝ', 'prefixes': [], 'suffixes': [], 'ending': 'as'},
            ...         {'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'o'}
            ...     )
            ... ]
            >>> m1.score_triples_full(triples)
            [0.923, 0.156]  # Food doesn't eat dogs!
        """
        if not triples:
            return []

        # Batch encode
        subj_embs = []
        verb_embs = []
        obj_embs = []

        for subj, verb, obj in triples:
            with torch.no_grad():
                subj_embs.append(self.comp_emb.encode_word(**subj))
                verb_embs.append(self.comp_emb.encode_word(**verb))
                obj_embs.append(self.comp_emb.encode_word(**obj))

        # Stack and score
        subj_batch = torch.stack(subj_embs).to(self.device)
        verb_batch = torch.stack(verb_embs).to(self.device)
        obj_batch = torch.stack(obj_embs).to(self.device)

        with torch.no_grad():
            outputs = self.model(subj_batch, verb_batch, obj_batch)
            scores = outputs['triple_score'].squeeze(-1).cpu().tolist()

        return scores if isinstance(scores, list) else [scores]

    def filter_plausible(self, triples: List[Tuple[str, str, str]],
                        threshold: float = 0.5) -> List[Tuple[str, str, str]]:
        """
        Filter triples to keep only plausible ones.

        Args:
            triples: List of (subject, verb, object) tuples
            threshold: Minimum plausibility score (default: 0.5)

        Returns:
            List of plausible triples
        """
        scores = self.score_triples(triples)
        return [triple for triple, score in zip(triples, scores) if score >= threshold]

    def rank_by_plausibility(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[Tuple[str, str, str], float]]:
        """
        Rank triples by plausibility score (highest first).

        Args:
            triples: List of (subject, verb, object) tuples

        Returns:
            List of (triple, score) tuples sorted by score descending
        """
        scores = self.score_triples(triples)
        ranked = list(zip(triples, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_detailed_scores(self, subject: str, verb: str, obj: str) -> Dict[str, float]:
        """
        Get detailed scores for all components (backward compatible).

        Assumes default morphology.

        Args:
            subject: Subject root
            verb: Verb root
            obj: Object root

        Returns:
            Dict with keys: subj_verb_score, verb_obj_score, triple_score
        """
        # Encode with default morphology
        with torch.no_grad():
            subj_emb = self.comp_emb.encode_word(
                root=subject, prefixes=[], suffixes=[], ending='o'
            ).unsqueeze(0).to(self.device)

            verb_emb = self.comp_emb.encode_word(
                root=verb, prefixes=[], suffixes=[], ending='as'
            ).unsqueeze(0).to(self.device)

            obj_emb = self.comp_emb.encode_word(
                root=obj, prefixes=[], suffixes=[], ending='o'
            ).unsqueeze(0).to(self.device)

            # Score
            outputs = self.model(subj_emb, verb_emb, obj_emb)
            return {
                'subj_verb_score': outputs['subj_verb_score'].item(),
                'verb_obj_score': outputs['verb_obj_score'].item(),
                'triple_score': outputs['triple_score'].item()
            }

    def validate_answer(self, subject: str, verb: str, obj: str,
                       threshold: float = 0.5) -> Tuple[bool, float, str]:
        """
        Validate if an answer triple is plausible.

        Args:
            subject: Subject root
            verb: Verb root
            obj: Object root
            threshold: Minimum plausibility threshold

        Returns:
            (is_valid, score, explanation)
        """
        score = self.score_triple(subject, verb, obj)
        is_valid = score >= threshold

        if is_valid:
            explanation = f"Plausible (score: {score:.3f} ≥ {threshold})"
        else:
            explanation = f"Implausible (score: {score:.3f} < {threshold})"

        return is_valid, score, explanation
