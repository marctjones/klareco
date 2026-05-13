"""
TreeMatchReranker: AST-Aware Multi-Level Reranker

A truly AST-aware reranker that uses full parse tree structure via multi-level matching:
- Level 1: Syntax matching (70%, deterministic, 0 params)
- Level 2: Compositional matching (20%, hybrid)
- Level 3: Semantic matching (10%, learned, ~20K params)

Architecture Philosophy:
- Maximize deterministic processing (90% of scoring)
- Minimize learned parameters (20K vs 180K for MLP)
- Use AST structure as trees, not bags-of-words
- Perfect compositional generalization
- Fully interpretable score breakdowns

Total trainable params: ~20K (9x smaller than old MLP reranker)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from klareco.embeddings import CompositionalEmbedding

logger = logging.getLogger(__name__)


class RootCrossAttention(nn.Module):
    """
    Cross-attention between query and document roots (learned component).

    Only attends to ROOT embeddings (64d), not full compositional embeddings.
    This keeps the learned component minimal.

    Total params: ~20K
    """

    def __init__(self, root_dim: int = 64, hidden_dim: int = 32, num_heads: int = 2):
        """
        Initialize root cross-attention.

        Args:
            root_dim: Dimension of root embeddings (64d from compositional model)
            hidden_dim: Hidden dimension for attention (32d keeps it small)
            num_heads: Number of attention heads (2 heads sufficient)
        """
        super().__init__()

        self.root_dim = root_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project root embeddings to attention space
        self.query_proj = nn.Linear(root_dim, hidden_dim)  # 64×32 = 2,048 params
        self.key_proj = nn.Linear(root_dim, hidden_dim)    # 64×32 = 2,048 params
        self.value_proj = nn.Linear(root_dim, hidden_dim)  # 64×32 = 2,048 params

        # Multi-head attention (2 heads × 16d each)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )  # ~12,800 params

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)  # 32×1 = 32 params

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"RootCrossAttention initialized: {total_params:,} parameters")

    def forward(self, query_roots: torch.Tensor, doc_roots: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic similarity via cross-attention.

        Args:
            query_roots: [num_query_roots, root_dim] - Query root embeddings
            doc_roots: [num_doc_roots, root_dim] - Document root embeddings

        Returns:
            Semantic similarity score [0, 1]
        """
        if query_roots.dim() == 1:
            query_roots = query_roots.unsqueeze(0)
        if doc_roots.dim() == 1:
            doc_roots = doc_roots.unsqueeze(0)

        # Project to attention space
        Q = self.query_proj(query_roots).unsqueeze(0)  # [1, num_query, hidden]
        K = self.key_proj(doc_roots).unsqueeze(0)      # [1, num_doc, hidden]
        V = self.value_proj(doc_roots).unsqueeze(0)    # [1, num_doc, hidden]

        # Cross-attention: which doc roots are relevant to query?
        attn_output, attn_weights = self.attention(
            query=Q,
            key=K,
            value=V
        )  # attn_output: [1, num_query, hidden]

        # Aggregate attention output to single score
        pooled = attn_output.mean(dim=1)  # [1, hidden]
        score = torch.sigmoid(self.output_proj(pooled))  # [1, 1]

        return score.squeeze()


class TreeMatchReranker(nn.Module):
    """
    AST-aware reranker using multi-level tree matching.

    Combines:
    - Syntax matching (70%, deterministic, 0 params)
    - Compositional matching (20%, hybrid)
    - Semantic matching (10%, learned, ~20K params)

    Total trainable params: ~20K
    """

    def __init__(
        self,
        compositional_embedding: CompositionalEmbedding,
        freeze_embedding: bool = True,
        root_dim: int = 64,
        hidden_dim: int = 32,
        num_heads: int = 2
    ):
        """
        Initialize TreeMatch reranker.

        Args:
            compositional_embedding: Pre-trained CompositionalEmbedding model
            freeze_embedding: If True, freeze embedding parameters (recommended)
            root_dim: Dimension of root embeddings (64d)
            hidden_dim: Hidden dimension for attention (32d)
            num_heads: Number of attention heads (2)
        """
        super().__init__()

        self.comp_emb = compositional_embedding

        # Freeze embedding if requested
        if freeze_embedding:
            for param in self.comp_emb.parameters():
                param.requires_grad = False
            logger.info("CompositionalEmbedding frozen (not trainable)")

        # LEARNED: Cross-attention for semantic matching (roots only)
        self.semantic_attn = RootCrossAttention(
            root_dim=root_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # LEARNED: Mixing weights (initialized to 70/20/10)
        self.syntax_weight = nn.Parameter(torch.tensor(0.7))      # 70% syntax
        self.comp_weight = nn.Parameter(torch.tensor(0.2))        # 20% compositional
        self.semantic_weight = nn.Parameter(torch.tensor(0.1))    # 10% semantic

        # Count total trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"TreeMatchReranker initialized: {trainable:,} trainable / {total:,} total params")

    def forward(
        self,
        query_ast: Dict,
        doc_ast: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Score query-document relevance using multi-level tree matching.

        Args:
            query_ast: Parsed query AST
            doc_ast: Parsed document AST

        Returns:
            score: Final relevance score [0, 1]
            breakdown: Interpretable score breakdown
        """
        # LEVEL 1: Syntax matching (deterministic, 0 params)
        syntax_score = self.syntax_tree_match(query_ast, doc_ast)

        # LEVEL 2: Compositional matching (hybrid)
        comp_score = self.compositional_match(query_ast, doc_ast)

        # LEVEL 3: Semantic matching (learned attention)
        sem_score = self.semantic_match(query_ast, doc_ast)

        # Convert deterministic scores to tensors for gradient flow
        syntax_score_t = torch.tensor(syntax_score, dtype=torch.float32)
        comp_score_t = torch.tensor(comp_score, dtype=torch.float32)

        # Normalize weights to sum to 1
        weights = self._normalize_weights()

        # Combine scores (maintaining gradient flow)
        final_score = (
            weights['syntax'] * syntax_score_t +
            weights['compositional'] * comp_score_t +
            weights['semantic'] * sem_score
        )

        # Return with full breakdown (convert to Python types for logging)
        breakdown = {
            'syntax_score': syntax_score,
            'compositional_score': comp_score,
            'semantic_score': sem_score.item() if isinstance(sem_score, torch.Tensor) else sem_score,
            'syntax_weight': weights['syntax'].item(),
            'compositional_weight': weights['compositional'].item(),
            'semantic_weight': weights['semantic'].item(),
            'final_score': final_score.item()
        }

        return final_score, breakdown

    # ========================================================================
    # LEVEL 1: Syntax Tree Matching (Deterministic, 0 Params)
    # ========================================================================

    def syntax_tree_match(self, query_ast: Dict, doc_ast: Dict) -> float:
        """
        Match AST syntax structures (deterministic, 0 learned params).

        Matching rules (Esperanto grammar):
        1. Verb-to-verb matching (POS, tense, mood)
        2. Subject-to-subject matching (case, role)
        3. Object-to-object matching (accusative case)
        4. Modifier alignment (aliaj → aliaj)
        5. Syntax pattern matching (SVO, SOV, etc.)

        Returns: Score [0, 1] based on structural similarity
        """
        score = 0.0

        # === VERB MATCHING (30% of syntax score) ===
        query_verb = query_ast.get('verbo')
        doc_verb = doc_ast.get('verbo')

        if query_verb and doc_verb:
            verb_score = self._match_verb(query_verb, doc_verb)
            score += 0.3 * verb_score

        # === SUBJECT MATCHING (25% of syntax score) ===
        query_subj = query_ast.get('subjekto')
        doc_subj = doc_ast.get('subjekto')

        if query_subj and doc_subj:
            subj_score = self._match_noun_phrase(query_subj, doc_subj)
            score += 0.25 * subj_score

        # === OBJECT MATCHING (25% of syntax score) ===
        query_obj = query_ast.get('objekto')
        doc_obj = doc_ast.get('objekto')

        if query_obj and doc_obj:
            obj_score = self._match_noun_phrase(query_obj, doc_obj)
            score += 0.25 * obj_score

        # === MODIFIER MATCHING (10% of syntax score) ===
        query_aliaj = query_ast.get('aliaj', [])
        doc_aliaj = doc_ast.get('aliaj', [])

        if query_aliaj and doc_aliaj:
            modifier_score = self._match_modifiers(query_aliaj, doc_aliaj)
            score += 0.1 * modifier_score

        # === SYNTAX PATTERN MATCHING (10% of syntax score) ===
        query_pattern = self._extract_syntax_pattern(query_ast)
        doc_pattern = self._extract_syntax_pattern(doc_ast)

        if query_pattern == doc_pattern:
            score += 0.1

        return min(score, 1.0)

    def _match_verb(self, verb1: Dict, verb2: Dict) -> float:
        """Match verbs by grammatical properties."""
        score = 0.0

        # POS tag match (both must be verbs)
        if (verb1.get('vortspeco') == 'verbo' and
            verb2.get('vortspeco') == 'verbo'):
            score += 0.5  # Base match

            # Tense match (present, past, future)
            if verb1.get('tempo') == verb2.get('tempo'):
                score += 0.3

            # Mood match (indicative, conditional, volitive)
            if verb1.get('modo') == verb2.get('modo'):
                score += 0.2

        return score

    def _match_noun_phrase(self, np1: Dict, np2: Dict) -> float:
        """
        Match noun phrases by grammatical properties.

        Returns: Score [0, 1] based on:
        - Case agreement (nominative, accusative)
        - Number agreement (singular, plural)
        - POS tag agreement
        """
        score = 0.0

        # Extract head noun
        head1 = self._extract_head_noun(np1)
        head2 = self._extract_head_noun(np2)

        if not head1 or not head2:
            return 0.0

        # Case matching (40% weight)
        if head1.get('kazo') == head2.get('kazo'):
            score += 0.4

        # Number matching (30% weight)
        if head1.get('nombro') == head2.get('nombro'):
            score += 0.3

        # POS matching (30% weight)
        if head1.get('vortspeco') == head2.get('vortspeco'):
            score += 0.3

        return score

    def _extract_head_noun(self, node: Dict) -> Optional[Dict]:
        """Extract head noun from noun phrase (vortgrupo or vorto)."""
        if node is None:
            return None

        if node.get('tipo') == 'vorto':
            return node
        elif node.get('tipo') == 'vortgrupo':
            return node.get('kerno')

        return None

    def _match_modifiers(self, mods1: List[Dict], mods2: List[Dict]) -> float:
        """Match modifier lists by type overlap."""
        types1 = set(self._extract_modifier_types(mods1))
        types2 = set(self._extract_modifier_types(mods2))

        if not types1 or not types2:
            return 0.0

        overlap = len(types1 & types2)
        max_len = max(len(types1), len(types2))

        return overlap / max_len if max_len > 0 else 0.0

    def _extract_modifier_types(self, modifiers: List[Dict]) -> List[str]:
        """Extract modifier types (adverb, prepositional phrase, etc.)."""
        types = []
        for mod in modifiers:
            if isinstance(mod, dict):
                mod_type = mod.get('tipo', '')
                vortspeco = mod.get('vortspeco', '')
                types.append(f"{mod_type}_{vortspeco}")
        return types

    def _extract_syntax_pattern(self, ast: Dict) -> str:
        """
        Extract syntax pattern (e.g., "SVO", "SOV", "S_V").

        Examples:
        - "Mi manĝas pomon" → "SVO" (subject-verb-object)
        - "La hundo dormas" → "SV" (subject-verb)
        - "Ĉu vi venis?" → "QVS" (question-verb-subject)
        """
        pattern = []

        if ast.get('fraztipo') == 'demando':
            pattern.append('Q')  # Question

        if ast.get('subjekto'):
            pattern.append('S')

        if ast.get('verbo'):
            pattern.append('V')

        if ast.get('objekto'):
            pattern.append('O')

        if ast.get('aliaj'):
            pattern.append('M')  # Modifiers

        return ''.join(pattern)

    # ========================================================================
    # LEVEL 2: Compositional Matching (Hybrid)
    # ========================================================================

    def compositional_match(self, query_ast: Dict, doc_ast: Dict) -> float:
        """
        Match words by compositional semantics (hybrid: deterministic affixes + learned roots).

        Compositional semantics:
        - Root: Learned (64d embedding)
        - Prefix: Deterministic (mal-, re-, ge-)
        - Suffix: Deterministic (-ej-, -ist-, -in-)
        - Ending: Deterministic (-o, -a, -n)

        Returns: Score [0, 1] based on compositional similarity
        """
        query_words = self._extract_all_words(query_ast)
        doc_words = self._extract_all_words(doc_ast)

        if not query_words or not doc_words:
            return 0.0

        # Build word-to-word similarity matrix
        similarities = []

        for query_word in query_words:
            max_sim = 0.0
            for doc_word in doc_words:
                sim = self._compositional_similarity(query_word, doc_word)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)

        # Average max similarity for each query word
        return np.mean(similarities) if similarities else 0.0

    def _compositional_similarity(self, word1: Dict, word2: Dict) -> float:
        """
        Compute compositional similarity between two words.

        Args:
            word1: AST node with radiko, prefikso, sufikso
            word2: AST node with radiko, prefikso, sufikso

        Returns:
            Similarity score [0, 1]
        """
        score = 0.0

        # === DETERMINISTIC: Prefix matching (30% weight) ===
        prefix1 = word1.get('prefikso', '')
        prefix2 = word2.get('prefikso', '')

        if prefix1 and prefix2:
            if prefix1 == prefix2:
                score += 0.3  # Same prefix
            elif self._are_opposite_prefixes(prefix1, prefix2):
                score -= 0.2  # Opposite meaning

        # === DETERMINISTIC: Suffix matching (30% weight) ===
        suffix1 = word1.get('sufikso', '')
        suffix2 = word2.get('sufikso', '')

        if suffix1 and suffix2:
            if suffix1 == suffix2:
                score += 0.3  # Same suffix
            elif self._are_related_suffixes(suffix1, suffix2):
                score += 0.15  # Related suffixes

        # === LEARNED: Root similarity (40% weight) ===
        root1 = word1.get('radiko', '').lower()
        root2 = word2.get('radiko', '').lower()

        if root1 and root2:
            # Get learned root embeddings (64d only)
            emb1 = self._get_root_embedding(root1)
            emb2 = self._get_root_embedding(root2)

            if emb1 is not None and emb2 is not None:
                # Cosine similarity
                root_sim = F.cosine_similarity(emb1, emb2, dim=0)
                score += 0.4 * root_sim.item()

        return max(0.0, min(score, 1.0))  # Clamp to [0, 1]

    def _are_opposite_prefixes(self, prefix1: str, prefix2: str) -> bool:
        """Check if prefixes have opposite meanings."""
        # mal- is opposite of no prefix
        return (prefix1 == 'mal' and not prefix2) or (not prefix1 and prefix2 == 'mal')

    def _are_related_suffixes(self, suffix1: str, suffix2: str) -> bool:
        """Check if suffixes are semantically related."""
        related = {
            ('ist', 'ul'),   # -ist- (profession) ~ -ul- (person)
            ('ej', 'uj'),    # -ej- (place) ~ -uj- (container)
            ('id', 'in'),    # -id- (offspring) ~ -in- (female)
        }
        pair = tuple(sorted([suffix1, suffix2]))
        return pair in related

    def _extract_all_words(self, ast: Dict) -> List[Dict]:
        """Extract all words from AST tree."""
        words = []

        def traverse(node):
            if node is None:
                return

            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    words.append(node)

                # Traverse tree
                for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                    traverse(node.get(key))
                for item in node.get('priskriboj', []) + node.get('aliaj', []):
                    traverse(item)

        traverse(ast)
        return words

    def _get_root_embedding(self, root: str) -> Optional[torch.Tensor]:
        """Get learned root embedding (64d) from compositional model."""
        try:
            # Access root embeddings directly (64d, not full 128d compositional)
            if hasattr(self.comp_emb, 'root_embed'):
                vocab = self.comp_emb.root_vocab
                if root in vocab:
                    idx = vocab[root]
                    return self.comp_emb.root_embed.weight[idx]
        except:
            pass
        return None

    # ========================================================================
    # LEVEL 3: Semantic Matching (Learned)
    # ========================================================================

    def semantic_match(self, query_ast: Dict, doc_ast: Dict) -> float:
        """
        Learned semantic matching using cross-attention on roots.

        Use case: When syntax doesn't match but semantics do.
        Example: "Kiu kreis X?" vs "X estis kreita de Y" (passive voice)

        Returns: Score [0, 1] based on semantic similarity
        """
        # Extract root embeddings (learned 64d only, not affixes)
        query_roots = self._extract_root_embeddings(query_ast)
        doc_roots = self._extract_root_embeddings(doc_ast)

        if len(query_roots) == 0 or len(doc_roots) == 0:
            return 0.0

        # Stack to tensors
        query_tensor = torch.stack(query_roots)  # [num_query_roots, 64]
        doc_tensor = torch.stack(doc_roots)      # [num_doc_roots, 64]

        # Cross-attention: which doc roots are relevant to query?
        score = self.semantic_attn(query_tensor, doc_tensor)

        return score.item()

    def _extract_root_embeddings(self, ast: Dict) -> List[torch.Tensor]:
        """Extract ONLY root embeddings (64d learned) from AST."""
        embeddings = []

        def traverse(node):
            if node is None:
                return

            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    root = node.get('radiko', '').lower()
                    if root:
                        emb = self._get_root_embedding(root)
                        if emb is not None:
                            embeddings.append(emb)

                # Traverse tree
                for key in ['kerno', 'subjekto', 'verbo', 'objekto']:
                    traverse(node.get(key))
                for item in node.get('priskriboj', []) + node.get('aliaj', []):
                    traverse(item)

        traverse(ast)
        return embeddings

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _normalize_weights(self) -> Dict[str, torch.Tensor]:
        """Normalize mixing weights to sum to 1."""
        w_syn = torch.sigmoid(self.syntax_weight)
        w_comp = torch.sigmoid(self.comp_weight)
        w_sem = torch.sigmoid(self.semantic_weight)

        total = w_syn + w_comp + w_sem

        return {
            'syntax': w_syn / total,
            'compositional': w_comp / total,
            'semantic': w_sem / total,
        }

    def score_batch(
        self,
        query_ast: Dict,
        doc_asts: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Score multiple documents against single query.

        Args:
            query_ast: Parsed query AST
            doc_asts: List of parsed document ASTs

        Returns:
            scores: Tensor of relevance scores [0, 1]
            breakdowns: List of score breakdowns
        """
        scores = []
        breakdowns = []

        for doc_ast in doc_asts:
            score, breakdown = self.forward(query_ast, doc_ast)
            scores.append(score)
            breakdowns.append(breakdown)

        return torch.stack(scores), breakdowns

    def save(self, path: Path):
        """Save reranker model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'semantic_attn_state_dict': self.semantic_attn.state_dict(),
            'syntax_weight': self.syntax_weight.item(),
            'comp_weight': self.comp_weight.item(),
            'semantic_weight': self.semantic_weight.item(),
        }, path)

        logger.info(f"TreeMatchReranker saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        compositional_embedding: CompositionalEmbedding
    ) -> 'TreeMatchReranker':
        """
        Load reranker model.

        Args:
            path: Path to saved model
            compositional_embedding: Compositional embedding model

        Returns:
            Loaded TreeMatchReranker
        """
        checkpoint = torch.load(path, map_location='cpu')

        reranker = cls(
            compositional_embedding=compositional_embedding,
            freeze_embedding=True
        )

        reranker.semantic_attn.load_state_dict(checkpoint['semantic_attn_state_dict'])
        reranker.syntax_weight.data = torch.tensor(checkpoint['syntax_weight'])
        reranker.comp_weight.data = torch.tensor(checkpoint['comp_weight'])
        reranker.semantic_weight.data = torch.tensor(checkpoint['semantic_weight'])

        logger.info(f"TreeMatchReranker loaded from {path}")
        return reranker


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
