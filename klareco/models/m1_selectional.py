"""
M1 Selectional Preference Model

Learns compatibility between roots in grammatical roles:
- Can X be the subject of verb Y?
- Can verb Y take Z as object?
- Is triple (subject, verb, object) plausible?

Architecture:
- Input: 3 x 64D embeddings (subject, verb, object from Stage 1)
- Hidden: 128D with ReLU activation
- Output: 3 scores [0-1] (subject-verb, verb-object, triple plausibility)
- Total: ~10M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class M1SelectionalPreference(nn.Module):
    """
    M1 Selectional Preference Model.

    Scores compatibility of subject-verb-object triples using Stage 1 embeddings.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Initialize M1 model.

        Args:
            embedding_dim: Dimension of input embeddings (from Stage 1)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Separate encoders for each role
        self.subject_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.verb_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.object_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Compatibility scorers
        # Subject-verb compatibility
        self.subj_verb_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Verb-object compatibility
        self.verb_obj_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Triple plausibility (uses all three)
        self.triple_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, subject_emb: torch.Tensor, verb_emb: torch.Tensor,
                object_emb: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            subject_emb: Subject embeddings [batch, embedding_dim]
            verb_emb: Verb embeddings [batch, embedding_dim]
            object_emb: Object embeddings [batch, embedding_dim]

        Returns:
            dict with keys:
                - subj_verb_score: Subject-verb compatibility [batch, 1]
                - verb_obj_score: Verb-object compatibility [batch, 1]
                - triple_score: Overall triple plausibility [batch, 1]
        """
        # Encode each role
        subj_hidden = self.subject_encoder(subject_emb)  # [batch, hidden]
        verb_hidden = self.verb_encoder(verb_emb)        # [batch, hidden]
        obj_hidden = self.object_encoder(object_emb)     # [batch, hidden]

        # Compute pairwise scores
        subj_verb_cat = torch.cat([subj_hidden, verb_hidden], dim=1)  # [batch, 2*hidden]
        verb_obj_cat = torch.cat([verb_hidden, obj_hidden], dim=1)    # [batch, 2*hidden]

        subj_verb_score = self.subj_verb_scorer(subj_verb_cat)  # [batch, 1]
        verb_obj_score = self.verb_obj_scorer(verb_obj_cat)     # [batch, 1]

        # Compute triple score
        triple_cat = torch.cat([subj_hidden, verb_hidden, obj_hidden], dim=1)  # [batch, 3*hidden]
        triple_score = self.triple_scorer(triple_cat)  # [batch, 1]

        return {
            'subj_verb_score': subj_verb_score,
            'verb_obj_score': verb_obj_score,
            'triple_score': triple_score
        }

    def predict(self, subject_emb: torch.Tensor, verb_emb: torch.Tensor,
                object_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict plausibility for inference.

        Args:
            subject_emb: Subject embeddings
            verb_emb: Verb embeddings
            object_emb: Object embeddings

        Returns:
            Triple plausibility scores [batch, 1]
        """
        with torch.no_grad():
            outputs = self.forward(subject_emb, verb_emb, object_emb)
            return outputs['triple_score']

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class M1Loss(nn.Module):
    """
    Combined loss for M1 training.

    Combines three objectives:
    1. Subject-verb compatibility
    2. Verb-object compatibility
    3. Triple plausibility

    All losses are binary cross-entropy with target labels.
    """

    def __init__(self, subj_verb_weight: float = 0.3, verb_obj_weight: float = 0.3,
                 triple_weight: float = 0.4):
        """
        Initialize loss.

        Args:
            subj_verb_weight: Weight for subject-verb loss
            verb_obj_weight: Weight for verb-object loss
            triple_weight: Weight for triple plausibility loss
        """
        super().__init__()
        self.subj_verb_weight = subj_verb_weight
        self.verb_obj_weight = verb_obj_weight
        self.triple_weight = triple_weight
        self.bce = nn.BCELoss()

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs (dict with subj_verb_score, verb_obj_score, triple_score)
            labels: Target labels [batch, 1] (0 or 1)

        Returns:
            dict with:
                - loss: Total weighted loss
                - subj_verb_loss: Subject-verb BCE loss
                - verb_obj_loss: Verb-object BCE loss
                - triple_loss: Triple plausibility BCE loss
        """
        # All three scores should match the label (plausible=1, implausible=0)
        subj_verb_loss = self.bce(outputs['subj_verb_score'], labels)
        verb_obj_loss = self.bce(outputs['verb_obj_score'], labels)
        triple_loss = self.bce(outputs['triple_score'], labels)

        # Weighted combination
        total_loss = (self.subj_verb_weight * subj_verb_loss +
                      self.verb_obj_weight * verb_obj_loss +
                      self.triple_weight * triple_loss)

        return {
            'loss': total_loss,
            'subj_verb_loss': subj_verb_loss,
            'verb_obj_loss': verb_obj_loss,
            'triple_loss': triple_loss
        }
