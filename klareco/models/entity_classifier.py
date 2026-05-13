"""
Entity Type Classifier using GNN on AST Graphs.

Architecture:
- Input: AST node → PyG graph + deterministic priors
- GNN Encoder: 2-layer GCN/GAT processes graph structure
- Prior Fusion: Concatenate deterministic features with GNN output
- Classifier Heads: Separate heads for each Tier 3 family
  - PersonType (7 classes)
  - LocationType (7 classes)
  - TimeType (5 classes)
  - ThingType (7 classes)

Philosophy: Maximize use of deterministic priors (71% coverage),
learn only the semantic gap (29%) where deterministic features are uncertain.

Model size: ~1-2M params (minimal, focused on semantic gap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch-geometric not installed. Install with: pip install torch-geometric")

from klareco.semantic_enrichment.taxonomy import (
    TopLevelCategory, EntityType, PersonType, LocationType, TimeType, ThingType
)

logger = logging.getLogger(__name__)


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for AST graphs.

    Processes syntactic structure of word in context to capture
    compositional semantics beyond deterministic features.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize GNN encoder.

        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers (default 2)
            dropout: Dropout rate
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric required for GNN. Install: pip install torch-geometric")

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode AST graph to embedding.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Graph edges [2, num_edges]
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activation(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)

        # Global pooling (aggregate all nodes in graph)
        if batch is None:
            # Single graph: mean pool all nodes
            graph_embedding = x.mean(dim=0, keepdim=True)
        else:
            # Batched graphs: mean pool per graph
            graph_embedding = global_mean_pool(x, batch)

        # Project to output dimension
        graph_embedding = self.output_proj(graph_embedding)

        return graph_embedding


class EntityTypeClassifier(nn.Module):
    """
    Entity type classifier using GNN + deterministic priors.

    Combines:
    1. GNN encoding of AST structure (~1M params)
    2. Deterministic prior features (0 params)
    3. Separate classifier heads for each Tier 3 family (~100K params)

    Total: ~1.2M params (minimal, focused on 29% semantic gap)
    """

    def __init__(
        self,
        node_feature_dim: int,
        deterministic_feature_dim: int = 6,  # tier1, tier2, tier3, confidence + 2 evidence
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 64,
        gnn_layers: int = 2,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize entity type classifier.

        Args:
            node_feature_dim: Dimension of AST node features
            deterministic_feature_dim: Dimension of deterministic prior features
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_layers: Number of GNN layers
            classifier_hidden_dim: Classifier hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.deterministic_feature_dim = deterministic_feature_dim

        # GNN encoder for AST structure
        self.gnn_encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )

        # Combined feature dimension (GNN + deterministic priors)
        combined_dim = gnn_output_dim + deterministic_feature_dim

        # Shared feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Separate classifier heads for each Tier 3 family
        self.person_head = nn.Linear(classifier_hidden_dim, len(PersonType))
        self.location_head = nn.Linear(classifier_hidden_dim, len(LocationType))
        self.time_head = nn.Linear(classifier_hidden_dim, len(TimeType))
        self.thing_head = nn.Linear(classifier_hidden_dim, len(ThingType))

        # Tier 2 type predictor (used to select correct head)
        self.tier2_classifier = nn.Linear(classifier_hidden_dim, len(EntityType))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        deterministic_priors: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        tier2_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Graph edges [2, num_edges]
            deterministic_priors: Deterministic features [batch_size, det_feature_dim]
            batch: Batch assignment for each node
            tier2_target: Ground truth tier2 types (for training) [batch_size]

        Returns:
            Dictionary with:
            - tier2_logits: Tier 2 type predictions [batch_size, 18]
            - person_logits: Person type predictions [batch_size, 7]
            - location_logits: Location type predictions [batch_size, 7]
            - time_logits: Time type predictions [batch_size, 5]
            - thing_logits: Thing type predictions [batch_size, 7]
        """
        # Encode AST structure with GNN
        graph_embedding = self.gnn_encoder(x, edge_index, batch)  # [batch_size, gnn_output_dim]

        # Concatenate with deterministic priors
        combined_features = torch.cat([graph_embedding, deterministic_priors], dim=1)

        # Fuse features
        fused = self.fusion(combined_features)  # [batch_size, classifier_hidden_dim]

        # Tier 2 classification
        tier2_logits = self.tier2_classifier(fused)

        # Tier 3 classification (separate heads)
        person_logits = self.person_head(fused)
        location_logits = self.location_head(fused)
        time_logits = self.time_head(fused)
        thing_logits = self.thing_head(fused)

        return {
            'tier2_logits': tier2_logits,
            'person_logits': person_logits,
            'location_logits': location_logits,
            'time_logits': time_logits,
            'thing_logits': thing_logits,
        }

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        deterministic_priors: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode: predict entity types.

        Args:
            x: Node features
            edge_index: Graph edges
            deterministic_priors: Deterministic features
            batch: Batch assignment

        Returns:
            Dictionary with:
            - tier2_type: Predicted tier2 types [batch_size]
            - tier3_type: Predicted tier3 types [batch_size]
            - tier2_confidence: Tier2 confidence scores [batch_size]
            - tier3_confidence: Tier3 confidence scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, edge_index, deterministic_priors, batch)

            # Tier 2 prediction
            tier2_probs = F.softmax(outputs['tier2_logits'], dim=1)
            tier2_confidence, tier2_pred = tier2_probs.max(dim=1)

            # Tier 3 prediction: select head based on tier2 prediction
            batch_size = tier2_pred.size(0)
            tier3_type = torch.zeros(batch_size, dtype=torch.long)
            tier3_confidence = torch.zeros(batch_size)

            for i in range(batch_size):
                tier2 = tier2_pred[i].item()

                # Map tier2 index to EntityType
                tier2_enum = list(EntityType)[tier2]

                # Select appropriate head
                if tier2_enum == EntityType.PERSON:
                    probs = F.softmax(outputs['person_logits'][i], dim=0)
                    conf, pred = probs.max(dim=0)
                    tier3_type[i] = pred
                    tier3_confidence[i] = conf
                elif tier2_enum in [EntityType.LOCATION, EntityType.FACILITY, EntityType.GPE]:
                    probs = F.softmax(outputs['location_logits'][i], dim=0)
                    conf, pred = probs.max(dim=0)
                    tier3_type[i] = pred
                    tier3_confidence[i] = conf
                elif tier2_enum in [EntityType.TIME_POINT, EntityType.TIME_DURATION]:
                    probs = F.softmax(outputs['time_logits'][i], dim=0)
                    conf, pred = probs.max(dim=0)
                    tier3_type[i] = pred
                    tier3_confidence[i] = conf
                elif tier2_enum in [EntityType.THING, EntityType.CONCEPT, EntityType.EVENT]:
                    probs = F.softmax(outputs['thing_logits'][i], dim=0)
                    conf, pred = probs.max(dim=0)
                    tier3_type[i] = pred
                    tier3_confidence[i] = conf
                else:
                    # Default to thing for other types
                    probs = F.softmax(outputs['thing_logits'][i], dim=0)
                    conf, pred = probs.max(dim=0)
                    tier3_type[i] = pred
                    tier3_confidence[i] = conf

            return {
                'tier2_type': tier2_pred,
                'tier3_type': tier3_type,
                'tier2_confidence': tier2_confidence,
                'tier3_confidence': tier3_confidence,
            }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_deterministic_feature_vector(deterministic_priors: Dict) -> torch.Tensor:
    """
    Convert deterministic priors dict to feature tensor.

    Args:
        deterministic_priors: Dict with keys:
            - tier1_category: str or None
            - tier2_type: str or None
            - tier3_type: str or None
            - confidence: float
            - evidence: dict with affix, ending, etc.

    Returns:
        Feature tensor [6]
        - [0-2]: One-hot tier1 (3 dims: entity/attribute/quantity, or 0 if None)
        - [3]: Tier2 index (0 if None)
        - [4]: Confidence
        - [5]: Has affix evidence (0 or 1)
    """
    features = torch.zeros(6)

    # Tier 1 one-hot (simplified to 3 main categories)
    tier1 = deterministic_priors.get('tier1_category')
    if tier1:
        tier1_mapping = {'entity': 0, 'attribute': 1, 'quantity': 2}
        if tier1 in tier1_mapping:
            features[tier1_mapping[tier1]] = 1.0

    # Tier 2 index
    tier2 = deterministic_priors.get('tier2_type')
    if tier2:
        try:
            tier2_enum = EntityType(tier2)
            features[3] = float(list(EntityType).index(tier2_enum))
        except (ValueError, KeyError):
            features[3] = 0.0

    # Confidence
    features[4] = deterministic_priors.get('confidence', 0.0)

    # Has affix evidence
    evidence = deterministic_priors.get('evidence', {})
    has_affix = 1.0 if evidence.get('affix') else 0.0
    features[5] = has_affix

    return features


# =============================================================================
# Model Loading/Saving
# =============================================================================

def save_checkpoint(
    model: EntityTypeClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_path: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'model_config': {
            'node_feature_dim': model.node_feature_dim,
            'deterministic_feature_dim': model.deterministic_feature_dim,
        }
    }

    # Atomic save
    temp_path = checkpoint_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    device: str = 'cpu'
) -> Tuple[EntityTypeClassifier, Dict]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        (model, metrics) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['model_config']
    model = EntityTypeClassifier(
        node_feature_dim=config['node_feature_dim'],
        deterministic_feature_dim=config['deterministic_feature_dim']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    metrics = checkpoint.get('metrics', {})
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Metrics: {metrics}")

    return model, metrics


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GNNEncoder',
    'EntityTypeClassifier',
    'create_deterministic_feature_vector',
    'save_checkpoint',
    'load_checkpoint',
]
