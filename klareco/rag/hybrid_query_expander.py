"""
Hybrid Query Expander - Combines Deterministic Synonyms + Learned Associations

Two-track query expansion for RAG retrieval:
1. Track 1: ReVo synonyms (deterministic, high precision)
2. Track 2: Embedding associations (learned, high recall)

Aligns with Klareco thesis: maximize deterministic processing, minimal learned parameters.
"""

import torch
import torch.nn.functional as F
import kuzu
from pathlib import Path
from typing import Set, Dict


class HybridQueryExpander:
    """
    Hybrid query expander using both deterministic and learned methods.
    
    Args:
        embedding_path: Path to root embeddings checkpoint
        db_path: Path to Kuzu database with ReVo relations
        embedding_k: Number of embedding neighbors to retrieve
        embedding_threshold: Minimum similarity for embedding expansion
        use_revo: Enable ReVo synonym expansion
        use_embeddings: Enable embedding association expansion
    """
    
    def __init__(
        self,
        embedding_path: Path,
        db_path: Path,
        embedding_k: int = 5,
        embedding_threshold: float = 0.4,
        use_revo: bool = True,
        use_embeddings: bool = True
    ):
        self.use_revo = use_revo
        self.use_embeddings = use_embeddings
        
        # Load embeddings
        if use_embeddings:
            print(f"Loading embeddings from {embedding_path}...")
            checkpoint = torch.load(embedding_path, map_location='cpu', weights_only=False)
            
            self.embeddings = F.normalize(checkpoint['embeddings'], p=2, dim=1)
            self.vocab = checkpoint['vocab']
            self.root_to_idx = checkpoint['root_to_idx']
            self.embedding_k = embedding_k
            self.embedding_threshold = embedding_threshold
        
        # Connect to Kuzu for ReVo relations
        if use_revo:
            print(f"Connecting to Kuzu database: {db_path}...")
            db = kuzu.Database(str(db_path))
            self.conn = kuzu.Connection(db)
    
    def get_revo_synonyms(self, root: str) -> Set[str]:
        """Get deterministic synonyms from ReVo."""
        if not self.use_revo:
            return set()
        
        synonyms = set()
        
        try:
            # Query all ReVo semantic relations
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[rel:REVO_SINONIMO|REVO_HIPERNIMO]->(s:Radiko)
                RETURN s.radiko
            """)
            
            while result.has_next():
                synonyms.add(result.get_next()[0])
        except Exception:
            pass
        
        return synonyms
    
    def get_embedding_associations(self, root: str) -> Set[str]:
        """Get learned associations from embeddings."""
        if not self.use_embeddings:
            return set()
        
        if root not in self.root_to_idx:
            return set()
        
        target_idx = self.root_to_idx[root]
        target_emb = self.embeddings[target_idx]
        
        # Compute similarities
        similarities = self.embeddings @ target_emb
        
        # Get top k (excluding self)
        top_k_indices = similarities.argsort(descending=True)[1:self.embedding_k+1]
        
        associations = set()
        for idx in top_k_indices:
            sim = similarities[idx].item()
            if sim > self.embedding_threshold:
                associations.add(self.vocab[idx])
        
        return associations
    
    def expand(self, roots: Set[str]) -> Dict[str, Set[str]]:
        """
        Expand query roots using hybrid approach.
        
        Args:
            roots: Set of original query roots
            
        Returns:
            Dictionary with:
                - 'original': Original roots
                - 'revo_synonyms': ReVo deterministic synonyms
                - 'embedding_associations': Learned associations
                - 'all': Union of all expanded roots
        """
        expansion = {
            'original': set(roots),
            'revo_synonyms': set(),
            'embedding_associations': set()
        }
        
        for root in roots:
            # Track 1: Deterministic synonyms
            if self.use_revo:
                revo_syns = self.get_revo_synonyms(root)
                expansion['revo_synonyms'].update(revo_syns)
            
            # Track 2: Learned associations
            if self.use_embeddings:
                emb_assoc = self.get_embedding_associations(root)
                expansion['embedding_associations'].update(emb_assoc)
        
        # Compute union
        expansion['all'] = (
            expansion['original'] | 
            expansion['revo_synonyms'] | 
            expansion['embedding_associations']
        )
        
        return expansion
