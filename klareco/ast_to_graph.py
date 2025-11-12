"""
AST to Graph Conversion for GNN Encoder.

Converts Klareco's morpheme-based ASTs into graph representations
suitable for PyTorch Geometric (PyG) processing.

Graph Structure:
- Nodes: Individual AST elements (words, morphemes)
- Edges: Syntactic and morphological relationships
- Node features: Embedded morpheme information
- Edge types: has_subject, has_verb, has_object, modifies, etc.
"""

import torch
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch-geometric not installed. Install with: pip install torch-geometric")


# Edge type enumeration
EDGE_TYPES = {
    'has_subject': 0,
    'has_verb': 1,
    'has_object': 2,
    'has_modifier': 3,
    'modifies': 4,
    'has_article': 5,
    'has_root': 6,
    'has_prefix': 7,
    'has_suffix': 8,
    'has_description': 9,
}

# POS tag enumeration
POS_TAGS = {
    'substantivo': 0,
    'adjektivo': 1,
    'verbo': 2,
    'adverbo': 3,
    'pronomo': 4,
    'prepozicio': 5,
    'konjunkcio': 6,
    'interjekcio': 7,
    'artikolo': 8,
    'nomo': 9,  # proper name
    'unknown': 10,
}

# Number enumeration
NUMBER_TAGS = {
    'singularo': 0,
    'pluralo': 1,
}

# Case enumeration
CASE_TAGS = {
    'nominativo': 0,
    'akuzativo': 1,
}


class ASTToGraphConverter:
    """Converts Klareco ASTs to PyTorch Geometric Data objects."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize converter.

        Args:
            embed_dim: Dimension for morpheme embeddings
        """
        self.embed_dim = embed_dim
        self.morpheme_vocab = {}  # Will be populated from corpus
        self.next_morpheme_id = 0

    def get_morpheme_id(self, morpheme: str) -> int:
        """
        Get or create ID for a morpheme.

        Args:
            morpheme: Morpheme string (root, prefix, or suffix)

        Returns:
            Integer ID for the morpheme
        """
        if morpheme not in self.morpheme_vocab:
            self.morpheme_vocab[morpheme] = self.next_morpheme_id
            self.next_morpheme_id += 1
        return self.morpheme_vocab[morpheme]

    def extract_node_features(self, node: Dict) -> torch.Tensor:
        """
        Extract feature vector from AST node.

        Args:
            node: AST node dictionary

        Returns:
            Feature tensor (270d by default)
        """
        features = []

        # Root embedding ID (128d placeholder - will be replaced by actual embedding)
        root = node.get('radiko', '')
        root_id = self.get_morpheme_id(root) if root else 0
        features.append(float(root_id))

        # POS tag (one-hot, 11d)
        pos = node.get('vortspeco', 'unknown')
        pos_onehot = [0.0] * len(POS_TAGS)
        pos_onehot[POS_TAGS.get(pos, POS_TAGS['unknown'])] = 1.0
        features.extend(pos_onehot)

        # Number (one-hot, 2d)
        number = node.get('nombro', 'singularo')
        number_onehot = [0.0] * len(NUMBER_TAGS)
        number_onehot[NUMBER_TAGS.get(number, 0)] = 1.0
        features.extend(number_onehot)

        # Case (one-hot, 2d)
        case = node.get('kazo', 'nominativo')
        case_onehot = [0.0] * len(CASE_TAGS)
        case_onehot[CASE_TAGS.get(case, 0)] = 1.0
        features.extend(case_onehot)

        # Prefix ID (64d placeholder)
        prefix = node.get('prefikso', '')
        prefix_id = self.get_morpheme_id(prefix) if prefix else 0
        features.append(float(prefix_id))

        # Suffixes (mean of suffix IDs, 64d placeholder)
        suffixes = node.get('sufiksoj', [])
        if suffixes:
            suffix_ids = [self.get_morpheme_id(s) for s in suffixes]
            suffix_mean = np.mean(suffix_ids)
        else:
            suffix_mean = 0.0
        features.append(suffix_mean)

        # Parse status (1d: 0=success, 1=failed)
        parse_status = 0.0 if node.get('parse_status') == 'success' else 1.0
        features.append(parse_status)

        return torch.tensor(features, dtype=torch.float)

    def ast_to_graph(self, ast: Dict) -> Optional[Data]:
        """
        Convert AST to PyTorch Geometric Data object.

        Args:
            ast: Klareco AST dictionary

        Returns:
            PyG Data object with nodes, edges, and features
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric is required. Install with: pip install torch-geometric")

        if ast.get('tipo') != 'frazo':
            raise ValueError(f"Expected AST tipo='frazo', got {ast.get('tipo')}")

        nodes = []
        node_features = []
        edges = []
        edge_types = []
        node_id = 0

        # Root node: sentence itself
        sentence_node_id = node_id
        node_id += 1
        nodes.append({'type': 'sentence', 'id': sentence_node_id})
        # Sentence node features (placeholder - aggregate of children)
        node_features.append(torch.zeros(19))  # 19 features (excluding morpheme IDs)

        # Process subject
        if ast.get('subjekto'):
            subject_node_id = node_id
            node_id += 1

            # Add subject node
            if ast['subjekto'].get('tipo') == 'vortgrupo':
                # Word group (noun phrase)
                subject = ast['subjekto']
                kerno = subject.get('kerno')
                if kerno:
                    nodes.append({'type': 'subject', 'id': subject_node_id, 'data': kerno})
                    node_features.append(self.extract_node_features(kerno))

                    # Edge: sentence → subject
                    edges.append([sentence_node_id, subject_node_id])
                    edge_types.append(EDGE_TYPES['has_subject'])

                    # Process descriptions (adjectives)
                    for adj in subject.get('priskriboj', []):
                        adj_node_id = node_id
                        node_id += 1
                        nodes.append({'type': 'adjective', 'id': adj_node_id, 'data': adj})
                        node_features.append(self.extract_node_features(adj))

                        # Edge: adjective → subject (modifies)
                        edges.append([adj_node_id, subject_node_id])
                        edge_types.append(EDGE_TYPES['modifies'])

                    # Article
                    if 'artikolo' in subject:
                        article_node_id = node_id
                        node_id += 1
                        article_features = torch.zeros(19)
                        article_features[POS_TAGS['artikolo'] + 1] = 1.0  # +1 because first feature is root_id
                        nodes.append({'type': 'article', 'id': article_node_id})
                        node_features.append(article_features)

                        # Edge: subject → article
                        edges.append([subject_node_id, article_node_id])
                        edge_types.append(EDGE_TYPES['has_article'])

        # Process verb
        if ast.get('verbo'):
            verb = ast['verbo']
            verb_node_id = node_id
            node_id += 1

            nodes.append({'type': 'verb', 'id': verb_node_id, 'data': verb})
            node_features.append(self.extract_node_features(verb))

            # Edge: sentence → verb
            edges.append([sentence_node_id, verb_node_id])
            edge_types.append(EDGE_TYPES['has_verb'])

        # Process object
        if ast.get('objekto'):
            object_node_id = node_id
            node_id += 1

            if ast['objekto'].get('tipo') == 'vortgrupo':
                # Word group (noun phrase)
                obj = ast['objekto']
                kerno = obj.get('kerno')
                if kerno:
                    nodes.append({'type': 'object', 'id': object_node_id, 'data': kerno})
                    node_features.append(self.extract_node_features(kerno))

                    # Edge: sentence → object
                    edges.append([sentence_node_id, object_node_id])
                    edge_types.append(EDGE_TYPES['has_object'])

                    # Process descriptions (adjectives)
                    for adj in obj.get('priskriboj', []):
                        adj_node_id = node_id
                        node_id += 1
                        nodes.append({'type': 'adjective', 'id': adj_node_id, 'data': adj})
                        node_features.append(self.extract_node_features(adj))

                        # Edge: adjective → object (modifies)
                        edges.append([adj_node_id, object_node_id])
                        edge_types.append(EDGE_TYPES['modifies'])

                    # Article
                    if 'artikolo' in obj:
                        article_node_id = node_id
                        node_id += 1
                        article_features = torch.zeros(19)
                        article_features[POS_TAGS['artikolo'] + 1] = 1.0
                        nodes.append({'type': 'article', 'id': article_node_id})
                        node_features.append(article_features)

                        # Edge: object → article
                        edges.append([object_node_id, article_node_id])
                        edge_types.append(EDGE_TYPES['has_article'])

        # Process other words (aliaj)
        for alia in ast.get('aliaj', []):
            alia_node_id = node_id
            node_id += 1

            nodes.append({'type': 'other', 'id': alia_node_id, 'data': alia})
            node_features.append(self.extract_node_features(alia))

            # Edge: sentence → other
            edges.append([sentence_node_id, alia_node_id])
            edge_types.append(EDGE_TYPES['has_modifier'])

        # Convert to PyG Data object
        if not edges:
            # Degenerate case: single-node graph
            x = torch.stack(node_features)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        else:
            x = torch.stack(node_features)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes)
        )

        return data

    def batch_ast_to_graph(self, asts: List[Dict]) -> List[Data]:
        """
        Convert multiple ASTs to graphs.

        Args:
            asts: List of AST dictionaries

        Returns:
            List of PyG Data objects
        """
        graphs = []
        for ast in asts:
            try:
                graph = self.ast_to_graph(ast)
                if graph is not None:
                    graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to convert AST to graph: {e}")
                continue
        return graphs


def main():
    """Test AST to graph conversion."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from klareco.parser import parse

    # Test sentence
    sentence = "La hundo vidas la grandan katon."
    print(f"Parsing: {sentence}")

    ast = parse(sentence)
    print(f"AST: {ast['tipo']}")

    # Convert to graph
    converter = ASTToGraphConverter()
    graph = converter.ast_to_graph(ast)

    print(f"\nGraph:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape}")
    print(f"  Edge types: {graph.edge_attr.shape}")

    print("\nNode features (first node):")
    print(graph.x[0])

    print("\nEdge index:")
    print(graph.edge_index)

    print("\nEdge types:")
    print(graph.edge_attr)

    print("\n✅ AST to graph conversion successful!")


if __name__ == '__main__':
    main()
