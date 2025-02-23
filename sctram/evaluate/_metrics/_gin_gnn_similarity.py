#!/usr/bin/env python3

import sys
working_directory = "/Users/kemalinecik/git_nosync/sctram"
sys.path.append(working_directory)

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sctram.evaluate._metrics.validators import validate_between_minus_plus_1


def gin_gnn_similarity(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, threshold:float, validate_result: bool, remove_self_loops: bool = True, seed: int = 0):
    """Computes deterministic GNN-based similarity with GIN architecture.
    
    It is between two adjacency matrices using enhanced structural features and a robust graph embedding methodology. 
    Implements a theoretically grounded approach combining:
    1. Multi-faceted node features capturing various graph topological properties
    2. Fixed-weight GIN architecture with proven WL-test equivalence
    3. Multi-scale structural aggregation through deep message passing
    4. Comprehensive graph-level embedding strategies
    
    Mathematical Justification:
        - GIN architecture (Xu et al., ICLR 2019) provides injective multiset aggregation, theoretically
        capable of distinguishing any graphs detectable by the WL test
        - Feature engineering captures complementary aspects of node importance (PageRank), local clustering 
        (triangles), and core network structure (k-core)
        - Multiple aggregation strategies (sum/mean/max pooling) create permutation-invariant graph embeddings
        that preserve different statistical moments of node features
        - Cosine similarity in final embedding space provides normalized similarity measure invariant to 
        absolute magnitude differences
      
    Statistical Robustness:
        - Deterministic through fixed initialization and feature normalization
        - Combines 7 complementary node features to reduce feature bias
        - 4-layer architecture captures multi-scale structural patterns
        - Multiple pooling strategies increase sensitivity to both local and global structure
        - Feature standardization mitigates feature scale variance
    
    Complexity Considerations:
        - O(n^3) features (eigenvector centrality) acceptable for n ≤ 60
        - Fixed neural architecture ensures constant inference time
        - GPU acceleration support through PyTorch
    
    Parameters:
        given_adjacency_matrix (np.ndarray): The reference square adjacency matrix.
        inferred_adjacency_matrix (np.ndarray): The inferred square adjacency matrix.
        remove_self_loops (bool, optional): If True, diagonal entries (self-loops) are set to 0. Default is True.
        threshold (float): Binarization threshold to convert weighted edges into binary edges. 
        validate_result (bool): If True, validates that the computed distance is zero or positive. 
        seed (int): Random seed for reproducibility
        
    Advantages:
        - Leverages a fixed GIN architecture with proven Weisfeiler-Lehman test equivalence to capture graph structure.
        - Utilizes multiple node features and aggregation methods (mean, max, add, and an interaction term) to summarize graph embeddings.
        - Provides a deterministic and reproducible similarity measure between two graphs.

    Limitations:
        - Sensitive to the choice of the binarization threshold; an inappropriate threshold can affect the similarity score.
        - All node features are treated equally, without weighting for potential differences in node or edge significance.
        - Computational complexity may increase with graph size due to the use of networkx for feature extraction and PyG for processing.
        
    Returns:
        float: Cosine similarity between graph embeddings (ranges -1 to 1)
    """
    # Binarize matrices
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    given_binary = (given_adjacency_matrix >= threshold).astype(int)

    if remove_self_loops:
        np.fill_diagonal(given_binary, 0)
        np.fill_diagonal(inferred_binary, 0)

    # Set deterministic environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_graph_data(adj_matrix):
        """Create PyG Data object with comprehensive structural features"""
        G = nx.from_numpy_array(adj_matrix)
        
        # Compute node features
        deg = list(dict(G.degree).values())
        clust = list(nx.clustering(G).values())
        
        try:  # Eigenvector centrality
            eigen = list(nx.eigenvector_centrality(G, max_iter=1000).values())
        except nx.PowerIterationFailedConvergence:
            eigen = [0.0]*len(G)
            
        pagerank = list(nx.pagerank(G).values())
        core = list(nx.core_number(G).values())
        triangles = list(nx.triangles(G).values())
        avg_degree = [np.mean([G.degree[n] for n in G.neighbors(i)]) if G.degree[i] > 0 else 0 for i in G.nodes]
        
        # Assemble features with error-resistant normalization
        features = np.array([deg, clust, eigen, pagerank, core, triangles, avg_degree]).T
        means = np.nanmean(features, axis=0, keepdims=True)
        stds = np.nanstd(features, axis=0, keepdims=True) + 1e-6
        features = (features - means) / stds
        
        # Handle edge_index creation for empty graphs
        edges = list(G.edges)
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index
        )

    class StructuralGIN(torch.nn.Module):
        """Fixed GIN architecture with enhanced capacity"""
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            
            def create_mlp(in_dim, out_dim):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, out_dim),
                    torch.nn.ReLU()  # Added non-linearity between layers
                )
            
            self.convs = torch.nn.ModuleList([
                GINConv(create_mlp(7, 32)),  # 7 input features
                GINConv(create_mlp(32, 64)),
                GINConv(create_mlp(64, 128)),
                GINConv(create_mlp(128, 256))
            ])

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = conv(x, edge_index)
            return x

    # Create and process graph data
    data1 = create_graph_data(given_binary).to(device)
    data2 = create_graph_data(inferred_binary).to(device)
    
    model = StructuralGIN().to(device)
    model.eval()

    with torch.no_grad():
        emb1, emb2 = model(data1), model(data2)
        
        def aggregate(emb):
            return torch.cat([
                global_mean_pool(emb, None),
                global_max_pool(emb, None),
                global_add_pool(emb, None),
                global_max_pool(emb, None) * global_mean_pool(emb, None)  # Interaction term
            ], dim=1)
            
        emb1 = aggregate(emb1)
        emb2 = aggregate(emb2)

    score = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
    
    if validate_result:
        validate_between_minus_plus_1(score=score)
    
    return score

# Tests

def test_identical_graphs():
    """Test that identical graphs yield maximum similarity (1.0)."""
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    score = gin_gnn_similarity(adj, adj, validate_result=False, seed=0, threshold=0.5)
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

def test_disconnected_graphs():
    """Test completely disconnected graphs (identity) return 1.0."""
    adj = np.zeros((3, 3), dtype=np.float32)
    score = gin_gnn_similarity(adj, adj, validate_result=False, seed=0, threshold=0.5)
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

def test_opposite_structured_graphs():
    """Test structurally opposite graphs with seed=42 (deterministic check)."""
    adj1 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)  # Complete graph
    adj2 = np.zeros_like(adj1)  # Disconnected
    score = gin_gnn_similarity(adj1, adj2, validate_result=False, seed=42, threshold=0.5)
    # Precomputed with seed=42 (deterministic expectation)
    expected_score = 0.91375494  # Model-specific output
    assert np.isclose(score, expected_score, atol=1e-1), f"Expected {expected_score}, got {score}"

def test_isomorphic_graphs():
    """Test isomorphic graphs return similarity 1.0."""
    adj1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)  # Line graph
    adj2 = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)  # Permuted
    score = gin_gnn_similarity(adj1, adj2, validate_result=False, seed=0, threshold=0.5)
    assert np.isclose(score, 1.0, atol=1e-3), f"Expected 1.0, got {score}"

def test_determinism():
    """Test deterministic outputs across multiple runs."""
    adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
    score1 = gin_gnn_similarity(adj, adj, validate_result=False, seed=42, threshold=0.5)
    score2 = gin_gnn_similarity(adj, adj, validate_result=False, seed=42, threshold=0.5)
    assert np.isclose(score1, score2, atol=1e-6), f"Scores differ: {score1} vs {score2}"

def test_feature_variation_sensitivity():
    """Test sensitivity to feature variations (non-identical but similar graphs)."""
    adj1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    adj2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)  # Modified
    score = gin_gnn_similarity(adj1, adj2, validate_result=False, seed=0, threshold=0.5)
    # Expect similarity less than identical but significantly above random
    assert 0.5 < score < 1.0, f"Unexpected score: {score}"

def test_validation_flag():
    """Test validation function is triggered correctly."""
    adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
    try:
        gin_gnn_similarity(adj, adj, validate_result=True, seed=0, threshold=0.5)
    except ValueError as e:
        assert False, f"Validation should pass but failed: {e}"

def test_single_node_graphs():
    """Test graphs with a single node (trivial case)."""
    adj = np.array([[0]], dtype=np.float32)
    score = gin_gnn_similarity(adj, adj, threshold=0.5, validate_result=False, seed=0)
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

def test_threshold_effect():
    """Test thresholding correctly affects edge binarization."""
    given = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)
    inferred_high = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)  # Same binarized as given at 0.5
    inferred_low = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)    # All edges at 0.5 threshold
    
    score_same = gin_gnn_similarity(given, inferred_high, threshold=0.5, validate_result=False, seed=42)
    score_diff = gin_gnn_similarity(given, inferred_low, threshold=0.5, validate_result=False, seed=42)
    assert score_same > score_diff, "Threshold should distinguish edge presence"

def test_empty_vs_nonempty():
    """Test empty graph vs connected graph."""
    adj_empty = np.zeros((3, 3), dtype=np.float32)
    adj_connected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    score = gin_gnn_similarity(adj_empty, adj_connected, threshold=0.5, validate_result=False, seed=42)
    assert score < 1.0, "Empty vs connected graphs should have low similarity"

def test_star_vs_ring():
    """Test structural difference between star and ring graphs."""
    # Star graph (center node 0)
    star = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.float32)
    # Ring graph
    ring = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.float32)
    score = gin_gnn_similarity(star, ring, threshold=0.5, validate_result=False, seed=42)
    assert 0.0 < score < 1.0, f"Star and ring similarity out of expected range: {score}"

def test_weighted_matrices_threshold():
    """Test threshold impact on weighted matrices."""
    given = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
    # At threshold 0.5, given becomes [[1,0],[0,1]]
    inferred_above = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)  # Binarizes to same as given
    inferred_below = np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)  # Binarizes to different structure
    score_same = gin_gnn_similarity(given, inferred_above, threshold=0.5, validate_result=False, seed=0)
    score_diff = gin_gnn_similarity(given, inferred_below, threshold=0.5, validate_result=False, seed=0)
    assert score_same > score_diff, "Threshold should reflect structural similarity"

def test_self_loops():
    """Test graphs with self-loops vs edges between nodes."""
    adj_self = np.array([[1, 0], [0, 1]], dtype=np.float32)    # Self-loops only
    adj_edge = np.array([[0, 1], [1, 0]], dtype=np.float32)    # Edge between nodes
    score = gin_gnn_similarity(adj_self, adj_edge, threshold=0.5, validate_result=False, seed=0)
    assert score < 1.0, "Self-loops vs edges should have different similarity"

if __name__ == "__main__":
    test_identical_graphs()
    test_disconnected_graphs()
    test_opposite_structured_graphs()
    test_isomorphic_graphs()
    test_determinism()
    test_feature_variation_sensitivity()
    test_validation_flag()
    # 
    test_single_node_graphs()
    test_threshold_effect()
    test_empty_vs_nonempty()
    test_star_vs_ring()
    test_weighted_matrices_threshold()
    test_self_loops()
    print("All tests passed.")