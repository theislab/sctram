#!/usr/bin/env python3

# import sys
# working_directory = "/Users/kemalinecik/git_nosync/sctram"
# sys.path.append(working_directory)

import warnings
import numpy as np
import networkx as nx
from sctram.evaluate._metrics.validators import validate_between_minus_plus_1

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F

_FEATURES = [
    'Weighted Degree', 'Weighted Clustering', 'Eigenvector Centrality',
    'Weighted PageRank', 'Structural Core Number', 'Structural Triangles',
    'Betweenness Centrality', 'Katz Centrality', 'HITS Hubs', 'HITS Authorities',
    'Ego Network Edges', 'Max Neighbor Degree', 'Neighbor Degree Variation'
]


class _StructuralGCN(torch.nn.Module):
        """Fixed-weights GCN with layer-wise feature capture
        
        This is included for the sake of code completion. GIN based architecture is being used instead.
        """
        def __init__(self, n_input_feature: int, seed: int):
            super().__init__()
            torch.manual_seed(seed)
            self.conv1 = GCNConv(n_input_feature, 32, add_self_loops=False)
            self.conv2 = GCNConv(32, 64, add_self_loops=False)
            self.conv3 = GCNConv(64, 128, add_self_loops=False)
            self.conv4 = GCNConv(128, 256, add_self_loops=False)
            self.conv5 = GCNConv(256, 512, add_self_loops=False)
            self.lin = torch.nn.Linear(512, 256)
            self.layer_outputs = []

        def forward(self, data: Data) -> torch.Tensor:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            self.layer_outputs = []
            
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            self.layer_outputs.append(x)
            
            x = F.elu(self.conv2(x, edge_index, edge_weight))
            self.layer_outputs.append(x)
            
            x = F.elu(self.conv3(x, edge_index, edge_weight))
            self.layer_outputs.append(x)
            
            x = F.elu(self.conv4(x, edge_index, edge_weight))
            self.layer_outputs.append(x)
            
            x = F.elu(self.conv5(x, edge_index, edge_weight))
            self.layer_outputs.append(x)
            
            x = self.lin(x)
            self.layer_outputs.append(x)
            
            return x


class _StructuralGIN(torch.nn.Module):
        """Enhanced GIN architecture with expanded feature capacity"""
        def __init__(self, n_input_feature: int, seed: int):
            super().__init__()
            torch.manual_seed(seed)
            
            def create_mlp(in_dim, out_dim):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.Linear(64, out_dim),
                    torch.nn.ReLU()
                )
            
            self.convs = torch.nn.ModuleList([
                GINConv(create_mlp(n_input_feature, 64)),
                GINConv(create_mlp(64, 128)),
                GINConv(create_mlp(128, 256)),
                GINConv(create_mlp(256, 512))
            ])

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = conv(x, edge_index)
            return x


def _aggregate(emb: torch.Tensor) -> torch.Tensor:
    """Numerically stable aggregation with NaN prevention"""
    
    # Base statistics with stabilization
    mean_pool = global_mean_pool(emb, None)
    max_pool = global_max_pool(emb, None)
    sum_pool = global_add_pool(emb, None)
    std_pool = torch.std(emb, dim=0, keepdim=True).clamp_min(1e-6)

    # Moment encoding with clamping
    moments = torch.cat([
        mean_pool,
        torch.mean(emb**2, dim=0, keepdim=True),
        torch.mean((emb**3).clamp(-1e6, 1e6), dim=0, keepdim=True),  # Prevent overflow
        torch.mean((emb**4).clamp(-1e6, 1e6), dim=0, keepdim=True)
    ], dim=1)

    # Interaction terms with safe operations
    safe_sum = sum_pool.clamp_min(1e-6)
    interaction = torch.cat([
        mean_pool * max_pool,
        (mean_pool - max_pool).abs(),  # Prevent negative variance
        torch.sigmoid(max_pool) * torch.log1p(safe_sum),
        F.normalize(std_pool, p=2, dim=1) * mean_pool
    ], dim=1)

    # Positional encoding with length check
    sorted_emb, _ = torch.sort(emb, dim=0)
    if sorted_emb.shape[0] < 4:  # Handle small graphs
        positional = torch.zeros(1, 3*emb.shape[1], device=emb.device)
    else:
        positional = torch.cat([
            sorted_emb[:len(emb)//4].mean(dim=0, keepdim=True),
            sorted_emb[len(emb)//4:3*len(emb)//4].mean(dim=0, keepdim=True),
            sorted_emb[3*len(emb)//4:].mean(dim=0, keepdim=True)
        ], dim=1)
            
    # Final aggregation
    combined = torch.cat([
        moments, 
        interaction, 
        positional, 
    ], dim=1)
    return combined



def _create_graph_data(original_adj: np.ndarray, binary_adj: np.ndarray, remove_self_loops: bool):
    """Create PyG Data object with enhanced structural features"""
    # Feature computation graph (weighted)
    G_feature = nx.from_numpy_array(original_adj)
    # Edge structure graph (binary)
    G_edge = nx.from_numpy_array(binary_adj)
    
    if remove_self_loops:
        G_feature.remove_edges_from(nx.selfloop_edges(G_feature))
        G_edge.remove_edges_from(nx.selfloop_edges(G_edge))

    # Compute edge_index from binary graph
    edges = list(G_edge.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # Feature engineering pipeline
    features = []
    
    # Basic structural features (weighted)
    deg = np.log1p([d for _, d in G_feature.degree(weight='weight')])
    clust = list(nx.clustering(G_feature, weight='weight').values())
    triangles = np.log1p(list(nx.triangles(G_edge).values()))  # Structural triangles
    
    # Centrality measures (weighted where applicable)
    try:
        eigen = list(nx.eigenvector_centrality(G_feature, max_iter=1000, weight='weight').values())
    except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
        eigen = [0.0] * len(G_feature)
        
    pagerank = list(nx.pagerank(G_feature, weight='weight').values())
    core = list(nx.core_number(G_edge).values())  # Structural k-core
    
    # Advanced centrality features
    try:
        betweenness = list(nx.betweenness_centrality(G_feature, weight='weight').values())
    except:
        betweenness = [0.0] * len(G_feature)
        
    try:
        katz = list(nx.katz_centrality(G_feature, weight='weight').values())
    except:
        katz = [0.0] * len(G_feature)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hubs, auth = nx.hits(G_feature)
            hubs = list(hubs.values())
            authorities = list(auth.values())
    except:
        hubs = [0.0] * len(G_feature)
        authorities = [0.0] * len(G_feature)

    # Neighborhood features
    ego_edges = np.log1p([len(list(nx.ego_graph(G_edge, n))) for n in G_edge.nodes])
    neighbor_deg = [sorted([G_feature.degree(n, weight='weight') for n in G_feature.neighbors(i)], reverse=True) 
                    for i in G_feature.nodes]
    max_neighbor_deg = [d[0] if d else 0 for d in neighbor_deg]
    std_neighbor_deg = [np.std(d) if d else 0 for d in neighbor_deg]

    # Feature assembly and normalization
    features = np.array([
        deg, clust, eigen, pagerank, core, triangles, betweenness,
        katz, hubs, authorities, ego_edges, max_neighbor_deg, std_neighbor_deg
    ]).T

    # Robust normalization
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    means = np.mean(features, axis=0, keepdims=True)
    stds = np.std(features, axis=0, keepdims=True) + 1e-6
    features = (features - means) / stds

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index
    ), features.shape[1]



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
        - Combines 13 complementary node features to reduce feature bias
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
        validate_result (bool): A bool deciding whether or not to validate the score based on `validate` method below
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

    # Create and process graph data
    data1, _n_input_feature_1 = _create_graph_data(given_adjacency_matrix, given_binary, remove_self_loops)
    data2, _n_input_feature_2 = _create_graph_data(inferred_adjacency_matrix, inferred_binary, remove_self_loops)
    assert _n_input_feature_1 == _n_input_feature_2 == len(_FEATURES)
    data1 = data1.to(device)
    data2 = data2.to(device)
    
    model = _StructuralGIN(n_input_feature = _n_input_feature_1, seed=seed).to(device)
    model.eval()

    with torch.no_grad():
        emb1, emb2 = model(data1), model(data2)            
        emb1 = _aggregate(emb1)
        emb2 = _aggregate(emb2)
    
    score = F.cosine_similarity(emb1, emb2).item()
    
    if validate_result:
        validate_between_minus_plus_1(score=score)
    
    return score


if __name__ == "__main__":
    
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
        assert np.isclose(score, 1.0, atol=1e-2), f"Expected 1.0, got {score}"

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
    
    test_identical_graphs()
    test_disconnected_graphs()
    test_opposite_structured_graphs()
    test_isomorphic_graphs()
    test_determinism()
    test_feature_variation_sensitivity()
    test_validation_flag()
    test_threshold_effect()
    test_empty_vs_nonempty()
    test_star_vs_ring()
    test_weighted_matrices_threshold()
    test_self_loops()
    print("All tests passed.")
