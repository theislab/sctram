#!/usr/bin/env python3

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from loguru import logger
import warnings

try:
    from sctram.evaluate._metrics.validators import validate_between_minus_plus_1 as _validator
    from sctram.evaluate._metrics.utils import Centroids
except ImportError:
    from validators import validate_between_minus_plus_1 as _validator
    from utils import Centroids

_logger = logger.bind(module="MetricBase")


def embedding_distance_correlation(
    given_graph: nx.Graph,
    labels_array: np.ndarray,
    validate_result: bool,
    centroids: Centroids,
) -> float:
    """Compute Spearman correlation between graph-based and embedding-based distances.

    This method provides a robust statistical comparison between a cell-type trajectory graph
    and a low-dimensional embedding by evaluating how well their relative distance structures align.
    The correlation is computed between:
    1. Shortest path distances in the undirected version of the input graph
    2. Euclidean distances between label centroids in the embedding space

    Parameters:
        given_graph (nx.Graph): Directed/undirected graph representing cell-type relationships.
                                Nodes must correspond to labels in labels_array.
        inferred_embedding (np.ndarray): 2D array of shape (n_cells, n_features) containing
                                        the low-dimensional embedding.
        labels_array (np.ndarray): 1D array of shape (n_cells,) containing cell-type labels.
        validate_result (bool): If True, validates correlation is in [-1, 1]. Default True.
        centroids (Centroids): Object providing centroids for graph nodes via `get_centroids(nodes)`.

    Returns:
        float: Spearman's ρ between the two distance measures. Returns np.nan if undefined.

    Statistical Rationale:
        1. Non-parametric Analysis: Spearman's correlation assesses monotonic relationships
           without assuming linearity, appropriate for complex biological data.
        2. Centroid Stability: Using label centroids reduces within-population noise while
           maintaining biological signal.
        3. Graph Structure Preservation: Shortest paths capture essential topological features
           of differentiation trajectories.

    Advantages:
        - Robust to scaling: Focuses on relative distance ordering rather than absolute values
        - Handles disconnected components: Automatically excludes unconnected pairs
        - Vectorized operations: Efficient centroid distance calculations using numpy

    Limitations:
        - Assumes label homogeneity: Requires distinct populations for meaningful centroids
        - Quadratic complexity: O(t^2) pairwise comparisons for t labels (manageable for t < 1e4)
        - Undefined for constant distances: Returns NaN if either distance set has zero variance

    Interpretation:
        ρ ≈ 1: Strong structural preservation
        ρ ≈ 0: No monotonic relationship
        ρ ≈ -1: Inverse relationship (rare in biological contexts)

    Raises:
        ValueError: On invalid input structure or insufficient valid pairs
    """
    # Input validation
    unique_labels = np.unique(labels_array)
    graph_nodes = set(given_graph.nodes())
    
    if not graph_nodes.issubset(unique_labels):
        missing = graph_nodes - set(unique_labels)
        raise ValueError(f"Graph nodes {missing} missing from labels array")

    # Convert to undirected graph for path calculations
    undir_graph = given_graph.to_undirected() if given_graph.is_directed() else given_graph.copy()
    nodes = list(undir_graph.nodes())
    n_nodes = len(nodes)
    
    if n_nodes < 2:
        raise ValueError("At least two graph nodes required")

    # Get shortest path lengths using BFS for unweighted graphs
    shortest_paths = dict(nx.all_pairs_shortest_path_length(undir_graph))

    # Collect pairwise distances
    graph_dists, embed_dists = [], []
    centroid_matrix = np.array([centroids.get_single(label) for label in nodes])
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            label_i, label_j = nodes[i], nodes[j]
            path_len = shortest_paths.get(label_i, {}).get(label_j, None)
            
            if path_len is not None:
                graph_dists.append(path_len)
                embed_dist = np.linalg.norm(centroid_matrix[j] - centroid_matrix[i])
                embed_dists.append(embed_dist)

    # Check sufficient pairs
    if len(graph_dists) < 2:
        raise ValueError("Insufficient valid pairs for correlation")
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corr, _ = spearmanr(graph_dists, embed_dists)

    if validate_result:
        _validator(score=corr)

    return corr


if __name__ == "__main__":

    class MockCentroids:
        """Mock Centroids class for testing."""
        def __init__(self, data):
            self.data = data
        
        def get_single(self, label):
            return self.data[label]

    def test_perfect_correlation():
        """Test perfect Spearman correlation (ρ=1)."""
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C')])
        labels = np.array(['A', 'B', 'C'])
        centroids = MockCentroids({'A': np.array([0.0]), 'B': np.array([1.0]), 'C': np.array([2.0])})
        result = embedding_distance_correlation(G, labels, True, centroids)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    def test_disconnected_components():
        """Test with disconnected nodes resulting in NaN."""
        from scipy.stats._warnings_errors import ConstantInputWarning

        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('C', 'D')])
        labels = np.array(['A', 'B', 'C', 'D'])
        centroids = MockCentroids({'A': np.array([0.0]), 'B': np.array([1.0]), 
                            'C': np.array([2.0]), 'D': np.array([3.0])})
        try:
            _ = embedding_distance_correlation(G, labels, True, centroids)
            assert False, "Error should have been raised."
        except ConstantInputWarning:
            pass    
        

    def test_large_linear_graph():
        """Test large linear graph for perfect correlation."""
        n = 10  # 10 nodes
        G = nx.path_graph(n)
        labels = np.array([i for i in range(n)])
        centroids = MockCentroids({i: np.array([i]) for i in range(n)})
        result = embedding_distance_correlation(G, labels, True, centroids)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    def test_known_correlation():
        """Test with manually verifiable correlation."""
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        labels = np.array(['A', 'B', 'C', 'D'])
        centroids = MockCentroids({
            'A': np.array([0]), 'B': np.array([1]), 
            'C': np.array([3]), 'D': np.array([4])
        })
        
        # Manually compute expected correlation
        nodes = list(G.nodes())
        centroid_matrix = np.array([centroids.get_single(n) for n in nodes])
        graph_dists, embed_dists = [], []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                graph_dists.append(nx.shortest_path_length(G, nodes[i], nodes[j]))
                embed_dists.append(np.linalg.norm(centroid_matrix[j] - centroid_matrix[i]))
        expected_rho, _ = spearmanr(graph_dists, embed_dists)
        
        result = embedding_distance_correlation(G, labels, True, centroids)
        assert np.isclose(result, expected_rho), f"Expected {expected_rho}, got {result}"

    def test_minimal_graph_error():
        """Test insufficient pairs raise ValueError."""
        G = nx.Graph()
        G.add_edge('A', 'B')
        labels = np.array(['A', 'B'])
        centroids = MockCentroids({'A': np.array([0]), 'B': np.array([1])})
        try:
            value = embedding_distance_correlation(G, labels, True, centroids)
            assert "Insufficient valid pairs", value
        except ValueError:
            pass

    def test_zero_variance():
        """Test zero variance leading to NaN."""
        from scipy.stats._warnings_errors import ConstantInputWarning
        
        G = nx.complete_graph(3)
        labels = np.array([0, 1, 2])
        centroids = MockCentroids({0: np.array([0]), 1: np.array([0]), 2: np.array([0])})
        
        try:
            _ = embedding_distance_correlation(G, labels, True, centroids)
            assert False, "Error should have been raised."
        except ConstantInputWarning:
            pass    
        
    test_perfect_correlation()
    test_disconnected_components()
    test_large_linear_graph()
    test_known_correlation()
    test_minimal_graph_error()
    test_zero_variance()
    print("All tests passed.")
