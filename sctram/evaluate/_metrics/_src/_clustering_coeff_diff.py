#!/usr/bin/env python3

import numpy as np
import networkx as nx

try:
    from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def clustering_coeff_diff(given_adjacency_matrix: np.ndarray, 
                          inferred_adjacency_matrix: np.ndarray, 
                          validate_result: bool) -> float:
    """Computes the absolute difference in average clustering coefficients between two graphs.
    
    This method evaluates the local connectivity patterns by comparing the average clustering 
    coefficients computed from two square adjacency matrices. For a mathematical and statistical 
    audience, note that the clustering coefficient quantifies the tendency of nodes to form triangles, 
    thereby indicating local clustering in the network.
    
    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A flag indicating whether the computed difference should be validated as zero or positive.
    
    Returns:
        float: The absolute difference between the average clustering coefficients of the two graphs.
    
    Advantages:
        - Captures differences in local connectivity and the presence of tightly-knit communities.
        - Sensitive to changes in triangle formation that reflect local graph structure.
    
    Limitations:
        - Only accounts for local connectivity; global structural differences may not be reflected.
        - Sensitive to edge modifications that affect triangle counts (a factor that might be influenced by noise).
    
    Interpretation:
        - Clustering Coefficient gives a sense of how nodes tend to cluster together, thus providing a measure 
            of local cohesiveness in a network. Calculating the absolute difference in average clustering 
            coefficients between two graphs effectively measures how much the local connectivity patterns have 
            changed. This is particularly useful in studies where you might be comparing the structural dynamics 
            of networks over time or under different conditions.
        - A result of 0 indicates identical average clustering coefficients and, hence, similar local connectivity.
        - A larger value implies greater dissimilarity in the local clustering patterns between the two graphs.
    """
    g1 = nx.from_numpy_array(given_adjacency_matrix)
    g2 = nx.from_numpy_array(inferred_adjacency_matrix)
    clustering_g1 = nx.average_clustering(g1, weight="weight")
    clustering_g2 = nx.average_clustering(g2, weight="weight")
    score = abs(clustering_g1 - clustering_g2)
    
    if validate_result:
        _validator(score=score)
    
    return score

if __name__ == "__main__":
    
    def test_identical_matrices():
        """Test that identical matrices yield a difference of 0."""
        adj = np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]], dtype=np.float64)
        diff = clustering_coeff_diff(adj, adj, False)
        assert np.isclose(diff, 0.0, atol=1e-9), f"Expected 0.0, got {diff}"

    def test_zero_vs_triangle():
        """Test all-zero matrix vs. triangle matrix yields difference of 1.0."""
        adj_zero = np.zeros((3, 3), dtype=np.float64)
        adj_triangle = np.array([[0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0]], dtype=np.float64)
        diff = clustering_coeff_diff(adj_zero, adj_triangle, False)
        assert np.isclose(diff, 1.0, atol=1e-9), f"Expected 1.0, got {diff}"

    def test_line_vs_triangle():
        """Test line graph vs. triangle graph yields difference of 1.0."""
        adj_line = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.float64)
        adj_triangle = np.array([[0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0]], dtype=np.float64)
        diff = clustering_coeff_diff(adj_line, adj_triangle, False)
        assert np.isclose(diff, 1.0, atol=1e-9), f"Expected 1.0, got {diff}"

    def test_partial_clustering():
        """Test graph with partial clustering against complete graph."""
        adj_partial = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=np.float64)
        adj_complete = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
        expected_diff = abs(1.0 - (5/6))
        diff = clustering_coeff_diff(adj_partial, adj_complete, False)
        assert np.isclose(diff, expected_diff, atol=1e-9), f"Expected {expected_diff}, got {diff}"

    def test_large_complete_vs_cycle():
        """Test large complete graph vs. cycle graph yields difference of 1.0."""
        n = 100
        # Complete graph adjacency matrix
        adj_complete = np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64)
        # Cycle graph adjacency matrix
        rows, cols = np.indices((n, n))
        adj_cycle = ((cols == (rows + 1) % n) | (cols == (rows - 1) % n)).astype(np.float64)
        np.fill_diagonal(adj_cycle, 0)
        # Compute difference
        diff = clustering_coeff_diff(adj_complete, adj_cycle, False)
        assert np.isclose(diff, 1.0, atol=1e-9), f"Expected 1.0, got {diff}"
    
    test_identical_matrices()
    test_zero_vs_triangle()
    test_line_vs_triangle()
    test_partial_clustering()
    test_large_complete_vs_cycle()
    print("All tests passed.")