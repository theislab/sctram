#!/usr/bin/env python3

import numpy as np
import networkx as nx

try:
    from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def graph_edit_distance(given_adjacency_matrix: np.ndarray,
                        inferred_adjacency_matrix: np.ndarray,
                        threshold: float,
                        validate_result: bool) -> float:
    """Compute the Graph Edit Distance (GED) between two square adjacency matrices.
    
    This function calculates the Graph Edit Distance (GED), which quantifies the minimum number of 
    edit operations (edge additions or deletions) required to transform one graph into the other. 
    The input graphs are represented by their square adjacency matrices, and the inferred matrix 
    is first binarized using a provided threshold before comparison.
    
    Parameters:
        given_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix 
                                               of the first graph.
        inferred_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix 
                                                  of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates the computed GED using a validator function.
    
    Returns:
        float: The Graph Edit Distance, a non-negative scalar representing the minimum number of edit 
               operations required to transform one graph into the other. 
    
    Advantages:
        - Provides a comprehensive measure of structural dissimilarity.
        - Reflects the exact number of changes needed for graph transformation.
    
    Limitations:
        - Computationally intensive for larger graphs.
        - Highly sensitive to both edge additions and deletions.
        - May not be feasible for graphs with a large number of nodes.
    
    Interpretation:
        - A GED of 0 indicates that the graphs are structurally identical.
        - Higher values indicate more substantial differences in graph structure.
        - The computed value represents the minimum number of edit operations needed for transformation.
    """
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    given_edges = set(tuple(sorted(e)) for e in np.argwhere(given_adjacency_matrix) if e[0] != e[1])
    inferred_binary = (inferred_binary >= threshold).astype(int)
    inferred_edges = set(tuple(sorted(e)) for e in np.argwhere(inferred_binary) if e[0] != e[1])
    ged = len(given_edges.symmetric_difference(inferred_edges))
    
    if validate_result:
        _validator(ged)
        
    return ged


if __name__ == "__main__":
    
    def test_identical_matrices():
        """Test when given and inferred matrices are identical."""
        given = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=int)
        inferred = given.astype(float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 0, f"Expected 0, got {ged}"

    def test_one_edge_addition():
        """Test when inferred has one extra edge compared to given."""
        given = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]], dtype=int)
        inferred = np.array([[0, 0.6, 0.0],
                            [0.6, 0, 0.7],
                            [0.0, 0.7, 0]], dtype=float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 1, f"Expected 1, got {ged}"

    def test_two_edge_difference():
        """Test when given and inferred differ by two edges."""
        given = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=int)
        inferred = np.array([[0, 0.6, 0.6],
                            [0.6, 0, 0.4],
                            [0.6, 0.4, 0]], dtype=float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 2, f"Expected 2, got {ged}"

    def test_complete_vs_empty():
        """Test complete graph against empty graph."""
        n = 4
        given = np.zeros((n, n), dtype=int)
        inferred = np.ones((n, n)) * 0.6
        np.fill_diagonal(inferred, 0)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 6, f"Expected 6, got {ged}"

    def test_large_matrix_chain_plus_edge():
        """Test large matrix with chain structure plus an extra edge."""
        n = 100
        given = np.zeros((n, n), dtype=int)
        for i in range(n-1):
            given[i, i+1] = 1
            given[i+1, i] = 1
        inferred = given.copy().astype(float)
        inferred[0, 99] = 0.6
        inferred[99, 0] = 0.6
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 1, f"Expected 1, got {ged}"

    def test_threshold_edge_case():
        """Test when inferred values are exactly at the threshold."""
        given = np.zeros((2, 2), dtype=int)
        inferred = np.array([[0, 0.5], [0.5, 0]], dtype=float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 1, f"Expected 1, got {ged}"

    def test_empty_matrices():
        """Test both matrices are empty."""
        given = np.zeros((3, 3), dtype=int)
        inferred = np.zeros((3, 3), dtype=float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 0, f"Expected 0, got {ged}"

    def test_triangle_vs_empty():
        """Test triangle graph against empty graph."""
        given = np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]], dtype=int)
        inferred = np.zeros((3, 3), dtype=float)
        threshold = 0.5
        ged = graph_edit_distance(given, inferred, threshold, validate_result=False)
        assert ged == 3, f"Expected 3, got {ged}"

    
    test_identical_matrices()
    test_one_edge_addition()
    test_two_edge_difference()
    test_complete_vs_empty()
    test_large_matrix_chain_plus_edge()
    test_threshold_edge_case()
    test_empty_matrices()
    test_triangle_vs_empty()
    print("All tests passed!")
