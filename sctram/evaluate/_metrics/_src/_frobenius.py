#!/usr/bin/env python3

import numpy as np

try:
    from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def frobenius(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool) -> float:
    """Frobenius norm of the difference.    

    Calculates the Frobenius norm of the difference between two adjacency matrices,
    which quantifies the structural dissimilarity between two graphs represented by these matrices.
    The Frobenius norm is defined as the square root of the sum of the absolute squares of its elements,
    which in this context measures the difference in connectivity between two graphs.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A bool deciding whether or not validate the score based on `validate` method below

    Returns:
        float: The Frobenius norm of the matrix difference, which is a non-negative scalar.

    Advantages:
        - Captures global structural differences between two matrices, providing a comprehensive measure of graph dissimilarity.
        - Sensitive to both the addition and deletion of edges, offering a robust metric for comparing graphs.

    Limitations:
        - Does not indicate specific locations of differences within the graph.
        - All elements of the matrix contribute equally to the score, without considering potential variances in node or edge significance.

    Interpretation:
        - A score of 0 indicates identical graphs with no structural differences.
        - Lower scores represent minimal differences, suggesting high similarity between the graphs.
        - Higher scores indicate greater differences, implying significant structural dissimilarity.
        - The score range can vary depending on the size of the matrices and the nature of the graph structures. 
            The actual range and interpretation of scores should be contextualized to the specific graphs 
            being analyzed, particularly their size and density.
    """
    diff_matrix = given_adjacency_matrix - inferred_adjacency_matrix
    score = np.linalg.norm(diff_matrix, "fro")
    
    if validate_result:
        # always need to provide validate
        _validator(score=score)
    
    return score 


if __name__ == "__main__":
    
    def test_identical_matrices():
        """Test when both adjacency matrices are identical, expecting a score of 0.0."""
        given = np.zeros((5, 5))
        inferred = np.zeros((5, 5))
        expected = 0.0
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_single_edge_difference():
        """Test matrices differing by a single symmetric edge (two elements)."""
        given = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        inferred = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        expected = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_non_integer_differences():
        """Test matrices with non-integer differences."""
        given = np.array([[0.5, 0.5], [0.5, 0.5]])
        inferred = np.array([[0.3, 0.3], [0.3, 0.3]])
        expected = np.sqrt(4 * (0.2 ** 2))  # sqrt(4 * 0.04) = 0.4
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, 0.4, decimal=6)

    def test_all_ones_vs_zeros():
        """Test a complete graph against an empty graph (all elements differ)."""
        size = 10
        given = np.zeros((size, size))
        inferred = np.ones((size, size))
        expected = np.sqrt(size * size * 1.0)  # sqrt(100) = 10.0
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_large_matrix():
        """Test with large matrices (100x100) where each element differs by 0.1."""
        size = 100
        given = np.zeros((size, size))
        inferred = np.full((size, size), 0.1)
        expected = np.sqrt(size**2 * (0.1**2))  # sqrt(10000 * 0.01) = 10.0
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_mixed_differences():
        """Test matrices with mixed positive and negative differences."""
        given = np.array([[1.5, 2.3], [2.3, 0.0]])
        inferred = np.array([[1.5, 2.0], [2.0, 0.5]])
        # Manual calculation: sum of squares = 0.3^2 + 0.3^2 + (-0.5)^2 = 0.09 + 0.09 + 0.25 = 0.43
        expected = np.sqrt(0.43)  # ≈ 0.655743852
        score = frobenius(given, inferred, validate_result=True)
        np.testing.assert_almost_equal(score, expected, decimal=6)
    
    test_identical_matrices()
    test_single_edge_difference()
    test_non_integer_differences()
    test_all_ones_vs_zeros()
    test_large_matrix()
    test_mixed_differences()
    print("All tests passed.")
