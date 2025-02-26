#!/usr/bin/env python3

import numpy as np
    
try:
    from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def spectral_distance(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool) -> float:
    """Calculates the Spectral Distance between two adjacency matrices.

    Spectral Distance measures the dissimilarity between two graphs by comparing the eigenvalues of their adjacency matrices.
    It calculates the Euclidean (L2) norm of the difference between the sorted eigenvalues of both matrices.

    Parameters:
        given_adjacency_matrix (np.ndarray): Adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): Adjacency matrix of the second graph.
        validate_result (bool): Whether to validate the resulting score using the validation method below.

    Returns:
        float: The calculated Spectral Distance, a non-negative scalar value representing the topological dissimilarity between the two graphs.

    Advantages:
        - Captures global structural properties such as connectivity and expansion.
        - Reflects overall graph topology differences effectively.

    Limitations:
        - Lacks localized edge-specific information.
        - Sensitive to minor changes that can significantly alter the spectral characteristics.

    Interpretation:
        - A score of 0 indicates no difference in the eigenvalues, thus no topological difference.
        - Higher scores indicate greater differences, suggesting significant topological dissimilarity between the graphs.
        - Interpretation should consider the nature of the graphs, including their size and the extent of connectivity changes.
    """
    # Compute eigenvalues
    eigen_g1 = np.linalg.eigvals(given_adjacency_matrix)
    eigen_g2 = np.linalg.eigvals(inferred_adjacency_matrix)
    # Sort eigenvalues for alignment
    eigen_g1_sorted = np.sort_complex(eigen_g1)
    eigen_g2_sorted = np.sort_complex(eigen_g2)
    # Compute Euclidean (L2) norm of eigenvalue differences
    score = np.linalg.norm(eigen_g1_sorted - eigen_g2_sorted, ord=2)
    
    if validate_result:
        _validator(score=score)
    
    return score


if __name__ == "__main__":
    
    def test_identical_matrices():
        """Test that identical matrices yield a spectral distance of 0."""
        A = np.eye(3)
        score = spectral_distance(A, A, validate_result=False)
        np.testing.assert_almost_equal(score, 0.0, decimal=6)

    def test_permuted_adjacency_matrices():
        """Test that isomorphic graphs (permuted adjacency matrices) have distance 0."""
        A = np.array([[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]])
        # Permute rows and columns to create isomorphic graph
        B = A[[2, 1, 0], :][:, [2, 1, 0]]
        score = spectral_distance(A, B, validate_result=False)
        np.testing.assert_almost_equal(score, 0.0, decimal=6)

    def test_zero_vs_identity():
        """Test spectral distance between zero matrix and identity matrix."""
        A = np.zeros((3, 3))
        B = np.eye(3)
        expected = np.sqrt(3)  # sqrt( (1^2)*3 )
        score = spectral_distance(A, B, validate_result=False)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_manual_2x2_case():
        """Test manually calculated spectral distance for 2x2 matrices."""
        A = np.array([[0, 1], 
                    [1, 0]])
        B = np.array([[0, 2], 
                    [2, 0]])
        expected = np.sqrt(2)
        score = spectral_distance(A, B, validate_result=False)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_large_diagonal_difference():
        """Test large diagonal matrices with a single eigenvalue difference."""
        size = 1000
        diag_A = np.arange(1, size + 1)
        diag_B = diag_A.copy()
        diag_B[-1] += 1  # Alter the last eigenvalue
        A = np.diag(diag_A)
        B = np.diag(diag_B)
        expected = 1.0  # Only the last eigenvalue differs by 1
        score = spectral_distance(A, B, validate_result=False)
        np.testing.assert_almost_equal(score, expected, decimal=6)

    def test_reversed_diagonal():
        """Test matrices with reversed diagonal entries (same sorted eigenvalues)."""
        A = np.diag([3, 2, 1])
        B = np.diag([1, 2, 3])
        score = spectral_distance(A, B, validate_result=False)
        np.testing.assert_almost_equal(score, 0.0, decimal=6)
    
    test_identical_matrices()
    test_permuted_adjacency_matrices()
    test_zero_vs_identity()
    test_manual_2x2_case()
    test_large_diagonal_difference()
    test_reversed_diagonal()
    print("All tests passed.")
