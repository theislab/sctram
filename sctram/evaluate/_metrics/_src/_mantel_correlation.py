#!/usr/bin/env python3

import networkx as nx
import numpy as np
from skbio.stats.distance import DistanceMatrix, mantel

try:
    from sctram.evaluate._metrics._src.validators import validate_between_minus_plus_1 as _validator
except ImportError:
    from validators import validate_between_minus_plus_1 as _validator


def mantel_correlation(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool,
    permutations: int = 10000,
    seed: int = 0,
):
    """Calculates the Mantel test statistic between two adjacency matrices.

    The Mantel test statistically assesses the correlation between distance matrices derived from
    two adjacency matrices, providing a measure of similarity between the underlying graph structures.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A bool deciding whether or not validate the score based on `validate` method.
        repetition (int): Number of permutations to average over for Mante calculation.
        seed (int, optional): Seed to provide Mantel function.

    Returns:
        float: The Mantel correlation coefficient, a measure of similarity between the two graphs.

    Advantages:
        - Provides a statistical measure of similarity between two network structures.
        - Accounts for spatial or structural dependencies via distance matrices.

    Limitations:
        - Computationally intensive, particularly for large matrices.
        - Assumes distance matrices are meaningful representations of the graphs' structure.

    Interpretation:
        - A correlation coefficient close to 1 indicates high similarity.
        - A correlation coefficient close to -1 suggests high dissimilarity.
        - A correlation coefficient around 0 indicates no correlation between the matrices.
    """
    g1 = nx.from_numpy_array(given_adjacency_matrix)
    g2 = nx.from_numpy_array(inferred_adjacency_matrix)

    # Compute all-pairs shortest path distance matrices
    try:
        dm_g1 = nx.floyd_warshall_numpy(g1)
        dm_g2 = nx.floyd_warshall_numpy(g2)
    except nx.NetworkXError as e:
        raise ValueError(f"Error computing shortest paths: {e}")

    # Check for disconnected graphs (infinite distances)
    if np.isinf(dm_g1).any() or np.isinf(dm_g2).any():
        raise ValueError("Disconnected graph detected. Mantel requires fully connected graphs.")

    # Convert to DistanceMatrix objects
    dm_given = DistanceMatrix(dm_g1)
    dm_inferred = DistanceMatrix(dm_g2)

    # Calculate Mantel test statistic
    score = mantel(dm_given, dm_inferred, method="spearman", permutations=permutations, seed=seed)[0]

    if validate_result:
        _validator(score=score)

    return score


if __name__ == "__main__":

    def test_identical_matrices():
        """Test that identical matrices yield a Mantel correlation of 1.0."""
        A = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        np.fill_diagonal(A, 0)  # Ensure diagonal is zero
        A = (A + A.T) / 2  # Ensure symmetry
        result = mantel_correlation(A, A, validate_result=True, permutations=0)
        assert np.isclose(result, 1.0, atol=1e-7), f"Expected 1.0, got {result}"

    def test_scaled_adjacency_matrices():
        """Test that scaling edge weights results in a correlation of 1.0."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = 2 * A  # All edge weights doubled
        result = mantel_correlation(A, B, validate_result=False, permutations=0)
        assert np.isclose(result, 1.0, atol=1e-7), f"Expected 1.0, got {result}"

    def test_spearman_correlation():
        """Test a manually verifiable case with expected correlation of -0.5."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        result = mantel_correlation(A, B, validate_result=False, permutations=0)
        expected = -0.5
        assert np.isclose(result, expected, atol=1e-7), f"Expected {expected}, got {result}"

    def test_disconnected_graph_raises_error():
        """Test that disconnected graphs raise a ValueError."""
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # Node 2 is disconnected
        B = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Fully connected
        try:
            mantel_correlation(A, B, validate_result=False)
            assert False, "Expected ValueError for disconnected graph"
        except ValueError as e:
            assert "disconnected" in str(e).lower()

    def test_large_matrix_identity():
        """Test a large matrix (10x10 chain) where correlation should be 1.0."""
        n = 10
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        result = mantel_correlation(A, A, validate_result=False, permutations=0)
        assert np.isclose(result, 1.0, atol=1e-7), f"Expected 1.0, got {result}"

    test_identical_matrices()
    test_scaled_adjacency_matrices()
    test_spearman_correlation()
    test_disconnected_graph_raises_error()
    test_large_matrix_identity()
    print("All tests passed.")
