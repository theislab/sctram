#!/usr/bin/env python3

import numpy as np

try:
    from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator


def accuracy(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, threshold: float, validate_result: bool) -> float:
    """Calculates the accuracy of the inferred adjacency matrix.

    Accuracy is defined as the proportion of correctly inferred edges (both present and absent)
    relative to the total number of possible edges in the adjacency matrices representing graphs.
    Implementation looks for being sufficiently close to the given adjacency matrix, considering a specified tolerance.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): A bool deciding whether or not to validate the score based on `validate` method below.

    Returns:
        float: A single scalar value between 0 and 1 representing accuracy.

    Advantages:
        - Provides a straightforward measure of overall correctness.
        - Balances true positives and true negatives.

    Limitations:
        - Can be misleading in cases of class imbalance (e.g., sparse graphs).
        - Does not distinguish between types of errors (false positives vs. false negatives).

    Interpretation:
        - A score of 1 indicates perfect accuracy with all edges correctly inferred.
        - Lower scores indicate lesser accuracy, with more errors in edge inference.
        - The score provides a simple, global measure of how well the graph's structure has been inferred.
    """
    # Binarize the inferred adjacency matrix using the provided threshold.
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)

    # Calculate total elements and matching elements.
    total_elements = given_adjacency_matrix.size
    matching_elements = np.sum(given_adjacency_matrix == inferred_binary)
    score = matching_elements / total_elements

    if validate_result:
        _validator(score=score)
    
    return score


if __name__ == "__main__":
    
    def test_perfect_match():
        """Test when the inferred matrix perfectly matches the given matrix."""
        given = np.ones((5, 5), dtype=int)
        inferred = np.ones((5, 5)) * 0.6  # Threshold 0.5 converts to 1
        threshold = 0.5
        expected = 1.0
        assert accuracy(given, inferred, threshold, False) == expected

    def test_no_match():
        """Test when the inferred matrix is completely wrong."""
        given = np.ones((2, 2), dtype=int)
        inferred = np.zeros((2, 2))
        threshold = 0.5
        expected = 0.0
        assert accuracy(given, inferred, threshold, False) == expected

    def test_partial_match():
        """Test a case with partial correctness."""
        given = np.array([[1, 0], [0, 1]])
        inferred = np.array([[0.6, 0.6], [0.3, 0.7]])
        threshold = 0.5
        # Inferred becomes [[1,1], [0,1]] → 3/4 correct
        expected = 0.75
        assert accuracy(given, inferred, threshold, False) == expected

    def test_threshold_edge():
        """Test elements exactly at the threshold."""
        given = np.array([[1, 0], [0, 1]])
        inferred = np.array([[0.5, 0.4], [0.3, 0.5]])
        threshold = 0.5
        # Inferred becomes [[1,0], [0,1]] → perfect match
        expected = 1.0
        assert accuracy(given, inferred, threshold, False) == expected

    def test_large_matrix():
        """Test with large matrices for scalability and correctness."""
        given = np.ones((100, 100), dtype=int)
        inferred = np.empty((100, 100))
        # First 5000 elements above threshold, rest below
        inferred.flat[:5000] = 0.6
        inferred.flat[5000:] = 0.4
        threshold = 0.5
        # 5000 correct (1s), 5000 incorrect (0s) → 0.5 accuracy
        expected = 0.5
        assert accuracy(given, inferred, threshold, False) == expected

    def test_mixed_ones_and_zeros():
        """Test a matrix with a mix of correct and incorrect entries."""
        given = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        inferred = np.array([
            [0.7, 0.3, 0.2, 0.1],  # → [1,0,0,0]
            [0.1, 0.8, 0.3, 0.0],  # → [0,1,0,0]
            [0.4, 0.2, 0.4, 0.0],  # → [0,0,0,0]
            [0.0, 0.0, 0.0, 0.9]   # → [0,0,0,1]
        ])
        threshold = 0.5
        # Correct matches: 14 out of 16 → 0.875
        expected = 14 / 16
        assert accuracy(given, inferred, threshold, False) == expected

    def test_non_square_matrix():
        """Test non-square matrices (though adjacency matrices are typically square)."""
        given = np.array([[1, 0, 1], [0, 1, 0]])
        inferred = np.array([[0.6, 0.4, 0.5], [0.3, 0.7, 0.2]])
        threshold = 0.5
        # Inferred becomes [[1,0,1], [0,1,0]] → perfect match
        expected = 1.0
        assert accuracy(given, inferred, threshold, False) == expected

    test_perfect_match()
    test_no_match()
    test_partial_match()
    test_threshold_edge()
    test_large_matrix()
    test_mixed_ones_and_zeros()
    test_non_square_matrix()
    print("All tests passed!")
