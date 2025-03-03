#!/usr/bin/env python3

import numpy as np

from sctram.evaluate._metrics._src.validators import validate_zero_or_positive


def hamming_distance(
    given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, threshold: float, validate_result: bool
) -> int:
    """Calculates the Hamming Distance between two adjacency matrices.

    The Hamming Distance is the count of differing elements (i.e., mismatched edges) between two square adjacency matrices,
    representing the number of edge differences between two graphs.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): A bool deciding whether or not to validate the result based on `validate` method below.

    Returns:
        int: The total count of differing matrix elements, which is a non-negative integer.

    Advantages:
        - Simple and direct measure of edge discrepancies between two graph structures.
        - Easy to compute and interpret.

    Limitations:
        - Does not account for the significance or impact of specific differences.
        - Treats all differences equally regardless of their structural importance.
        - Sensitive to any edge addition or deletion, but does not differentiate between types of edge differences.

    Interpretation:
        - A score of 0 indicates identical graphs with no edge differences.
        - Higher scores indicate more differences, implying greater dissimilarity between the graphs.
        - The range of possible scores is from 0 to n^2 where n is the number of vertices in the graphs (assuming square matrices).
    """
    # Binarize the inferred adjacency matrix using the provided threshold.
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    hamming_score = np.sum(given_adjacency_matrix != inferred_binary)

    if validate_result:
        validate_zero_or_positive(score=hamming_score)

    return hamming_score
