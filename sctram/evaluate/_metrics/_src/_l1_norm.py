#!/usr/bin/env python3

import numpy as np

from sctram.evaluate._metrics._src.validators import validate_zero_or_positive


def l1_norm(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool) -> float:
    """Calculates the L1 norm (Manhattan distance) of the difference between two adjacency matrices.

    The L1 norm sums the absolute differences of the corresponding elements in the matrices,
    representing total edge discrepancies and providing a simple and interpretable measure.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A bool deciding whether or not to validate the score based on `validate` method below.

    Returns:
        float: The L1 norm of the matrix difference, which is a non-negative scalar.

    Advantages:
        - Simple and interpretable measure of total edge discrepancies.
        - Sensitive to both the magnitude and number of differences.

    Limitations:
        - Does not localize where differences occur within the graph.
        - May not adequately capture the structural impact of specific edge differences due to the uniform weighting of discrepancies.

    Interpretation:
        - A score of 0 indicates identical graphs with no edge differences.
        - Lower scores represent minimal differences, suggesting high similarity between the graphs.
        - Higher scores indicate greater differences, implying significant structural dissimilarity.
        - As with the Frobenius norm, the score range can vary depending on the size and density of the matrices.
    """
    diff_matrix = given_adjacency_matrix - inferred_adjacency_matrix
    score = np.sum(np.abs(diff_matrix))

    if validate_result:
        validate_zero_or_positive(score=score)

    return score
