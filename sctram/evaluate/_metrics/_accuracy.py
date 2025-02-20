#!/usr/bin/env python3

import numpy as np
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1


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
        validate_inclusive_between_0_1(score=score)
    
    return score
