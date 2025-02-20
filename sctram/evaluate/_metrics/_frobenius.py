#!/usr/bin/env python3

import numpy as np
from sctram.evaluate._metrics.validators import validate_zero_or_positive


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
        validate_zero_or_positive(score=score)
    
    return score 

