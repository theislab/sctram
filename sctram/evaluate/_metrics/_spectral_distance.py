#!/usr/bin/env python3

import numpy as np
from sctram.evaluate._metrics.validators import validate_zero_or_positive


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
        validate_zero_or_positive(score=score)
    
    return score
