#!/usr/bin/env python3

import numpy as np

from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1


def jaccard_similarity(
    given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, threshold: float, validate_result: bool
) -> float:
    """Calculates the Jaccard Similarity between the edge sets of two graphs represented by adjacency matrices.

    Jaccard Similarity is the ratio of the number of common edges to the total number of unique edges in both graphs.

    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): A bool deciding whether or not to validate the score based on `validate` method below

    Returns:
        float: The Jaccard similarity score, a single scalar value between 0 and 1.

    Advantages:
        - Intuitive measure of edge-wise similarity.
        - Easy to interpret as a percentage.

    Limitations:
        - Does not account for the importance or weight of specific edges.
        - Binary measure; does not capture edge weights if present.
        - Sensitive to the presence or absence of edges.
        - Ignores edge multiplicity in MultiGraphs.

    Interpretation:
        - A score of 0 indicates no overlap between the graphs' edges, implying no similarity.
        - A score of 1 indicates complete overlap, implying identical edge sets.
        - Values closer to 1 suggest higher similarity between the graphs.
    """
    # Binarize the inferred adjacency matrix using the provided threshold.
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)

    set_g1 = set(zip(*np.where(given_adjacency_matrix)))
    set_g2 = set(zip(*np.where(inferred_binary)))
    intersection = set_g1.intersection(set_g2)
    union = set_g1.union(set_g2)
    jaccard = len(intersection) / len(union) if union else 1.0  # Both graphs have no edges

    if validate_result:
        validate_inclusive_between_0_1(score=jaccard)

    return jaccard
