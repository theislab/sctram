#!/usr/bin/env python3

import networkx as nx
import numpy as np
from sctram.evaluate._metrics.validators import validate_between_minus_plus_1


def mantel_correlation(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool, permutations: int = 10000, seed: int = 0):
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
    from skbio.stats.distance import DistanceMatrix, mantel
    
    # Convert adjacency matrices to graphs
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
    score = mantel(dm_given, dm_inferred, method="pearson", permutations=permutations, seed=seed)[0]

    if validate_result:
        validate_between_minus_plus_1(score=score)

    return score

