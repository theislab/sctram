#!/usr/bin/env python3

import networkx as nx
import numpy as np
from sctram.evaluate._metrics.validators import validate_zero_or_positive
from typing import Optional

def average_shortest_path_difference(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool, threshold: Optional[float] = None) -> float:
    """Compute the absolute difference in average shortest path lengths between two connected graphs.

    This method constructs graphs from two square adjacency matrices and computes the absolute
    difference of the average shortest path lengths between all connected pairs of nodes in each graph. 
    It is a global metric capturing differences in graph connectivity and navigability.

    Parameters:
        given_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): If True, the resulting score is validated to be zero or positive using a helper validator.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.

    Returns:
        float: The absolute difference between the average shortest path lengths of the two graphs.
               Returns NaN if the graphs are disconnected (though this is pre-checked and raises an error).

    Advantages:
        - Captures global structural differences in connectivity and navigability.
        - Provides a single scalar value that summarizes differences in path lengths.

    Limitations:
        - Requires both graphs to be fully connected; otherwise, an error is raised.
        - Sensitive to minor edge modifications that can cause significant changes in average shortest path length.
        - Does not capture the distribution of individual path lengths beyond the average.
        - Assumes input matrices are valid representations of undirected graphs.

    Interpretation:
        - A value of zero indicates that the two graphs have identical average shortest path lengths.
        - A larger value suggests greater differences in the global connectivity between the two graphs.
        - The metric is particularly useful when comparing the overall efficiency of information flow or transport in networks.

    Raises:
        ValueError: If either input matrix is not square or if the dimensions of the two matrices do not match.
        ValueError: If one or both graphs are disconnected.

    """
    if threshold is not None:
        # Binarize the inferred adjacency matrix using the provided threshold.
        inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
        g_inferred = nx.from_numpy_array(inferred_binary)
    else:
        # Default option.
        g_inferred = nx.from_numpy_array(inferred_adjacency_matrix)
    
    g_given = nx.from_numpy_array(given_adjacency_matrix)
    avg_given = nx.average_shortest_path_length(g_given, method="floyd-warshall-numpy", weight="weight")
    avg_inferred = nx.average_shortest_path_length(g_inferred, method="floyd-warshall-numpy", weight="weight", )
    
    if np.isinf(avg_given).any() or np.isinf(avg_inferred).any():
        raise ValueError("Disconnected graph detected. Average shortest path difference requires fully connected graphs.")
    if np.isnan(avg_given).any() or np.isnan(avg_inferred).any():
        raise ValueError("One or both graphs are disconnected. Average shortest path difference is undefined.")
    
    score = abs(avg_given - avg_inferred)

    if validate_result:
        validate_zero_or_positive(score=score)

    return score


