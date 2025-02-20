#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sctram.evaluate._metrics.validators import validate_zero_or_positive


def graph_edit_distance(given_adjacency_matrix: np.ndarray,
                        inferred_adjacency_matrix: np.ndarray,
                        threshold: float,
                        validate_result: bool) -> float:
    """Compute the Graph Edit Distance (GED) between two square adjacency matrices.
    
    This function calculates the Graph Edit Distance (GED), which quantifies the minimum number of 
    edit operations (edge additions or deletions) required to transform one graph into the other. 
    The input graphs are represented by their square adjacency matrices, and the inferred matrix 
    is first binarized using a provided threshold before comparison.
    
    Parameters:
        given_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix 
                                               of the first graph.
        inferred_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix 
                                                  of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates the computed GED using a validator function.
    
    Returns:
        float: The Graph Edit Distance, a non-negative scalar representing the minimum number of edit 
               operations required to transform one graph into the other. 
    
    Advantages:
        - Provides a comprehensive measure of structural dissimilarity.
        - Reflects the exact number of changes needed for graph transformation.
    
    Limitations:
        - Computationally intensive for larger graphs.
        - Highly sensitive to both edge additions and deletions.
        - May not be feasible for graphs with a large number of nodes.
    
    Interpretation:
        - A GED of 0 indicates that the graphs are structurally identical.
        - Higher values indicate more substantial differences in graph structure.
        - The computed value represents the minimum number of edit operations needed for transformation.
    """
    # Convert the given adjacency matrix to a NetworkX graph. This is already binary
    g1 = nx.from_numpy_array(given_adjacency_matrix, create_using=nx.Graph)
    
    # Binarize the inferred adjacency matrix using the provided threshold.
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    g2 = nx.from_numpy_array(inferred_binary, create_using=nx.Graph)
    ged = nx.graph_edit_distance(g1, g2)
    
    if validate_result:
        validate_zero_or_positive(ged)
        
    return ged
