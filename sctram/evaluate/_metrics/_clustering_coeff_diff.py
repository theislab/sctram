#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sctram.evaluate._metrics.validators import validate_zero_or_positive

def clustering_coeff_diff(given_adjacency_matrix: np.ndarray, 
                          inferred_adjacency_matrix: np.ndarray, 
                          validate_result: bool) -> float:
    """
    Computes the absolute difference in average clustering coefficients between two graphs.
    
    This method evaluates the local connectivity patterns by comparing the average clustering 
    coefficients computed from two square adjacency matrices. For a mathematical and statistical 
    audience, note that the clustering coefficient quantifies the tendency of nodes to form triangles, 
    thereby indicating local clustering in the network.
    
    Parameters:
        given_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A flag indicating whether the computed difference should be validated as zero or positive.
    
    Returns:
        float: The absolute difference between the average clustering coefficients of the two graphs.
    
    Advantages:
        - Captures differences in local connectivity and the presence of tightly-knit communities.
        - Sensitive to changes in triangle formation that reflect local graph structure.
    
    Limitations:
        - Only accounts for local connectivity; global structural differences may not be reflected.
        - Sensitive to edge modifications that affect triangle counts (a factor that might be influenced by noise).
    
    Interpretation:
        - Clustering Coefficient gives a sense of how nodes tend to cluster together, thus providing a measure 
            of local cohesiveness in a network. Calculating the absolute difference in average clustering 
            coefficients between two graphs effectively measures how much the local connectivity patterns have 
            changed. This is particularly useful in studies where you might be comparing the structural dynamics 
            of networks over time or under different conditions.
        - A result of 0 indicates identical average clustering coefficients and, hence, similar local connectivity.
        - A larger value implies greater dissimilarity in the local clustering patterns between the two graphs.
    """
    g1 = nx.from_numpy_array(given_adjacency_matrix)
    g2 = nx.from_numpy_array(inferred_adjacency_matrix)
    clustering_g1 = nx.average_clustering(g1, weight="weight")
    clustering_g2 = nx.average_clustering(g2, weight="weight")
    score = abs(clustering_g1 - clustering_g2)
    
    if validate_result:
        validate_zero_or_positive(score=score)
    
    return score
