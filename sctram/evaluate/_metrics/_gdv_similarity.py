#!/usr/bin/env python3

import sys
working_directory = "/Users/kemalinecik/git_nosync/sctram"
sys.path.append(working_directory)

import numpy as np
import itertools
import numpy as np
from scipy.stats import entropy
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1


def gdv_similarity(
        given_adjacency_matrix: np.ndarray, 
        inferred_adjacency_matrix: np.ndarray, 
        threshold: float, 
        
        validate_result: bool
    ) -> float:
    """Compute a similarity score between two graphs based on their GDVs.
    
    The method does not assume any underlying distribution for the network structure; it is entirely data-driven, 
    making it robust in many settings.
    
    Note: The approach can be extended with bootstrapping or permutation tests to assess the statistical 
    significance of the observed similarities, further solidifying its rigorous basis.

    Approach:
        - Graphlet Definition: Focus on 3-node and 4-node graphlets to capture linear (paths) and branching (stars) structures.
        - GDV Calculation: For each node, compute the frequency of each graphlet type it participates in.
        - Statistical Comparison: Use Jensen-Shannon Divergence (JSD) to compare the distribution of graphlet counts between the two graphs, aggregating results into a single similarity score.

    Explanation:
        - Graphlet Calculation:
            - `compute_gdvs` generates GDVs for each node by enumerating all 3-node and 4-node connected subgraphs 
                (graphlets) and counting their occurrences.
            - `is_connected` checks if a subgraph is connected using BFS.
        - Statistical Comparison:
            - `compute_jsd` computes the Jensen-Shannon Divergence between the distributions of graphlet
                counts for two graphs.
            - `gdv_similarity` aggregates JSD values across all graphlet types to produce a final similarity 
                score between 0 (dissimilar) and 1 (identical).
    
    Parameters:
        given_adjacency_matrix (np.ndarray): A square numpy array, symmetric adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A square numpy array, symmetric adjacency matrix of the second graph.
        threshold (float): A threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): A boolean flag indicating whether to validate that the resulting score range.
    
    Returns:
        float: A similarity score between 0 (completely dissimilar) and 1 (identical)
        
    Interpretation:
        - Converting the divergence to a similarity score on a [0, 1] scale gives a clear, interpretable metric: 
            1 means identical local structure distributions and 0 indicates maximum divergence.
    """
    # Binarize the inferred adjacency matrix using the provided threshold.
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    # Compute the Graphlet Degree Vectors for both graphs considering 3-node and 4-node graphlets
    gdv1 = compute_gdvs(given_adjacency_matrix, [3, 4])
    gdv2 = compute_gdvs(inferred_binary, [3, 4])
    
    graphlet_types_indices = 8  # Total number of graphlet types defined. See below function.
    jsds = []
    # For each graphlet type, compare the distributions of counts between the two graphs
    for type_idx in range(graphlet_types_indices):
        # Extract counts of the current graphlet type from each node's GDV
        counts1 = [gdv[type_idx] for gdv in gdv1.values()]
        counts2 = [gdv[type_idx] for gdv in gdv2.values()]
        jsd_value = compute_jsd(counts1, counts2)
        jsds.append(jsd_value)
    
    # Average the JSD values across all graphlet types
    avg_jsd = np.mean(jsds)
    # Transform the divergence into a similarity score between 0 and 1
    similarity = 1 - avg_jsd
    
    if validate_result:
        validate_inclusive_between_0_1(similarity)
        
    return similarity


def is_connected(nodes, adj):
    """Check if a set of nodes forms a connected subgraph.
    
    Parameters:
    - nodes: tuple/list of node indices
    - adj: dictionary mapping each node to its set of neighbors
    
    Returns:
    - True if the subgraph induced by 'nodes' is connected; False otherwise.
    """
    visited = set()
    # Start BFS from the first node in the combination
    start_node = nodes[0]
    queue = [start_node]
    visited.add(start_node)
    while queue:
        current = queue.pop(0)
        for neighbor in adj[current]:
            # Only consider neighbors that are in the subgraph and haven't been visited
            if neighbor in nodes and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    # The subgraph is connected if we've visited all nodes in 'nodes'
    return len(visited) == len(nodes)


def compute_gdvs(adj_matrix, k_values):
    """Compute Graphlet Degree Vectors (GDVs) for each node in the graph.
    
    Graphlets are small subgraphs (e.g., 3-node paths, triangles, 4-node stars) that summarize local connectivity 
    patterns. In many network applications—including single-cell genomics—the local wiring of nodes (i.e., 
    which nodes tend to cluster, form cycles, or create branching structures) carries important information about 
    overall network organization.
    
    By counting the frequency of each graphlet type that a node participates in, we obtain a Graphlet Degree 
    Vector (GDV) that serves as a rich feature representation of that node's local environment. By focusing on 3- and 
    4-node graphlets, the method extracts meaningful local connectivity patterns that are relevant in various 
    fields, including genomics.
    
    Parameters:
    - adj_matrix: symmetric adjacency matrix of the graph (list of lists or 2D array)
    - k_values: list of sizes (e.g., [3, 4]) of graphlets to consider
    
    Returns:
    - Dictionary mapping node indices to their GDV (list of counts for each graphlet type)
    """
    n = len(adj_matrix)
    # Build an adjacency list for faster lookup
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            # Avoid self-loops and add edge if present
            if adj_matrix[i][j] == 1 and i != j:
                adj[i].add(j)
    
    # Define mapping for graphlet types to indices in the GDV vector
    graphlet_types = {
        (3, 'path'): 0,
        (3, 'triangle'): 1,
        (4, 'path'): 2,
        (4, 'cycle'): 3,
        (4, 'star'): 4,
        (4, 'complete'): 5,
        (4, 'diamond'): 6,
        (4, 'paw'): 7,
    }
    num_types = len(graphlet_types)
    # Initialize GDVs: each node gets a list of zeros, one for each graphlet type
    gdvs = {node: [0] * num_types for node in range(n)}
    
    # Iterate over desired graphlet sizes (only 3 and 4 considered here)
    for k in k_values:
        
        if k not in [3, 4]:
            raise ValueError("Unsupported type for graphlet size")
        
        # Enumerate all combinations of k nodes
        for nodes in itertools.combinations(range(n), k):
            
            if not is_connected(nodes, adj):
                # Only consider connected subgraphs
                continue  
            
            if k == 3:
                edge_count = 0
                # Count edges among the 3 nodes
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        if nodes[j] in adj[nodes[i]]:
                            edge_count += 1
                
                # Classify the 3-node graphlet based on the number of edges
                if edge_count == 2:
                    graphlet_type = (3, 'path')
                elif edge_count == 3:
                    graphlet_type = (3, 'triangle')
                else:
                    continue
            
            elif k == 4:
                edge_count = 0
                degrees = []
                # Compute degrees and count edges for the 4-node subgraph
                for node in nodes:
                    degree = 0
                    for other in nodes:
                        if other != node and other in adj[node]:
                            degree += 1
                    degrees.append(degree)
                    # Count each edge only once (other > node ensures no double counting)
                    for other in nodes:
                        if other > node and other in adj[node]:
                            edge_count += 1
                degrees.sort()  # Sort the degree sequence to help classify the graphlet
                
                # Classify based on edge count and degree sequence
                if edge_count == 3:
                    if degrees == [1, 1, 1, 3]:
                        graphlet_type = (4, 'star')
                    elif degrees == [1, 1, 2, 2]:
                        graphlet_type = (4, 'path')
                    else:
                        continue
                elif edge_count == 4:
                    if degrees == [2, 2, 2, 2]:
                        graphlet_type = (4, 'cycle')
                    elif degrees == [1, 2, 3, 3]:
                        graphlet_type = (4, 'paw')
                    else:
                        continue
                elif edge_count == 5:
                    if degrees == [2, 3, 3, 3]:
                        graphlet_type = (4, 'diamond')
                    else:
                        continue
                elif edge_count == 6:
                    graphlet_type = (4, 'complete')
                else:
                    continue
            
            # Map the graphlet type to its index and update GDVs for all nodes in this subgraph
            type_idx = graphlet_types[graphlet_type]
            for node in nodes:
                gdvs[node][type_idx] += 1
    return gdvs


def compute_jsd(counts1, counts2):
    """Compute the Jensen-Shannon Divergence (JSD) between two lists of counts.
    
    JSD is a symmetrized and smoothed version of the Kullback-Leibler divergence. It measures the similarity 
    between two probability distributions while always yielding a finite value, which makes it attractive 
    for comparing distributions of graphlet counts.
    
    The divergence is bounded (usually between 0 and 1 when normalized) and symmetric. This ensures that 
    the similarity metric (1 - JSD) is both interpretable and consistent regardless of the order of the graphs.
    By calculating JSD for each graphlet type and averaging across them, we create a holistic metric that 
    summarizes differences in local structural patterns across the entire network.
    
    Parameters:
    - counts1, counts2: lists of counts for a particular graphlet type
    
    Returns:
    - A scalar JSD value (lower values indicate more similar distributions)
    """
    # Create a unified set of count values from both lists
    all_counts = list(set(counts1 + counts2))
    if not all_counts and not counts1 and not counts2:
        return 0.0
    all_counts.sort()
    # Compute frequency counts for each unique value in both distributions
    freq1 = [counts1.count(c) for c in all_counts]
    freq2 = [counts2.count(c) for c in all_counts]
    total1 = len(counts1) if counts1 else 1  # Avoid division by zero
    total2 = len(counts2) if counts2 else 1
    # Convert frequencies to probability distributions
    prob1 = [f / total1 for f in freq1]
    prob2 = [f / total2 for f in freq2]
    # Compute the average distribution
    m = 0.5 * (np.array(prob1) + np.array(prob2))
    m = np.where(m == 0, 1e-10, m)
    # Replace zeros with a small value to avoid log(0) issues in entropy calculations
    prob1 = np.where(prob1 == 0, 1e-10, prob1)
    prob2 = np.where(prob2 == 0, 1e-10, prob2)
    # Calculate the JSD as the average of the KL divergences from each distribution to m
    jsd = 0.5 * (entropy(prob1, m) + entropy(prob2, m))
    return jsd
