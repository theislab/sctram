#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
from scipy import sparse
import warnings
from typing import Optional
from loguru import logger

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator


_logger = logger.bind(name="MetricBase")


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool,
    skip_single_branches: bool = True,
    spectral_n_init: int = 20,
    min_cells_per_branch: int = 5
) -> float:
    """Advanced Directed Trajectory Validation using Spectral Consensus (D-TVS).

    Evaluates embedding's ability to preserve directed trajectory branches by:
    1. Identifying bifurcation points in directed graph structure
    2. Establishing ground truth through directed reachability analysis
    3. Validating embedding clusters using spectral graph theory

    Parameters:
        given_graph (nx.DiGraph): Biological trajectory with directed edges
        labels_array (np.ndarray): Cell type labels (n_cells,)
        precomputed_embedded_connectivities (sparse.csr_matrix): Scanpy KNN connectivities
        validate_result (bool): Validate output score in [0,1]
        skip_single_branches (bool): Skip points with <2 biological branches
        spectral_n_init (int): Spectral clustering initializations
        min_cells_per_branch (int): Minimum cells required per branch

    Returns:
        float: Weighted F1-score (0-1) averaged over valid branch points

    Statistical Rationale:
        1. Directional Fidelity: Uses directed graph structure to identify true
           biological bifurcations (nodes with out-degree ≥2)
        2. Reachability Ground Truth: For each branch point, defines expected
           clusters through descendants in directed acyclic graph (DAG)
        3. Spectral Consensus: Combines manifold learning (spectral clustering)
           with graph theory for robust cluster validation
        4. Noise Robustness: Majority voting with neighbor exclusion handles
           embedding noise and label uncertainty

    Complexity Management:
        - O(kN) neighbor lookups using precomputed KNN
        - O(m) graph operations for branch analysis (m = graph edges)
        - O(n²/c) spectral clustering (c = cells per branch point)

    Advantages:
        - Respects directed biological relationships
        - Handles complex branching patterns
        - Robust to local embedding noise
        - Theoretically grounded in spectral graph theory

    Limitations:
        - Requires meaningful KNN parameters
        - Assumes bifurcation points have distinct descendants
        - Depends on label accuracy for ground truth
    """
    _logger.warning("`trajectory_cardinality_validation` is not tested.")

    # Identify directed branch points (biological bifurcations)
    branch_points = [node for node in given_graph.nodes 
                    if given_graph.out_degree(node) >= 2]
    if not branch_points:
        return np.nan  # No biological branches
    
    total_f1 = 0.0
    valid_branches = 0

    for bp in branch_points:
        # Get immediate children in directed graph
        children = list(given_graph.successors(bp))
        if len(children) < 2:
            continue

        # Find unique biological branches using directed reachability
        branch_sets = []
        for child in children:
            # Get all descendants excluding those through other branches
            descendants = set()
            stack = [child]
            while stack:
                node = stack.pop()
                if node == bp:  # Prevent back traversal
                    continue
                descendants.add(node)
                # Only follow paths that don't return to branch point
                stack.extend(n for n in given_graph.successors(node) 
                            if n not in descendants and n != bp)
            if descendants:
                branch_sets.append(descendants)

        # Remove duplicate branches (shared descendants)
        unique_branches = []
        seen = set()
        for branch in branch_sets:
            frozen = frozenset(branch)
            if frozen not in seen:
                seen.add(frozen)
                unique_branches.append(branch)
        n_branches = len(unique_branches)
        
        if n_branches < 2:
            if skip_single_branches:
                continue
            else:
                raise ValueError(f"Branch point {bp} leads to {n_branches} unique branches")

        # Get cells of branch point type
        bp_cells = np.where(labels_array == bp)[0]
        if len(bp_cells) < min_cells_per_branch * n_branches:
            warnings.warn(f"Skipping {bp}: insufficient cells ({len(bp_cells)})")
            continue

        # Map nodes to branches (-1 for non-branch nodes)
        branch_mapper = {}
        for bid, branch in enumerate(unique_branches):
            for node in branch:
                branch_mapper[node] = bid

        # Determine expected clusters through neighbor voting
        expected_labels = []
        valid_indices = []
        for cell_idx in bp_cells:
            # Get neighbors from precomputed KNN (cell_idx is row index)
            start = precomputed_embedded_connectivities.indptr[cell_idx]
            end = precomputed_embedded_connectivities.indptr[cell_idx+1]
            neighbors = precomputed_embedded_connectivities.indices[start:end]
            
            # Exclude self and non-branch cells
            votes = []
            for nbr_idx in neighbors:
                nbr_label = labels_array[nbr_idx]
                if nbr_label == bp:  # Ignore same-type neighbors
                    continue
                if nbr_label in branch_mapper:
                    votes.append(branch_mapper[nbr_label])
            
            if len(votes) == 0:
                continue  # No informative neighbors
            
            # Majority vote with random tie-break
            unique, counts = np.unique(votes, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            chosen = np.random.choice(candidates)  # Random tie resolution
            expected_labels.append(chosen)
            valid_indices.append(cell_idx)

        # Check cluster validity
        if len(expected_labels) < min_cells_per_branch * n_branches:
            warnings.warn(f"Skipping {bp}: insufficient valid cells")
            continue
        if len(np.unique(expected_labels)) < n_branches:
            warnings.warn(f"Skipping {bp}: incomplete branch coverage")
            continue

        # Extract connectivity subset for valid cells
        cell_indices = np.array(valid_indices)
        affinity = precomputed_embedded_connectivities[cell_indices][:, cell_indices]

        # Spectral clustering with automatic eigen-gap detection
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            clustering = SpectralClustering(
                n_clusters=n_branches,
                affinity='precomputed',
                n_init=spectral_n_init,
                random_state=0
            ).fit(affinity)

        # Calculate weighted F1-score
        f1 = f1_score(expected_labels, clustering.labels_, average='weighted')
        total_f1 += f1
        valid_branches += 1

    if valid_branches == 0:
        return np.nan
    
    final_score = total_f1 / valid_branches
    if validate_result:
        _validator(score=final_score)
    
    return final_score
