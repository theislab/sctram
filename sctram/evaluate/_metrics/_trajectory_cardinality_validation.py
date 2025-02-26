#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
from scipy import sparse
import warnings
from typing import Optional, Dict, Set, List
from loguru import logger
from collections import defaultdict

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
    from sctram.evaluate._metrics.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    from utils import convert_scanpy_neighbors_to_indices


_logger = logger.bind(name="TrajectoryCardinalityValidator")


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    k_neighbors: int,
    validate_result: bool = True,
    skip_single_branches: bool = True,
    spectral_n_init: int = 20,
    min_cells_per_branch: int = 10,
    random_state: Optional[int] = None,
    exclude_self_from_neighbors: bool = True
) -> float:
    """Advanced Directed Trajectory Validation using Spectral Consensus (D-TVS).

    Evaluates how well an embedding preserves trajectory structure by comparing
    spectral clusters with graph-derived ground truth through:
    1. Graph-theoretic identification of biological bifurcations
    2. Directed reachability analysis for ground truth branch definitions
    3. Spectral validation using manifold learning on KNN graph

    Parameters:
        given_graph (nx.DiGraph): Directed trajectory graph (cell type transitions)
        labels_array (np.ndarray): Cell type labels (n_cells,)
        precomputed_embedded_connectivities (sparse.csr_matrix): Scanpy KNN connectivities
        k_neighbors (int): Number of neighbors used in KNN graph construction
        validate_result (bool): Validate output score is in [0,1]
        skip_single_branches (bool): Skip branch points with <2 valid branches
        spectral_n_init (int): Spectral clustering initializations
        min_cells_per_branch (int): Minimum cells required per branch
        random_state (int): Random seed for reproducibility
        exclude_self_from_neighbors (bool): Exclude self in neighbor indices

    Returns:
        float: Weighted F1 score (0-1) averaged over valid branch points

    Statistical Rationale:
        1. Directional Fidelity: Utilizes directed edges to identify true bifurcations
        2. Ground Truth: Branch membership defined via DAG reachability (biologically valid)
        3. Spectral Validation: Combines manifold learning with graph structure
        4. Noise Handling: Neighbor voting with exclusion of same-type cells reduces local noise
        5. Consensus Approach: Majority voting with deterministic tie-breaking ensures reproducibility

    Complexity Analysis:
        - Branch identification: O(m) for m edges
        - Neighbor processing: O(nk) for n cells and k neighbors
        - Spectral clustering: O(c³) per branch point (c = cells per branch)
    """
    # Convert KNN connectivities to neighbor indices matrix
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=k_neighbors - 1,
        include_itself=not exclude_self_from_neighbors
    )

    # Identify branch points (nodes with out-degree ≥2)
    branch_points = [node for node in given_graph.nodes 
                    if given_graph.out_degree(node) >= 2]
    if not branch_points:
        _logger.warning("No biological branch points found")
        return np.nan
    
    total_f1 = 0.0
    valid_branches = 0
    node_to_label = {i: lbl for i, lbl in enumerate(np.unique(labels_array))}
    label_to_nodes = defaultdict(set)
    for idx, lbl in enumerate(labels_array):
        label_to_nodes[lbl].add(idx)

    for bp in branch_points:
        _logger.debug(f"Processing branch point: {bp}")

        # Get biological branches via DAG reachability
        children = list(given_graph.successors(bp))
        if len(children) < 2:
            continue

        # Collect unique branches using directed descendants
        branch_sets = []
        for child in children:
            descendants = set()
            stack = [child]
            visited = set()
            while stack:
                node = stack.pop()
                if node in visited or node == bp:
                    continue
                visited.add(node)
                descendants.add(node)
                # Follow successors not leading back to branch point
                for succ in given_graph.successors(node):
                    if succ != bp and succ not in visited:
                        stack.append(succ)
            if descendants:
                branch_sets.append(descendants)

        # Remove duplicate branches
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
                _logger.debug(f"Skipping {bp}: only {n_branches} unique branches")
                continue
            else:
                raise ValueError(f"Branch {bp} has {n_branches} unique branches")

        # Create branch label mapping
        branch_mapper: Dict[str, int] = {}
        for bid, branch in enumerate(unique_branches):
            for node in branch:
                if node in branch_mapper:
                    _logger.warning(f"Node {node} appears in multiple branches")
                branch_mapper[node] = bid

        # Get cells of branch point type
        bp_cell_indices = label_to_nodes.get(bp, set())
        if not bp_cell_indices:
            _logger.warning(f"No cells found for branch point {bp}")
            continue
        bp_cells = np.array(sorted(bp_cell_indices))
        if len(bp_cells) < min_cells_per_branch * n_branches:
            _logger.warning(f"Skipping {bp}: insufficient cells")
            continue

        # Neighbor voting to determine expected clusters
        expected_labels = []
        valid_indices = []
        for cell_idx in bp_cells:
            neighbors = embedded_neighbors[cell_idx]
            votes = []
            for nbr_idx in neighbors:
                nbr_label = labels_array[nbr_idx]
                if nbr_label == bp:
                    continue  # Exclude same-type neighbors
                if nbr_label in branch_mapper:
                    votes.append(branch_mapper[nbr_label])
            
            if not votes:
                continue  # No informative neighbors
            
            # Deterministic majority vote
            unique, counts = np.unique(votes, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            chosen = min(candidates)  # Deterministic tie-break
            expected_labels.append(chosen)
            valid_indices.append(cell_idx)

        # Check validity after voting
        valid_indices = np.array(valid_indices)
        if len(valid_indices) < min_cells_per_branch * n_branches:
            _logger.warning(f"Skipping {bp}: insufficient valid cells")
            continue
        unique_expected = np.unique(expected_labels)
        if len(unique_expected) < n_branches:
            _logger.warning(f"Skipping {bp}: incomplete branch coverage")
            continue

        # Subset affinity matrix for valid cells
        cell_mask = np.isin(bp_cells, valid_indices)
        subset_cells = bp_cells[cell_mask]
        affinity_subset = precomputed_embedded_connectivities[subset_cells][:, subset_cells]

        # Spectral clustering with validation
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clustering = SpectralClustering(
                    n_clusters=n_branches,
                    affinity='precomputed',
                    n_init=spectral_n_init,
                    random_state=random_state
                ).fit(affinity_subset)
            cluster_labels = clustering.labels_
        except Exception as e:
            _logger.error(f"Spectral clustering failed for {bp}: {str(e)}")
            continue

        # Calculate weighted F1 score
        f1 = f1_score(expected_labels, cluster_labels, average='weighted')
        total_f1 += f1
        valid_branches += 1
        _logger.debug(f"Branch {bp} F1: {f1:.3f}")

    if valid_branches == 0:
        _logger.warning("No valid branches processed")
        return np.nan
    
    final_score = total_f1 / valid_branches
    if validate_result:
        _validator(score=final_score)
    
    return final_score
