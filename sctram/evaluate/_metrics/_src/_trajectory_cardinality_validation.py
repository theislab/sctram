#!/usr/bin/env python3

import warnings
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score

try:
    from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator

_logger = logger.bind(name="MetricsBase")


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool = True,
    skip_single_branches: bool = False,
    spectral_n_init: int = 50,
    min_cells_per_branch: int = 15,
    random_state: Optional[int] = None,
) -> float:
    """Computes the Directed Trajectory Validation Score (DTVS) between a cell embedding and a trajectory graph.

    DTVS (Directed Trajectory Validation Score) is a robust metric designed to quantitatively evaluate how
    well a cell embedding (e.g., from single-cell genomics data) aligns with a biological trajectory graph.
    It focuses on validating whether the local structure of the embedding reflects the branching patterns defined
    in the trajectory graph, which represents hypothesized developmental pathways.

    This advanced metric evaluates topological consistency by comparing spectral clustering results in the embedding
    with biological branches derived from neighbor voting. Designed for single-cell genomics data, it handles:
    - Complex branching trajectories
    - High-dimensional embeddings
    - Sparse biological graphs

    Statistical Rationale:
    1. Local Consistency: Assumes cells near branch points should have neighbors predictive of downstream branches
    2. Manifold Alignment: Spectral clustering captures non-linear structures in the embedding
    3. Weighted Scoring: Uses F1-score to handle class imbalance between branches

    Args:
        given_graph: Biological trajectory as directed acyclic graph (DAG)
        labels_array: Cell-type labels for each cell
        precomputed_embedded_connectivities: Scanpy neighbors connectivities matrix
        validate_result: Ensure output is in [0,1]
        skip_single_branches: Skip branch points with <2 branches
        spectral_n_init: Spectral clustering initializations for stability
        min_cells_per_branch: Minimum cells required per branch
        random_state: Random seed for reproducibility

    Returns:
        float: DTVS score between 0 (no alignment) and 1 (perfect alignment)

    Raises:
        ValueError: On invalid input dimensions or empty graph
    """
    # Input validation
    n_cells = labels_array.shape[0]
    if precomputed_embedded_connectivities.shape != (n_cells, n_cells):
        raise ValueError("Connectivities matrix must be n_cells x n_cells")

    if not given_graph.nodes:
        raise ValueError("Input graph contains no nodes")

    # Precompute label indices
    label_indices = defaultdict(set)
    for idx, lbl in enumerate(labels_array):
        label_indices[lbl].add(idx)

    # Identify valid branch points
    branch_points = [node for node in given_graph.nodes if given_graph.out_degree(node) >= 2]
    if not branch_points:
        raise ValueError("No biological branch points found")

    total_score = 0.0
    processed_branches = 0

    for bp in branch_points:
        # Get all immediate children
        children = list(given_graph.successors(bp))
        if len(children) < 2:
            _logger.debug(f"Skipping {bp}: insufficient children ({len(children)})")
            continue

        # Extract unique biological branches
        branch_sets = []
        visited_global = set()
        for child in children:
            if child in visited_global:
                _logger.debug(f"Child {child} already processed")
                continue
            # Collect all descendants not revisiting branch point
            descendants = set(nx.descendants(given_graph, child))
            descendants.add(child)
            # Exclude nodes from other branches and branch point
            clean_desc = descendants - {bp}
            if not clean_desc:
                _logger.debug(f"No descendants remaining after cleaning for child {child}")
                continue
            branch_sets.append(clean_desc)
            visited_global.update(clean_desc)

        # Remove duplicate branches using containment checks
        unique_branches = []
        for branch in branch_sets:
            is_unique = True
            for existing in unique_branches:
                if branch.issubset(existing) or existing.issubset(branch):
                    is_unique = False
                    break
            if is_unique:
                unique_branches.append(branch)

        n_branches = len(unique_branches)
        if n_branches < 2:
            if skip_single_branches:
                _logger.debug(f"Skipping {bp}: {n_branches} unique branches")
                continue
            else:
                raise ValueError(f"Branch point {bp} has {n_branches} branches")

        # Create branch membership mapping
        branch_map = {}
        for bid, nodes in enumerate(unique_branches):
            for node in nodes:
                if node in branch_map:
                    _logger.warning(f"Node {node} in multiple branches")
                branch_map[node] = bid

        # Get cells for current branch point
        bp_cells = np.array(sorted(label_indices.get(bp, [])))
        if len(bp_cells) < min_cells_per_branch * n_branches:
            _logger.debug(f"Skipping {bp}: insufficient cells")
            continue

        # Neighbor voting process
        expected_labels = []
        valid_cells = []
        for cell_idx in bp_cells:
            # Extract neighbors from precomputed CSR
            start = precomputed_embedded_connectivities.indptr[cell_idx]
            end = precomputed_embedded_connectivities.indptr[cell_idx + 1]
            neighbors = precomputed_embedded_connectivities.indices[start:end]

            # Collect votes from neighbors in downstream branches
            votes = []
            for nbr in neighbors:
                nbr_label = labels_array[nbr]
                if nbr_label == bp:
                    continue  # Ignore same-type neighbors
                if nbr_label in branch_map:
                    votes.append(branch_map[nbr_label])

            if not votes:
                continue  # No informative neighbors

            # Stable voting with sorted tie-break
            unique, counts = np.unique(votes, return_counts=True)
            max_votes = counts.max()
            candidates = unique[counts == max_votes]
            chosen_branch = sorted(candidates)[0]  # Deterministic tie-break

            expected_labels.append(chosen_branch)
            valid_cells.append(cell_idx)

        # Check minimum cell requirements
        if not valid_cells:
            _logger.debug(f"Skipping {bp}: 0 valid cells after voting")
            continue

        # Convert to numpy arrays for processing
        expected_labels_np = np.array(expected_labels)
        valid_cells_np = np.array(valid_cells)

        # Filter out underpopulated branches
        unique_labels, counts = np.unique(expected_labels_np, return_counts=True)
        sufficient_mask = counts >= min_cells_per_branch
        sufficient_labels = unique_labels[sufficient_mask]

        # Create mask for cells in sufficiently populated branches
        cell_mask = np.isin(expected_labels_np, sufficient_labels)
        filtered_expected = expected_labels_np[cell_mask]
        filtered_valid_cells = valid_cells_np[cell_mask]

        # Check after filtering
        if len(filtered_valid_cells) == 0:
            _logger.debug(f"Skipping {bp}: 0 cells remaining after branch filtering")
            continue

        filtered_unique, filtered_counts = np.unique(filtered_expected, return_counts=True)
        n_filtered_branches = len(filtered_unique)

        # Determine if penalty should be applied
        apply_penalty = False
        if n_filtered_branches != n_branches:
            _logger.debug(f"Branch {bp}: Expected {n_branches} branches, found {n_filtered_branches} after filtering")
            apply_penalty = True
        if n_filtered_branches < 2:
            _logger.debug(f"Branch {bp}: Insufficient branches ({n_filtered_branches}) after filtering")
            apply_penalty = True

        # Spectral clustering on filtered cells
        try:
            subset_indices = filtered_valid_cells
            affinity = precomputed_embedded_connectivities[subset_indices, :][:, subset_indices]

            # Ensure connectivity
            if affinity.nnz == 0:
                raise ValueError("Affinity matrix is completely disconnected")

            # Perform clustering
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clusterer = SpectralClustering(
                    n_clusters=n_branches, affinity="precomputed", n_init=spectral_n_init, random_state=random_state
                )
                cluster_labels = clusterer.fit_predict(affinity.toarray())

            # Calculate alignment score
            base_f1 = f1_score(filtered_expected, cluster_labels, average="weighted")

            if apply_penalty:
                # Apply a harmonic penalty if the full branching structure wasn't met
                detected_branches = n_filtered_branches
                penalty = (2 * detected_branches) / (n_branches + detected_branches)  # harmonic penalty
                raw_f1 = base_f1 * penalty
                f1 = max(0.0, min(1.0, raw_f1))
                _logger.debug(f"Applied penalty {penalty:.2f} to branch {bp}")
            else:
                f1 = base_f1

            total_score += f1
            processed_branches += 1
            _logger.debug(f"Processed {bp}: F1={f1:.3f} (Cells: {len(filtered_valid_cells)})")

        except Exception as e:
            _logger.error(f"Clustering failed for {bp}: {str(e)}")
            continue

    if processed_branches == 0:
        raise ValueError("No valid branches processed")

    final_score = total_score / processed_branches
    if validate_result:
        _validator(final_score)

    return final_score
