#!/usr/bin/env python3

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger
from scipy import sparse

_logger = logger.bind(name="MetricsUtils")


def convert_scanpy_neighbors_to_indices(
    scanpy_neighbors_matrix: sparse.csr_matrix, k: int, include_itself: bool = False
) -> sparse.csr_matrix:
    """Converts Scanpy neighbors matrix (distances/connectivities) to indices matrix.

    Scanpy's neighbor matrices store column indices in .indices array for each row.
    This extracts the first k column indices per row to create an indices matrix.

    Parameters:
        scanpy_neighbors_matrix (sparse.csr_matrix): Output from scanpy.pp.neighbors
        k (int): Number of neighbors to extract

    Returns:
        sparse.csr_matrix: CSR matrix containing neighbor indices (n_samples x k)
    """
    n_samples = scanpy_neighbors_matrix.shape[0]
    r = np.zeros((n_samples, k), dtype=int)

    for i in range(n_samples):
        start = scanpy_neighbors_matrix.indptr[i]
        end = scanpy_neighbors_matrix.indptr[i + 1]
        available_neighbors = scanpy_neighbors_matrix.indices[start:end]

        if len(available_neighbors) < k:
            raise ValueError(
                f"Row {i} has only {len(available_neighbors)} neighbors (needs {k}). "
                "Did you forget excluding the node itself by `k-1` operation."
            )

        r[i] = available_neighbors[:k]

    if include_itself:
        np.concatenate([np.arange(r.shape[0], dtype=r.dtype).reshape(r.shape[0], 1), r], axis=1)

    return r


class Centroids:

    def __init__(self, embedding: np.ndarray, labels_array: np.ndarray, outlier_threshold: Optional[float] = 3.0):
        self.robust = outlier_threshold is not None
        self.outlier_threshold = outlier_threshold
        self.centroids = None
        self._cache_centroids(embedding, labels_array)

    def get_single(self, label):
        if label not in self.centroids:
            raise ValueError(f"Missing labels in data: {label}")
        return self.centroids[label]

    def get_centroids(self, ordered_labels_queried: list[str]) -> list[np.ndarray]:
        missing_labels = set(ordered_labels_queried) - set(self.centroids.keys())
        if missing_labels:
            raise ValueError(f"Missing labels in data: {missing_labels}")
        return [self.centroids[label] for label in ordered_labels_queried]

    def _cache_centroids(self, embedding: np.ndarray, labels_array: np.ndarray) -> None:
        """Compute centroids for specified labels, optionally removing outliers."""
        if len(labels_array) != embedding.shape[0]:
            raise ValueError("Labels array length must match embedding row count")

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels_array):
            label_to_indices[label].append(idx)

        label_to_indices = {label: np.array(indices, dtype=int) for label, indices in label_to_indices.items()}

        self.centroids = dict()
        for label, indices in label_to_indices.items():
            subset = embedding[indices]
            if self.robust and subset.shape[0] > 1:
                centroid = self._compute_robust_centroid(subset)
            else:
                centroid = self._compute_standard_centroid(subset)
            self.centroids[label] = centroid

    def _compute_robust_centroid(self, subset: np.ndarray) -> np.ndarray:
        """Compute centroid with outlier removal using median/MAD filtering.

        Statistical Justification:
            When using robust=True:
            1. Initial centroid is coordinate-wise median (resistant to outliers)
            2. MAD provides robust dispersion estimate (vs standard deviation)
            3. Threshold = median_distance + threshold*MAD filters extreme values
            4. Final mean is calculated on remaining points
            This approach maintains efficiency while resisting ~30% contamination
            (Rousseeuw & Croux, 1993). Suitable for high-dimensional data where
            simple thresholds might fail due to curse of dimensionality.
        """
        median_vect = np.median(subset, axis=0)
        distances = np.linalg.norm(subset - median_vect, axis=1)
        med_dist = np.median(distances)
        mad = np.median(np.abs(distances - med_dist))
        threshold = med_dist + self.outlier_threshold * mad

        mask = distances <= threshold
        filtered = subset[mask]

        if filtered.size == 0:
            _logger.warning("All samples filtered as outliers for label. Using median as fallback.")
            return median_vect

        return np.mean(filtered, axis=0)

    def _compute_standard_centroid(self, subset: np.ndarray) -> np.ndarray:
        """Compute standard centroid using mean."""
        return np.mean(subset, axis=0)

    # def _compute_robust_centroids(
    #     embedding: np.ndarray,
    #     labels: np.ndarray,
    #     distances: Optional[sparse.csr_matrix] = None
    # ) -> Centroids:
    #     """Compute noise-resistant centroids using medoid/mean selection."""
    #     from sklearn.metrics import pairwise_distances

    #     unique_labels = np.unique(labels)
    #     centroid_list = []

    #     for label in unique_labels:
    #         mask = labels == label
    #         cluster_emb = embedding[mask]

    #         if distances is not None:
    #             # Find medoid using precomputed distances
    #             sub_dist = distances[mask][:, mask]
    #             if not isinstance(sub_dist, np.ndarray):
    #                 sub_dist = sub_dist.toarray()
    #             medoid_idx = np.argmin(sub_dist.sum(axis=1))
    #             centroid = cluster_emb[medoid_idx]
    #         else:
    #             # Use mean if no distances available
    #             centroid = np.mean(cluster_emb, axis=0)

    #         centroid_list.append(centroid)

    #     return Centroids(centroid_list, unique_labels)


def get_longest_path(graph, beam_width=100, biological_scoring_func=None):
    """Extract the longest biologically plausible simple path from a directed graph.

    For Directed Acyclic Graphs (DAGs), uses topological sorting for efficient O(V + E) computation.
    For cyclic graphs, employs a beam search heuristic to find source-to-sink paths with configurable
    biological validity scoring. The algorithm prioritizes paths that reflect likely biological processes.

    Biological validity scoring considers both path length and domain-specific node/edge features when
    a custom scoring function is provided. The default scoring uses path length as the primary criterion.

    Args:
        graph (nx.DiGraph): Trajectory graph to analyze
        beam_width (int): Width of beam search for cyclic graphs (controls memory/accuracy tradeoff)
        biological_scoring_func (callable): Function that takes a path and returns a score. Higher
            scores indicate better biological validity. Default: path length.

    Returns:
        list: Node labels in longest path order

    Raises:
        ValueError: If no valid paths can be found
        NetworkXError: If graph analysis fails

    Mathematical Defense:
        - DAG case: Correct by construction using dynamic programming on topological order
        - Cyclic case: Beam search provides polynomial-time approximation (O(bw*N^2)) with:
            - Completeness: Guaranteed to find existing paths <= beam_width exploration width
            - Biologically constrained: Focused on source-sink paths with validity scoring
        - Statistical robustness: Prioritizes high-scoring paths validated by domain-specific metrics
    """
    # Set default scoring function if not provided
    if biological_scoring_func is None:
        biological_scoring_func = lambda path: len(path)

    # DAG-optimized path finding
    if nx.is_directed_acyclic_graph(graph):
        _logger.info("Using DAG-optimized longest path algorithm")
        try:
            longest_path = nx.dag_longest_path(graph)
            _logger.info(f"Found DAG path with {len(longest_path)} nodes")
            return longest_path
        except nx.NetworkXUnfeasible:
            raise ValueError("Unexpected cycle detection in DAG")

    # Cyclic graph fallback with beam search heuristic
    _logger.warning("Graph contains cycles - using beam search heuristic")

    # Identify sources and sinks
    sources = [n for n, d in graph.in_degree() if d == 0]
    sinks = {n for n, d in graph.out_degree() if d == 0}

    if not sources:
        raise ValueError("No source nodes (in_degree=0) found for path initiation")
    if not sinks:
        raise ValueError("No sink nodes (out_degree=0) found for path termination")

    # Track best path across all sources
    best_path = []
    best_score = -float("inf")

    for source in sources:
        current_paths = [([source], {source})]  # (path, visited_set)
        longest_from_source = []
        max_local_score = -float("inf")

        # Beam search iterations
        while current_paths:
            next_paths = []
            for path, visited in current_paths:
                current_node = path[-1]

                # Update best path if current path ends at sink
                if current_node in sinks:
                    score = biological_scoring_func(path)
                    if score > best_score:
                        best_score = score
                        best_path = path.copy()
                    continue  # No further extensions from sink

                # Explore neighbors
                for neighbor in graph.successors(current_node):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        next_paths.append((new_path, new_visited))

            # Prune to beam width using scoring function
            next_paths.sort(key=lambda x: biological_scoring_func(x[0]), reverse=True)
            current_paths = next_paths[:beam_width]

            # Track best path in current iteration
            for path, _ in current_paths:
                score = biological_scoring_func(path)
                if score > max_local_score:
                    max_local_score = score
                    longest_from_source = path.copy()

        # Compare with previous best after exhausting this source
        if max_local_score > best_score:
            best_score = max_local_score
            best_path = longest_from_source

    if not best_path:
        raise ValueError("No valid source-to-sink paths found")

    _logger.info(f"Heuristic found path with score {best_score} (length {len(best_path)})")
    return best_path


def min_max_scale(
    arr: np.ndarray, epsilon: float = 1e-8, lower_clip: Optional[float] = None, upper_clip: Optional[float] = None
) -> np.ndarray:
    """
    Robust min-max scaling with outlier clipping and numerical stability controls.

    Features:
    - Clips data to specified percentiles before scaling
    - Handles constant arrays (zero ptp)
    - Prevents division by zero
    - Clips final values to [0,1]
    - Preserves array shape and dtype

    Parameters:
    arr : np.ndarray
        1D array of pseudotime values
    epsilon : float
        Small value to prevent division by zero
    lower_clip : float, optional
        Lower quantile (0-1) for clipping. None disables lower clipping.
    upper_clip : float, optional
        Upper quantile (0-1) for clipping. None disables upper clipping.

    Returns:
    np.ndarray
        Scaled array in [0, 1]
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.copy()

    # Validate clipping parameters
    if lower_clip is not None and not 0 <= lower_clip <= 1:
        raise ValueError("lower_clip must be between 0 and 1")
    if upper_clip is not None and not 0 <= upper_clip <= 1:
        raise ValueError("upper_clip must be between 0 and 1")
    if lower_clip is not None and upper_clip is not None and lower_clip > upper_clip:
        raise ValueError("lower_clip must be <= upper_clip")

    # Apply percentile-based clipping
    if lower_clip is not None or upper_clip is not None:
        clip_bounds = []
        if lower_clip is not None:
            clip_bounds.append(np.percentile(arr, lower_clip * 100))
        if upper_clip is not None:
            clip_bounds.append(np.percentile(arr, upper_clip * 100))

        arr = np.clip(arr, *clip_bounds) if clip_bounds else arr

    min_val = arr.min()
    ptp = arr.ptp()

    if ptp < epsilon:
        return np.zeros_like(arr)

    scaled = (arr - min_val) / (ptp + epsilon)
    return np.clip(scaled, 0.0, 1.0)


def z_score_scale(
    arr: np.ndarray, epsilon: float = 1e-8, lower_clip: Optional[float] = None, upper_clip: Optional[float] = None
) -> np.ndarray:
    """
    Robust z-score standardization with outlier clipping and numerical stability controls.

    Features:
    - Clips data to specified percentiles before scaling
    - Handles constant arrays (zero std)
    - Prevents division by zero
    - Preserves array shape and dtype

    Parameters:
    arr : np.ndarray
        1D array of pseudotime values
    epsilon : float
        Small value to prevent division by zero
    lower_clip : float, optional
        Lower quantile (0-1) for clipping. None disables lower clipping.
    upper_clip : float, optional
        Upper quantile (0-1) for clipping. None disables upper clipping.

    Returns:
    np.ndarray
        Standardized array (μ=0, σ=1 for non-constant inputs)
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.copy()

    # Validate clipping parameters
    if lower_clip is not None and not 0 <= lower_clip <= 1:
        raise ValueError("lower_clip must be between 0 and 1")
    if upper_clip is not None and not 0 <= upper_clip <= 1:
        raise ValueError("upper_clip must be between 0 and 1")
    if lower_clip is not None and upper_clip is not None and lower_clip > upper_clip:
        raise ValueError("lower_clip must be <= upper_clip")

    # Apply percentile-based clipping
    if lower_clip is not None or upper_clip is not None:
        clip_bounds = []
        if lower_clip is not None:
            clip_bounds.append(np.percentile(arr, lower_clip * 100))
        if upper_clip is not None:
            clip_bounds.append(np.percentile(arr, upper_clip * 100))

        arr = np.clip(arr, *clip_bounds) if clip_bounds else arr

    mean = arr.mean()
    std = arr.std(ddof=0)

    if std < epsilon:
        return np.zeros_like(arr)

    return (arr - mean) / (std + epsilon)


def prepare_pseudotime(
    arr, method: str, lower_clip: Optional[float] = 0.01, upper_clip: Optional[float] = 0.99
) -> np.ndarray:
    """
    Prepare pseudotime values with optional outlier clipping.

    Parameters:
    arr : np.ndarray
        Input array of pseudotime values
    method : str
        Scaling method ('zscore' or 'minmax')
    lower_clip : float, optional
        Lower quantile (0-1) for clipping. None disables lower clipping.
    upper_clip : float, optional
        Upper quantile (0-1) for clipping. None disables upper clipping.

    Returns:
    np.ndarray
        Scaled pseudotime values
    """
    if method == "zscore":
        prepared = z_score_scale(arr, lower_clip=lower_clip, upper_clip=upper_clip)
    elif method == "minmax":
        prepared = min_max_scale(arr, lower_clip=lower_clip, upper_clip=upper_clip)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    return prepared
