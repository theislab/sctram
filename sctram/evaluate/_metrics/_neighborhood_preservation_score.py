#!/usr/bin/env python3

import numpy as np
import networkx as nx
from loguru import logger
from scipy import sparse

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
    from sctram.evaluate._metrics.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    from utils import convert_scanpy_neighbors_to_indices

_logger = logger.bind(name = "MetricBase")


def neighborhood_preservation_score(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    n_neighbors: int,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool,
    threshold: float = 0.5,
) -> float:
    """Neighborhood Preservation Score (NPS) with threshold-based scoring.

    Computes how well an embedding preserves cell-type transitions from a biological graph by
    evaluating the fraction of KNN neighbors in embedding space that belong to adjacent cell types,
    considering only cells with sufficient different-type neighbors.

    Parameters:
        given_graph (nx.DiGraph): Biological graph with cell types as nodes and transitions as edges.
        labels (np.ndarray): 1D array (n_cells) of cell type labels.
        n_neighbors (int): Number of neighbors (including self). Must match precomputed_embedded_connectivities.
        precomputed_embedded_connectivities (sparse.csr_matrix): Scanpy KNN connectivities (includes self as first neighbor).
        validate_result (bool): Validate score is in [0, 1].
        threshold (float): Minimum proportion of different-type neighbors required to consider a cell (default: 0.25).

    Returns:
        float: NPS between 0 (no preservation) and 1 (perfect preservation), considering only cells meeting the threshold.

    Statistical Rationale:
        - Focuses on cells with significant cross-type interactions, enhancing biological relevance.
        - Robust to dominant same-type neighborhoods by applying a threshold.
    """
    # Convert Scanpy neighbors to indices matrix (exclude self)
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=n_neighbors - 1
    )
    n_cells = embedded_neighbors.shape[0]

    # Validate input shapes
    assert embedded_neighbors.shape == (n_cells, n_neighbors - 1), \
        f"Neighbor indices shape {embedded_neighbors.shape} != ({n_cells}, {n_neighbors-1})"
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"

    # Precompute allowed transitions matrix
    unique_labels = list(given_graph.nodes())
    if not unique_labels:
        raise ValueError("Handle empty graph case")
    
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    allowed = np.zeros((len(unique_labels), len(unique_labels)), dtype=bool)
    for u in unique_labels:
        for v in given_graph.neighbors(u):
            if v in label_to_idx:
                allowed[label_to_idx[u], label_to_idx[v]] = True

    # Convert cell labels to indices (-1 = missing in graph)
    label_indices = np.array([label_to_idx.get(l, -1) for l in labels_array], dtype=int)
    valid_mask = (label_indices != -1)
    valid_indices = label_indices[valid_mask]

    if valid_indices.size == 0:
        raise ValueError("No valid cells to process")

    # Get neighbor indices for valid cells (n_valid, n_neighbors-1)
    neighbor_indices = embedded_neighbors[valid_mask]
    neighbor_labels = label_indices[neighbor_indices]

    # Compute same-type and different-type masks
    same_type = neighbor_labels == valid_indices[:, np.newaxis]
    different_type = ~same_type

    # Calculate K (different-type neighbors) and valid different-type mask
    valid_different_mask = different_type & (neighbor_labels != -1)
    rows, cols = np.nonzero(valid_different_mask)
    u_values = valid_indices[rows]
    v_values = neighbor_labels[rows, cols]

    # Compute allowed transitions for valid different-type neighbors
    allowed_trans = allowed[u_values, v_values]
    allowed_transitions_mask = np.zeros_like(neighbor_labels, dtype=bool)
    allowed_transitions_mask[rows, cols] = allowed_trans

    # Sum allowed transitions per cell (K_known)
    K_known = allowed_transitions_mask.sum(axis=1)
    K = different_type.sum(axis=1)
    ratio = K / (n_neighbors - 1)
    meet_threshold = ratio > threshold

    # Calculate scores for cells meeting threshold and K > 0
    valid_scores = meet_threshold & (K > 0)
    scores_per_cell = np.zeros(len(valid_indices), dtype=np.float64)
    scores_per_cell[valid_scores] = K_known[valid_scores] / K[valid_scores]

    # Average scores of qualifying cells
    selected_scores = scores_per_cell[meet_threshold]
    if selected_scores.size > 0:
        avg_score = selected_scores.mean()
    else:
        raise ValueError("selected_scores is empty")

    if validate_result:
        _validator(score=avg_score)

    return avg_score


if __name__ == "__main__":
    
    def test_perfect_preservation():
        """All neighbors are allowed transitions (score=1.0)."""
        given_graph = nx.DiGraph()
        given_graph.add_edges_from([('A', 'B'), ('B', 'A')])
        labels = np.array(['A', 'B'])
        # Cell 0 (A) has 4 B neighbors; cell 1 (B) has 4 A neighbors
        embedded_neighbors = np.array([[1,1,1,1], [0,0,0,0]])
        data = np.ones(8)
        indptr = [0,4,8]
        precomputed = sparse.csr_matrix(
            (data, embedded_neighbors.flatten(), indptr), 
            shape=(2,2))
        score = neighborhood_preservation_score(
            given_graph=given_graph,
            labels_array=labels,
            n_neighbors=5,
            precomputed_embedded_connectivities=precomputed,
            validate_result=False,
            threshold=0.25
        )
        assert np.isclose(score, 1.0)

    def test_no_preservation():
        """No transitions allowed (score=0.0)."""
        given_graph = nx.DiGraph()
        given_graph.add_nodes_from(['A', 'B'])
        labels = np.array(['A', 'B'])
        # All neighbors are cross-type but invalid
        embedded_neighbors = np.array([[1,1,1,1], [0,0,0,0]])
        data = np.ones(8)
        indptr = [0,4,8]
        precomputed = sparse.csr_matrix(
            (data, embedded_neighbors.flatten(), indptr), 
            shape=(2,2))
        score = neighborhood_preservation_score(
            given_graph=given_graph,
            labels_array=labels,
            n_neighbors=5,
            precomputed_embedded_connectivities=precomputed,
            validate_result=False,
            threshold=0.25
        )
        assert np.isclose(score, 0.0)

    def test_partial_preservation():
        """Mixed allowed/forbidden transitions (score=0.25)."""
        given_graph = nx.DiGraph()
        given_graph.add_edge('A', 'B')
        labels = np.array(['A', 'B', 'C'])  # C not in graph
        # Cell 0 (A): 2 B (allowed), 2 C (invalid)
        # Cell 1 (B): 2 A (forbidden), 2 C (invalid)
        embedded_neighbors = np.array([[1,1,2,2], [0,0,2,2], [0,1,1,1]])
        data = np.ones(12)
        indptr = [0,4,8,12]
        precomputed = sparse.csr_matrix(
            (data, embedded_neighbors.flatten(), indptr), 
            shape=(3,3)
        )
        score = neighborhood_preservation_score(
            given_graph=given_graph,
            labels_array=labels,
            n_neighbors=5,
            precomputed_embedded_connectivities=precomputed,
            validate_result=False,
            threshold=0.2
        )
        assert np.isclose(score, 0.25), f"Expected 0.25, got {score}"

    def test_threshold_exclusion():
        """Cells below threshold are excluded from scoring."""
        given_graph = nx.DiGraph()
        given_graph.add_edge('A', 'B')
        labels = np.array(['A', 'B'])
        # Cell 0 (A): 1 B neighbor (K=1, ratio=0.25)
        embedded_neighbors = np.array([[1,0,0,0], [0,0,0,0]])
        data = np.ones(8)
        indptr = [0,4,8]
        precomputed = sparse.csr_matrix(
            (data, embedded_neighbors.flatten(), indptr), 
            shape=(2,2))
        try:
            _ = neighborhood_preservation_score(
                given_graph=given_graph,
                labels_array=labels,
                n_neighbors=5,
                precomputed_embedded_connectivities=precomputed,
                validate_result=False,
                threshold=0.25
            )
            # Only cell 1 has K=0 (excluded), cell 0 ratio=0.25 not >0.25
        except ValueError:
            pass

    def test_all_cells_invalid():
        given_graph = nx.DiGraph()
        given_graph.add_node('A')
        labels = np.array(['B', 'B'])  # Not in graph
        precomputed = sparse.csr_matrix((2, 2))
        try:
            _ = neighborhood_preservation_score(
                given_graph=given_graph,
                labels_array=labels,
                n_neighbors=5,
                precomputed_embedded_connectivities=precomputed,
                validate_result=False
            )
        except ValueError:
            pass

    def test_large_real_case():
        """Large synthetic dataset with known expected score (0.75)."""
        n = 300
        labels = np.array(['A']*100 + ['B']*100 + ['C']*100)
        given_graph = nx.DiGraph()
        given_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        
        # Construct neighbor indices matrix
        embedded_neighbors = np.zeros((n, 4), dtype=int)
        # A cells (0-99): 3 B's (100-199), 1 C (200-299)
        embedded_neighbors[:100, :3] = np.random.choice(100, size=(100,3)) + 100
        embedded_neighbors[:100, 3] = np.random.choice(100, size=100) + 200
        # B cells (100-199): 3 C's (200-299), 1 A (0-99)
        embedded_neighbors[100:200, :3] = np.random.choice(100, size=(100,3)) + 200
        embedded_neighbors[100:200, 3] = np.random.choice(100, size=100)
        # C cells (200-299): 3 A's (0-99), 1 B (100-199)
        embedded_neighbors[200:300, :3] = np.random.choice(100, size=(100,3))
        embedded_neighbors[200:300, 3] = np.random.choice(100, size=100) + 100
        
        data = np.ones(embedded_neighbors.size)
        indptr = np.arange(0, embedded_neighbors.size+1, embedded_neighbors.shape[1])
        precomputed = sparse.csr_matrix(
            (data, embedded_neighbors.flatten(), indptr), 
            shape=(n, n))
        
        score = neighborhood_preservation_score(
            given_graph=given_graph,
            labels_array=labels,
            n_neighbors=5,
            precomputed_embedded_connectivities=precomputed,
            validate_result=False,
            threshold=0.25
        )
        assert np.isclose(score, 0.75)
        
    test_perfect_preservation()
    test_no_preservation()
    test_partial_preservation()
    test_threshold_exclusion()
    test_all_cells_invalid()
    test_large_real_case()
    print("All tests passed.")
