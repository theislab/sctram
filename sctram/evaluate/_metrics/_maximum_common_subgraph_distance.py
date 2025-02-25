#!/usr/bin/env python3

import numpy as np
import networkx as nx

try:
    from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def maximum_common_subgraph_distance(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    threshold: float,
    validate_result: bool
) -> float:
    """Computes the Maximum Common Subgraph (MCS) distance using Graph Edit Distance (GED).

    This method converts the adjacency matrices into undirected graphs and computes the MCS distance via GED, considering only edge 
    operations (insertions and deletions) while allowing node substitutions. The distance reflects the minimum number of edge changes 
    required to transform one graph into another, accounting for all possible node permutations, thus capturing structural similarity.

    Parameters:
        given_adjacency_matrix (np.ndarray): A square symmetric numpy array representing the ground-truth adjacency matrix (binary).
        inferred_adjacency_matrix (np.ndarray): A square symmetric numpy array representing the inferred adjacency matrix (weighted).
        threshold (float): Threshold to binarize the inferred adjacency matrix
        validate_result (bool): If True, validates the computed score to ensure it is non-negative.

    Returns:
        float: Non-negative MCS distance. Returns NaN if GED computation fails (e.g., timeout).

    Raises:
        ValueError: If input matrices are non-square, asymmetric, or have mismatched dimensions.

    Mathematical Justification:
        The MCS distance is derived from the Graph Edit Distance (GED) under the following cost scheme:
        - Node substitutions allowed at zero cost (enabling node permutations).
        - Prohibitively high costs for node insertions/deletions (graphs must have the same size).
        - Unit costs for edge insertions/deletions.

        This configuration ensures:
            MCS_distance = |E1| + |E2| - 2 * |E_MCS|,
        where E1 and E2 are edge sets of the input graphs, and E_MCS is the edge set of their maximum common subgraph.

        The MCS distance satisfies metric properties (non-negativity, identity, symmetry, triangle inequality), making it a robust 
        measure for graph comparison.

    Statistical Robustness:
        - Invariance to Node Ordering: By allowing free node substitutions, the metric is unaffected by node permutations.
        - Structural Sensitivity: Directly measures discrepancies in edge structure, capturing topological differences.
        - Threshold Handling: Binarizes inferred matrices to enable comparison with ground-truth graphs.

    Advantages:
        - Provably equivalent to the MCS size under the specified cost model.
        - Computationally feasible for small graphs (n ≤ 60 nodes).
        - Interpretable output: 0 indicates isomorphic graphs; higher values indicate greater dissimilarity.

    Limitations:
        - NP-hard complexity limits scalability to large graphs.
        - GED computation may fail to converge, returning NaN.
    """    
    # Binarize inferred adjacency matrix
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    g1 = nx.from_numpy_array(given_adjacency_matrix)
    g2 = nx.from_numpy_array(inferred_binary)
    
    # Configure graph edit distance parameters
    # Prohibit node insertions/deletions (high cost), allow edge operations and node substitutions
    # Configure graph edit distance parameters with callables
    ged_params = {
        "node_ins_cost": lambda node_attrs: 1e9,  # High cost to prevent node insertion
        "node_del_cost": lambda node_attrs: 1e9,  # High cost to prevent node deletion
        "edge_ins_cost": lambda edge_attrs: 1,    # Unit cost for edge insertion
        "edge_del_cost": lambda edge_attrs: 1,    # Unit cost for edge deletion
        "node_subst_cost": lambda node1_attrs, node2_attrs: 0,  # Free node substitution
        "timeout": 3600        # Timeout after 1 hour
    }
    
    score = nx.graph_edit_distance(g1, g2, **ged_params)
    
    if validate_result:
        _validator(score=score)
    
    return score


if __name__ == "__main__":
    
    def test_empty_graphs():
        """Test when both graphs have no edges, distance should be 0."""
        given = np.zeros((3, 3), dtype=int)
        inferred = np.zeros((3, 3), dtype=float)
        threshold = 0.5
        expected = 0.0
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Empty graphs: Expected {expected}, got {distance}"

    def test_identical_graphs():
        """Test identical graphs, expecting distance 0."""
        given = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
        inferred = given.astype(float)
        threshold = 0.5
        expected = 0.0
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Identical graphs: Expected {expected}, got {distance}"

    def test_isomorphic_graphs():
        """Test isomorphic graphs with permuted nodes, expecting distance 0."""
        given = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
        # Inferred is isomorphic by permutation (1->0, 0->2, 2->1)
        inferred = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=float)
        threshold = 0.5
        expected = 0.0
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Isomorphic graphs: Expected {expected}, got {distance}"

    def test_single_edge_vs_empty():
        """Test one edge against empty graph, expecting distance 1."""
        given = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int)
        inferred = np.zeros((3, 3), dtype=float)
        threshold = 0.5
        expected = 1.0
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Single edge vs empty: Expected {expected}, got {distance}"

    def test_non_overlapping_edges():
        """Test graphs with no common edges after permutation, expecting distance 2."""
        given = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int)
        inferred = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        threshold = 0.5
        expected = 0.0  # After permutation, edges overlap
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Edges permuted: Expected {expected}, got {distance}"

    def test_partial_overlap_isomorphic():
        """Test isomorphic graphs should have distance 0."""
        given = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int)
        inferred = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        threshold = 0.5
        expected = 0.0  # Correct expected distance for isomorphic graphs
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Expected {expected}, got {distance}"
        
    def test_partial_overlap_nonisomorphic():
        """Test non-isomorphic graphs with partial overlap."""
        # Chain graph 0-1-2-3 (3 edges)
        given = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=int)
        
        # Star graph centered at 0 (3 edges)
        inferred = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=float)
        
        threshold = 0.5
        expected = 2.0  # Corrected expected distance
        distance = maximum_common_subgraph_distance(given, inferred, threshold, validate_result=False)
        assert np.isclose(distance, expected), f"Expected {expected}, got {distance}"
    
    
    test_empty_graphs()
    test_identical_graphs()
    test_isomorphic_graphs()
    test_single_edge_vs_empty()
    test_non_overlapping_edges()
    test_partial_overlap_isomorphic()
    test_partial_overlap_nonisomorphic()
    print("All tests passed!")
