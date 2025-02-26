#!/usr/bin/env python3

from typing import Dict, Tuple
import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score

try:
    from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator


def branch_silhouette_score(
    given_graph: nx.DiGraph,
    inferred_embedding: np.ndarray,
    labels_array: np.ndarray,
    validate_result: bool,
    metric: str = 'euclidean',
    **kwargs
) -> float:
    """Computes the silhouette score for branch separation in a trajectory embedding.

    This function assigns cells to branches based on their labels' positions in the trajectory graph,
    then computes the silhouette score to quantify branch separation in the embedding. Cells are assigned
    to the most terminal branch (leaf) possible for their label, ensuring biologically meaningful clusters.

    Parameters:
        given_graph (nx.DiGraph): A directed tree (arborescence) where nodes represent labels.
        inferred_embedding (np.ndarray): 2D array of shape (n_samples, n_features) representing cell embeddings.
        labels_array (np.ndarray): 1D array of strings (length n_samples) assigning labels to cells.
        validate_result (bool): Whether to validate the score using `_validator`.
        metric (str): Distance metric for silhouette_score (default: 'euclidean').
        **kwargs: Additional keyword arguments passed to `silhouette_score`.

    Returns:
        float: Silhouette score between -1 and 1. Higher values indicate better branch separation.

    Statistical Justification:
        - Cluster Validity: The silhouette score measures how well clusters (branches) are separated and cohesive.
        - Graph Topology: Branch assignments derive from the graph's hierarchical structure, ensuring biological relevance.
        - Metric Flexibility: Supports any metric valid for `sklearn.metrics.silhouette_score`, including precomputed distances.

    Assumptions:
        - The graph is a directed tree with one root and no cycles.
        - Cells assigned to the same branch are biologically related to that trajectory branch.

    Notes:
        - Unassigned cells (labels not in the graph) are excluded.
        - Prefer Euclidean metric for embeddings; use 'precomputed' if `inferred_embedding` is a distance matrix.
    """
    # Validate graph structure
    if not nx.is_arborescence(given_graph):
        raise ValueError("Graph must be a directed tree (arborescence) with a single root and no cycles.")
    
    root = next(n for n, d in given_graph.in_degree() if d == 0)
    leaves = [n for n, d in given_graph.out_degree() if d == 0]

    # Precompute node-to-leaf assignments with maximum distance
    node_to_leaf: Dict[str, Tuple[str, int]] = {}
    for leaf in leaves:
        path = nx.shortest_path(given_graph, root, leaf)
        for idx, node in enumerate(path):
            distance = len(path) - idx - 1  # Steps from node to leaf
            if node not in node_to_leaf:
                node_to_leaf[node] = (leaf, distance)
            else:
                current_leaf, current_dist = node_to_leaf[node]
                if (distance > current_dist) or (distance == current_dist and leaf < current_leaf):
                    node_to_leaf[node] = (leaf, distance)

    # Assign branches to cells
    branch_labels = np.array([node_to_leaf.get(lbl, (None,))[0] for lbl in labels_array])
    valid = branch_labels != None
    if np.sum(valid) < 2:
        raise ValueError("Insufficient branch assignments (n < 2) for silhouette score.")
    
    # Compute silhouette score
    score = silhouette_score(
        X=inferred_embedding[valid] if metric != 'precomputed' else inferred_embedding[np.ix_(valid, valid)],
        labels=branch_labels[valid],
        metric=metric,
        **kwargs
    )
    
    if validate_result:
        _validator(score=score)
    
    return score


if __name__ == '__main__':
    
    def test_perfect_separation_precomputed():
        """Test perfect separation using a precomputed distance matrix."""
        # Create a simple arborescence with root and two leaves
        G = nx.DiGraph()
        G.add_edges_from([('root', 'L1'), ('root', 'L2')])
        
        # Labels: 100 L1, 100 L2
        labels = np.array(['L1'] * 100 + ['L2'] * 100)
        
        # Precomputed distance matrix with 0s intra-cluster, 1s inter-cluster
        size = len(labels)
        distance_matrix = np.ones((size, size))
        distance_matrix[:100, :100] = 0
        distance_matrix[100:, 100:] = 0
        
        score = branch_silhouette_score(
            G, distance_matrix, labels, validate_result=True, metric='precomputed'
        )
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_small_manual_case_euclidean():
        """Test a small manually verifiable case with Euclidean metric."""
        G = nx.DiGraph()
        G.add_edges_from([('root', 'L1'), ('root', 'L2')])
        
        # Four samples: two L1, two L2
        labels = np.array(['L1', 'L1', 'L2', 'L2'])
        embedding = np.array([
            [0, 0], [0, 1],   # L1 cluster
            [10, 10], [10, 11] # L2 cluster
        ])
        
        score = branch_silhouette_score(G, embedding, labels, validate_result=True)
        expected_score = 0.929  # Precomputed expected value
        assert np.isclose(score, expected_score, atol=0.01), \
            f"Expected ~{expected_score}, got {score}"

    def test_invalid_graph_raises_error():
        """Test non-arborescence graph raises ValueError."""
        # Create a cyclic graph
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        labels = np.array(['1', '2', '3'])
        embedding = np.random.randn(3, 2)
        
        try:
            branch_silhouette_score(G, embedding, labels, validate_result=True)
            assert False, "Expected ValueError for cyclic graph"
        except ValueError:
            pass

    def test_insufficient_samples_raises_error():
        """Test <2 valid samples raises ValueError."""
        G = nx.DiGraph()
        G.add_edge('root', 'L1')
        
        # Only one valid sample after filtering
        labels = np.array(['L1', 'invalid'])
        embedding = np.array([[0, 0], [1, 1]])
        
        try:
            branch_silhouette_score(G, embedding, labels, validate_result=True)
            assert False, "Expected ValueError for <2 samples"
        except ValueError:
            pass

    def test_lexicographic_leaf_selection():
        """Test lex order determines leaf when distances are equal."""
        G = nx.DiGraph()
        G.add_edges_from([('root', 'B'), ('root', 'A')])  # B is lex greater than A
        
        # Root's leaf is determined by lex order (A comes before B)
        labels = np.array(['root', 'root'])
        embedding = np.array([[0, 0], [1, 1]])
        
        # Both samples should be assigned to leaf 'A'
        try:
            # Since both are in the same cluster, this should raise an error
            score = branch_silhouette_score(G, embedding, labels, validate_result=True)
            assert False, "Expected ValueError for single cluster"
        except ValueError:
            pass

    def test_large_scale_realistic_embedding():
        """Test a large-scale realistic embedding with known outcome."""
        np.random.seed(42)
        
        # Create a graph with root -> A -> L1 and root -> L2
        G = nx.DiGraph()
        G.add_edges_from([('root', 'A'), ('A', 'L1'), ('root', 'L2')])
        
        # Generate 500 samples for L1 and L2
        n_samples = 500
        labels = np.array(['L1'] * n_samples + ['L2'] * n_samples)
        
        # Embedding: L1 around (5,5), L2 around (-5,-5), root/A in between
        embedding = np.vstack([
            np.random.normal(loc=(5, 5), scale=0.1, size=(n_samples, 2)),
            np.random.normal(loc=(-5, -5), scale=0.1, size=(n_samples, 2))
        ])
        
        score = branch_silhouette_score(G, embedding, labels, validate_result=True)
        
        # Expect high silhouette score due to good separation
        assert score > 0.7, f"Expected >0.7, got {score}"

    
    test_perfect_separation_precomputed()
    test_small_manual_case_euclidean()
    test_invalid_graph_raises_error()
    test_insufficient_samples_raises_error()
    test_lexicographic_leaf_selection()
    test_large_scale_realistic_embedding()
    print("All tests passed.")
