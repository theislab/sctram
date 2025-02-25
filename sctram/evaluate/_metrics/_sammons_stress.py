#!/usr/bin/env python3

import numpy as np
import networkx as nx
import itertools
from scipy.spatial.distance import pdist
from loguru import logger

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
    from sctram.evaluate._metrics.utils import Centroids
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    from utils import Centroids


_logger = logger.bind(name="MetricBase")


def sammons_stress(
    given_graph: nx.DiGraph,
    centroids: Centroids,
    validate_result: bool,
    embedding_metric: str = "cosine"
) -> float:
    """Scaled Sammon's stress.
    
    Computes the scaled Sammon's stress between graph distances and optimally scaled 
    embedding centroid distances. This metric evaluates structural preservation with 
    emphasis on local relationships, using an optimal scaling factor derived to minimize stress.

    Mathematical Formulation:
        Stress = ( Σ_{i<j} [ (d_ij - s*e_ij)^2 / d_ij ] ) / Σ_{i<j} d_ij
        where s = (Σ e_ij) / (Σ (e_ij^2 / d_ij)) minimizes the stress.

    Parameters:
        given_graph (nx.DiGraph): Graph where nodes represent labels (e.g., cell types).
        centroids (Centroids): Object providing centroids for graph nodes via `get_centroids(nodes)`.
        validate_result (bool): A bool deciding whether or not validate the score based on `validate` method.
        embedding_metric (str): Distance metric for embedding ('euclidean' or 'cosine').

    Returns:
        float: Sammon's stress. Lower values (typically <0.1) indicate strong structural preservation.

    Statistical Justification:
        - Optimal Scaling: The scaling factor `s` is analytically derived to minimize stress,
          ensuring invariance to arbitrary embedding scales.
        - Local Emphasis: Weighting by 1/d_ij prioritizes accurate preservation of small
          distances critical for biological transitions.
        - Normalization: Stress is relative to total graph distance, enabling cross-dataset comparison.

    Limitations:
        - Assumes graph connectivity reflects biological reality; disconnected components are excluded.
        - Centroid-based distances may overlook within-label heterogeneity.
    """
    # Compute pairwise distances from graph and embedding
    graph_distances, embedding_distances = _get_distances_matrices(
        given_graph, centroids, embedding_metric
    )

    # Filter invalid (NaN) and zero graph distances
    valid_mask = ~np.isnan(graph_distances)
    if np.sum(valid_mask) < 2:
        raise ValueError("Insufficient valid node pairs for stress calculation.")
    graph_d = graph_distances[valid_mask]
    embedding_d = embedding_distances[valid_mask]

    nonzero_mask = graph_d > 0
    graph_d_nonzero = graph_d[nonzero_mask]
    embedding_d_nonzero = embedding_d[nonzero_mask]

    if len(graph_d_nonzero) < 2:
        raise ValueError("Insufficient non-zero graph distance pairs.")

    # Compute optimal scaling factor
    sum_e = np.sum(embedding_d_nonzero)
    sum_e2_over_d = np.sum((embedding_d_nonzero ** 2) / graph_d_nonzero)
    if sum_e2_over_d <= 0:
        raise ValueError("Optimal scaling denominator non-positive. Check embedding distances.")
    scale_factor = sum_e / sum_e2_over_d

    # Scale embedding distances and compute stress
    scaled_embedding = embedding_d_nonzero * scale_factor
    squared_errors = (graph_d_nonzero - scaled_embedding) ** 2
    weighted_errors = squared_errors / graph_d_nonzero

    stress = np.sum(weighted_errors) / np.sum(graph_d_nonzero)

    if validate_result:
        _validator(score=stress)

    return stress


def _get_distances_matrices(
    given_graph: nx.Graph,
    centroids: Centroids,
    embedding_metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Computes pairwise graph shortest-path distances and embedding centroid distances.

    Parameters:
        given_graph (nx.Graph): Graph with nodes representing labels.
        centroids (Centroids): Provides centroids for each node.
        embedding_metric (str): Distance metric for embeddings ('euclidean' or 'cosine').

    Returns:
        tuple: (graph_distances, embedding_distances) as condensed distance matrices.
    """
    nodes = list(given_graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least 2 nodes.")

    # Compute centroids and pairwise embedding distances
    centroid_coords = np.array(centroids.get_centroids(nodes))
    embedding_distances = pdist(centroid_coords, metric=embedding_metric)

    # Compute all-pairs shortest paths in the graph
    path_lengths = dict(nx.all_pairs_shortest_path_length(given_graph))
    graph_distances = []
    for i, j in itertools.combinations(nodes, 2):
        dist = path_lengths.get(i, {}).get(j, np.nan)
        if np.isnan(dist):
            _logger.warning(f"No path between {i} and {j}. Assigning NaN.")
        graph_distances.append(dist)

    return np.array(graph_distances), np.array(embedding_distances)


if __name__ == "__main__":

    def test_three_nodes_perfect_scaling():
        """Test three nodes in a line with perfect scaling, stress should be zero."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1)
        G.add_edge("B", "C", weight=1)
        
        class MockCentroids:
            def get_centroids(self, nodes):
                return [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        
        centroids = MockCentroids()
        stress = sammons_stress(G, centroids, validate_result=True, embedding_metric="euclidean")
        assert np.isclose(stress, 0.0), stress

    def test_three_nodes_non_zero_stress():
        """Test three nodes with non-proportional distances, resulting in non-zero stress."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1)
        G.add_edge("B", "C", weight=1)
        
        class MockCentroids:
            def get_centroids(self, nodes):
                # Return centroids as a list in the order of input nodes
                centroid_map = {
                    "A": np.array([0.0]),
                    "B": np.array([1.0]),
                    "C": np.array([3.0]),
                }
                return [centroid_map[node] for node in nodes]
        
        centroids = MockCentroids()
        
        # Compute expected stress manually
        # Graph distances (A-B=1, A-C=2, B-C=1)
        graph_d = np.array([1, 2, 1])
        # Embedding distances (A-B=1, A-C=3, B-C=2)
        embed_d = np.array([1.0, 3.0, 2.0])
        
        # Optimal scaling factor
        sum_e = embed_d.sum()
        sum_e2_over_d = (embed_d**2 / graph_d).sum()
        scale = sum_e / sum_e2_over_d  # 6 / 9.5 ≈ 0.631579
        
        # Scaled embedding distances
        scaled_embed = embed_d * scale
        
        # Stress calculation
        stress = ((graph_d - scaled_embed)**2 / graph_d).sum() / graph_d.sum()
        expected_stress = stress  # Directly use computed value instead of hardcoding
        
        # Validate
        computed_stress = sammons_stress(G, centroids, validate_result=True, embedding_metric="euclidean")
        assert np.isclose(computed_stress, expected_stress, atol=1e-4), f"Expected {expected_stress}, got {computed_stress}"

    def test_nan_handling_insufficient_pairs():
        """Test handling of NaN distances leading to insufficient valid pairs, expecting an error."""
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])
        G.add_edge("A", "B", weight=1)
        G.add_edge("C", "B", weight=1)  # No path from A to C or B to C
        
        class MockCentroids:
            def get_centroids(self, nodes):
                return [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        
        centroids = MockCentroids()
            
        try:
            sammons_stress(G, centroids, validate_result=True, embedding_metric="euclidean")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_large_scale_perfect_scaling():
        """Large-scale test with 100 nodes in a line, stress should be zero."""
        n_nodes = 100
        G = nx.DiGraph()
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1, weight=1)
        
        # Embedding positions: 0, 2, 4, ..., 198 (spaced 2 units apart)
        positions = {i: np.array([i * 2.0]) for i in range(n_nodes)}
        
        class MockCentroids:
            def get_centroids(self, nodes):
                return [positions[node] for node in nodes]
        
        centroids = MockCentroids()
        stress = sammons_stress(G, centroids, validate_result=True, embedding_metric="euclidean")
        assert np.isclose(stress, 0.0), stress

    test_three_nodes_perfect_scaling()
    test_three_nodes_non_zero_stress()
    test_nan_handling_insufficient_pairs()
    test_large_scale_perfect_scaling()
    print("All tests passed.")
