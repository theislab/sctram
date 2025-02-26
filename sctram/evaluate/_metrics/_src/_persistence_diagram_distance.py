#!/usr/bin/env python3

import numpy as np
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from scipy.sparse.csgraph import shortest_path

try:
    from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator



def _compute_diagrams(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute processed persistence diagrams from adjacency matrix"""
    adj_max = adj.max() if adj.max() != 0 else 1.0
    adj_norm = adj / adj_max

    # Compute distance as reciprocal of normalized adjacency for edges, infinity otherwise
    epsilon = 1e-9  # Avoid division by zero
    distance = np.divide(1.0, adj_norm + epsilon, where=adj_norm > 0, out=np.full_like(adj_norm, np.inf))
    np.fill_diagonal(distance, 0.0)

    # Compute shortest paths
    shortest_dist = shortest_path(distance, directed=False)

    # Replace infinite distances with a large finite value
    max_finite = np.max(shortest_dist[shortest_dist < np.inf]) if np.any(shortest_dist < np.inf) else 1.0
    shortest_dist[shortest_dist == np.inf] = 2 * max_finite + 1e-3

    rips = gd.RipsComplex(distance_matrix=shortest_dist)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    dgm0, dgm1 = [], []
    for dim, (birth, death) in persistence:
        if dim == 0:
            dgm0.append((birth, death))
        elif dim == 1:
            dgm1.append((birth, death))
    
    return _process_diagram(dgm0), _process_diagram(dgm1)

def _process_diagram(diagram: list) -> np.ndarray:
    """Handle infinite persistence and convert to numpy array"""
    finite_deaths = [d for _, d in diagram if d != np.inf]
    if finite_deaths:
        max_finite = max(finite_deaths)
        epsilon = 1e-6
    else:
        # All deaths are infinite; use max birth plus a large value
        max_birth = max([b for b, d in diagram]) if diagram else 0.0
        max_finite = max_birth + 1.0  # Adjusted to ensure significant persistence
        epsilon = 1e-6
    
    processed = []
    for birth, death in diagram:
        if death == np.inf:
            adjusted_death = max(birth + epsilon, max_finite + epsilon)
        else:
            adjusted_death = death
        processed.append((birth, adjusted_death))
    
    return np.array(processed)


def persistence_diagram_distance(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool
) -> float:
    """Computes topological distance between adjacency matrices using persistent homology
    and Wasserstein distance between persistence diagrams.

    Mathematical Justification:
    1. Normalization: Adjacency matrices are normalized to [0,1] using max-scaling to ensure
       scale invariance and proper distance conversion.
    2. Persistent Homology: Computes Vietoris-Rips persistence for dimensions 0 (connected components)
       and 1 (loops). This captures multiscale topological features essential for understanding
       graph structure evolution.
    3. Stability: Wasserstein distance (Earth Mover's Distance) provides stable comparison of
       persistence diagrams under small perturbations, respecting both feature locations and
       their persistence strengths.
    4. Dimensional Synthesis: Combined 0D+1D analysis detects both connectivity differences
       (component structure) and cyclic structure variations (loop presence/absence).
    
    Parameters:
        ref_adj: Reference adjacency matrix (n x n), symmetric, non-negative
        test_adj: Test adjacency matrix (n x n), symmetric, non-negative
        validate: Enable result validation checks
    """
    # Compute diagrams for both matrices
    dgm0_ref, dgm1_ref = _compute_diagrams(given_adjacency_matrix)
    dgm0_test, dgm1_test = _compute_diagrams(inferred_adjacency_matrix)

    w0 = wasserstein_distance(dgm0_ref, dgm0_test, order=1)  #, enable_autodiff=True)
    w1 = wasserstein_distance(dgm1_ref, dgm1_test, order=1)  #, enable_autodiff=True)
    
    # Convert total distance to similarity score
    score = w0 + w1

    if validate_result:
        _validator(score=score)
    
    return score


if __name__ == "__main__":

    def test_identical_matrices():
        """Test that identical matrices yield zero distance."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        distance = persistence_diagram_distance(adj, adj, validate_result=True)
        assert np.isclose(distance, 0.0), f"Expected 0, got {distance}"

    def test_zero_matrices():
        """Test that all-zero matrices yield zero distance."""
        adj = np.zeros((3, 3), dtype=np.float64)
        distance = persistence_diagram_distance(adj, adj, validate_result=True)
        assert np.isclose(distance, 0.0), f"Expected 0, got {distance}"

    def test_scale_invariance():
        """Test that scaling doesn't affect the normalized distance."""
        adj_ref = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=np.float64)
        adj_test = adj_ref * 0.5  # Scale down
        distance = persistence_diagram_distance(adj_ref, adj_test, validate_result=True)
        assert np.isclose(distance, 0.0), f"Expected 0, got {distance}"

    def test_disconnected_vs_connected():
        """Test distance between disconnected and connected graphs (0D difference)."""
        # Reference: Two disconnected edges (0-1 and 2-3)
        ref_adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        # Test: Fully connected (complete graph)
        test_adj = np.ones((4, 4), dtype=np.float64) - np.eye(4)
        distance = persistence_diagram_distance(ref_adj, test_adj, validate_result=True)
        assert distance > 0.0, f"Expected positive distance, got {distance}"

    def test_cycle_vs_no_cycle():
        """Test distance between cyclic and acyclic graphs (1D difference)."""
        # Reference: 4-node cycle (square without diagonals)
        ref_adj = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float64)
        
        # Test: Line graph (0-1-2-3, acyclic)
        test_adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        
        distance = persistence_diagram_distance(ref_adj, test_adj, validate_result=False)
        assert np.isclose(distance, 0.5, atol=1e-3), f"Expected ~0.5, got {distance}"

    def test_perturbed_matrix():
        """Test small perturbation results in small distance (stability check)."""
        np.random.seed(42)  # Reproducibility
        base_adj = np.random.rand(4, 4)
        base_adj = (base_adj + base_adj.T) / 2  # Symmetrize
        np.fill_diagonal(base_adj, 0)
        
        # Add small noise
        noise = np.random.normal(0, 0.01, base_adj.shape)
        noise = (noise + noise.T) / 2  # Symmetrize noise
        test_adj = np.clip(base_adj + noise, 0, None)  # Ensure non-negativity
        np.fill_diagonal(test_adj, 0)
        
        distance = persistence_diagram_distance(base_adj, test_adj, validate_result=True)
        assert distance < 0.5, f"Expected small distance, got {distance}"

    def test_non_zero_vs_zero():
        """Test non-zero vs all-zero matrix yields positive distance."""
        ref_adj = np.zeros((3, 3), dtype=np.float64)
        test_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        distance = persistence_diagram_distance(ref_adj, test_adj, validate_result=True)
        assert distance > 0.0, f"Expected positive distance, got {distance}"

    test_identical_matrices()
    test_zero_matrices()
    test_scale_invariance()
    test_disconnected_vs_connected()
    test_cycle_vs_no_cycle()
    test_perturbed_matrix()
    test_non_zero_vs_zero()
    print("All tests passed!")
