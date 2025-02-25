#!/usr/bin/env python3

import numpy as np
from scipy.stats import wasserstein_distance

try:
    from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def _compute_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Compute the graph Laplacian of a square adjacency matrix.

    Parameters:
        adj (np.ndarray): A square numpy array representing the adjacency matrix of a graph.

    Returns:
        np.ndarray: The graph Laplacian computed as D - adj, where D is the diagonal degree matrix.
    """
    D = np.diag(adj.sum(axis=1))
    return D - adj


def laplacian_spectral_emd(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool
) -> float:
    """Compute the Earth Mover's Distance (EMD) between the normalized Laplacian eigenvalue spectra of two graphs.

    It compares two square adjacency matrices by computing their graph Laplacians, extracting the eigenvalue spectra, 
    normalizing these spectra, and then quantifying their difference using the Earth Mover's Distance (EMD).
    This approach captures global structural differences between the graphs.

    Parameters:
        given_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix of the first graph.
        inferred_adjacency_matrix (np.ndarray): A square numpy array representing the adjacency matrix of the second graph.
        validate_result (bool): A boolean flag indicating whether to validate that the resulting EMD is zero or positive.

    Returns:
        float: The Earth Mover's Distance (EMD) between the normalized Laplacian eigenvalue spectra.

    Advantages:
        - Captures global structural differences by comparing the eigenvalue spectra of graph Laplacians.
        - Sensitive to variations in connectivity and community structure.
        - Applicable to graphs with different sizes and densities, as long as the matrices are square.

    Limitations:
        - Computationally intensive for large graphs due to eigenvalue computation (O(n^3) complexity).
        - Assumes undirected graphs with symmetric adjacency matrices.
        - Sensitive to disconnected or empty graphs, which may result in a NaN value.

    Interpretation:
        - A lower EMD indicates that the graphs have more similar normalized Laplacian spectra and thus similar structural properties.
        - A higher EMD suggests significant differences in the topology of the graphs.
        - An EMD of 0 means the graphs have identical normalized eigenvalue spectra, implying identical graph structure.
    """
    L1 = _compute_laplacian(given_adjacency_matrix)
    L2 = _compute_laplacian(inferred_adjacency_matrix)
    
    # Compute eigenvalues in ascending order for symmetric matrices
    eig1 = np.linalg.eigvalsh(L1)
    eig2 = np.linalg.eigvalsh(L2)
    
    # Total sums of eigenvalues (trace of Laplacian) used for normalization.
    total1 = eig1.sum()
    total2 = eig2.sum()
    if total1 <= 0 or total2 <= 0:
        raise ValueError("Adjacency graphs are either disconnected or empty")
    
    eig1_norm = eig1 / total1
    eig2_norm = eig2 / total2
    
    # Compute EMD between normalized spectra
    score = wasserstein_distance(eig1_norm, eig2_norm)
    
    if validate_result:
        _validator(score)
        
    return score
    
    
if __name__ == "__main__":
    
    def test_identity_matrices_raise_error():
        """Test that identity matrices (self-loops only) raise ValueError due to zero Laplacian trace."""
        n = 3
        adj = np.eye(n)
        try:
            laplacian_spectral_emd(adj, adj, validate_result=False)
            assert False, "Expected ValueError due to zero trace Laplacian"
        except ValueError:
            pass

    def test_empty_matrices_raise_error():
        """Test empty adjacency matrices (all zeros) raise ValueError."""
        n = 2
        adj = np.zeros((n, n))
        try:
            laplacian_spectral_emd(adj, adj, validate_result=False)
            assert False, "Expected ValueError for empty graph"
        except ValueError:
            pass

    def test_identical_matrices_zero_emd():
        """Test that identical adjacency matrices result in EMD of zero."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        emd = laplacian_spectral_emd(adj, adj, validate_result=True)
        assert emd == 0.0, f"EMD should be 0 for identical matrices, got {emd}"

    def test_line_vs_cycle_emd():
        """Test EMD between 3-node line graph and 3-node cycle graph."""
        # 3-node line graph adjacency
        line_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        # 3-node cycle (triangle) adjacency
        cycle_adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        
        emd = laplacian_spectral_emd(line_adj, cycle_adj, validate_result=False)
        expected_emd = 1/6  # Precomputed expected value
        assert np.isclose(emd, expected_emd, atol=1e-6), f"Expected EMD {expected_emd}, got {emd}"

    def test_non_square_matrices_raise_error():
        """Test that non-square matrices raise an error during Laplacian computation."""
        adj1 = np.array([[0, 1], [1, 0]])  # Square
        adj2 = np.array([[0, 1, 1], [1, 0, 0]])  # 2x3, non-square
        try:
            laplacian_spectral_emd(adj1, adj2, validate_result=False)
            assert False, "Expected error for non-square adjacency matrix"
        except ValueError:
            pass

    def test_block_diagonal_matrices():
        """Test EMD between two block diagonal matrices with known structure."""
        # Two 3-node line graphs (total 6 nodes)
        block3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=float)
        adj1 = np.zeros((6, 6))
        adj1[:3, :3] = block3
        adj1[3:, 3:] = block3.copy()

        # Three 2-node line graphs (total 6 nodes)
        adj2 = np.zeros((6, 6))
        for i in range(0, 6, 2):
            adj2[i, i+1] = 1
            adj2[i+1, i] = 1

        emd = laplacian_spectral_emd(adj1, adj2, validate_result=False)
        expected_emd = 5/72  # Precomputed expected value
        assert np.isclose(emd, expected_emd, atol=1e-6), f"Expected EMD {expected_emd}, got {emd}"

    def test_large_complete_graphs_zero_emd():
        """Test that two large complete graphs have zero EMD."""
        n = 100
        # Create a complete graph adjacency (excluding self-loops)
        adj = np.ones((n, n)) - np.eye(n)
        emd = laplacian_spectral_emd(adj, adj, validate_result=False)
        assert emd == 0.0, f"EMD should be 0 for identical large graphs, got {emd}"
    
    test_identity_matrices_raise_error()
    test_empty_matrices_raise_error()
    test_identical_matrices_zero_emd()
    test_line_vs_cycle_emd()
    test_non_square_matrices_raise_error()
    test_block_diagonal_matrices()
    test_large_complete_graphs_zero_emd()
    print("All tests passed.")