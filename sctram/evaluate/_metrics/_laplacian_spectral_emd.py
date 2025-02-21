#!/usr/bin/env python3

import numpy as np
from scipy.stats import wasserstein_distance
from sctram.evaluate._metrics.validators import validate_zero_or_positive


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
        validate_zero_or_positive(score)
        
    return score
    
    