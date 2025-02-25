#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigvals

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    

def random_walk_kernel_distance(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool,
    lambda_param: float = 0.1,
    k_max: int = 20,
    include_k0: bool = False,
    check_spectral_radius: bool = True,
) -> float:
    """Computes the Random Walk Kernel Distance between two symmetric adjacency matrices.

    This distance measures the dissimilarity between graphs by comparing their random walk
    structures. It leverages the normalized random walk kernel, which accounts for all
    possible walks up to length k_max, with exponential decay to ensure convergence.

    Parameters:
        given_adjacency (np.ndarray): Square, symmetric adjacency matrix of the first graph.
        inferred_adjacency (np.ndarray): Square, symmetric adjacency matrix of the second graph.
        lambda_param (float): Decay factor for longer walks. Must be < 1/(ρ(A)ρ(B)) for convergence.
                              Default: 0.1.
        k_max (int): Maximum walk length to consider. Larger values better approximate the infinite
                     series but increase computational cost. Default: 20.
        include_k0 (bool): Whether to include walks of length 0 (identity matrix). Default: False.
        check_spectral_radius (bool): Check λ < 1/(ρ(A)ρ(B)) to ensure convergence. Default: True.
        validate_result (bool): Validate the output distance is in [0, 1]. Default: True.

    Returns:
        float: Distance in [0, 1]. Returns np.nan if undefined (e.g., zero denominator).

    Raises:
        ValueError: For invalid matrix inputs or λ exceeding convergence threshold.

    Notes:
        - The kernel is normalized using the Cauchy-Schwarz inequality, ensuring 0 ≤ distance ≤ 1.
        - Statistical robustness comes from considering all walk lengths, capturing global structure.
        - Complexity is O(k_max * n^3) due to matrix multiplications, manageable for n ≤ 60.

    Mathematical Justification:
        The random walk kernel K(G, H) = ∑_{k=0}^∞ λ^k trace(A^k B^k) compares walk counts between
        graphs. The normalization K_norm = K(G,H)/√(K(G,G)K(H,H)) accounts for graph size and density.
        Distance = 1 - K_norm provides a metric where 0 indicates identical walk structures.
        λ controls the trade-off between local (short walks) and global (long walks) structure.
        Convergence is guaranteed if λ < 1/(ρ(A)ρ(B)), where ρ is the spectral radius.
    """
    n = given_adjacency_matrix.shape[0]
    
    # Check convergence condition if required
    if check_spectral_radius:
        rho_A = np.max(np.abs(eigvals(given_adjacency_matrix)))
        rho_B = np.max(np.abs(eigvals(inferred_adjacency_matrix)))
        lambda_max = 1.0 / (rho_A * rho_B)
        if lambda_param >= lambda_max:
            raise ValueError(f"lambda_param must be < {lambda_max:.4f} for convergence. Current: {lambda_param}.")

    # Precompute matrix powers for A and B up to k_max
    A_powers = []
    current_A = given_adjacency_matrix.copy()
    for _ in range(k_max):
        A_powers.append(current_A)
        current_A = current_A @ given_adjacency_matrix  # A^1, A^2, ..., A^k_max

    B_powers = []
    current_B = inferred_adjacency_matrix.copy()
    for _ in range(k_max):
        B_powers.append(current_B)
        current_B = current_B @ inferred_adjacency_matrix

    # Initialize kernel sums
    K_GH, K_GG, K_HH = 0.0, 0.0, 0.0
    start_k = 0 if include_k0 else 1

    for k in range(start_k, k_max + 1):
        if k == 0:
            A_k = np.eye(n)
            B_k = np.eye(n)
        else:
            A_k = A_powers[k-1]  # A_powers[0] = A^1, ..., A_powers[k_max-1] = A^k_max
            B_k = B_powers[k-1]

        trace_GH = np.trace(A_k @ B_k)
        trace_GG = np.trace(A_k @ A_k)
        trace_HH = np.trace(B_k @ B_k)

        weight = lambda_param ** k
        K_GH += weight * trace_GH
        K_GG += weight * trace_GG
        K_HH += weight * trace_HH

    # Handle potential division by zero
    denominator = np.sqrt(K_GG * K_HH)
    if denominator <= np.finfo(float).eps:
        return np.nan
    
    normalized_kernel = K_GH / denominator
    distance = 1.0 - normalized_kernel

    if validate_result:
        _validator(score=distance)
    
    return distance


if __name__ == "__main__":

    def test_identical_matrices():
        """Test that identical matrices yield a distance of 0."""
        A = np.array([[0, 1], [1, 0]])
        distance = random_walk_kernel_distance(
            A, A, 
            lambda_param=0.1, 
            validate_result=False, 
            check_spectral_radius=False
        )
        assert np.isclose(distance, 0.0), f"Expected 0.0, got {distance}"

    def test_orthogonal_matrices():
        """Test matrices with non-overlapping structures yield a distance of 1."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        distance = random_walk_kernel_distance(
            A, B, 
            lambda_param=0.1, 
            k_max=20, 
            include_k0=False, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isclose(distance, 1.0), f"Expected 1.0, got {distance}"

    def test_empty_matrices():
        """Test empty matrices with and without include_k0."""
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        
        # include_k0=False should result in division by zero (NaN)
        distance_nan = random_walk_kernel_distance(
            A, B, 
            include_k0=False, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isnan(distance_nan), f"Expected NaN, got {distance_nan}"
        
        # include_k0=True should yield distance 0
        distance_zero = random_walk_kernel_distance(
            A, B, 
            include_k0=True, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isclose(distance_zero, 0.0), f"Expected 0.0, got {distance_zero}"

    def test_spectral_radius_exception():
        """Test that exceeding the spectral radius raises an error."""
        A = np.array([[0, 1], [1, 0]])
        B = A.copy()
        lambda_param = 1.0  # Exceeds 1/(ρ(A)ρ(B)) = 1.0
        
        try:
            random_walk_kernel_distance(
                A, B, 
                lambda_param=lambda_param, 
                check_spectral_radius=True, 
                validate_result=False
            )
            assert False, "Expected ValueError due to spectral radius"
        except ValueError:
            pass

    def test_include_k0():
        """Test the effect of including k=0 walks."""
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        
        # include_k0=True adds identity matrices, leading to valid comparison
        distance_with = random_walk_kernel_distance(
            A, B, 
            include_k0=True, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isclose(distance_with, 0.0), f"Expected 0.0, got {distance_with}"
        
        # include_k0=False results in invalid comparison (NaN)
        distance_without = random_walk_kernel_distance(
            A, B, 
            include_k0=False, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isnan(distance_without), f"Expected NaN, got {distance_without}"

    def test_lambda_zero():
        """Test lambda=0 with include_k0=True results in distance 0."""
        A = np.array([[0, 1], [1, 0]])
        distance = random_walk_kernel_distance(
            A, A, 
            lambda_param=0.0, 
            include_k0=True, 
            check_spectral_radius=False, 
            validate_result=False
        )
        assert np.isclose(distance, 0.0), f"Expected 0.0, got {distance}"

    def test_validate_result():
        """Test validation ensures the result is within [0, 1]."""
        A = np.array([[0, 1], [1, 0]])
        B = np.array([[0, 0.5], [0.5, 0]])
        
        # This configuration should yield a valid distance between 0 and 1
        distance = random_walk_kernel_distance(
            A, B, 
            lambda_param=0.1, 
            validate_result=True
        )
        assert 0 <= distance <= 1, f"Distance {distance} not in [0, 1]"
        
    def test_diagonal_matrices_manual():
        """Test diagonal matrices with manually computed expected distance."""
        A = np.diag([1.0, 0.5])
        B = np.diag([0.5, 1.0])
        # Manually compute expected kernel values for lambda=0.5, k_max=2, include_k0=True
        # K_GH = 2*(1 + 0.5*0.5 + (0.5^2)*(0.5^2)) = 2.625
        # K_GG = K_HH = 2*(1 + 0.5*(1 + 0.25) + 0.25*(1 + 0.0625)) = 2.890625
        # Normalized kernel = 2.625 / 2.890625 ≈ 0.908
        # Distance = 1 - 0.908 ≈ 0.092
        expected_distance = 1 - (2.625 / 2.890625)
        distance = random_walk_kernel_distance(
            A, B,
            lambda_param=0.5,
            k_max=2,
            include_k0=True,
            check_spectral_radius=False,
            validate_result=False
        )
        assert np.isclose(distance, expected_distance, rtol=1e-4), \
            f"Expected {expected_distance:.4f}, got {distance}"

    def test_off_diagonal_weighted_matrices():
        """Test matrices with different edge weights but same structure."""
        A = np.array([[0, 1], [1, 0]])
        B = np.array([[0, 2], [2, 0]])
        # Manual calculation for lambda=0.5, k_max=2, include_k0=False
        # K_GH = 0.5*4 + 0.25*8 = 4
        # K_GG = 0.5*2 + 0.25*2 = 1.5
        # K_HH = 0.5*8 + 0.25*32 = 12
        # Normalized kernel = 4 / sqrt(1.5*12) ≈ 0.9428 → distance ≈ 0.0572
        expected_distance = 1 - (4 / np.sqrt(1.5 * 12))
        distance = random_walk_kernel_distance(
            A, B,
            lambda_param=0.5,
            k_max=2,
            include_k0=False,
            check_spectral_radius=False,
            validate_result=False
        )
        assert np.isclose(distance, expected_distance, rtol=1e-4), \
            f"Expected {expected_distance:.4f}, got {distance}"

    def test_k0_only():
        """Test only k=0 is considered when k_max=0 and include_k0=True."""
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        # K_GH = trace(I@I) = 2, K_GG=K_HH=2 → normalized kernel = 1.0 → distance=0
        distance = random_walk_kernel_distance(
            A, B,
            lambda_param=0.5,
            k_max=0,
            include_k0=True,
            check_spectral_radius=False,
            validate_result=False
        )
        assert np.isclose(distance, 0.0), f"Expected 0.0, got {distance}"

    test_identical_matrices()
    test_orthogonal_matrices()
    test_empty_matrices()
    test_spectral_radius_exception()
    test_include_k0()
    test_lambda_zero()
    test_validate_result()
    test_diagonal_matrices_manual()
    test_off_diagonal_weighted_matrices()
    test_k0_only()
    print("All tests passed!")
