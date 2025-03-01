#!/usr/bin/env python3

import numpy as np
from loguru import logger
from scipy.linalg import eigvals

from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator

_logger = logger.bind(name="MetricsBase")


def random_walk_kernel_distance(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool,
    target_lambda: float = 0.1,  # User's desired lambda
    max_k_max: int = 100,  # User's desired k_max
    safety_factor: float = 0.9,  # Multiplier for lambda_max (e.g., 0.9 = 90% of lambda_max)
    **kwargs,
) -> float:
    # Compute spectral radii
    rho_A = np.max(np.abs(eigvals(given_adjacency_matrix)))
    rho_B = np.max(np.abs(eigvals(inferred_adjacency_matrix)))
    lambda_max = 1.0 / (rho_A * rho_B)

    # Adjust lambda_param to stay within safe bounds
    safe_lambda = min(target_lambda, safety_factor * lambda_max)

    # Adjust k_max based on machine precision.
    # Avoids redundant computations for terms that vanish numerically
    eps = np.finfo(float).eps
    effective_k_max = int(np.ceil(np.log(eps) / np.log(safe_lambda))) if safe_lambda != 0 else 0
    safe_k_max = min(max_k_max, effective_k_max)

    return _random_walk_kernel_distance(
        given_adjacency_matrix,
        inferred_adjacency_matrix,
        validate_result=validate_result,
        lambda_param=safe_lambda,
        k_max=safe_k_max,
        check_spectral_radius=False,  # Already enforced
        **kwargs,
    )


def _random_walk_kernel_distance(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    validate_result: bool,
    lambda_param: float = 0.01,
    k_max: int = 400,
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
                     series but increase computational cost. Default: 10000.
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
            A_k = A_powers[k - 1]  # A_powers[0] = A^1, ..., A_powers[k_max-1] = A^k_max
            B_k = B_powers[k - 1]

        trace_GH = np.trace(A_k @ B_k)
        trace_GG = np.trace(A_k @ A_k)
        trace_HH = np.trace(B_k @ B_k)

        weight = lambda_param**k
        K_GH += weight * trace_GH
        K_GG += weight * trace_GG
        K_HH += weight * trace_HH

    # Handle potential division by zero
    denominator = np.sqrt(K_GG * K_HH)
    if denominator <= np.finfo(float).eps:
        return ValueError

    normalized_kernel = K_GH / denominator
    # Clamp to [0, 1] to handle numerical instability or truncation effects
    normalized_kernel = np.clip(normalized_kernel, 0.0, 1.0)
    distance = 1.0 - normalized_kernel

    if validate_result:
        _validator(score=distance)

    return distance
