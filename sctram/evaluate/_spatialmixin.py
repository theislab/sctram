#!/usr/bin/env python3

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph


class SpatialMetricsMixin:
    """Mixin class providing spatial autocorrelation metrics for trajectory evaluation.

    This mixin implements various spatial autocorrelation metrics, including Moran's I,
    Geary's C, Local Moran's I (LISA), and Getis-Ord Gi*. It supports different types of
    trajectory representations such as adjacency matrices, pseudotime vectors, and
    diffusion map embeddings. The mixin handles the computation of spatial weights based
    on the input type and provides methods to calculate each metric.

    Mathematical Foundations:
        - Moran's I: Measures global spatial autocorrelation.
        - Geary's C: Measures spatial autocorrelation with emphasis on local differences.
        - Local Moran's I (LISA): Assesses spatial autocorrelation at individual units.
        - Getis-Ord Gi*: Identifies spatial clusters (hotspots and coldspots).

    Computational Considerations:
        - Utilizes sparse matrix operations for efficiency.
        - Designed to handle large datasets.
        - Avoids reliance on external spatial libraries like `pysal`.
    """

    def compute_spatial_weights(self, data: Any, input_type: str, **kwargs) -> csr_matrix:
        """Computes spatial weights based on the input data type.

        Spatial weights define the spatial relationships between units. The computation
        method varies depending on whether the input is an adjacency matrix, pseudotime vector,
        or diffusion map embedding.

        Args:
            data (Any): The data representing the trajectory.
                        - For 'adjacency': 2D numpy array (adjacency matrix).
                        - For 'pseudotime': 1D numpy array (pseudotime values).
                        - For 'diffmap': 2D numpy array (diffusion map embeddings).
            input_type (str): The type of input data ('adjacency', 'pseudotime', 'diffmap').
            kwargs: Additional parameters for spatial weights computation.
                      - For 'pseudotime':
                          - 'weight_method': 'inverse' or 'gaussian'.
                          - 'sigma': Parameter for 'gaussian' weighting.
                      - For 'diffmap':
                          - 'weight_method': 'knn' or 'rbf'.
                          - 'k': Number of neighbors for 'knn'.
                          - 'gamma': Parameter for 'rbf' weighting.

        Returns:
            csr_matrix: The computed spatial weights matrix in sparse CSR format.

        Raises:
            ValueError: If the input_type is unsupported or required parameters are missing.
        """
        if input_type == "adjacency":
            # Assume 'data' is adjacency matrix
            adjacency_matrix = data
            if not isinstance(adjacency_matrix, np.ndarray):
                raise ValueError("Adjacency matrix must be a numpy array.")
            if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
                raise ValueError("Adjacency matrix must be a square 2D array.")
            spatial_weights = csr_matrix(adjacency_matrix)
            row_sums = spatial_weights.sum(axis=1).A1  # Convert to 1D array
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            spatial_weights = spatial_weights.multiply(1 / row_sums[:, np.newaxis])
        elif input_type == "pseudotime":
            # Assume 'data' is a 1D pseudotime vector
            pseudotime = data
            if not isinstance(pseudotime, np.ndarray):
                raise ValueError("Pseudotime data must be a numpy array.")
            if pseudotime.ndim != 1:
                raise ValueError("Pseudotime data must be a 1D array.")
            weight_method = kwargs.get("weight_method", "inverse")  # 'inverse' or 'gaussian'
            if weight_method == "inverse":
                distance_matrix = np.abs(pseudotime[:, np.newaxis] - pseudotime[np.newaxis, :])
                spatial_weights = 1 / (distance_matrix + 1e-5)  # Add small value to avoid division by zero
            elif weight_method == "gaussian":
                sigma = kwargs.get("sigma", 1.0)
                distance_matrix = np.abs(pseudotime[:, np.newaxis] - pseudotime[np.newaxis, :])
                spatial_weights = np.exp(-(distance_matrix**2) / (2 * sigma**2))
            else:
                raise ValueError(f"Unknown weight_method {weight_method!r} for pseudotime.")
            spatial_weights = csr_matrix(spatial_weights)
            row_sums = spatial_weights.sum(axis=1).A1
            row_sums[row_sums == 0] = 1
            spatial_weights = spatial_weights.multiply(1 / row_sums[:, np.newaxis])
        elif input_type == "diffmap":
            # Assume 'data' is diffusion map embeddings
            embedding = data
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Diffusion map embeddings must be a numpy array.")
            if embedding.ndim != 2:
                raise ValueError("Diffusion map embeddings must be a 2D array.")

            weight_method = kwargs.get("weight_method", "knn")  # 'knn' or 'rbf'
            if weight_method == "knn":
                k = kwargs.get("k", 10)
                spatial_weights = kneighbors_graph(embedding, n_neighbors=k, mode="connectivity", include_self=True)
            elif weight_method == "rbf":
                gamma = kwargs.get("gamma", 1.0)
                spatial_weights = rbf_kernel(embedding, gamma=gamma)
                spatial_weights = csr_matrix(spatial_weights)
            else:
                raise ValueError(f"Unknown weight_method {weight_method!r} for diffmap.")

            if weight_method == "knn":
                row_sums = spatial_weights.sum(axis=1).A1
                row_sums[row_sums == 0] = 1
                spatial_weights = spatial_weights.multiply(1 / row_sums[:, np.newaxis])
        else:
            raise ValueError(
                f"Unsupported input_type {input_type!r}. Supported types: 'adjacency', 'pseudotime', 'diffmap'."
            )
        return spatial_weights

    def calculate_morans_i(self, x: np.ndarray, spatial_weights: csr_matrix) -> float:
        """Calculates Moran's I for the given data and spatial weights.

        Moran's I is a measure of global spatial autocorrelation, indicating whether similar
        values are clustered together or dispersed across the spatial units.

        Mathematical Formulation:
            I = (N / W) * (Σi Σj w_ij (x_i - x̄)(x_j - x̄)) / Σi (x_i - x̄)^2

        Where:
            - N: Number of spatial units.
            - W: Sum of all spatial weights (Σi Σj w_ij).
            - x_i: Value at spatial unit i.
            - x̄: Mean of all x_i.
            - w_ij: Spatial weight between spatial units i and j.

        Args:
            x (np.ndarray): 1D array of data values.
            spatial_weights (csr_matrix): Sparse matrix of spatial weights.

        Returns:
            float: Moran's I statistic.

        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input x must be a numpy array.")
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if not isinstance(spatial_weights, csr_matrix):
            raise ValueError("Spatial weights must be a scipy.sparse CSR matrix.")
        n = len(x)
        w = spatial_weights.sum()
        if w == 0:
            raise ValueError("Sum of spatial weights W must not be zero.")
        x_mean = np.mean(x)
        x_diff = x - x_mean
        numerator = x_diff @ (spatial_weights @ x_diff)
        denominator = np.sum(x_diff**2)
        morans_i = (n / w) * (numerator / denominator)
        return morans_i

    def calculate_gearys_c(self, x: np.ndarray, spatial_weights: csr_matrix) -> float:
        """Calculates Geary's C for the given data and spatial weights.

        Geary's C is another measure of spatial autocorrelation, emphasizing local
        differences between neighboring spatial units.

        Mathematical Formulation:
            C = ((N - 1) / (2W)) * (Σi Σj w_ij (x_i - x_j)^2) / Σi (x_i - x̄)^2

        Where:
            - N: Number of spatial units.
            - W: Sum of all spatial weights (Σi Σj w_ij).
            - x_i: Value at spatial unit i.
            - x_j: Value at spatial unit j.
            - x̄: Mean of all x_i.

        Args:
            x (np.ndarray): 1D array of data values.
            spatial_weights (csr_matrix): Sparse matrix of spatial weights.

        Returns:
            float: Geary's C statistic.

        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input x must be a numpy array.")
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if not isinstance(spatial_weights, csr_matrix):
            raise ValueError("Spatial weights must be a scipy.sparse CSR matrix.")
        n = len(x)
        w = spatial_weights.sum()
        if w == 0:
            raise ValueError("Sum of spatial weights W must not be zero.")
        x_mean = np.mean(x)
        x_diff = x - x_mean
        # Compute (x_i - x_j)^2 for all i, j
        # Efficient computation using sparse matrix operations
        # (x_i - x_j)^2 = x_i^2 + x_j^2 - 2 * x_i * x_j
        # Compute x_i^2 * w_ij
        x_i_sq = x_diff**2
        x_j_sq = x_diff**2
        # Compute x_i * w_ij * x_j
        cross_term = 2 * (x_diff @ (spatial_weights @ x_diff))
        # Numerator: Σi Σj w_ij (x_i - x_j)^2 = Σi Σj w_ij x_i^2 + Σi Σj w_ij x_j^2 - 2 Σi Σj w_ij x_i x_j
        # Since w_ij is symmetric, Σi Σj w_ij x_i^2 = Σi x_i^2 Σj w_ij = Σi x_i^2 * row_sums
        # Similarly for Σi Σj w_ij x_j^2
        sum_wx_sq = (spatial_weights.multiply(x_i_sq[:, np.newaxis])).sum() + (spatial_weights.multiply(x_j_sq).sum())
        # Since spatial_weights is symmetric, sum_wx_sq = 2 * Σi Σj w_ij x_i^2
        # Numerator is Σi Σj w_ij (x_i - x_j)^2 = 2 * Σi Σj w_ij x_i^2 - 2 * Σi Σj w_ij x_i x_j
        numerator = sum_wx_sq - cross_term
        denominator = 2 * np.sum(x_diff**2)
        gearys_c = ((n - 1) / (2 * w)) * (numerator / denominator)
        return gearys_c

    def calculate_lisa(self, x: np.ndarray, spatial_weights: csr_matrix) -> np.ndarray:
        """Calculates Local Moran's I (LISA) for each spatial unit.

        Local Moran's I assesses spatial autocorrelation at the individual unit level,
        identifying hotspots and coldspots.

        Mathematical Formulation:
            I_i = ((x_i - x̄) / S^2) * Σj w_ij (x_j - x̄)

        Where:
            - x_i: Value at spatial unit i.
            - x_j: Value at spatial unit j.
            - x̄: Mean of all x_i.
            - S^2: Variance of all x_i.
            - w_ij: Spatial weight between spatial units i and j.

        Args:
            x (np.ndarray): 1D array of data values.
            spatial_weights (csr_matrix): Sparse matrix of spatial weights.

        Returns:
            np.ndarray: Array of Local Moran's I values for each spatial unit.

        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input x must be a numpy array.")
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if not isinstance(spatial_weights, csr_matrix):
            raise ValueError("Spatial weights must be a scipy.sparse CSR matrix.")
        _ = len(x)
        x_mean = np.mean(x)
        s2 = np.var(x, ddof=1)
        if s2 == 0:
            raise ValueError("Variance of x must not be zero.")
        x_diff = x - x_mean
        # Compute spatial lag: Σj w_ij (x_j - x̄)
        spatial_lag = spatial_weights @ x_diff
        lisa = (x_diff / s2) * spatial_lag
        return lisa

    def calculate_getis_ord_gi_star(self, x: np.ndarray, spatial_weights: csr_matrix) -> np.ndarray:
        """Calculates the Getis-Ord Gi* statistic for each spatial unit.

        Getis-Ord Gi* identifies statistically significant spatial clusters of high or low values,
        known as hotspots and coldspots.

        Mathematical Formulation:
            G_i^* = [Σj w_ij x_j - x̄ Σj w_ij] / [S * sqrt((n Σj w_ij^2 - (Σj w_ij)^2) / (n - 1))]

        Where:
            - x_j: Value at spatial unit j.
            - x̄: Mean of all x_j.
            - S: Standard deviation of all x_j.
            - n: Number of spatial units.
            - w_ij: Spatial weight between spatial units i and j.

        Args:
            x (np.ndarray): 1D array of data values.
            spatial_weights (csr_matrix): Sparse matrix of spatial weights.

        Returns:
            np.ndarray: Array of Getis-Ord Gi* values for each spatial unit.

        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input x must be a numpy array.")
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if not isinstance(spatial_weights, csr_matrix):
            raise ValueError("Spatial weights must be a scipy.sparse CSR matrix.")
        n = len(x)
        x_mean = np.mean(x)
        s = np.std(x, ddof=1)
        if s == 0:
            raise ValueError("Standard deviation of x must not be zero.")
        # Σj w_ij x_j for each i
        sum_wx = spatial_weights @ x
        # Σj w_ij for each i
        sum_w = spatial_weights.sum(axis=1).A1
        numerator = sum_wx - (x_mean * sum_w)
        # Compute denominator
        sum_w2 = spatial_weights.multiply(spatial_weights).sum(axis=1).A1
        denominator = s * np.sqrt((n * sum_w2 - sum_w**2) / (n - 1))
        # Handle division by zero
        denominator[denominator == 0] = 1e-10
        gi_star = numerator / denominator
        return gi_star
