#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.
# Note: The codebase here actually belongs to the previous version of the codebase. 
# It was kept as reference.

from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
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

    def _compute_spatial_weights(
        self,
        data: Any,
        input_type: str,
        weight_method: Optional[str] = None,
        k: Optional[int] = 90,  # Set default here
        sigma: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> csr_matrix:
        """Compute spatial weights based on the input data type.

        Spatial weights define the spatial relationships between units. The computation
        method varies depending on whether the input is an adjacency matrix, pseudotime vector,
        or embedding (e.g., diffusion map embeddings).

        Args:
            data (Any): The data representing the trajectory.
                - For 'adjacency': 2D numpy array (adjacency matrix).
                - For 'pseudotime': 1D numpy array (pseudotime values).
                - For 'embedding': 2D numpy array (embeddings such as diffusion maps or PCA).
                - For 'precalculated': Already computed spatial weights matrix.
            input_type (str): The type of input data. Must be one of:
                - 'adjacency'
                - 'pseudotime'
                - 'embedding'
                - 'precalculated'
            weight_method (Optional[str]): Method to compute weights.
                - For 'pseudotime':
                    - 'inverse' (default)
                    - 'gaussian'
                - For 'embedding':
                    - 'knn' (default)
                    - 'rbf'
                - Ignored for 'adjacency' and 'precalculated'.
            k (Optional[int]): Number of neighbors for 'knn' weighting. Default is 10.
                Applicable only when input_type is 'embedding' and weight_method is 'knn'.
            sigma (Optional[float]): Parameter for 'gaussian' weighting. Default is 1.0.
                Applicable only when input_type is 'pseudotime' and weight_method is 'gaussian'.
            gamma (Optional[float]): Parameter for 'rbf' weighting. Default is 1.0.
                Applicable only when input_type is 'embedding' and weight_method is 'rbf'.

        Returns:
            csr_matrix: The computed spatial weights matrix in sparse CSR format.

        Raises:
            ValueError: If input_type is unsupported, required parameters are missing,
                        or input data is invalid.
        """
        supported_input_types = {"adjacency", "pseudotime", "embedding", "precalculated"}
        if input_type not in supported_input_types:
            raise ValueError(f"Unsupported input_type {input_type!r}. Supported types are: {supported_input_types}.")

        if input_type == "precalculated":
            if not isspmatrix_csr(data):
                raise ValueError("For 'precalculated' input_type, data must be a scipy.sparse CSR matrix.")
            return data.copy()

        spatial_weights = None

        if input_type == "adjacency":
            spatial_weights = self._compute_adjacency_weights(data)

        elif input_type == "pseudotime":
            spatial_weights = self._compute_pseudotime_weights(
                data, weight_method=weight_method or "inverse", sigma=sigma
            )

        elif input_type == "embedding":
            spatial_weights = self._compute_embedding_weights(
                data, weight_method=weight_method or "knn", k=k, gamma=gamma
            )

        # Normalize the spatial weights
        spatial_weights = self._normalize_weights(spatial_weights)

        return spatial_weights

    def _compute_adjacency_weights(self, adjacency_matrix: np.ndarray) -> csr_matrix:
        """Compute spatial weights from an adjacency matrix.

        Args:
            adjacency_matrix (np.ndarray): 2D square numpy array representing adjacency.

        Raises:
            ValueError: If the `adjacency_matrix` is not in correct format.

        Returns:
            csr_matrix: Normalized spatial weights matrix.
        """
        if not isinstance(adjacency_matrix, np.ndarray):
            raise ValueError("Adjacency matrix must be a numpy array.")
        if adjacency_matrix.ndim != 2:
            raise ValueError("Adjacency matrix must be 2-dimensional.")
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square (same number of rows and columns).")
        if not np.allclose(adjacency_matrix, adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetric.")
        spatial_weights = csr_matrix(adjacency_matrix)
        return spatial_weights

    def _compute_pseudotime_weights(
        self, pseudotime: np.ndarray, weight_method: str, sigma: Optional[float]
    ) -> csr_matrix:
        """Compute spatial weights based on pseudotime data.

        Args:
            pseudotime (np.ndarray): 1D array of pseudotime values.
            weight_method (str): Method to compute weights ('inverse' or 'gaussian').
            sigma (Optional[float]): Parameter for 'gaussian' weighting.

        Returns:
            csr_matrix: Spatial weights matrix.

        Raises:
            ValueError: If weight_method is invalid or required parameters are missing.
        """
        if not isinstance(pseudotime, np.ndarray):
            raise ValueError("Pseudotime data must be a numpy array.")
        if pseudotime.ndim != 1:
            raise ValueError("Pseudotime data must be a 1D array.")
        if np.isnan(pseudotime).any() or np.isinf(pseudotime).any():
            raise ValueError("Pseudotime data contains NaN or Inf values.")

        if weight_method == "inverse":
            distance_matrix = np.abs(pseudotime[:, np.newaxis] - pseudotime[np.newaxis, :])
            weights = 1.0 / (distance_matrix + 1e-5)  # Avoid division by zero
        elif weight_method == "gaussian":
            sigma = sigma if sigma is not None else 1.0  # Default value
            if sigma <= 0:
                raise ValueError("Sigma must be positive for 'gaussian' weight_method.")
            distance_matrix = np.abs(pseudotime[:, np.newaxis] - pseudotime[np.newaxis, :])
            weights = np.exp(-(distance_matrix**2) / (2 * sigma**2))
        else:
            raise ValueError(f"Unknown weight_method {weight_method!r} for 'pseudotime' input_type.")

        return csr_matrix(weights)

    def _compute_embedding_weights(
        self, embedding: np.ndarray, weight_method: str, k: Optional[int], gamma: Optional[float]
    ) -> csr_matrix:
        """Compute spatial weights based on embedding data.

        Args:
            embedding (np.ndarray): 2D array of embeddings (e.g., diffusion maps, PCA).
            weight_method (str): Method to compute weights ('knn' or 'rbf').
            k (Optional[int]): Number of neighbors for 'knn' weighting.
            gamma (Optional[float]): Parameter for 'rbf' weighting.

        Returns:
            csr_matrix: Spatial weights matrix.

        Raises:
            ValueError: If weight_method is invalid or required parameters are missing.
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding data must be a numpy array.")
        if embedding.ndim != 2:
            raise ValueError("Embedding data must be a 2D array.")
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Embedding data contains NaN or Inf values.")

        if weight_method == "knn":
            if k is None or k <= 0:
                raise ValueError("Number of neighbors k must be positive for 'knn' weight_method.")
            spatial_weights = kneighbors_graph(
                embedding, n_neighbors=k, mode="connectivity", include_self=True, n_jobs=-1
            )
        elif weight_method == "rbf":
            gamma = gamma if gamma is not None else 1.0  # Default gamma
            if gamma is None or gamma <= 0:
                raise ValueError("Gamma must be positive for 'rbf' weight_method.")
            weights = rbf_kernel(embedding, gamma=gamma)
            spatial_weights = csr_matrix(weights)
        else:
            raise ValueError(f"Unknown weight_method {weight_method!r} for 'embedding' input_type.")

        return spatial_weights

    def _normalize_weights(self, weights: csr_matrix) -> csr_matrix:
        """Normalize spatial weights by row to ensure that the sum of weights for each unit is 1.

        Args:
            weights (csr_matrix): Spatial weights matrix.

        Returns:
            csr_matrix: Normalized spatial weights matrix.
        """
        row_sums = np.array(weights.sum(axis=1)).flatten()
        # To avoid division by zero, set zero sums to one (isolated units will have zero weights)
        with np.errstate(divide="ignore"):
            inv_row_sums = 1.0 / row_sums
            inv_row_sums[np.isinf(inv_row_sums)] = 0.0
        diagonal_inv = csr_matrix(
            (inv_row_sums, (np.arange(len(inv_row_sums)), np.arange(len(inv_row_sums)))),
            shape=(len(inv_row_sums), len(inv_row_sums)),
        )
        normalized_weights = diagonal_inv.dot(weights)
        return normalized_weights

    def calculate_morans_i(self, x: np.ndarray, spatial_weights: csr_matrix) -> float:
        """Calculates Moran's I for the given data and spatial weights.

        Moran's I is a measure of spatial autocorrelation that assesses the degree to which
        similar values are clustered together in a spatial context. It provides an indication
        of whether the pattern expressed is clustered, dispersed, or random.

        Mathematical Formulation:
            I = (N / W) * (Σi Σj w_ij (x_i - x̄)(x_j - x̄)) / Σi (x_i - x̄)^2

        Where:
            - N: Number of spatial units.
            - W: Sum of all spatial weights (Σi Σj w_ij).
            - x_i: Value at spatial unit i.
            - x̄: Mean of all x_i.
            - w_ij: Spatial weight between spatial units i and j.

        Advantages:
            - Provides a global measure of spatial autocorrelation, summarizing the overall
              spatial pattern of the data.
            - Intuitive interpretation where values close to +1 indicate strong clustering,
              values around 0 suggest randomness, and values close to -1 indicate dispersion.
            - Useful for identifying the presence of spatial clusters in the data.

        Limitations:
            - Assumes linear relationships between the spatial units, which may not capture
              more complex spatial dependencies.
            - Sensitive to the specification of the spatial weights matrix; different
              weight matrices can lead to different results.
            - Does not identify where clusters or outliers are located, only the overall
              degree of autocorrelation.

        Sensitivities:
            - Highly sensitive to the choice and structure of the spatial weights matrix,
              which defines the spatial relationships between units.
            - Influenced by outliers or extreme values in the data, which can skew the
              autocorrelation measure.

        Result:
            - A single scalar value representing Moran's I statistic.
            - Values range from -1 (indicating perfect dispersion) to +1 (indicating perfect
              clustering), with 0 suggesting no spatial autocorrelation.

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

        Geary's C is a measure of spatial autocorrelation that focuses more on local differences
        between values. Unlike Moran's I, which emphasizes global patterns, Geary's C is more
        sensitive to changes in individual pairs of neighboring spatial units.

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

        Advantages:
            - Provides a complementary perspective to Moran's I by emphasizing local spatial
              variations and differences.
            - More sensitive to local anomalies and outliers compared to global autocorrelation measures.
            - Useful for detecting areas with significant local dissimilarities.

        Limitations:
            - Like Moran's I, Geary's C is sensitive to the choice of spatial weights matrix.
            - Interpretation can be less intuitive, as values less than 1 indicate positive
              spatial autocorrelation, values equal to 1 indicate no spatial autocorrelation,
              and values greater than 1 indicate negative spatial autocorrelation.
            - Does not provide information on the location of spatial autocorrelation.

        Sensitivities:
            - Highly dependent on the spatial weights matrix, which defines the notion of neighborhood.
            - Influenced by the presence of spatial outliers, which can disproportionately affect the measure.

        Result:
            - A single scalar value representing Geary's C statistic.
            - Values range from 0 to 2, where values below 1 indicate positive spatial autocorrelation,
              values equal to 1 suggest no spatial autocorrelation, and values above 1 indicate
              negative spatial autocorrelation.

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

        Local Moran's I, also known as LISA (Local Indicators of Spatial Association), measures spatial
        autocorrelation at the local level. It identifies specific locations where high or low values
        are clustered, highlighting local clusters and spatial outliers.

        Mathematical Formulation:
            I_i = ((x_i - x̄) / S^2) * Σj w_ij (x_j - x̄)

        Where:
            - x_i: Value at spatial unit i.
            - x_j: Value at spatial unit j.
            - x̄: Mean of all x_i.
            - S^2: Variance of all x_i.
            - w_ij: Spatial weight between spatial units i and j.

        Advantages:
            - Provides spatially explicit information, allowing for the identification of specific
              locations with significant spatial autocorrelation.
            - Can detect local clusters of high or low values, as well as spatial outliers.
            - Enhances the understanding of spatial patterns by revealing localized spatial dependencies.

        Limitations:
            - Requires multiple hypothesis testing corrections due to the multiple local tests being performed.
            - Interpretation can be complex, especially when dealing with multiple significant local
              indicators.
            - Sensitive to the choice of spatial weights matrix, which influences the identification of
              local clusters and outliers.

        Sensitivities:
            - Highly dependent on the spatial weights matrix, affecting which neighbors are considered.
            - Influenced by the presence of local outliers, which can affect the identification of clusters.

        Result:
            - An array or similar structure containing Local Moran's I values for each spatial unit.
            - Each value indicates the degree of local spatial autocorrelation, with higher absolute
              values suggesting stronger local clustering or outlier status.

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

        The Getis-Ord Gi* statistic measures the degree of clustering of high or low values in the data.
        It identifies "hotspots" (clusters of high values) and "coldspots" (clusters of low values) within
        the spatial context.

        Mathematical Formulation:
            G_i^* = [Σj w_ij x_j - x̄ Σj w_ij] / [S * sqrt((n Σj w_ij^2 - (Σj w_ij)^2) / (n - 1))]

        Where:
            - x_j: Value at spatial unit j.
            - x̄: Mean of all x_j.
            - S: Standard deviation of all x_j.
            - n: Number of spatial units.
            - w_ij: Spatial weight between spatial units i and j.

        Advantages:
            - Effectively identifies local clusters of high or low values, providing spatially explicit
              insights into hotspots and coldspots.
            - Useful for detecting areas with significant concentration of extreme values.
            - Can be applied to various types of spatial data, including counts, rates, and measurements.

        Limitations:
            - Requires careful interpretation, as statistical significance does not always imply practical
              significance.
            - Sensitive to the specification of the spatial weights matrix, which defines neighborhood relationships.
            - May identify clusters influenced by the overall distribution of the data, potentially overlooking
              smaller or less intense clusters.

        Sensitivities:
            - Highly dependent on the spatial weights matrix, affecting the identification of spatial clusters.
            - Influenced by the presence of spatial outliers and the overall distribution of data values.

        Result:
            - An array or similar structure containing Getis-Ord Gi* values for each spatial unit.
            - Each value indicates the degree of clustering of high or low values around the corresponding
              location, with higher positive values indicating hotspots and lower negative values indicating
              coldspots.

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
