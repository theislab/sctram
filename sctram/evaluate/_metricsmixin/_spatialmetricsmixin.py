#!/usr/bin/env python3

from typing import Any, Callable, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

from sctram.utils._utils import Utils


class SpatialMetricsMixin:
    """Mixin class providing spatial autocorrelation metrics for trajectory evaluation.

    This mixin implements various spatial autocorrelation metrics, including Moran's I,
    Geary's C, Local Moran's I (LISA), and Getis-Ord Gi*. The mixin handles the computation of spatial weights
    and data preperation based on the input type and provides methods to calculate each metric.

    Mathematical Foundations:
        - Moran's I: Measures global spatial autocorrelation.
        - Geary's C: Measures spatial autocorrelation with emphasis on local differences.
        - Local Moran's I (LISA): Assesses spatial autocorrelation at individual units.
        - Getis-Ord Gi*: Identifies spatial clusters (hotspots and coldspots).

    Computational Considerations:
        - Avoids reliance on external spatial libraries like `pysal`.
    """

    def _compute_embedding_weights(self) -> csr_matrix:
        cell_labels = self.labels  # self.labels for embedding evaluate class is simply labels for the embedding.
        adjacency_labels = np.array(
            list(self.prepared_after_subset_given.nodes())
        ).flatten()  # the order does not matter here
        adj_matrix = Utils.adjacency_graph_to_matrix(
            g=self.prepared_after_subset_given, nodelist_filter_and_order=adjacency_labels
        )
        Utils.validate_adjacency_matrix(adj_matrix)

        return self._create_sparse_cell_adjacency(
            adj_matrix=adj_matrix, adjacency_labels=adjacency_labels, cell_labels=cell_labels
        )

    def _compute_pseudotime_weights(self):
        unique_labels = np.unique(self.subset_labels)
        subset_adjacency_matrix = Utils.adjacency_graph_to_matrix(
            g=self.subset_given, nodelist_filter_and_order=unique_labels
        )
        Utils.validate_adjacency_matrix(subset_adjacency_matrix)

        return self._create_sparse_cell_adjacency(
            adj_matrix=subset_adjacency_matrix, adjacency_labels=unique_labels, cell_labels=self.subset_labels
        )

    def _create_sparse_cell_adjacency(
        self, adj_matrix: np.ndarray, adjacency_labels: np.ndarray, cell_labels: np.ndarray
    ) -> csr_matrix:
        """Create an nxn sparse cell-cell adjacency matrix from an lxl PAGA adjacency matrix.

        This implementation groups cells by their label and then, for each pair of labels,
        creates the block of cell-cell interactions. This avoids iterating over all n^2 cell pairs.

        Args:
            cell_labels: array-like of shape (n,) or (n,1). Array containing the label for each cell.
            adj_matrix: array-like of shape (l, l). e.g. the PAGA connectivity (adjacency) matrix among the l labels.
            adjacency_labels : array-like of shape (l,) or (l,1). The list of label names corresponding to
                the rows/columns of paga_adj.

        Returns:
            csr_matrix: A sparse cell-cell adjacency matrix where for each pair of cells (i, j)
                cell_adj[i, j] = paga_adj[label_index(cell_i), label_index(cell_j)]
                where label_index(cell) is determined by the mapping defined in adjacency_labels.
        """
        if not isinstance(adjacency_labels, np.ndarray) or not isinstance(cell_labels, np.ndarray):
            raise ValueError("Expected both 'cell_labels' and 'adjacency_labels' to be a numpy array.")
        if adjacency_labels.ndim != 1 or cell_labels.ndim != 1:
            raise ValueError("Both 'adjacency_labels' and 'cell_labels' must be one-dimensional arrays.")
        if len(adjacency_labels) != len(adj_matrix):
            raise ValueError("Mismatch between the number of labels in 'adjacency_labels' and the size of the data.")

        # Build a mapping from label value to its index in adjacency_labels.
        label_to_index = {label: idx for idx, label in enumerate(adjacency_labels)}

        # Map each cell's label to its index in the PAGA matrix.
        mapped_labels = []
        for _label in cell_labels:
            try:
                mapped_labels.append(label_to_index[_label])
            except KeyError:
                mapped_labels.append(np.nan)
        mapped_labels = np.array(mapped_labels)

        n = cell_labels.size
        l = adjacency_labels.size

        # Group cell indices by their mapped label (which is an integer in 0,...,l-1)
        groups = {}
        for cell_idx, lab_idx in enumerate(mapped_labels):
            groups.setdefault(lab_idx, []).append(cell_idx)
        for key in groups:
            groups[key] = np.array(groups[key], dtype=np.int64)

        # Prepare lists to accumulate sparse matrix data.
        row_inds = []
        col_inds = []
        data_vals = []

        # Loop over label pairs.
        # We only add blocks when paga_adj[i, j] is nonzero and when both groups exist.
        for i in range(l):
            if i not in groups:
                continue  # No cell has label adjacency_labels[i]
            idx_i = groups[i]
            for j in range(l):
                if j not in groups:
                    continue  # No cell has label adjacency_labels[j]
                val = adj_matrix[i, j]
                if val == 0:
                    continue  # Skip blocks that contribute no value

                idx_j = groups[j]
                # Create the block indices:
                # Every cell in group i connects to every cell in group j.
                # We use np.repeat and np.tile to create the full index arrays.
                block_rows = np.repeat(idx_i, idx_j.size)
                block_cols = np.tile(idx_j, idx_i.size)
                block_data = np.full(block_rows.shape, val)

                row_inds.append(block_rows)
                col_inds.append(block_cols)
                data_vals.append(block_data)

        # If no blocks were added, return an empty sparse matrix.
        if row_inds:
            row_inds = np.concatenate(row_inds)
            col_inds = np.concatenate(col_inds)
            data_vals = np.concatenate(data_vals)
        else:
            row_inds = np.array([], dtype=np.int64)
            col_inds = np.array([], dtype=np.int64)
            data_vals = np.array([], dtype=adj_matrix.dtype)

        # Create a COO-format sparse matrix and then convert to CSR.
        cell_adj = coo_matrix((data_vals, (row_inds, col_inds)), shape=(n, n)).tocsr()
        return cell_adj

    def _normalize_weights(self, weights: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, csr_matrix]:
        """Normalize spatial weights to ensure appropriate scaling.

        For 2D numpy arrays or CSR matrices, normalizes each row to sum to 1.

        Args:
            weights: Input spatial weights, which can be a 1D/2D numpy array or CSR matrix.

        Returns:
            Normalized weights of the same type as input.

        Raises:
            ValueError: If the input has unsupported dimensions.
        """
        if isspmatrix_csr(weights):
            # Normalize each row in the CSR matrix to sum to 1
            row_sums = np.array(weights.sum(axis=1)).flatten()
            inv_row_sums = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums != 0)
            normalized_weights = weights.multiply(inv_row_sums[:, np.newaxis])
            return normalized_weights.tocsr()
        else:
            raise ValueError(f"Unsupported weights type {type(weights)}. Must be numpy array or CSR matrix.")

    def _calculate_spatial_autocorrelation(self, metric: str, input_type: str, normalize_weights: bool = True):
        """Calculates the specified spatial metric using SpatialMetricsMixin.

        Args:
            metric (str): The spatial metric to calculate. Must be one of:
                - "morans_i"
                - "gearys_c"
                - "local_morans_i"
                - "getis_ord_gi_star"
        """
        metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
            "morans_i": self._calculate_morans_i,
            "gearys_c": self._calculate_gearys_c,
            # Get the mean to have a single value. Note that this makes the metric a bit arguable.
            "lisa": lambda data, weights: np.mean(self._calculate_lisa(data, weights)),
            "getis_ord_gi_star": lambda data, weights: np.mean(self._calculate_getis_ord_gi_star(data, weights)),
        }
        metric_func = metrics.get(metric)
        if not metric_func:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(metrics.keys())}.")

        method_print_str = metric.replace("_", " ").title()
        try:
            if input_type == "pseudotime":
                spatial_weights = self._compute_pseudotime_weights()
            elif input_type == "embedding":
                spatial_weights = self._compute_embedding_weights()
            else:
                raise ValueError(f"Unsupported input_type {input_type!r}")

            if normalize_weights:
                spatial_weights = self._normalize_weights(spatial_weights)
            x = self.prepared_after_subset_inferred

            if not isinstance(x, np.ndarray):
                raise ValueError("x must be either a numpy array or a csr_matrix.")
            if not isinstance(spatial_weights, csr_matrix):
                raise ValueError(
                    f"'spatial_weights' must be either a numpy array or a csr_matrix: {type(spatial_weights)!r}"
                )
            if spatial_weights.ndim != 2:
                raise ValueError("'spatial_weights' must be 2-dimensional.")
            if x.shape[0] != spatial_weights.shape[0]:
                raise ValueError("Mismatch in number of rows between x and spatial weights.")

            if input_type == "embedding":
                if x.ndim != 2:
                    raise ValueError("For elementwise computation, x must be 2-dimensional.")
                # Compute metric element-wise across columns. Slicing works for both numpy arrays and csr_matrix
                results = [metric_func(x[:, col], spatial_weights) for col in range(x.shape[1])]
                value = np.mean(results)
            elif input_type == "pseudotime":
                if x.ndim != 1:
                    raise ValueError("For global computation, x must be 1-dimensional.")
                value = metric_func(x, spatial_weights)
            else:
                raise ValueError(f"Unsupported input_type {input_type!r}")

            self.result[metric] = value
            self.logger.debug(f"{method_print_str!r} calculated for {input_type!r}): {value}")
        except Exception as e:
            self.logger.error(f"Failed to calculate {method_print_str!r} for {input_type!r}: {e}")
            self.result[metric] = np.nan

    def _calculate_morans_i(self, x: np.ndarray, spatial_weights: csr_matrix) -> float:
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

    def _calculate_gearys_c(self, x: np.ndarray, spatial_weights: csr_matrix) -> float:
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
        n = len(x)
        w = spatial_weights.sum()
        if w == 0:
            raise ValueError("Sum of spatial weights W must not be zero.")

        x_mean = np.mean(x)
        x_diff = x - x_mean
        denominator = 2 * np.sum(x_diff**2)

        # Compute (x_i - x_j)^2 for all i, j
        # Efficient computation using sparse matrix operations:
        # (x_i - x_j)^2 = x_i^2 + x_j^2 - 2 * x_i * x_j
        x_diff_squared = x_diff**2

        # Σi Σj w_ij x_i^2
        sum_wx_i_sq = spatial_weights.multiply(x_diff_squared[:, np.newaxis]).sum()

        # Σi Σj w_ij x_j^2
        sum_wx_j_sq = spatial_weights.multiply(x_diff_squared).sum()

        # Σi Σj w_ij x_i x_j
        cross_term = x_diff @ (spatial_weights @ x_diff)

        # Σi Σj w_ij (x_i - x_j)^2 = Σi Σj w_ij x_i^2 + Σi Σj w_ij x_j^2 - 2 Σi Σj w_ij x_i x_j
        numerator = sum_wx_i_sq + sum_wx_j_sq - 2 * cross_term

        gearys_c = ((n - 1) / (2 * w)) * (numerator / denominator)
        return gearys_c

    def _calculate_lisa(self, x: np.ndarray, spatial_weights: csr_matrix) -> np.ndarray:
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
        x_mean = np.mean(x)
        s2 = np.var(x, ddof=1)
        if s2 == 0:
            raise ValueError("Variance of x must not be zero.")

        x_diff = x - x_mean
        # Compute spatial lag: Σj w_ij (x_j - x̄)
        spatial_lag = spatial_weights @ x_diff
        # Calculate Local Moran's I for each unit
        lisa = (x_diff / s2) * spatial_lag
        return np.array(lisa).flatten()

    def _calculate_getis_ord_gi_star(self, x: np.ndarray, spatial_weights: csr_matrix) -> np.ndarray:
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
        n = len(x)
        x_mean = np.mean(x)
        s = np.std(x, ddof=1)
        if s == 0:
            raise ValueError("Standard deviation of x must not be zero.")

        # Σj w_ij x_j for each i
        sum_wx = spatial_weights @ x
        sum_wx = np.array(sum_wx).flatten()

        # Σj w_ij for each i
        sum_w = np.array(spatial_weights.sum(axis=1)).flatten()

        # Numerator: Σj w_ij x_j - x̄ Σj w_ij
        numerator = sum_wx - (x_mean * sum_w)

        # Σj w_ij^2 for each i
        sum_w2 = spatial_weights.power(2).sum(axis=1).A1

        # Denominator: S * sqrt((n Σj w_ij^2 - (Σj w_ij)^2) / (n - 1))
        denominator = s * np.sqrt((n * sum_w2 - sum_w**2) / (n - 1))

        # Handle division by zero by setting denominator to a small number
        denominator[denominator == 0] = 1e-10
        gi_star = numerator / denominator
        return np.array(gi_star).flatten()
