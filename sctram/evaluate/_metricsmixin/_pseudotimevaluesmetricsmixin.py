#!/usr/bin/env python3

import logging
from typing import Any, Dict, List

import numpy as np
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, mutual_info_score, r2_score

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class PseudotimeValuesMetricsMixin(MetricsMixinBase, SpatialMetricsMixin):
    """Metrics for comparing two 1d numpya arrays."""

    available_metrics = [
        "pearson",
        "spearman",
        "kendall",
        "mse",
        "mae",
        "r2",
        "concordance_index",
        "dynamic_time_warping",
        "wasserstein_distance",
        "mutual_information",
        "cumulative_density_difference",
        # New spatial metrics
        "morans_i",
        "gearys_c",
        "local_morans_i",
        "getis_ord_gi_star",
    ]

    # Declare expected attributes with type annotations
    prepared_after_subset_given: np.ndarray
    prepared_after_subset_inferred: np.ndarray
    result: Dict[str, Any]
    metrics: List[str]
    method_params: Dict[str, Any]
    logger: logging.Logger

    def _calculate(self):
        """Performs the evaluation by comparing the pseudotime arrays using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "pearson":
                self._calculate_pearson()
            elif metric == "spearman":
                self._calculate_spearman()
            elif metric == "kendall":
                self._calculate_kendall()
            elif metric == "mse":
                self._calculate_mse()
            elif metric == "mae":
                self._calculate_mae()
            elif metric == "r2":
                self._calculate_r2()
            elif metric == "concordance_index":
                self._calculate_concordance_index()
            elif metric == "dynamic_time_warping":
                self._calculate_dynamic_time_warping()
            elif metric == "wasserstein_distance":
                self._calculate_wasserstein_distance()
            elif metric == "mutual_information":
                self._calculate_mutual_information()
            elif metric == "cumulative_density_difference":
                self._calculate_cumulative_density_difference()
            elif metric == "morans_i":
                self._calculate_morans_i()
            elif metric == "gearys_c":
                self._calculate_gearys_c()
            elif metric == "local_morans_i":
                self._calculate_local_morans_i()
            elif metric == "getis_ord_gi_star":
                self._calculate_getis_ord_gi_star()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _calculate_pearson(self):
        """Calculates the Pearson correlation coefficient between the inferred and reference pseudotime.

        Advantage:
            - Captures how well the inferred pseudotime correlates with the true progression along the trajectory.
            - Measures linear relationship between two variables.
            - Sensitive to linear associations.
            - Reflects the global structure of the trajectory.

        Difference:
            - Captures only linear correlations, not non-linear relationships.
            - Based on geodesic distances derived from the given trajectory graph.

        Sensitivity:
            - Sensitive to outliers.
            - Assumes both variables are normally distributed.

        What it Measures:
            - The degree of linear correlation between the inferred and reference pseudotime.
            - Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
        """
        corr, p_value = pearsonr(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        self.result["pearson"] = corr
        self.result["pearson_p_value"] = p_value
        self.logger.debug(f"Pearson correlation: {corr}, p-value: {p_value}")

    def _calculate_spearman(self):
        """Calculates the monotonicity by Spearman correlation between the inferred and reference pseudotime.

        Advantage:
            - Measures monotonic relationships, both linear and non-linear.
            - Useful for assessing consistent progression.
            - Less sensitive to outliers than Pearson correlation.

        Difference:
            - Based on rank-ordering of data rather than raw values.

        Sensitivity:
            - Sensitive to changes in the order of data points.
            - Robust to non-normal distributions.

        What it Measures:
            - The degree to which the relationship between two variables can be described using a monotonic function.
            - Values range from -1 to 1, similar to Pearson correlation.
        """
        corr, p_value = spearmanr(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        self.result["spearman"] = corr
        self.result["spearman_p_value"] = p_value
        self.logger.debug(f"Spearman correlation: {corr}, p-value: {p_value}")

    def _calculate_kendall(self):
        """Calculates the Kendall's tau correlation coefficient between the inferred and reference pseudotime.

        Advantage:
            - Non-parametric measure of rank correlation.
            - Measures the strength of dependence between two variables.

        Difference:
            - Considers the number of concordant and discordant pairs.

        Sensitivity:
            - Sensitive to changes in the ordering of data points.
            - Less affected by ties in data compared to Spearman correlation.

        What it Measures:
            - The probability that the orderings of the two variables are consistent.
            - Values range from -1 to 1.
        """
        tau, p_value = kendalltau(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        self.result["kendall"] = tau
        self.result["kendall_p_value"] = p_value
        self.logger.debug(f"Kendall's tau: {tau}, p-value: {p_value}")

    def _calculate_mse(self):
        """Calculates the Mean Squared Error (MSE) between the inferred and reference pseudotime.

        Advantage:
            - Penalizes larger errors more than smaller ones due to squaring.
            - Useful for capturing the magnitude of prediction errors.

        Difference:
            - Focuses on the average of squared differences.

        Sensitivity:
            - Highly sensitive to outliers because errors are squared.
            - Requires data on the same scale.

        What it Measures:
            - The average of the squares of the differences between the inferred and reference pseudotime.
            - Lower values indicate better agreement.
        """
        mse = mean_squared_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mse"] = mse
        self.logger.debug(f"Mean Squared Error: {mse}")

    def _calculate_mae(self):
        """Calculates the Mean Absolute Error (MAE) between the inferred and reference pseudotime.

        Advantage:
            - Provides a linear score which does not emphasize large errors.
            - More robust to outliers than MSE.

        Difference:
            - Focuses on the average of absolute differences.

        Sensitivity:
            - Less sensitive to outliers compared to MSE.
            - Requires data on the same scale.

        What it Measures:
            - The average absolute difference between the inferred and reference pseudotime.
            - Lower values indicate better agreement.
        """
        mae = mean_absolute_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mae"] = mae
        self.logger.debug(f"Mean Absolute Error: {mae}")

    def _calculate_r2(self):
        """Calculates the R-squared (coefficient of determination) between the inferred and reference pseudotime.

        Advantage:
            - Indicates the proportion of variance in the reference pseudotime that is predictable from the inferred pseudotime.
            - Commonly used in regression analysis.

        Difference:
            - Provides a relative measure compared to the variance in the data.

        Sensitivity:
            - Sensitive to the range of data.
            - Can be misleading with non-linear relationships.

        What it Measures:
            - The goodness of fit of the inferred pseudotime to the reference pseudotime.
            - Values range from negative infinity to 1; higher values indicate better fit.
        """
        r2 = r2_score(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["r2"] = r2
        self.logger.debug(f"R-squared: {r2}")

    def _calculate_concordance_index(self):
        """Calculates the Concordance Index between the inferred and reference pseudotime.

        Advantage:
            - Measures the agreement between the ordering of inferred and reference pseudotime.
            - Handles censored data, commonly used in survival analysis.

        Difference:
            - Focuses on the pairwise comparison of ordering.

        Sensitivity:
            - Sensitive to the correct ranking of pairs.
            - Less affected by the exact values of pseudotime.

        What it Measures:
            - The probability that, for a randomly chosen pair of samples, the sample with the
                higher observed pseudotime also has a higher inferred pseudotime.
            - Values range from 0 to 1; higher values indicate better concordance.
        """
        try:
            n = len(self.prepared_after_subset_given)
            if n < 2:
                self.logger.warning("Not enough samples to compute Concordance Index.")
                self.result["concordance_index"] = np.nan
                return

            # Create pairwise comparison matrices
            # Using broadcasting to create matrices T_i and T_j for all i < j
            t_i = self.prepared_after_subset_given[:, np.newaxis]
            t_j = self.prepared_after_subset_given[np.newaxis, :]
            p_i = self.prepared_after_subset_inferred[:, np.newaxis]
            p_j = self.prepared_after_subset_inferred[np.newaxis, :]

            # Boolean matrices indicating concordant and discordant pairs
            concordant = ((t_i < t_j) & (p_i < p_j)) | ((t_i > t_j) & (p_i > p_j))
            discordant = ((t_i < t_j) & (p_i > p_j)) | ((t_i > t_j) & (p_i < p_j))
            usable = (t_i != t_j) & (p_i != p_j)

            # Consider only the upper triangle of the matrices to avoid duplicate pairs and self-pairs
            upper_tri = np.triu(np.ones_like(concordant, dtype=bool), k=1)

            # Calculate counts
            concordant_count = np.sum(concordant & usable & upper_tri)
            discordant_count = np.sum(discordant & usable & upper_tri)
            usable_count = np.sum(usable & upper_tri)

            self.logger.debug(f"Concordant pairs: {concordant_count}")
            self.logger.debug(f"Discordant pairs: {discordant_count}")
            self.logger.debug(f"Usable pairs: {usable_count}")

            if usable_count == 0:
                self.logger.warning("No usable pairs found to compute Concordance Index.")
                ci = np.nan
            else:
                ci = concordant_count / usable_count

            self.result["concordance_index"] = ci
            self.logger.debug(f"Concordance Index: {ci}")

        except Exception as e:
            self.logger.error(f"Error computing Concordance Index: {e}")
            self.result["concordance_index"] = np.nan

    def _calculate_dynamic_time_warping(self):
        """Calculates the Dynamic Time Warping (DTW) distance between the inferred and reference pseudotime.

        Advantage:
            - Captures similarities between sequences that may vary in time or speed.
            - Aligns sequences by warping the time axis optimally.

        Difference:
            - Considers the optimal alignment between sequences.

        Sensitivity:
            - Sensitive to the shape of the sequences.
            - Can be influenced by local variations.

        What it Measures:
            - The minimal cumulative distance required to match the inferred and reference pseudotime sequences.
            - Lower values indicate better alignment.
        """
        try:
            from dtw import dtw  # type: ignore

            distance, _, _, _ = dtw(
                self.prepared_after_subset_inferred,
                self.prepared_after_subset_given,
                dist=lambda x, y: abs(x - y),
            )
            self.result["dynamic_time_warping"] = distance
            self.logger.debug(f"Dynamic Time Warping distance: {distance}")
        except ImportError:
            self.logger.error("DTW package is not installed. Dynamic Time Warping cannot be computed.")
            self.result["dynamic_time_warping"] = np.nan

    def _calculate_wasserstein_distance(self):
        """Calculates the Wasserstein distance between the inferred and reference pseudotime distributions.

        Advantage:
            - Measures the distance between two probability distributions.
            - Sensitive to the overall shape and location differences.

        Difference:
            - Also known as Earth Mover's Distance; considers the 'cost' of transforming one distribution into another.

        Sensitivity:
            - Sensitive to shifts and spreads in distributions.
            - Reflects both location and dispersion differences.

        What it Measures:
            - The minimum amount of 'work' needed to transform the inferred pseudotime distribution into the reference distribution.
            - Lower values indicate more similar distributions.
        """
        wd = wasserstein_distance(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        self.result["wasserstein_distance"] = wd
        self.logger.debug(f"Wasserstein distance: {wd}")

    def _calculate_mutual_information(self):
        """Calculates the Mutual Information between the inferred and reference pseudotime.

        Advantage:
            - Measures any kind of dependency between variables, both linear and non-linear.
            - Non-parametric and model-free.

        Difference:
            - Based on information theory; captures shared information.

        Sensitivity:
            - Sensitive to the amount of shared information.
            - Requires discretization of continuous variables.

        What it Measures:
            - The reduction in uncertainty about one variable given knowledge of the other.
            - Higher values indicate greater dependency.
        """
        # Discretize the pseudotime values
        bins = self.method_params.get("mi_bins", 10)
        inferred_discrete = np.digitize(self.prepared_after_subset_inferred, bins=np.linspace(0, 1, bins))
        reference_discrete = np.digitize(self.prepared_after_subset_given, bins=np.linspace(0, 1, bins))
        mi = mutual_info_score(inferred_discrete, reference_discrete)
        self.result["mutual_information"] = mi
        self.logger.debug(f"Mutual Information: {mi}")

    def _calculate_cumulative_density_difference(self):
        """Calculates the difference between the cumulative density functions of the inferred and reference pseudotime.

        Advantage:
            - Non-parametric test to compare distributions.
            - Sensitive to differences in both location and shape of distributions.

        Difference:
            - Uses the Kolmogorov-Smirnov statistic.

        Sensitivity:
            - Sensitive to any differences between the cumulative distributions.
            - Reflects both global and local discrepancies.

        What it Measures:
            - The maximum difference between the cumulative distributions of inferred and reference pseudotime.
            - Values range from 0 to 1; higher values indicate greater differences.
        """
        statistic, p_value = ks_2samp(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        self.result["cumulative_density_difference"] = statistic
        self.result["cumulative_density_p_value"] = p_value
        self.logger.debug(f"Cumulative density difference (KS statistic): {statistic}, p-value: {p_value}")

    def _calculate_morans_i(self):
        """Calculates Moran's I for the adjacency matrix. See the method in `SpatialMixin` class."""
        x = self.prepared_after_subset_given
        spatial_weights = self._compute_spatial_weights(x, "pseudotime", **self.method_params)
        morans_i = self.calculate_morans_i(x, spatial_weights)
        self.result["morans_i"] = morans_i
        self.logger.debug(f"Moran's I: {morans_i}")

    def _calculate_gearys_c(self):
        """Calculates Geary's C for the adjacency matrix. See the method in `SpatialMixin` class."""
        x = self.prepared_after_subset_given
        spatial_weights = self._compute_spatial_weights(x, "pseudotime", **self.method_params)
        gearys_c = self.calculate_gearys_c(x, spatial_weights)
        self.result["gearys_c"] = gearys_c
        self.logger.debug(f"Geary's C: {gearys_c}")

    def _calculate_local_morans_i(self):
        """Calculates Local Moran's I (LISA) for the adjacency matrix. See the method in `SpatialMixin` class."""
        x = self.prepared_after_subset_given
        spatial_weights = self._compute_spatial_weights(x, "pseudotime", **self.method_params)
        lisa = self.calculate_lisa(x, spatial_weights)
        self.result["local_morans_i"] = lisa
        self.logger.debug(f"Local Moran's I: {lisa}")

    def _calculate_getis_ord_gi_star(self):
        """Calculates the Getis-Ord Gi* statistic for the adjacency matrix. See the method in `SpatialMixin` class."""
        x = self.prepared_after_subset_given
        spatial_weights = self._compute_spatial_weights(x, "pseudotime", **self.method_params)
        gi_star = self.calculate_getis_ord_gi_star(x, spatial_weights)
        self.result["getis_ord_gi_star"] = gi_star
        self.logger.debug(f"Getis-Ord Gi* statistic: {gi_star}")
