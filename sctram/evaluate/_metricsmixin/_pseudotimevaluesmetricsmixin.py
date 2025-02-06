#!/usr/bin/env python3

from typing import Any, Dict, List

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr, wasserstein_distance, gaussian_kde, cramervonmises_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, normalized_mutual_info_score, r2_score

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
        "r2_with_spline",
        "concordance_index",
        "dynamic_time_warping",
        "wasserstein_distance",
        "mutual_information",
        "mutual_information_kde",
        "cumulative_density_difference",
        "morans_i",
        "gearys_c",
        "lisa",
        "getis_ord_gi_star",
    ]

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
            elif metric == "r2_with_spline":
                self._calculate_r2_with_spline()
            elif metric == "concordance_index":
                self._calculate_concordance_index()
            elif metric == "dynamic_time_warping":
                self._calculate_dynamic_time_warping()
            elif metric == "wasserstein_distance":
                self._calculate_wasserstein_distance()
            elif metric == "mutual_information":
                self._calculate_mutual_information()
            elif metric == "mutual_information_kde":
                self._calculate_mutual_information_kde()
            elif metric == "cumulative_density_difference":
                self._calculate_cumulative_density_difference()
            elif metric == "morans_i":
                self._calculate_spatial_autocorrelation(metric, input_type="pseudotime")
            elif metric == "gearys_c":
                self._calculate_spatial_autocorrelation(metric, input_type="pseudotime")
            elif metric == "lisa":
                self._calculate_spatial_autocorrelation(metric, input_type="pseudotime")
            elif metric == "getis_ord_gi_star":
                self._calculate_spatial_autocorrelation(metric, input_type="pseudotime")
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

    def _calculate_r2_with_spline(self):
        k = 3

        # Sort the data based on x in ascending order
        sorted_indices = np.argsort(self.prepared_after_subset_inferred)
        x_sorted = self.prepared_after_subset_inferred[sorted_indices]
        y_sorted = self.prepared_after_subset_given[sorted_indices]
        
        uspline = UnivariateSpline(x_sorted, y_sorted, k=k)  # Fit the spline to the sorted data
        y_pred = uspline(self.prepared_after_subset_inferred)  # Predict y values using the original (unsorted) x data
        non_linear_r2 = r2_score(self.prepared_after_subset_given, y_pred)  # Calculate R-squared
        
        # Store the result
        self.result["r2_with_spline"] = non_linear_r2
        self.logger.debug(f"R-squared with Spline: {non_linear_r2}")


    def _calculate_concordance_index(self):
        """Calculates the Concordance Index between the inferred and reference pseudotime.
        
        This method estimates the probability that, for a randomly chosen pair of samples,
        the sample with the higher observed pseudotime also has a higher inferred pseudotime.
        A higher Concordance Index indicates better agreement between the inferred and reference
        orderings of samples. The index ranges from 0 (no concordance) to 1 (perfect concordance).
        
        Requires:
            - self.prepared_after_subset_given: array of reference pseudotimes.
            - self.prepared_after_subset_inferred: array of inferred pseudotimes.
        
        Updates:
            - self.result["concordance_index"]: stores the calculated Concordance Index.
        
        Raises:
            - ValueError: If the input arrays are not of the same length or are empty.
        """
        try:  # TODO: make all metrics' docstring like the one above: add requires and updates.
            n = len(self.prepared_after_subset_given)
            if n < 2:
                self.logger.warning("Not enough samples to compute Concordance Index.")
                self.result["concordance_index"] = np.nan
                return
            
            if len(self.prepared_after_subset_given) != len(self.prepared_after_subset_inferred):
                raise ValueError("Input arrays must be of the same length.")

            # Pairwise comparisons
            t_i = self.prepared_after_subset_given[:, np.newaxis]
            t_j = self.prepared_after_subset_given[np.newaxis, :]
            p_i = self.prepared_after_subset_inferred[:, np.newaxis]
            p_j = self.prepared_after_subset_inferred[np.newaxis, :]

            # Concordance and discordance conditions
            concordant = ((t_i < t_j) & (p_i < p_j)) | ((t_i > t_j) & (p_i > p_j))
            discordant = ((t_i < t_j) & (p_i > p_j)) | ((t_i > t_j) & (p_i < p_j))
            usable = (t_i != t_j) & (p_i != p_j)

            # Use only upper triangle to avoid duplicates
            upper_tri = np.triu(np.ones_like(concordant, dtype=bool), k=1)

            # Counting concordant and discordant pairs
            concordant_count = np.sum(concordant & usable & upper_tri)
            discordant_count = np.sum(discordant & usable & upper_tri)
            usable_count = np.sum(usable & upper_tri)

            self.logger.debug(
                f"Concordant pairs: {concordant_count}, "
                f"Discordant pairs: {discordant_count}, Usable pairs: {usable_count}"
            )

            if usable_count == 0:
                self.logger.warning("No usable pairs found to compute Concordance Index.")
                ci = np.nan
            else:
                ci = concordant_count / usable_count

            self.result["concordance_index"] = ci
            self.logger.info(f"Calculated Concordance Index: {ci}")

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
            from fastdtw import fastdtw  # type: ignore

            distance, path = fastdtw(
                self.prepared_after_subset_inferred,
                self.prepared_after_subset_given,
            )
            normalized_dtw = distance / len(path)
            self.result["dynamic_time_warping"] = normalized_dtw
            self.logger.debug(f"Dynamic Time Warping distance: {normalized_dtw}")
        except (ImportError, ModuleNotFoundError):
            self.logger.debug(f"'fastdtw' library is not found. Using fallback DTW implementation")
            x = np.array(self.prepared_after_subset_inferred)
            y = np.array(self.prepared_after_subset_given)
            
            # Create cost matrix
            n, m = len(x), len(y)
            dtw_matrix = np.full((n+1, m+1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(x[i-1] - y[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],    # Insertion
                        dtw_matrix[i, j-1],    # Deletion
                        dtw_matrix[i-1, j-1]   # Match
                    )

            # Backtrack to find path length
            i, j = n, m
            path_length = 0
            while i > 0 or j > 0:
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    min_val = min(dtw_matrix[i-1, j], 
                                dtw_matrix[i, j-1],
                                dtw_matrix[i-1, j-1])
                    if dtw_matrix[i-1, j-1] == min_val:
                        i -= 1
                        j -= 1
                    elif dtw_matrix[i-1, j] == min_val:
                        i -= 1
                    else:
                        j -= 1
                path_length += 1

            # Normalize by path length to match fastdtw's behavior
            normalized_dtw = dtw_matrix[n, m] / path_length if path_length > 0 else 0
            self.result["dynamic_time_warping"] = normalized_dtw
            self.logger.debug(f"Fallback DTW distance: {normalized_dtw}")
        except Exception as e:
            self.logger.debug(f"Error computing Dynamic Time Warping: {e}")
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
        def freedman_diaconis_bins(data):
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            n = len(data)
            bin_width = 2 * iqr / (n ** (1/3))
            bins = max(1, int((np.max(data) - np.min(data)) / bin_width))
            return bins

        bins_inferred = freedman_diaconis_bins(self.prepared_after_subset_inferred)
        bins_given = freedman_diaconis_bins(self.prepared_after_subset_given)
        bins = max(bins_inferred, bins_given)

        inferred_discrete = np.digitize(
            self.prepared_after_subset_inferred, 
            bins=np.histogram_bin_edges(self.prepared_after_subset_inferred, bins=bins))
        reference_discrete = np.digitize(
            self.prepared_after_subset_given, 
            bins=np.histogram_bin_edges(self.prepared_after_subset_given, bins=bins))
        nmi = normalized_mutual_info_score(inferred_discrete, reference_discrete)
        self.result["normalized_mutual_information"] = nmi

    def _calculate_mutual_information_kde(self):
        """KDE-based Mutual Information."""
        # Estimate marginal KDEs
        kde_inferred = gaussian_kde(self.prepared_after_subset_inferred)
        kde_given = gaussian_kde(self.prepared_after_subset_given)

        # Estimate joint KDE
        joint_data = np.vstack([self.prepared_after_subset_inferred, self.prepared_after_subset_given])
        joint_kde = gaussian_kde(joint_data)

        # Compute log-densities
        log_p_inferred = kde_inferred.logpdf(self.prepared_after_subset_inferred)
        log_p_given = kde_given.logpdf(self.prepared_after_subset_given)
        log_p_joint = joint_kde.logpdf(joint_data)

        # Calculate mutual information
        mi = np.mean(log_p_joint - (log_p_inferred + log_p_given))
        self.result["mutual_information_kde"] = mi

    def _calculate_cumulative_density_difference(self):
        """Calculates the differences between the cumulative density functions (CDFs) using KS and CvM tests.

        This method provides a non-parametric way to compare the distribution of two datasets, sensitive to 
        differences in both the location and shape of the distributions. The method utilizes the KS statistic to
        capture the maximum difference at any point between the CDFs and the CvM statistic to measure the overall 
        squared differences across the entire range of data.

        Advantages:
            - Non-parametric: Does not assume a specific distribution of data.
            - Sensitivity: Capable of detecting both global discrepancies across the entire distribution and 
            local discrepancies at specific points within the distribution.

        What it Measures:
            - KS Statistic: The maximum absolute difference between the CDFs of the two datasets, 
            indicating the most significant single-point discrepancy.
            - CvM Statistic: An integral of the squared differences between the CDFs, reflecting the 
            overall distribution shape discrepancies.

        Outputs:
            - Updates the `result` dictionary with the KS and CvM statistics.
            - Logs the computed KS and CvM statistics for debugging purposes.
        """
        ks_statistic, _ = ks_2samp(self.prepared_after_subset_inferred, self.prepared_after_subset_given)
        cvm_stat = cramervonmises_2samp(self.prepared_after_subset_inferred, self.prepared_after_subset_given).statistic
        
        self.result["cumulative_density_difference"] = ks_statistic
        self.result["cramer_von_mises"] = cvm_stat
        self.logger.debug(f"Cumulative density difference (KS statistic): {ks_statistic}")
        self.logger.debug(f"Cumulative density difference (Cramér-von Mises statistic): {cvm_stat}")
