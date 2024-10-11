#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, mutual_info_score, r2_score

from sctram.evaluate._base import EvaluationBase


class PseudotimeEvaluation(EvaluationBase):
    """Evaluation method to compare inferred pseudotime with a given trajectory graph.

    This class compares the inferred pseudotime values (1D array) with the given trajectory,
    which is represented as a graph over cell types (`InputTrajectories`).
    The comparison is facilitated through the cell type labels associated with each data point.
    Multiple metrics are used to assess the similarity or agreement between the inferred pseudotime
    and the progression implied by the given trajectory.

    Metrics include both basic statistical measures and advanced methods:
    - Basic Metrics: Pearson correlation, Spearman correlation, Kendall's tau, MSE, MAE, R², Concordance Index.
    - Advanced Metrics: Dynamic Time Warping, Wasserstein distance, Mutual Information, Monotonicity,
      Geodesic distance correlation, Cumulative density difference.

    Each metric provides different insights into the relationship between the inferred and reference pseudotime,
    capturing aspects such as linear correlation, rank correlation, error magnitude, ordering consistency,
    distribution similarity, and more.
    """

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
    ]

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the pseudotime evaluation method."""  # noqa
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=prepare_params_before_subset,
            prepare_params_after_subset=prepare_params_after_subset,
        )
        self.logger.debug(f"Initialized PseudotimeEvaluation with metrics: {self.metrics}")

        if prepare_params_before_subset is None or "method" not in prepare_params_before_subset.keys():
            raise ValueError("Determine a method how to estimate pseudotime from adjacency matrix.")

    def _verify_inferred_trajectory(self, inferred_pseudotime: Any) -> np.ndarray:
        """Verifies the inferred pseudotime.

        Ensures that the inferred pseudotime is a 1D numpy array.

        Args:
            inferred_pseudotime (Any): The inferred pseudotime.

        Returns:
            np.ndarray: The verified inferred pseudotime.

        Raises:
            ValueError: If the inferred pseudotime is not a 1D numpy array.
        """
        if not isinstance(inferred_pseudotime, np.ndarray):
            raise ValueError("Inferred pseudotime must be a numpy array.")
        if inferred_pseudotime.ndim != 1:
            raise ValueError("Inferred pseudotime must be a 1D array.")
        return inferred_pseudotime

    def _prepare_before_subset(self, given_trajectory: Any, inferred_pseudotime: Any) -> Tuple[Any, Any]:
        """Prepares the data before subsetting.

        Computes a reference pseudotime for each data point based on the given trajectory and labels.

        Args:
            given_trajectory (Any): The given trajectory as a graph.
            inferred_pseudotime (Any): The inferred pseudotime.

        Returns:
            Tuple[Any, Any]: The prepared reference pseudotime and inferred pseudotime.
        """
        self.logger.debug("Computing reference pseudotime from given trajectory.")

        # TODO: given_trajectory should be converted into reference_pseudotime by `LabelAdjacencyPseudotimeConverter`.
        # TODO: also have a look at the required parameters, and correctly enter it in `__init__`.
        reference_pseudotime = given_trajectory

        return reference_pseudotime, inferred_pseudotime

    def _subset(self, reference_pseudotime: Any, inferred_pseudotime: Any) -> Tuple[Any, Any]:
        """Subsets the data based on `subset_params`.

        For pseudotime data, subsetting might involve selecting specific cells or ranges based on pseudotime values.

        Args:
            reference_pseudotime (Any): The prepared reference pseudotime before subsetting.
            inferred_pseudotime (Any): The prepared inferred pseudotime before subsetting.

        Returns:
            Tuple[Any, Any]: The subsetted reference and inferred pseudotime arrays.
        """
        indices = np.arange(len(reference_pseudotime))

        # Subsetting by pseudotime range specified in subset_params
        min_pt = self.subset_params.get("min_pseudotime", None)
        max_pt = self.subset_params.get("max_pseudotime", None)

        if min_pt is not None:
            indices = indices[reference_pseudotime >= min_pt]
        if max_pt is not None:
            indices = indices[reference_pseudotime <= max_pt]

        self.logger.debug(f"Subsetting data to indices: {indices}")

        subset_reference = reference_pseudotime[indices]
        subset_inferred = inferred_pseudotime[indices]

        # TODO: get the self.subset_params.get("labels", None), subset based on labels.

        return subset_reference, subset_inferred

    def _prepare_after_subset(self, reference_pseudotime: Any, inferred_pseudotime: Any) -> Tuple[Any, Any]:
        """Prepares the data after subsetting.

        Optionally normalizes the pseudotime values to a common scale.

        Args:
            reference_pseudotime (Any): The reference pseudotime.
            inferred_pseudotime (Any): The inferred pseudotime.

        Returns:
            Tuple[Any, Any]: The prepared reference pseudotime and inferred pseudotime.
        """
        self.logger.debug("Normalizing pseudotime values after subsetting.")

        if self.prepare_params_after_subset.get("normalize", True):
            # Normalize pseudotime values to [0,1]
            ref_min = reference_pseudotime.min()
            ref_max = reference_pseudotime.max()
            if ref_max > ref_min:
                reference_pseudotime_norm = (reference_pseudotime - ref_min) / (ref_max - ref_min)
            else:
                reference_pseudotime_norm = reference_pseudotime

            inf_min = inferred_pseudotime.min()
            inf_max = inferred_pseudotime.max()
            if inf_max > inf_min:
                inferred_pseudotime_norm = (inferred_pseudotime - inf_min) / (inf_max - inf_min)
            else:
                inferred_pseudotime_norm = inferred_pseudotime

            self.logger.debug("Pseudotime values normalized to [0,1].")
        else:
            reference_pseudotime_norm = reference_pseudotime
            inferred_pseudotime_norm = inferred_pseudotime
            self.logger.debug("Normalization skipped.")

        return reference_pseudotime_norm, inferred_pseudotime_norm

    def _verify_labels(self, labels: np.ndarray) -> np.ndarray:
        """Verifies the labels.

        Checks that labels are a 1D numpy array of the same length as the inferred pseudotime.

        Args:
            labels (np.ndarray): The labels corresponding to each data point.

        Returns:
            np.ndarray: The verified labels.

        Raises:
            ValueError: If labels is not a 1D array of appropriate length.
        """
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels must be a numpy array.")
        if labels.ndim != 1:
            raise ValueError("Labels must be a 1D array.")
        if labels.shape[0] != self.inferred_trajectory.shape[0]:
            raise ValueError("Labels must have the same length as inferred pseudotime.")
        return labels

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

    def get_result(self) -> Any:
        """Retrieves the result of the trajectory evaluation.

        Returns:
            Any: A dictionary containing the results of all evaluated metrics.

        Raises:
            ValueError: If the result is not available.
        """
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result
