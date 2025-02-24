#!/usr/bin/env python3

from sctram.evaluate._metrics import metrics as mmm

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class PseudotimeValuesMetricsMixin(MetricsMixinBase, SpatialMetricsMixin):
    """Metrics for comparing two 1d numpya arrays."""

    available_metrics = [
        "pearson_correlation",
        "spearman_correlation",
        "kendall_correlation",
        "mse",
        "mae",
        "r_squared_with_spline",
        "r_squared",
        "concordance_index",
        "dtw_distance",
        "wasserstein_distance",
        "normalized_mutual_information",
        "mutual_information_kde",
        "cdf_kolmogorov_smirnov",
        "cdf_cramer_von_mises",
        # spatial metrics missing
    ]

    def _calculate(self):
        """Performs the evaluation by comparing the pseudotime arrays using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            
            if metric in self.available_metrics:
                kwargs = dict(
                    given_pseudotime_array = self.prepared_after_subset_given,
                    inferred_pseudotime_array = self.prepared_after_subset_inferred,
                    validate_result = True
                )

                score, logger_message = mmm[metric]["with_desc"](**kwargs)
                self.result[metric] = score
                self.logger.info(logger_message)
                
            else:
                raise ValueError(f"Unknown metric {metric!r} specified.")
