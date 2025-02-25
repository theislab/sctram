#!/usr/bin/env python3

import numpy as np
from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metrics import metrics as mmm
from sctram.utils._utils import Utils as U

class AdjacencyMetricsMixin(MetricsMixinBase):
    """Metrics to compare two adjacency matrixes, that is, two networkx graphs."""

    available_metrics = [
        "frobenius",
        "l1_norm",
        "accuracy",
        "graph_edit_distance",
        "spectral_distance",
        "jaccard_similarity",
        "hamming_distance",
        "precision",
        "recall",
        "f1_score",
        "permutation_marginalized_ssim",
        "mantel_correlation",
        "average_shortest_path_difference",
        "laplacian_spectral_emd",
        "clustering_coeff_diff",
        "gdv_similarity",
        "weisfeiler_lehman_distance",
        "gin_gnn_similarity",
        "maximum_common_subgraph_distance",
        "random_walk_kernel_distance",
        "persistence_diagram_distance",
    ]
    
    _paga_threshold = 0.4

    def _calculate(self):
        """Performs the evaluation by comparing the adjacency matrices using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            
            if metric in self.available_metrics:

                kwargs = dict(
                    given_adjacency_matrix = self.prepared_after_subset_given,
                    inferred_adjacency_matrix = self.prepared_after_subset_inferred,
                )
                
                if U.requires_argument(mmm[metric]["base_before_val"], arg_name="threshold"):
                    score, logger_message = mmm[metric]["with_desc"](threshold=self._paga_threshold, **kwargs)
                else:
                    score, logger_message = mmm[metric]["with_desc"](**kwargs)
                    
                self.result[metric] = score
                self.logger.info(logger_message)
            else:
                raise ValueError(f"Unknown metric {metric!r} specified.")
                                