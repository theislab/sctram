#!/usr/bin/env python3

import numpy as np

from sctram.evaluate._metrics import metrics as mmm
from sctram.evaluate._metrics.utils import Centroids
from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.utils._constants import neighbors_key, connectivities_key
from loguru import logger

_logger = logger.bind(name = "MetricsMixin")


class EmbeddingTrajectoryMetricsMixin(MetricsMixinBase):
    """A mixin class to compute various metrics comparing an embedding to a trajectory graph. The aim is to find out
    whether the trajectory is actually involved in the embedding using a battery of metrics.

    Attributes:
        prepared_after_subset_inferred (np.ndarray): The n-dimensional embedding matrix (shape: [n_cells, n_dims]).
        prepared_after_subset_given (nx.MultiDiGraph): The trajectory graph as a NetworkX MultiDiGraph.
        labels (np.ndarray): 1D array of strings representing cell-type labels for each cell in the embedding.
        result (dict): Dictionary to store computed metric results.
        logger (loguru.logger): Logger for debugging and information messages.
    """

    available_metrics = [
        "branch_silhouette_score",
        "sammons_stress",
        "embedding_distance_correlation",
        "normalized_mean_curvature",
        "graph_based_trustworthiness",
        "neighborhood_preservation_score",
        "directionality_preservation",
        "wasserstein_distance_embedding",
        "trajectory_cardinality_validation",
        "morans_i_embedding",
        "gearys_c_embedding"
    ]

    def _calculate(self):
        """Performs the evaluation by comparing the embedding and trajectory graph using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        
        n_neighbors = self.prepared_after_subset_inferred.uns[neighbors_key]["params"]["n_neighbors"]
        connectivities = self.prepared_after_subset_inferred.obsp[connectivities_key]
        centroids = Centroids(
            embedding=self.prepared_after_subset_inferred.X,
            labels_array=self.labels,
            outlier_threshold = 3.0
        )
        
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            
            if metric not in self.available_metrics:
                raise ValueError(f"Unknown metric {metric!r} specified.")
                
            if metric == "branch_silhouette_score":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    inferred_embedding = self.prepared_after_subset_inferred.X,
                    labels_array = self.labels,
                    metric= 'euclidean'
                )
                
            elif metric == "sammons_stress":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    centroids = centroids,
                    embedding_metric = "cosine"
                )
                
            elif metric == "embedding_distance_correlation":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    centroids = centroids
                )
                
            elif metric == "normalized_mean_curvature":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    centroids = centroids,
                    min_path_length = 3,
                    n_sample_points = 1000,
                    aggregation_method = 'median'
                )
                
            elif metric == "graph_based_trustworthiness":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    n_neighbors = n_neighbors,
                    precomputed_embedded_connectivities = connectivities
                )
                
            elif metric == "neighborhood_preservation_score":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    n_neighbors = n_neighbors,
                    precomputed_embedded_connectivities = connectivities,
                    threshold = 0.25
                )
                
            elif metric == "directionality_preservation":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    inferred_embedding = self.prepared_after_subset_inferred.X,
                    labels_array = self.labels,
                    centroids = centroids,
                    n_neighbors = n_neighbors,
                    precomputed_embedded_connectivities = connectivities,
                    pca_components = 1,
                    k_cells = n_neighbors
                )
            
            elif metric == "wasserstein_distance_embedding":
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    n_neighbors = n_neighbors,
                    precomputed_embedded_connectivities = connectivities,
                )
            
            elif metric == "trajectory_cardinality_validation":
                _logger.warning("Trajectory cardinality validation is not tested extensively!")
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    precomputed_embedded_connectivities = connectivities,
                    skip_single_branches = True
                )
                
            elif metric in ["morans_i_embedding", "gearys_c_embedding"]:
                _logger.warning(f"Spatial autocorrelation implementation of {metric!r} could be problematic.")
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    inferred_embedding = self.prepared_after_subset_inferred.X,
                    normalize_weights = True,
                    force_to_implementation = "package"
                )
        
            else:
                raise ValueError(f"Metric {metric!r} not included.")

            self.result[metric] = score
            self.logger.info(logger_message)

    # TODO: _calculate_metric_stability
    
    # Not only here but also for other ones. sth like:
    
    # def _calculate_metric_stability(self, n_iter=10, subsample_ratio=0.8):
    #     """Compute coefficient of variation for metrics across subsamples"""
    #     # Quantify metric robustness to subsampling/noise.
    #     original_embedding = self.prepared_after_subset_inferred.copy()
    #     original_labels = self.labels.copy()
        
    #     for metric in self.metrics:
    #         values = []
    #         for _ in range(n_iter):
    #             # Subsample data
    #             idx = np.random.choice(len(original_embedding), 
    #                                  int(len(original_embedding)*subsample_ratio),
    #                                  replace=False)
    #             self.prepared_after_subset_inferred = original_embedding[idx]
    #             self.labels = original_labels[idx]
                
    #             self._calculate_single_metric(metric)
    #             values.append(self.result[metric])
            
    #         # Compute stability score
    #         mean_val = np.nanmean(values)
    #         std_val = np.nanstd(values)
    #         self.stability_results[metric] = std_val / (mean_val + 1e-12)
            
    #     # Restore original data
    #     self.prepared_after_subset_inferred = original_embedding
    #     self.labels = original_labels
