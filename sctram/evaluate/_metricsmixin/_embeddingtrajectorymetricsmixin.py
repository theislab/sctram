#!/usr/bin/env python3

from sctram.evaluate._metrics import metrics as mmm
from sctram.evaluate._metrics.utils import Centroids
from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin
from sctram.utils._constants import neighbors_key, connectivities_key


class EmbeddingTrajectoryMetricsMixin(MetricsMixinBase, SpatialMetricsMixin):
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
        # spatial metrics missing
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
        
            else:
                raise ValueError(f"Metric {metric!r} not included.")

            self.result[metric] = score
            self.logger.info(logger_message)
