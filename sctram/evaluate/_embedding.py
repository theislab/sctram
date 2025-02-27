#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import numpy as np
import anndata as ad

from sctram.evaluate._base import EvaluationBase
from sctram.utils._constants import neighbors_key, connectivities_key, distances_key
from sctram.evaluate._metrics import metrics as mmm
from sctram.evaluate._metrics._src.utils import Centroids



class EmbeddingsPairEvaluation(EvaluationBase):
    """Evaluation method to compare inferred embeddings with another embedding.

     This class compares the one embedding (numpy array of shape [n_samples, n_components])
    with another embedding, using data point labels, by means of various metrics to assess how well
    the embedding captures the trajectory structure.
    """

    pass  # TODO: complete the class.


class EmbeddingTrajectoryEvaluation(EvaluationBase):
    """Evaluation method to compare inferred embeddings with a given trajectory.

    This class compares the inferred embeddings (numpy array of shape [n_samples, n_components])
    with the given trajectory (networkx.MultiDiGraph), using various metrics to assess how well
    the embedding captures the trajectory structure.
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

    def __init__(
        self,
        method_params: Dict[str, Any],
        # subset_params: Optional[Dict[str, Any]] = None,
        # TODO: complete subset method for EmbeddingTrajectoryEvaluation
        # prepare_params: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the embedding evaluation method."""  # noqa
        super().__init__(
            method_params=method_params,
            subset_params=None,
            prepare_params_before_subset=None,
            prepare_params_after_subset=None,
        )
        self.logger.debug(f"Initialized EmbeddingTrajectoryEvaluation with metrics: {self.metrics}")

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
                score, logger_message = mmm[metric]["with_desc"](
                    given_graph = self.prepared_after_subset_given,
                    labels_array = self.labels,
                    precomputed_embedded_connectivities = connectivities,
                    skip_single_branches = True,
                    spectral_n_init = 50,
                    min_cells_per_branch = 50,
                    random_state = 0
                )
                
            elif metric in ["morans_i_embedding", "gearys_c_embedding"]:
                self.logger.warning(f"Implementation of the spatial metric {metric!r} may be problematic.")
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

    def get_result(self) -> Any:
        """Retrieves the result of the embedding based evaluation."""
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result

    def _verify_inferred_trajectory(self, inferred_embedding: ad.AnnData) -> ad.AnnData:
        """Verifies the embedding."""
        if not isinstance(inferred_embedding, ad.AnnData):
            raise ValueError("Embedding must be a anndata object.")
        if inferred_embedding.X.shape[0] <= 1 or inferred_embedding.X.shape[1] <= 1:
            raise ValueError("Embedding must have more than one row and one column.")
        if not np.all(np.isfinite(inferred_embedding.X)):
            raise ValueError("Embedding contains NaN or infinite values.")
        if any([k not in inferred_embedding.obsp.keys() for k in [connectivities_key, distances_key]]) or (neighbors_key not in inferred_embedding.uns.keys()):
            raise ValueError(f"Neigbors should have been calculated.")

        return inferred_embedding

    def _verify_labels_data_specific(self):
        """Verifies that labels is consistent with input trajectory and/or inferred embedding.

        The labels for the embedding evaluation should be with the same size as the number of data points.

        Raises:
            ValueError: there is inconsistency.
        """
        self.logger.debug("Checking the consistency between the given graph and labels.")
        if len(self.inferred_trajectory) != len(self.labels):
            raise ValueError("Datapoint amount in the inferred embedding does not match the number of given labels.")

    def _prepare_before_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the trajectories after subsetting.

        Raises:
            NotImplementedError: This method is not supposed to be running.
        """
        raise NotImplementedError("This method is not supposed to be running.")

    def _subset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subsets the trajectories based on `subset_params`.

        Raises:
            NotImplementedError: Will be implemented.
        """
        raise NotImplementedError("Will be implemented.")

    def _prepare_after_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the trajectories after subsetting.

        Raises:
            NotImplementedError: This method is not supposed to be running.
        """
        raise NotImplementedError("This method is not supposed to be running.")
