#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.

import numpy as np
from typing import Any, Dict, Optional, Tuple

from sctram._constants import sctram_operate_key
from sctram.evaluate._base import EvaluationBase
from sctram.evaluate._metricsmixin._embeddingspairmetricsmixin import EmbeddingsPairMetricsMixin
from sctram.evaluate._metricsmixin._embeddingtrajectorymetricsmixin import EmbeddingTrajectoryMetricsMixin


class EmbeddingsPairEvaluation(EmbeddingsPairMetricsMixin, EvaluationBase):
    """Evaluation method to compare inferred embeddings with another embedding.

     This class compares the one embedding (numpy array of shape [n_samples, n_components])
    with another embedding, using data point labels, by means of various metrics to assess how well
    the embedding captures the trajectory structure.
    """

    pass  # TODO: complete the class.


class EmbeddingTrajectoryEvaluation(EmbeddingTrajectoryMetricsMixin, EvaluationBase):
    """Evaluation method to compare inferred embeddings with a given trajectory.

    This class compares the inferred embeddings (numpy array of shape [n_samples, n_components])
    with the given trajectory (networkx.MultiDiGraph), using various metrics to assess how well
    the embedding captures the trajectory structure.
    """

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

    def get_result(self) -> Any:
        """Retrieves the result of the embedding based evaluation."""
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result
    
    def _verify_inferred_trajectory(self, inferred_embedding: np.ndarray) -> np.ndarray:
        """Verifies the embedding."""
        if not isinstance(inferred_embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array.")
        if inferred_embedding.ndim != 2:
            raise ValueError("Embedding must be a 2D array.")
        if inferred_embedding.shape[0] <= 1 or inferred_embedding.shape[1] <= 1:
            raise ValueError("Embedding must have more than one row and one column.")
        if not np.all(np.isfinite(inferred_embedding)):
            raise ValueError("Embedding contains NaN or infinite values.")
        if inferred_embedding.size == 0:
            raise ValueError("Embedding cannot be empty.")
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