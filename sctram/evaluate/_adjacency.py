#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sctram._constants import sctram_operate_key
from sctram._utils import Utils
from sctram.evaluate._base import EvaluationBase
from sctram.evaluate._metricsmixin._adjacencymetricsmixin import AdjacencyMetricsMixin


class AdjacencyMatrixEvaluation(Utils, AdjacencyMetricsMixin, EvaluationBase):
    """Evaluation method to compare given and inferred adjacency matrices using advanced metrics.

    This class compares the adjacency matrix of the given trajectory (converted from a NetworkX graph)
    with the inferred adjacency matrix using a variety of metrics that assess structural and functional
    similarities/differences from multiple perspectives.
    """

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        # prepare_params: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the adjacency matrix evaluation method."""  # noqa
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=None,
            prepare_params_after_subset=None,
        )
        # Prepare the input/inferred regardless of the prepare params, as there is no real parameter
        self.prepare_params_before_subset[sctram_operate_key] = True
        self.logger.debug(f"Initialized AdjacencyMatrixEvaluation with metrics: {self.metrics}")

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

    def _verify_inferred_trajectory(self, inferred_adjacency: np.ndarray) -> np.ndarray:
        """Verifies the inferred trajectory as adjacency matrix.

        Ensures that the inferred trajectory is in an acceptable format for comparison, is symmetric,
        has zeros on its diagonal, and contains values between 0 and 1 inclusive.

        Args:
            inferred_adjacency (np.ndarray): The inferred trajectory adjacency matrix.

        Returns:
            np.ndarray: The verified inferred trajectory.

        Raises:
            ValueError: If the inferred trajectory is not a numpy array, is not 2-dimensional, is not square,
                is not symmetric, has non-zero diagonal elements, or contains values outside the range [0, 1].
        """
        if not isinstance(inferred_adjacency, np.ndarray):
            raise ValueError("Inferred trajectory must be a numpy array.")
        if inferred_adjacency.ndim != 2:
            raise ValueError("Inferred trajectory must be a 2D array.")
        if inferred_adjacency.shape[0] != inferred_adjacency.shape[1]:
            raise ValueError("Inferred trajectory must be square.")
        if not np.array_equal(inferred_adjacency, inferred_adjacency.T):
            raise ValueError("Inferred trajectory must be symmetric.")
        if np.any(inferred_adjacency.diagonal() != 0):
            raise ValueError("All diagonal elements must be zero.")
        if not np.all((inferred_adjacency >= 0) & (inferred_adjacency <= 1)):
            raise ValueError("All values must be within the range [0, 1].")

        return inferred_adjacency

    def _verify_labels_data_specific(self):
        """Verifies that labels is consistent with input trajectory and/or inferred trajectory.

        The labels for the adjacency evaluation should be unique and with the same size as the number of nodes.

        Raises:
            ValueError: there is inconsistency.
        """
        self.logger.debug("Checking the consistency between the given graph and labels.")
        if len(self.given_trajectory.nodes()) != len(self.labels):
            raise ValueError("Number of nodes in the matrix does not match the number of given labels")
        elif set(self.given_trajectory.nodes()) != set(self.labels):
            raise ValueError("Node names in the graph and label array do not match")

    def _prepare_before_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the trajectories before subsetting by converting them to adjacency matrices.

        Converts both the given and inferred trajectories to adjacency matrices to ensure compatibility
        for comparison. Validates that both matrices have identical shapes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared given and inferred adjacency matrices.

        Raises:
            ValueError: If the adjacency matrices cannot be prepared due to incompatible shapes or types.
        """
        self.logger.debug("Converting given trajectory to adjacency matrix.")
        symetrical_graph = self.given_trajectory.to_symetrical_multidigraph()  # to have symetrical adjacency matrices
        given_adj_matrix = nx.to_numpy_array(symetrical_graph, nodelist=self.labels)
        self.logger.debug(f"Given adjacency matrix shape: {given_adj_matrix.shape}")
        # Note: this method assumes the inferred numpy array is already in correct order of labels.
        # See the paga return_mode `label` and `adjacency`.
        self.logger.debug(f"Inferred adjacency matrix is a numpy array with shape: {self.inferred_trajectory.shape}")

        if given_adj_matrix.shape != self.inferred_trajectory.shape:
            raise ValueError("Given and inferred adjacency matrices must have the same shape for comparison.")

        self.logger.debug("Assigned prepared adjacency matrices before subsetting.")
        return given_adj_matrix, self.inferred_trajectory.copy()

    def _subset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subsets the trajectories based on `subset_params`.

        For adjacency matrices, subsetting typically involves selecting a subset of labels (rows and columns).
        This method supports subsetting by specifying either labels to keep or labels to remove.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The subsetted given and inferred adjacency matrices.

        Raises:
            ValueError: If subsetting parameters are invalid or result in incompatible matrices.
        """
        self.logger.debug("Subsetting trajectories based on subset parameters.")
        labels_to_keep = self.sget(self.subset_params, "labels_to_keep", None)
        labels_to_remove = self.sget(self.subset_params, "labels_to_remove", None)

        if labels_to_keep is not None and labels_to_remove is None:
            self.logger.debug(f"Subsetting to keep nodes: {labels_to_keep}")
            if isinstance(labels_to_keep, (list, np.ndarray, pd.Index)):
                keep_indices = [label for label in labels_to_keep if label in self.labels]
                if len(keep_indices) == 0:
                    raise ValueError("No matching labels found to keep in subset.")
            else:
                raise ValueError("'labels_to_keep' must be a list, numpy array, or Pandas Index.")
        elif labels_to_remove is not None and labels_to_keep is None:
            self.logger.debug(f"Subsetting to remove labels: {labels_to_remove}")
            if isinstance(labels_to_remove, (list, np.ndarray, pd.Index)):
                indices_to_remove = [label for label in labels_to_remove if label in self.labels]
                if len(indices_to_remove) == 0:
                    raise ValueError("No matching labels found to remove in subset.")
                keep_indices = [label for label in self.labels if label not in indices_to_remove]
            else:
                raise ValueError("'labels_to_remove' must be a list, numpy array, or Pandas Index.")
        else:
            raise ValueError("Either 'labels_to_keep' or 'labels_to_remove' must be specified in subset_params.")

        subset_labels = np.array(keep_indices)
        subset_given = self.prepared_before_subset_given[np.ix_(keep_indices, keep_indices)]
        subset_inferred = self.prepared_before_subset_inferred[np.ix_(keep_indices, keep_indices)]

        self.logger.debug(f"Subset given adjacency matrix shape: {subset_given.shape}")
        self.logger.debug(f"Subset inferred adjacency matrix shape: {subset_inferred.shape}")
        return subset_given, subset_inferred, subset_labels

    def _prepare_after_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the trajectories after subsetting. `_prepare_before_subset` is used instead.

        Raises:
            NotImplementedError: This method is not supposed to be running.
        """
        raise NotImplementedError("This method is not supposed to be running.")


# Note: as an alternative, one may try creating another class accepting two adjacency matrix just like the one above,
# but converting them both using the coververters into the pseudotime. Then, the plan is to use pseudotime metrics.
