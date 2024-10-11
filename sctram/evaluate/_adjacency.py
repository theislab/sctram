#!/usr/bin/env python3

import logging
from typing import Any, Dict, Optional, Union, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sctram.evaluate._base import EvaluationBase


class AdjacencyMatrixEvaluation(EvaluationBase):
    """Evaluation method to compare given and inferred adjacency matrices.
    
    This class compares the adjacency matrix of the given trajectory (converted from a networkx graph)
    with the inferred adjacency matrix using specified metrics such as the Frobenius norm, L1 norm, or accuracy.
    """

    def __init__(
        self,
        method_params: Optional[Dict[str, Any]] = None,
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the adjacency matrix evaluation method.

        Args:
            method_params (Optional[Dict[str, Any]]): Parameters for the evaluation method.
                Expected keys:
                - 'metric': The metric to use for comparison. Options are 'frobenius', 'L1', 'accuracy'.
            subset_params (Optional[Dict[str, Any]]): Parameters for subsetting the data.
            prepare_params_before_subset (Optional[Dict[str, Any]]): Parameters for preparing the data before subsetting.
            prepare_params_after_subset (Optional[Dict[str, Any]]): Parameters for preparing the data after subsetting.
        """
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=prepare_params_before_subset,
            prepare_params_after_subset=prepare_params_after_subset,
        )
        self.metric = self.method_params.get('metric', 'frobenius')
        self.logger.debug(f"Initialized AdjacencyMatrixEvaluation with metric: {self.metric}")

    def _verify_inferred_trajectory(self, inferred: Any) -> Any:
        """Verifies the inferred trajectory.

        Args:
            inferred (Any): The inferred trajectory.

        Returns:
            Any: The verified inferred trajectory.

        Raises:
            ValueError: If the inferred trajectory is not in an acceptable format.
        """
        if isinstance(inferred, (nx.Graph, nx.MultiDiGraph)):
            self.logger.debug("Inferred trajectory is a networkx graph.")
            return inferred
        elif isinstance(inferred, (np.ndarray, pd.DataFrame)):
            self.logger.debug("Inferred trajectory is an adjacency matrix (numpy array or pandas DataFrame).")
            return inferred
        else:
            raise ValueError(
                "Inferred trajectory must be a networkx graph, numpy array, or pandas DataFrame representing an adjacency matrix."
            )

    def _prepare_before_subset(
        self, given_trajectory: Any, inferred_trajectory: Any
    ) -> Tuple[Any, Any]:
        """Prepares the trajectories before subsetting by converting them to adjacency matrices.

        Args:
            given_trajectory (Any): The given trajectory.
            inferred_trajectory (Any): The inferred trajectory.

        Returns:
            Tuple[Any, Any]: Prepared given and inferred adjacency matrices.

        Raises:
            ValueError: If the adjacency matrices cannot be prepared due to incompatible shapes or types.
        """
        self.logger.debug("Converting given trajectory to adjacency matrix.")
        # Convert the given trajectory (networkx graph) to adjacency matrix
        given_adj_matrix = nx.to_numpy_array(given_trajectory)
        self.logger.debug(f"Given adjacency matrix shape: {given_adj_matrix.shape}")

        self.logger.debug("Ensuring inferred trajectory is an adjacency matrix.")
        # Convert inferred trajectory to adjacency matrix if it's a graph
        if isinstance(inferred_trajectory, (nx.Graph, nx.MultiDiGraph)):
            inferred_adj_matrix = nx.to_numpy_array(inferred_trajectory)
            self.logger.debug(f"Inferred adjacency matrix converted from graph with shape: {inferred_adj_matrix.shape}")
        elif isinstance(inferred_trajectory, np.ndarray):
            inferred_adj_matrix = inferred_trajectory
            self.logger.debug(f"Inferred adjacency matrix is a numpy array with shape: {inferred_adj_matrix.shape}")
        elif isinstance(inferred_trajectory, pd.DataFrame):
            inferred_adj_matrix = inferred_trajectory.values
            self.logger.debug(f"Inferred adjacency matrix is a pandas DataFrame with shape: {inferred_adj_matrix.shape}")
        else:
            raise ValueError(
                "Inferred trajectory must be a networkx graph, numpy array, or pandas DataFrame representing an adjacency matrix."
            )

        # Ensure that both matrices have the same shape
        if given_adj_matrix.shape != inferred_adj_matrix.shape:
            raise ValueError("Given and inferred adjacency matrices must have the same shape for comparison.")

        self.logger.debug("Assigned prepared adjacency matrices before subsetting.")
        return given_adj_matrix, inferred_adj_matrix

    def _subset(
        self, given_trajectory: Any, inferred_trajectory: Any
    ) -> Tuple[Any, Any]:
        """Subsets the trajectories based on `subset_params`.

        For adjacency matrices, subsetting typically involves selecting a subset of nodes (rows and columns).

        Args:
            given_trajectory (Any): The prepared given adjacency matrix before subsetting.
            inferred_trajectory (Any): The prepared inferred adjacency matrix before subsetting.

        Returns:
            Tuple[Any, Any]: The subsetted given and inferred adjacency matrices.

        Raises:
            ValueError: If subsetting parameters are invalid.
        """
        if not self.subset_params:
            self.logger.debug("No subset parameters provided. Returning original trajectories.")
            return given_trajectory, inferred_trajectory

        self.logger.debug("Subsetting trajectories based on subset parameters.")
        nodes_to_keep = self.subset_params.get("nodes_to_keep", None)
        nodes_to_remove = self.subset_params.get("nodes_to_remove", None)

        if nodes_to_keep is not None:
            self.logger.debug(f"Subsetting to keep nodes: {nodes_to_keep}")
            if isinstance(nodes_to_keep, (list, np.ndarray, pd.Index)):
                indices = [i for i, node in enumerate(given_trajectory) if node in nodes_to_keep]
                if not indices:
                    raise ValueError("No matching nodes found to keep in subset.")
                subset_given = given_trajectory[np.ix_(indices, indices)]
                subset_inferred = inferred_trajectory[np.ix_(indices, indices)]
            else:
                raise ValueError("'nodes_to_keep' must be a list, numpy array, or pandas Index.")
        elif nodes_to_remove is not None:
            self.logger.debug(f"Subsetting to remove nodes: {nodes_to_remove}")
            if isinstance(nodes_to_remove, (list, np.ndarray, pd.Index)):
                total_nodes = given_trajectory.shape[0]
                all_indices = set(range(total_nodes))
                remove_indices = set()
                for node in nodes_to_remove:
                    if node in given_trajectory:
                        remove_indices.add(list(given_trajectory).index(node))
                keep_indices = sorted(all_indices - remove_indices)
                if not keep_indices:
                    raise ValueError("All nodes are removed in subset.")
                subset_given = given_trajectory[np.ix_(keep_indices, keep_indices)]
                subset_inferred = inferred_trajectory[np.ix_(keep_indices, keep_indices)]
            else:
                raise ValueError("'nodes_to_remove' must be a list, numpy array, or pandas Index.")
        else:
            raise ValueError("Either 'nodes_to_keep' or 'nodes_to_remove' must be specified in subset_params.")

        self.logger.debug(f"Subset given adjacency matrix shape: {subset_given.shape}")
        self.logger.debug(f"Subset inferred adjacency matrix shape: {subset_inferred.shape}")
        return subset_given, subset_inferred

    def _prepare_after_subset(
        self, subset_given: Any, subset_inferred: Any
    ) -> Tuple[Any, Any]:
        """Prepares the trajectories after subsetting.

        For adjacency matrices, this might involve normalization or other post-processing steps.

        Args:
            subset_given (Any): The subsetted given adjacency matrix.
            subset_inferred (Any): The subsetted inferred adjacency matrix.

        Returns:
            Tuple[Any, Any]: The prepared given and inferred adjacency matrices after subsetting.
        """
        self.logger.debug("Preparing adjacency matrices after subsetting (no additional preparation).")
        # If no additional preparation is needed after subsetting, simply return the subsetted matrices
        return subset_given, subset_inferred

    def _calculate(self):
        """Performs the evaluation by comparing the adjacency matrices using the specified metric."""
        given_adj_matrix, inferred_adj_matrix = self.prepared_after_subset_given, self.prepared_after_subset_inferred

        self.logger.debug(f"Calculating difference between adjacency matrices using metric '{self.metric}'.")

        if self.metric == 'frobenius':
            # Compute the Frobenius norm of the difference
            diff_matrix = given_adj_matrix - inferred_adj_matrix
            self.result = np.linalg.norm(diff_matrix, 'fro')
            self.logger.debug(f"Frobenius norm of difference: {self.result}")
        elif self.metric == 'L1':
            # Compute the L1 norm of the difference
            diff_matrix = given_adj_matrix - inferred_adj_matrix
            self.result = np.sum(np.abs(diff_matrix))
            self.logger.debug(f"L1 norm of difference: {self.result}")
        elif self.metric == 'accuracy':
            # Compute the proportion of matching edges (binary adjacency matrices assumed)
            total_elements = given_adj_matrix.size
            matching_elements = np.sum(given_adj_matrix == inferred_adj_matrix)
            self.result = matching_elements / total_elements
            self.logger.debug(f"Accuracy of adjacency matrices: {self.result}")
        else:
            raise ValueError(
                f"Unknown metric '{self.metric}' specified. Supported metrics are 'frobenius', 'L1', 'accuracy'."
            )

    def get_result(self) -> Any:
        """Retrieves the result of the trajectory evaluation.

        Returns:
            Any: The result of the evaluation.

        Raises:
            ValueError: If the result is not available.
        """
        if self.result is None:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result