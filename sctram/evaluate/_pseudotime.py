#!/usr/bin/env python3

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sctram.evaluate._base import EvaluationBase
from sctram.evaluate._converters._adjacency2pseudotime import LabelAdjacencyPseudotimeConverter
from sctram.evaluate._metricsmixin._pseudotimecategoricalmetricsmixin import PseudotimeCategoricalMetricsMixin
from sctram.evaluate._metricsmixin._pseudotimevaluesmetricsmixin import PseudotimeValuesMetricsMixin
from sctram.utils._constants import sctram_operate_key
from sctram.utils._utils import Utils


class PseudotimeEvaluationBase(EvaluationBase):
    """Evaluation method to compare inferred pseudotime with a given trajectory graph.

    There is two subclasses: PseudotimeValuesEvaluation and PseudotimeCategoricalEvaluation. They differ in how the
    final data prep is done and the metrics.
    """

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the pseudotime evaluation method."""  # noqa
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=None,
            prepare_params_after_subset=prepare_params,
        )
        # Prepare the input/inferred regardless of the prepare params, as there is no real parameter
        self.prepare_params_after_subset[sctram_operate_key] = True
        self.logger.debug(f"Initialized PseudotimeEvaluation with metrics: {self.metrics}")

    def get_result(self) -> Any:
        """Retrieves the result of the trajectory evaluation."""
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result

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

    def _verify_labels_data_specific(self):
        """Verifies that labels is consistent with input trajectory and/or inferred trajectory.

        The labels for the pseudotime evaluation should be with the same size as the number of data points.

        Raises:
            ValueError: there is inconsistency.
        """
        self.logger.debug("Checking the consistency between the given graph and labels.")
        if len(self.inferred_trajectory) != len(self.labels):
            raise ValueError("Datapoint amount in the inferred trajectory does not match the number of given labels.")

    def _prepare_before_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the trajectories after subsetting. `_prepare_after_subset` is used instead.

        Raises:
            NotImplementedError: This method is not supposed to be running.
        """
        raise NotImplementedError("This method is not supposed to be running.")

    def _subset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subsets the trajectories based on `subset_params`.

        For pseudotime array, subsetting typically involves selecting a subset of labels for the given graph and
        also subseting pseudotime array based on the same labels.
        This method supports subsetting by specifying either labels to keep or labels to remove.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The subsetted given and inferred adjacency matrices.

        Raises:
            ValueError: If subsetting parameters are invalid or result in incompatible matrices.
        """
        self.logger.debug("Subsetting trajectories based on subset parameters.")
        labels_to_keep = Utils.sget(
            dictionary=self.subset_params, key="labels_to_keep", default=None, logger=self.logger
        )
        labels_to_remove = Utils.sget(
            dictionary=self.subset_params, key="labels_to_remove", default=None, logger=self.logger
        )
        unique_labels = np.unique(self.labels)

        if labels_to_keep is not None and labels_to_remove is None:
            self.logger.debug(f"Subsetting to keep nodes: {labels_to_keep}")
            if isinstance(labels_to_keep, (list, np.ndarray, pd.Index)):
                keep_indices = [label for label in labels_to_keep if label in unique_labels]
                if len(keep_indices) == 0:
                    raise ValueError("No matching labels found to keep in subset.")
            else:
                raise ValueError("'labels_to_keep' must be a list, numpy array, or Pandas Index.")
        elif labels_to_remove is not None and labels_to_keep is None:
            self.logger.debug(f"Subsetting to remove labels: {labels_to_remove}")
            if isinstance(labels_to_remove, (list, np.ndarray, pd.Index)):
                indices_to_remove = [label for label in labels_to_remove if label in unique_labels]
                if len(indices_to_remove) == 0:
                    raise ValueError("No matching labels found to remove in subset.")
                keep_indices = [label for label in unique_labels if label not in indices_to_remove]
            else:
                raise ValueError("'labels_to_remove' must be a list, numpy array, or Pandas Index.")
        else:
            raise ValueError("Either 'labels_to_keep' or 'labels_to_remove' must be specified in subset_params.")

        bool_array = np.isin(self.labels, keep_indices)
        subset_given = self.prepared_before_subset_given.subgraph(keep_indices).copy()
        subset_inferred = self.prepared_before_subset_inferred[bool_array]
        subset_labels = self.labels[bool_array]

        self.logger.debug(f"Subset given adjacency matrix shape: {subset_given.shape}")
        self.logger.debug(f"Subset inferred adjacency matrix shape: {subset_inferred.shape}")
        return subset_given, subset_inferred, subset_labels

        # TODO: uniformize subsetting method across adjacency, pseudotime and embedding.

    @abstractmethod
    def _prepare_after_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """This need to be implemented in the subclasses.

        Raises:
            NotImplementedError: This is abstract method.
        """
        raise NotImplementedError(
            "This method should be implemented in 'PseudotimeCategoricalEvaluation' or 'PseudotimeValuesEvaluation'."
        )


class PseudotimeCategoricalEvaluation(PseudotimeCategoricalMetricsMixin, PseudotimeEvaluationBase):
    """Evaluation method to compare inferred pseudotime with a given trajectory graph.

    The data prep includes conversion of user defined adjacency matrix into an array of strings composed of labels
    using `...` class. At the end of the data prep, two numpy array will be obtained: one is numerical the other is
    categorical. The metrics are desingned to compare these two arrays.
    """

    def _prepare_after_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the data after subsetting."""
        # Convert subset_given (In) to adjacency matrix
        self.logger.debug("Converting `InputTrajectory` to adjacency matrix.")
        raise NotImplementedError("This class of metrics is not implemented yet.")
        # TODO: implement categorical pseudotime evaluation.


class PseudotimeValuesEvaluation(PseudotimeValuesMetricsMixin, PseudotimeEvaluationBase):
    """Evaluation method to compare inferred pseudotime with a given trajectory graph.

    The data prep includes conversion of user defined adjacency matrix into pseudotime
    values using `LabelAdjacencyPseudotimeConverter` class. At the end of the data prep, two numpy array
    will be obtained and both will be composed of floating numbers. The metrics are desingned to compare
    two numerical arrays.

    This class compares the inferred pseudotime values (1D array) with the given trajectory,
    which is represented as a graph over cell types (`InputTrajectories`).
    The comparison is facilitated through the cell type labels associated with each data point.
    Multiple metrics are used to assess the similarity or agreement between the inferred pseudotime
    and the progression implied by the given trajectory.

    Metrics include both basic statistical measures and relatively advanced methods:
    - Basic Metrics: Pearson correlation, Spearman correlation, Kendall's tau, MSE, MAE, R², Concordance Index.
    - Other Metrics: Dynamic Time Warping, Wasserstein distance, Mutual Information, Cumulative density difference.

    Each metric provides different insights into the relationship between the inferred and reference pseudotime,
    capturing aspects such as linear correlation, rank correlation, error magnitude, ordering consistency,
    distribution similarity, and more.
    """

    def _prepare_after_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the data after subsetting.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The prepared reference pseudotime and inferred pseudotime.

        Raises:
            ValueError: `label_pseudotime` is not given correctly.
            TypeError: `label_pseudotime` is not given correctly.
        """
        # Convert subset_given (In) to adjacency matrix
        self.logger.debug("Converting `InputTrajectory` to adjacency matrix.")
        unique_labels = np.unique(self.subset_labels)
        subset_adjacency_matrix = Utils.adjacency_graph_to_matrix(
            g=self.subset_given, nodelist_filter_and_order=unique_labels
        )
        self.logger.debug("Initializing `LabelAdjacencyPseudotimeConverter`.")
        converter = LabelAdjacencyPseudotimeConverter(
            label_adjacency_matrix=subset_adjacency_matrix,
            label_adjacency_matrix_labels=unique_labels,
            cell_labels=self.subset_labels,
        )

        # Retrieve pseudotime computation parameters from prepare_params_after_subset
        converter_parameters = Utils.sget(
            dictionary=self.prepare_params_after_subset, key="converter", default=dict(), logger=self.logger
        )
        method = Utils.sget(
            dictionary=converter_parameters, key="method", default="diffusion_with_damping", logger=self.logger
        )
        handle_disconnected = Utils.sget(
            dictionary=converter_parameters,
            key="handle_disconnected",
            default="assign_max_plus_one",
            logger=self.logger,
        )
        alternative_distance = Utils.sget(
            dictionary=converter_parameters, key="alternative_distance", default=None, logger=self.logger
        )
        # Needs root_label
        root_label = Utils.sget(dictionary=converter_parameters, key="root_label", default=None, logger=self.logger)
        root_label_index = np.where(unique_labels == root_label)[0][0] if root_label is not None else 0
        # Extract additional method-specific parameters
        method_specific_params = {
            k: v
            for k, v in converter_parameters.items()
            if k not in {"method", "handle_disconnected", "alternative_distance", "root_label"}
        }
        self.logger.debug(f"Computing reference pseudotime using method: {method}")
        label_pseudotime = converter.get_label_pseudotime(
            method=method,
            handle_disconnected=handle_disconnected,
            alternative_distance=alternative_distance,
            root_label_index=root_label_index,
            **method_specific_params,
        )
        if not isinstance(label_pseudotime, np.ndarray):
            raise TypeError("Label pseudotime must be a numpy array.")
        if label_pseudotime.ndim != 1:
            raise ValueError("Label pseudotime must be a one-dimensional array.")
        if label_pseudotime.size != len(unique_labels):
            raise ValueError("Label pseudotime size does not match the number of unique subset labels.")
        self.logger.debug("Computed label pseudotime successfully.")

        self.logger.debug("Assigning pseudotime to cells based on label pseudotime.")
        cell_pseudotime = converter.assign_cell_pseudotime(
            handle_disconnected=handle_disconnected, alternative_distance=alternative_distance
        )

        # Validate inferred pseudotime size
        if self.subset_inferred.size != cell_pseudotime.size:
            raise ValueError("Inferred pseudotime size does not match the number of cells in the subset.")

        # Normalize the reference and inferred pseudotime to [0, 1]
        self.logger.debug("Normalizing reference and inferred pseudotime to [0, 1].")
        scaler = MinMaxScaler()
        reference_normalized = scaler.fit_transform(cell_pseudotime.reshape(-1, 1)).flatten()
        inferred_normalized = scaler.fit_transform(self.subset_inferred.reshape(-1, 1)).flatten()
        self.logger.debug("Normalized pseudotime arrays successfully.")

        return reference_normalized, inferred_normalized


# Note: as an alternative, one may try creating another class accepting the inputs like the one above,
# but converting inferred pseutotime into adjacency matrix. Then, the plan is to use adjacency metrics.
